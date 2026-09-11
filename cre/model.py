"""Paper sections 2.2–3.1; implementation choices are documented in README.md."""
import importlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
import yaml


class VQTokenizer(nn.Module):
    """Frozen MAGE VQGAN. Input RGB tensors must be in [-1, 1]."""

    def __init__(self, source, config, checkpoint):
        super().__init__()
        source = Path(source).resolve()
        if not (source / 'taming/models/vqgan.py').is_file():
            raise ValueError('--vqgan-source must point to the MAGE repository')
        sys.path.insert(0, str(source))
        module = importlib.import_module('taming.models.vqgan')
        with open(config) as stream:
            params = yaml.safe_load(stream)['model']['params']
        params = dict(params)
        params.pop('ckpt_path', None)
        self.model = module.VQModel(**params)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state.get('state_dict', state), strict=True)
        self.vocab_size = params['n_embed']
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.shape[1:] != (3, 256, 256):
            raise ValueError('CRE requires RGB images at 256 x 256')
        # Keep nearest-code lookup in float32, including under AMP.
        with torch.autocast(device_type=images.device.type, enabled=False):
            quantized, _, info = self.model.encode(images.float())
        if quantized.shape[-2:] != (16, 16):
            raise ValueError('VQGAN must produce a 16 x 16 semantic-token grid')
        return info[2].reshape(images.shape[0], 256).long()


def transformer(width, depth, heads):
    # Explicit construction gives independent initial weights for every block.
    return nn.Sequential(*[
        nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.,
                                   activation='gelu', batch_first=True,
                                   norm_first=True, layer_norm_eps=1e-6)
        for _ in range(depth)
    ], nn.LayerNorm(width, eps=1e-6))


class CRE(nn.Module):
    def __init__(self, vocab_size, mask_ratio=.55, contrastive_weight=.2,
                 temperature=.2, projection_dim=256):
        super().__init__()
        if not 0 < mask_ratio < 1 or temperature <= 0 or contrastive_weight < 0:
            raise ValueError('Require 0 < mask_ratio < 1, temperature > 0, lambda >= 0')
        self.mask_ratio = mask_ratio
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.embedding = nn.Embedding(vocab_size, 768)
        self.encoder_pos = nn.Parameter(torch.zeros(1, 256, 768))
        self.encoder = transformer(768, 12, 12)
        self.decoder_embed = nn.Linear(768, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.decoder_pos = nn.Parameter(torch.zeros(1, 256, 512))
        self.decoder = transformer(512, 8, 16)
        self.prediction = nn.Linear(512, vocab_size)
        self.projector = nn.Sequential(nn.Linear(768, 4096), nn.ReLU(),
                                       nn.Linear(4096, projection_dim))
        if contrastive_weight == 0:
            self.projector.requires_grad_(False)
        for parameter in (self.encoder_pos, self.decoder_pos, self.mask_token):
            nn.init.normal_(parameter, std=.02)

    def encode(self, tokens):
        """Unmasked global-average features for downstream classification."""
        return self.encoder(self.embedding(tokens) + self.encoder_pos).mean(dim=1)

    def reconstruct(self, tokens):
        batch, length = tokens.shape
        keep = int(length * (1 - self.mask_ratio))
        order = torch.rand(batch, length, device=tokens.device).argsort(dim=1)
        restore = order.argsort(dim=1)
        visible = order[:, :keep]
        embedded = self.embedding(tokens) + self.encoder_pos
        latent = self.encoder(embedded.gather(1, visible[..., None].expand(-1, -1, 768)))
        decoded = torch.cat([self.decoder_embed(latent),
                             self.mask_token.expand(batch, length - keep, -1)], dim=1)
        decoded = decoded.gather(1, restore[..., None].expand(-1, -1, 512))
        logits = self.prediction(self.decoder(decoded + self.decoder_pos))
        mask = restore >= keep
        loss = F.cross_entropy(logits[mask].float(), tokens[mask])
        return loss, latent.mean(dim=1)

    def contrastive_loss(self, first, second):
        """Symmetric SimCLR InfoNCE; self excluded, positive included."""
        first, second = (F.normalize(self.projector(x).float(), dim=-1)
                         for x in (first, second))
        batch = first.shape[0]
        distributed = dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        # Autograd-aware collectives preserve gradients through remote negatives.
        all_first = torch.cat(all_gather(first)) if distributed else first
        all_second = torch.cat(all_gather(second)) if distributed else second
        count = all_first.shape[0]
        if count < 2:
            raise ValueError('Contrastive learning needs at least two images per global batch')
        anchors = torch.cat([first, second])
        candidates = torch.cat([all_first, all_second])
        logits = anchors @ candidates.T / self.temperature
        indices = torch.arange(batch, device=first.device) + rank * batch
        own = torch.cat([indices, indices + count])
        labels = torch.cat([indices + count, indices])
        logits = logits.scatter(1, own[:, None], float('-inf'))
        return F.cross_entropy(logits, labels)

    def forward(self, first, second):
        loss1, features1 = self.reconstruct(first)
        loss2, features2 = self.reconstruct(second)
        reconstruction = (loss1 + loss2) / 2
        contrastive = (self.contrastive_loss(features1, features2)
                       if self.contrastive_weight else reconstruction.new_zeros(()))
        return {'loss': reconstruction + self.contrastive_weight * contrastive,
                'reconstruction': reconstruction.detach(),
                'contrastive': contrastive.detach()}


class Classifier(nn.Module):
    def __init__(self, pretrained, num_classes, linear_probe=False):
        super().__init__()
        # Keep only the encoder; no unused decoder parameters in downstream DDP.
        self.embedding = pretrained.embedding
        self.encoder_pos = pretrained.encoder_pos
        self.encoder = pretrained.encoder
        self.linear_probe = linear_probe
        if linear_probe:
            self.requires_grad_(False)
        self.head = nn.Linear(768, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.linear_probe:
            self.embedding.eval()
            self.encoder.eval()
        return self

    def forward(self, tokens):
        features = self.encoder(self.embedding(tokens) + self.encoder_pos).mean(dim=1)
        return self.head(features)
