#!/usr/bin/env python
"""CRE pretraining, linear probing, fine-tuning and evaluation entry point."""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from cre.data import build_dataset
from cre.model import CRE, Classifier, VQTokenizer


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pretrain', 'linear', 'finetune', 'evaluate'])
    parser.add_argument('--data', required=True, help='Training image root / ImageFolder')
    parser.add_argument('--train-list', help='Optional path-label training manifest')
    parser.add_argument('--val-data', help='Validation image root / ImageFolder')
    parser.add_argument('--val-list', help='Optional validation manifest')
    parser.add_argument('--vqgan-source', required=True, help='Local MAGE checkout')
    parser.add_argument('--vqgan-config', required=True)
    parser.add_argument('--vqgan-checkpoint', required=True)
    parser.add_argument('--pretrained', help='CRE pretraining checkpoint for downstream tasks')
    parser.add_argument('--resume', help='Full checkpoint for resuming or evaluation')
    parser.add_argument('--output-dir', default='outputs/cre')
    parser.add_argument('--num-classes', type=int, default=199)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--warmup-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None, help='Global batch across ranks')
    parser.add_argument('--base-lr', type=float, default=None, help='Scaled by global batch / 256')
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--mask-ratio', type=float, default=.55)
    parser.add_argument('--contrastive-weight', type=float, default=.2)
    parser.add_argument('--temperature', type=float, default=.2)
    parser.add_argument('--projection-dim', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help='CUDA mixed precision')
    args = parser.parse_args()
    pretraining = args.mode == 'pretrain'
    defaults = {'epochs': 1600 if pretraining else 100,
                'warmup_epochs': 80 if pretraining else 5,
                'batch_size': 2048 if pretraining else 256,
                'base_lr': 1.5e-4 if pretraining else 1e-3}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error('Require epochs > 0 and 0 <= warmup-epochs < epochs')
    if args.batch_size <= 0 or args.num_classes <= 0 or args.workers < 0:
        parser.error('Invalid batch size, class count or worker count')
    if args.mode in ('linear', 'finetune') and not (args.pretrained or args.resume):
        parser.error('Downstream training requires --pretrained or --resume')
    if args.mode == 'evaluate' and not args.resume:
        parser.error('Evaluation requires a downstream --resume checkpoint')
    if not pretraining and not args.val_data:
        parser.error('Downstream tasks require --val-data')
    return args


def load_checkpoint(path):
    # Load only checkpoints from trusted sources (optimizer/RNG state uses pickle).
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, device=device, dtype=torch.long)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(tokenizer(images))
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum')
        pairs = labels * num_classes + logits.argmax(dim=1)
        confusion += torch.bincount(pairs, minlength=num_classes ** 2).reshape_as(confusion)
    if dist.is_initialized():
        dist.all_reduce(confusion)
        dist.all_reduce(loss_sum)
    tp = confusion.diag().float()
    support, predicted = confusion.sum(1), confusion.sum(0)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    return {'loss': (loss_sum / confusion.sum()).item(),
            'accuracy': (tp.sum() / confusion.sum()).item(),
            'macro_f1': f1.mean().item(),
            'weighted_f1': ((f1 * support).sum() / confusion.sum()).item()}


def main():
    args = arguments()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    if args.batch_size % world:
        raise ValueError('Global batch size must be divisible by world size')
    batch_size = args.batch_size // world
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    tokenizer = VQTokenizer(args.vqgan_source, args.vqgan_config,
                            args.vqgan_checkpoint).to(device)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    source = checkpoint or (load_checkpoint(args.pretrained) if args.pretrained else None)
    config = source['model_config'] if source else {
        'vocab_size': tokenizer.vocab_size, 'mask_ratio': args.mask_ratio,
        'contrastive_weight': args.contrastive_weight, 'temperature': args.temperature,
        'projection_dim': args.projection_dim}
    if tokenizer.vocab_size != config['vocab_size']:
        raise ValueError('Tokenizer vocabulary does not match the CRE checkpoint')
    pretraining = args.mode == 'pretrain'
    if checkpoint and ((checkpoint['mode'] == 'pretrain') != pretraining):
        raise ValueError('--resume must match task type; use --pretrained for transfer')
    if checkpoint and args.mode != 'evaluate' and checkpoint['mode'] != args.mode:
        raise ValueError('--resume must match the saved training mode')
    model = CRE(**config)
    if not pretraining:
        if source and not checkpoint:
            if source['mode'] != 'pretrain':
                raise ValueError('--pretrained must be a CRE pretraining checkpoint')
            model.load_state_dict(source['model'], strict=True)
        probe = (checkpoint['mode'] == 'linear') if checkpoint else args.mode == 'linear'
        model = Classifier(model, args.num_classes, linear_probe=probe)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    validation = None
    if not pretraining:
        validation = build_dataset(args.val_data, args.val_list, training=False)
        if max(validation.targets) >= args.num_classes:
            raise ValueError('Validation label exceeds --num-classes')
        if checkpoint and checkpoint.get('class_to_idx') is not None:
            if getattr(validation, 'class_to_idx', None) != checkpoint['class_to_idx']:
                raise ValueError('Validation class mapping differs from saved checkpoint')
        # No padding: every validation sample is evaluated exactly once.
        val_loader = DataLoader(validation, batch_size=batch_size,
                                sampler=range(rank, len(validation), world),
                                num_workers=args.workers, pin_memory=device.type == 'cuda')
    if args.mode == 'evaluate':
        metrics = evaluate(model, tokenizer, val_loader, device, args.num_classes)
        if rank == 0:
            print(json.dumps(metrics))
        if world > 1:
            dist.destroy_process_group()
        return
    training = build_dataset(args.data, args.train_list, pretraining=pretraining)
    class_to_idx = getattr(training, 'class_to_idx', None)
    if checkpoint and checkpoint.get('class_to_idx') != class_to_idx:
        raise ValueError('Training class mapping differs from saved checkpoint')
    if validation is not None:
        if max(training.targets) >= args.num_classes:
            raise ValueError('Training label exceeds --num-classes')
        if class_to_idx != getattr(validation, 'class_to_idx', None):
            raise ValueError('Train/validation class mappings differ; use consistent manifests')
    sampler = DistributedSampler(training, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.workers, drop_last=True,
                        pin_memory=device.type == 'cuda')
    if not len(loader):
        raise ValueError('Training split is smaller than one batch; reduce --batch-size')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.base_lr * args.batch_size / 256,
                                  betas=(.9, .95), weight_decay=args.weight_decay)
    amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
    raw_model = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'args.json').write_text(json.dumps(vars(args), indent=2))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if sampler:
            sampler.set_epoch(epoch)
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for step, (images, labels) in enumerate(loader):
            progress = epoch + step / len(loader)
            factor = (progress / args.warmup_epochs if progress < args.warmup_epochs else
                      .5 * (1 + math.cos(math.pi * (progress - args.warmup_epochs) /
                                          (args.epochs - args.warmup_epochs))))
            for group in optimizer.param_groups:
                group['lr'] = args.base_lr * args.batch_size / 256 * factor
            optimizer.zero_grad(set_to_none=True)
            if pretraining:
                tokens = [tokenizer(view.to(device, non_blocking=True)) for view in images]
            else:
                tokens = tokenizer(images.to(device, non_blocking=True))
                labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if pretraining:
                    result = model(*tokens)
                    loss = result['loss']
                else:
                    loss = nn.functional.cross_entropy(model(tokens), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss at epoch {}, step {}'.format(epoch, step))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals[0] += loss.detach()
            if pretraining:
                totals[1] += result['reconstruction']
                totals[2] += result['contrastive']
            if rank == 0 and step % 20 == 0:
                print('epoch={} step={}/{} loss={:.5f}'.format(epoch, step, len(loader), loss.item()), flush=True)
        if world > 1:
            dist.all_reduce(totals)
        metrics = dict(zip(['loss', 'reconstruction', 'contrastive'],
                           (totals / (len(loader) * world)).tolist()))
        if validation is not None:
            # Bypass DDP forward: validation ranks may have different batch counts.
            metrics['validation'] = evaluate(raw_model, tokenizer, val_loader, device, args.num_classes)
        if rank == 0:
            state = {'model': raw_model.state_dict(), 'model_config': config,
                     'mode': args.mode, 'epoch': epoch, 'args': vars(args),
                     'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                     'class_to_idx': class_to_idx}
            temporary = output / 'checkpoint.tmp'
            torch.save(state, temporary)
            temporary.replace(output / 'checkpoint-last.pth')
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch': epoch, **metrics}) + '\n')
            print(json.dumps({'epoch': epoch, **metrics}), flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
