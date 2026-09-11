# GPID-22: Self-Supervised Plant Pest and Disease Recognition

**Self-supervised transformer-based pre-training method with General Plant Infection dataset**

Zhengle Wang, Ruifeng Wang, Minjuan Wang, Tianyun Lai, and Man Zhang

[Paper](https://arxiv.org/abs/2407.14911) · [PDF](https://arxiv.org/pdf/2407.14911)

## Overview

GPID-22 is a plant pest and disease recognition dataset covering 22 plant species and 199 classes. The accompanying CRE framework combines masked semantic-token reconstruction with contrastive learning to learn representations for downstream classification.

This repository provides a PyTorch implementation with entry points for self-supervised pre-training, linear probing, fine-tuning, and evaluation.

## Method

CRE processes two independently augmented views of each image through a frozen VQGAN tokenizer. A Vision Transformer encodes visible semantic tokens, and a Transformer decoder predicts the masked token IDs. A two-layer projection head maps globally pooled encoder features into a contrastive embedding space.

The training objective is:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{reconstruction}} + \lambda\mathcal{L}_{\mathrm{contrastive}}, \qquad \lambda = 0.2.
$$

Reconstruction uses cross-entropy over masked positions. Contrastive learning uses normalized embeddings and symmetric InfoNCE with negatives gathered across distributed workers.

| Component | Configuration |
| --- | --- |
| Input resolution | 256 × 256 |
| Semantic token grid | 16 × 16 |
| Encoder | ViT-B: 12 layers, 768 dimensions, 12 attention heads |
| Decoder | 8 layers, 512 dimensions, 16 attention heads |
| Mask ratio | 0.55 |
| Projection head | 768 → 4096 → 256 |
| Contrastive temperature | 0.2 |
| Contrastive loss weight | 0.2 |

The decoder configuration, projection dimensions, temperature, and downstream optimization defaults are implementation choices where the paper does not specify a complete configuration. The encoder uses learnable positional embeddings and global average pooling. Transformer checkpoints from MAGE or MoCo are not directly interchangeable with CRE checkpoints.

## Installation

Use Python 3.10 and install the dependencies:

```bash
git clone https://github.com/WASSER2545/GPID-22.git
cd GPID-22
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cre.txt
```

For GPU execution, install PyTorch and torchvision builds compatible with your CUDA environment.

### VQGAN tokenizer

Obtain the MAGE source and its pretrained VQGAN tokenizer:

```bash
mkdir -p third_party checkpoints
git clone https://github.com/LTH14/mage.git third_party/mage
```

Download the **VQGAN tokenizer checkpoint** linked in the [MAGE preparation instructions](https://github.com/LTH14/mage#preparation) and save it as `checkpoints/vqgan.ckpt`. Use the matching configuration at `third_party/mage/config/vqgan.yaml`.

The tokenizer remains frozen during all stages. Use the same tokenizer configuration and weights for pre-training and downstream tasks. CRE checkpoints store the representation model separately from the tokenizer.

## Dataset

The paper describes GPID-22 as containing 205,371 images across 199 classes, with 164,700 pre-training images and 40,671 test images. Images, split manifests, and pretrained CRE checkpoints are not bundled with this repository.

### Directory format

Organize labeled images using the ImageFolder convention:

```text
data/GPID-22/
├── train/
│   ├── class_000/
│   ├── class_001/
│   └── ...
└── val/
    ├── class_000/
    ├── class_001/
    └── ...
```

Training and validation directories must share the same class-to-index mapping. Use a separate validation split when selecting hyperparameters and reserve the test split for final evaluation.

### Split manifests

Alternatively, supply UTF-8 manifests containing one image path and zero-based integer label per line:

```text
images/tomato/001.jpg 0
images/rice/002.jpg 1
```

Paths are relative to `--data` or `--val-data`. Pass `--train-list splits/train.txt` and `--val-list splits/val.txt` to use manifests. Labels must have consistent meanings across splits. The loader uses the supplied splits without generating a random partition.

## Usage

Define the shared tokenizer arguments in Bash:

```bash
TOKENIZER_ARGS=(
  --vqgan-source third_party/mage
  --vqgan-config third_party/mage/config/vqgan.yaml
  --vqgan-checkpoint checkpoints/vqgan.ckpt
)
```

### Self-supervised pre-training

```bash
torchrun --standalone --nproc_per_node=8 main_cre.py pretrain \
  --data data/GPID-22/train \
  "${TOKENIZER_ARGS[@]}" \
  --batch-size 2048 --epochs 1600 --warmup-epochs 80 \
  --base-lr 0.00015 --mask-ratio 0.55 --contrastive-weight 0.2 \
  --amp --output-dir outputs/cre-pretrain
```

Pre-training uses AdamW and cosine learning-rate decay after warmup. The learning rate is scaled as `base_lr × global_batch_size / 256`.

`--batch-size` denotes the global batch across all workers and must be divisible by the world size. Adjust it to available device memory; changing it also changes the contrastive negative pool. For a single process, use `python main_cre.py` instead of `torchrun`.

Set `--contrastive-weight 0` to train the reconstruction-only ablation.

### Linear probing

```bash
python main_cre.py linear \
  --data data/GPID-22/train --val-data data/GPID-22/val \
  "${TOKENIZER_ARGS[@]}" --num-classes 199 \
  --pretrained outputs/cre-pretrain/checkpoint-last.pth \
  --batch-size 256 --output-dir outputs/cre-linear
```

Linear probing freezes the representation encoder and trains a linear classification head on globally pooled, unmasked token features.

### Fine-tuning

```bash
python main_cre.py finetune \
  --data data/GPID-22/train --val-data data/GPID-22/val \
  "${TOKENIZER_ARGS[@]}" --num-classes 199 \
  --pretrained outputs/cre-pretrain/checkpoint-last.pth \
  --batch-size 256 --output-dir outputs/cre-finetune
```

Fine-tuning updates the encoder and classification head while keeping the tokenizer frozen. Set `--num-classes` to match the downstream dataset. Both downstream training modes support `torchrun`.

### Evaluation

```bash
python main_cre.py evaluate \
  --data data/GPID-22/train --val-data data/GPID-22/val \
  "${TOKENIZER_ARGS[@]}" --num-classes 199 \
  --resume outputs/cre-finetune/checkpoint-last.pth
```

Evaluation reports top-1 accuracy, macro-F1, and weighted-F1 as values in `[0, 1]`. Distributed evaluation aggregates the confusion matrix without duplicating samples. The shared CLI requires `--data`, but evaluation only reads `--val-data`.

### Checkpoints and resuming

Each training run writes:

- `checkpoint-last.pth`: model, optimizer, AMP scaler, epoch, and class mapping.
- `metrics.jsonl`: per-epoch losses and downstream validation metrics.
- `args.json`: command-line configuration.

Add `--resume outputs/<run>/checkpoint-last.pth` to the corresponding training command to resume from the next epoch. Keep the same tokenizer, data mapping, and optimization settings. The schedule follows the current command-line configuration; per-worker random states are not restored.

## Repository Structure

```text
cre/
├── data.py              # Dataset loading and image augmentation
├── model.py             # Tokenizer, CRE model, and classifier
└── __init__.py
main_cre.py              # Pre-training and downstream entry point
requirements-cre.txt     # Python dependencies
LICENSE
```

## Citation

If you use GPID-22 or this work in your research, please cite:

```bibtex
@article{wang2024selfsupervised,
  title={Self-supervised transformer-based pre-training method with General Plant Infection dataset},
  author={Wang, Zhengle and Wang, Ruifeng and Wang, Minjuan and Lai, Tianyun and Zhang, Man},
  journal={arXiv preprint arXiv:2407.14911},
  year={2024}
}
```

## Acknowledgments

This work builds on [MAGE](https://github.com/LTH14/mage), [VQGAN](https://github.com/CompVis/taming-transformers), and [SimCLR](https://github.com/google-research/simclr). The VQGAN tokenizer is loaded from the MAGE implementation.

## License

See [LICENSE](LICENSE) for the repository license. External dependencies and pretrained tokenizer assets are subject to their respective licenses.
