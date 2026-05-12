# jepa

Minimal, single-file PyTorch reimplementations of the JEPA family, with paired tutorials.

| File | Method | Dataset | LOC | Tutorial |
|---|---|---|---:|---|
| [`ijepa.py`](./ijepa.py) | I-JEPA | CIFAR-10 | 160 | [`ijepa_tutorial.md`](./ijepa_tutorial.md) |
| [`vjepa.py`](./vjepa.py) | V-JEPA | Moving MNIST | 188 | [`vjepa_tutorial.md`](./vjepa_tutorial.md) |
| [`vjepa2.py`](./vjepa2.py) | V-JEPA 2 + V-JEPA 2-AC | synthetic moving digits | 278 | [`vjepa2_tutorial.md`](./vjepa2_tutorial.md) |
| [`cjepa.py`](./cjepa.py) | C-JEPA | 3-digit bouncing video | 174 | [`cjepa_tutorial.md`](./cjepa_tutorial.md) |

Each algorithm file is **standalone** — only depends on `torch` and `torchvision`, no shared utilities. The matching `<algo>_extras.py` adds visualization (mask grids, loss curves, PCA/LDA/t-SNE evolution, linear probe).

## Quick start

```bash
git clone git@github.com:keon/jepa.git
cd jepa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pinned versions, see below

python ijepa.py                     # train I-JEPA only (no plots)
python ijepa_extras.py              # train + write all visualizations + linear probe
```

Runs on CUDA, MPS, or CPU. CIFAR-10 / MNIST datasets auto-download to `./data/`.

### Reproducibility

The repo pins exact versions in [`requirements.txt`](./requirements.txt) and [`pyproject.toml`](./pyproject.toml):

```
python >= 3.10  (tested on 3.13.5)
torch == 2.11.0
torchvision == 0.26.0
matplotlib == 3.10.9
scikit-learn == 1.8.0   # used by ijepa_extras for t-SNE
numpy == 2.4.4
pillow == 12.2.0
```

Install as a package instead of installing requirements directly:

```bash
pip install -e .
```

## What's where

```
.
├── ijepa.py / ijepa_extras.py           # I-JEPA on CIFAR-10
├── vjepa.py / vjepa_extras.py           # V-JEPA on Moving MNIST
├── vjepa2.py / vjepa2_extras.py         # V-JEPA 2 + V-JEPA 2-AC (synthetic)
├── cjepa.py / cjepa_extras.py           # C-JEPA on 3-digit bouncing video
├── ijepa_tutorial.md                    # walk-throughs that match the code
├── vjepa_tutorial.md
├── vjepa2_tutorial.md
├── cjepa_tutorial.md
├── papers/                              # the four source PDFs
├── samples/                             # mask grids, loss curves, PCA/LDA/t-SNE plots
└── figs/                                # paper figures referenced by tutorials
```

## The methods, in one paragraph each

**I-JEPA** ([Assran et al. 2023](https://arxiv.org/abs/2301.08243)) — predict embeddings of held-out image patches from embeddings of visible patches. EMA target encoder, multi-block masking, smooth-L1 loss. The canonical self-supervised JEPA.

**V-JEPA** ([Bardes et al. 2024](https://arxiv.org/abs/2404.08471)) — same recipe, but 3D tubelet patches over video. Two mask groups (short-range + long-range tubes), L1 loss, EMA 0.998 → 1.0.

**V-JEPA 2** ([Assran et al. 2025](https://arxiv.org/abs/2506.09985)) — two-phase: V-JEPA pretraining followed by **V-JEPA 2-AC**, an action-conditioned predictor trained on frozen-encoder latents with teacher forcing + rollout. The encoder is frozen in phase 2; no EMA.

**C-JEPA** ([Nam et al. 2026](https://arxiv.org/abs/2602.11389)) — object-level trajectory masking with an identity anchor at $t=0$. No EMA. Bidirectional transformer over flattened slot tokens. Built on top of a pretrained object-centric encoder (VideoSAUR in the paper; we use a frozen embedding lookup as a documented stand-in).

## Caveats

These are **educational** reimplementations:

- ViT-tiny, not ViT-Huge. CIFAR-10 / Moving MNIST / synthetic videos, not ImageNet / Kinetics.
- I-JEPA hits **~52.7% linear probe** on CIFAR-10 after 100 epochs. The paper's numbers come from ViT-H/14 on ImageNet for 300 epochs — different planet of compute.
- C-JEPA skips slot discovery (uses oracle positions). Real C-JEPA requires VideoSAUR pretraining (~100k steps) on top of frozen DINOv2 features.
- V-JEPA 2-AC's action-conditioning gap stays small in our toy because the data is too easy; the machinery is correct but the signal needs richer data to show up.

Each tutorial discloses the specific deviations from its source paper.

## License

MIT.
