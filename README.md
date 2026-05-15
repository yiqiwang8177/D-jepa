# jepa

Minimal, single-file PyTorch reimplementations of the JEPA family, with paired tutorials.

| File | Method | Dataset | LOC | Tutorial |
|---|---|---|---:|---|
| [`ijepa.py`](./ijepa.py) | I-JEPA | CIFAR-10 | 165 | [`ijepa_tutorial.md`](./ijepa_tutorial.md) |
| [`vjepa.py`](./vjepa.py) | V-JEPA | Moving MNIST | 194 | [`vjepa_tutorial.md`](./vjepa_tutorial.md) |
| [`vjepa2.py`](./vjepa2.py) | V-JEPA 2 + V-JEPA 2-AC | synthetic moving digits | 314 | [`vjepa2_tutorial.md`](./vjepa2_tutorial.md) |
| [`cjepa.py`](./cjepa.py) | C-JEPA | 3-digit bouncing video | 162 | [`cjepa_tutorial.md`](./cjepa_tutorial.md) |
| [`leworldmodel.py`](./leworldmodel.py) | LeWorldModel | synthetic moving digit | 223 | [`leworldmodel_tutorial.md`](./leworldmodel_tutorial.md) |

Each algorithm file is **standalone** — only depends on `torch` and `torchvision`, no shared utilities. The matching `<algo>_extras.py` adds visualization (mask grids, loss curves, PCA/LDA/t-SNE evolution, linear probe).

See [`FAITHFULNESS.md`](./FAITHFULNESS.md) for the load-bearing details each minimal implementation preserves and the educational substitutions it makes.

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
├── ijepa.py / ijepa_extras.py                       # I-JEPA on CIFAR-10
├── vjepa.py / vjepa_extras.py                       # V-JEPA on Moving MNIST
├── vjepa2.py / vjepa2_extras.py                     # V-JEPA 2 + V-JEPA 2-AC (synthetic)
├── cjepa.py / cjepa_extras.py                       # C-JEPA on 3-digit bouncing video
├── leworldmodel.py / leworldmodel_extras.py         # LeWorldModel (end-to-end JEPA, SIGReg)
├── ijepa_tutorial.md                                # walk-throughs that match the code
├── vjepa_tutorial.md
├── vjepa2_tutorial.md
├── cjepa_tutorial.md
├── leworldmodel_tutorial.md
├── FAITHFULNESS.md                       # preserved details + deliberate simplifications
├── papers/                              # source PDFs bundled with the repo
├── samples/                             # mask grids, loss curves, PCA/LDA/t-SNE plots
└── figs/                                # paper figures referenced by tutorials
```

## The methods, in one paragraph each

**I-JEPA** ([Assran et al. 2023](https://arxiv.org/abs/2301.08243)) — predict embeddings of held-out image patches from embeddings of visible patches. EMA target encoder, multi-block masking, smooth-L1 loss. The canonical self-supervised JEPA.

**V-JEPA** ([Bardes et al. 2024](https://arxiv.org/abs/2404.08471)) — same recipe, but 3D tubelet patches over video. Two mask groups (short-range + long-range tubes), L1 loss, EMA 0.998 → 1.0.

**V-JEPA 2** ([Assran et al. 2025](https://arxiv.org/abs/2506.09985)) — two-phase: V-JEPA pretraining followed by **V-JEPA 2-AC**, an action-conditioned predictor trained on frozen-encoder latents with teacher forcing + rollout. The encoder is frozen in phase 2; no EMA.

**C-JEPA** ([Nam et al. 2026](https://arxiv.org/abs/2602.11389)) — object-level trajectory masking with an identity anchor at $t=0$. No EMA. Bidirectional transformer over flattened slot tokens. Built on top of a pretrained object-centric encoder in the paper; here we use a frozen oracle position-slot embedding as a documented educational stand-in.

**LeWorldModel** ([Maes et al. 2026](https://arxiv.org/abs/2603.19312)) — end-to-end JEPA world model from pixels. No EMA, no stop-grad, no masking. The encoder and an action-conditioned AR predictor are jointly trained with two loss terms: next-embedding MSE plus a Sketch Isotropic Gaussian Regularizer (SIGReg) that prevents collapse by pushing the embedding marginals toward $\mathcal{N}(0, 1)$.

## Caveats

These are **educational** reimplementations:

- ViT-tiny, not ViT-Huge. CIFAR-10 / Moving MNIST / synthetic videos, not ImageNet / Kinetics.
- I-JEPA hits **~52.7% linear probe** on CIFAR-10 after 100 epochs. The paper's numbers come from ViT-H/14 on ImageNet for 300 epochs — different planet of compute.
- C-JEPA skips slot discovery (uses oracle positions). Real C-JEPA requires VideoSAUR/SAVi-style object-centric pretraining on top of visual features.
- V-JEPA 2-AC is a small block-causal, action/state-conditioned latent predictor, not Meta's 300M-parameter robot-action model; it preserves the teacher-forcing + rollout training shape.
- LeWorldModel includes the two-term objective and projection heads needed for SIGReg, but omits the paper's control/planning layer.

Each tutorial discloses the specific deviations from its source paper and keeps code snippets aligned with the minimized implementation.

## License

MIT.
