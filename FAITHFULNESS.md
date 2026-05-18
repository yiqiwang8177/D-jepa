# Faithfulness notes

These files are intentionally small educational implementations. They should preserve the
load-bearing algorithmic ideas, while documenting where they substitute tiny datasets or
small networks for the paper-scale systems.

| File | Preserved faithfully | Educational substitutions / limits |
| --- | --- | --- |
| `ijepa.py` | Context encoder, EMA target encoder, target masking at target-encoder output, multi-block image targets, predictor mask queries, latent loss. | CIFAR-10 and ViT-tiny instead of ImageNet and large ViTs; uses the official-code-style normalized targets and smooth-L1 loss. |
| `vjepa.py` | Tubelet encoder, EMA target encoder, short/long spatial tube masks spanning all time steps, L1 latent prediction. | MovingMNIST and tiny transformer; mask sampling preserves tube structure but may trim whole spatial tubes to keep batch tensors rectangular. |
| `vjepa2.py` | Two-stage training, frozen encoder in phase 2, block-causal action/state-conditioned predictor, teacher-forcing plus rollout loss. | Synthetic per-tubelet velocity actions and normalized position state tokens; no real robot proprio/extrinsics; tiny predictor instead of the paper's large robot-world-model predictor. |
| `vjepa2_1.py` | EMA target encoder, masked-target plus dense-context predictive loss, multi-layer hierarchical self-supervision, shared image/video backbone with modality embeddings. | MNIST + MovingMNIST instead of the paper's large image/video mixture; tiny ViT and predictor; single-process alternating image/video batches instead of the official mixed distributed loader; lightweight distance-weighted context loss and dense PCA visualizations instead of the full dense-task eval suite. |
| `cjepa.py` | Object-level history masking, earliest-frame identity anchor, future masking, bidirectional masked predictor, history + future MSE, no EMA. | Uses oracle grid-cell position slots, not VideoSAUR/SAVi/DINOv2 slots; this demonstrates the masking objective, not slot discovery. |
| `leworldmodel.py` | End-to-end encoder/predictor training, no EMA, no stop-grad, no masking, next-embedding MSE + SIGReg, action-conditioned causal predictor with zero-initialized AdaLN. | Synthetic moving digit data; small mean-pooled ViT instead of the paper's CLS-token ViT; no MPC/control evaluation. |

Known deliberate simplifications should stay visible in code comments and tutorials. If a
future change removes a load-bearing idea above, update this file and the paired tutorial.
