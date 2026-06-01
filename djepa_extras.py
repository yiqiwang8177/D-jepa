"""D-JEPA visualization utilities """

import os
import random
import numpy as np 

import matplotlib.pyplot as plt
import torch

from djepa import cfg, visualize_tokens_pca, sample_vjepa_masks, main as _main
from torch.utils.data import DataLoader
from djepa import RobomimicVideos

SAMPLES_DIR = "samples"

def _patches_to_pixel_mask(idx_iterable, t_grid, s_grid, t_patch, patch_size,
                           T_frames, img_size):
    mask = torch.zeros(T_frames, img_size, img_size)
    sg2 = s_grid * s_grid
    for p in idx_iterable:
        tt = p // sg2; r = (p % sg2) // s_grid; c = p % s_grid
        f0 = tt * t_patch
        mask[f0:f0 + t_patch,
             r * patch_size:(r + 1) * patch_size,
             c * patch_size:(c + 1) * patch_size] = 1.0
    return mask


def save_phase1_masks(video, ctx_idx, pred_idx, label, encoder, path, n_frames_show=5):
    v = (video[0, 0].detach().cpu() + 0.5).clamp(0, 1)
    T_frames, H, _ = v.shape
    fi_list = torch.linspace(0, T_frames - 1, n_frames_show).round().long().tolist()
    cm = _patches_to_pixel_mask(ctx_idx, encoder.t_grid, encoder.s_grid,
                                 encoder.t_patch, encoder.patch_size, T_frames, H)
    tm = _patches_to_pixel_mask(pred_idx, encoder.t_grid, encoder.s_grid,
                                 encoder.t_patch, encoder.patch_size, T_frames, H)
    fig, axes = plt.subplots(3, n_frames_show, figsize=(n_frames_show * 1.6, 5))
    fig.suptitle(f"V-JEPA 2 phase-1 mask group: {label}", fontsize=10)
    for col, fi in enumerate(fi_list):
        f = v[fi]
        axes[0, col].imshow(f.numpy(), cmap="gray", vmin=0, vmax=1); axes[0, col].axis("off")
        axes[1, col].imshow((f * cm[fi]).numpy(), cmap="gray", vmin=0, vmax=1); axes[1, col].axis("off")
        axes[2, col].imshow((f * tm[fi]).numpy(), cmap="gray", vmin=0, vmax=1); axes[2, col].axis("off")
        axes[0, col].set_title(f"t={fi}", fontsize=9)
    for r, lab in enumerate(["original", "context", "targets (union)"]):
        axes[r, 0].set_ylabel(lab, rotation=90, fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

def main(cfg):
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    cfg.phase1_epochs = 1
    cfg.episodes = 5
    cfg.use_wandb = False
    out = _main(cfg)

    # phase-1 mask snapshot
    # videos, centers = next(iter(out["loader"]))
    videos, flows = next(iter(out["loader"]))

    # centers = centers.numpy()  # (B, T, 2)
    encoder = out["encoder"]
    # groups = sample_vjepa_masks(videos.size(0), encoder.t_grid, encoder.s_grid,
    #                              rng=random.Random(42), center_Bs=centers)
    groups = sample_vjepa_masks(videos.size(0), encoder.t_grid, encoder.s_grid,
                                 rng=random.Random(42), flows=flows)
    for g in groups:
        save_phase1_masks(videos[0:1], g["ctx"][0], g["pred"][0],
                          f"{g['label']} ({g['n_blocks']} tubes, block {g['block_hw']})",
                          encoder,
                          f"{SAMPLES_DIR}/djepa_phase1_masks_{g['label']}.png")
    print(f"artifacts in ./{SAMPLES_DIR}/")

   
if __name__ == "__main__":
    cfg = cfg()
    main(cfg)
