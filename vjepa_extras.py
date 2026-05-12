"""V-JEPA visualization utilities (loss curves, mask grids per group)."""

import os
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from vjepa import MovingMNISTVideos, sample_vjepa_masks, train as _train

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


def save_group_masks(video, ctx_idx, pred_idx, label, encoder, path, n_frames_show=5):
    v = (video[0, 0].detach().cpu() + 0.5).clamp(0, 1)
    T_frames, H, _ = v.shape
    fi_list = torch.linspace(0, T_frames - 1, n_frames_show).round().long().tolist()
    cm = _patches_to_pixel_mask(ctx_idx, encoder.t_grid, encoder.s_grid,
                                 encoder.t_patch, encoder.patch_size, T_frames, H)
    tm = _patches_to_pixel_mask(pred_idx, encoder.t_grid, encoder.s_grid,
                                 encoder.t_patch, encoder.patch_size, T_frames, H)
    fig, axes = plt.subplots(3, n_frames_show, figsize=(n_frames_show * 1.6, 5))
    fig.suptitle(f"V-JEPA mask group: {label}", fontsize=10)
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


def save_loss_curves(losses_per_group, path):
    plt.figure(figsize=(6, 3.5))
    for label, losses in losses_per_group.items():
        plt.plot(losses, label=f"{label} group", alpha=0.85)
    plt.xlabel("step"); plt.ylabel("L1 loss")
    plt.title("V-JEPA training loss by mask group")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def main(epochs=3):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out = _train(epochs=epochs)

    # mask snapshot from a fresh video
    videos = next(iter(out["loader"]))
    B = videos.size(0)
    groups = sample_vjepa_masks(B, out["ctx_enc"].t_grid, out["ctx_enc"].s_grid,
                                 rng=random.Random(42))
    for g in groups:
        save_group_masks(videos[0:1], g["ctx"][0], g["pred"][0],
                         f"{g['label']} ({g['n_blocks']} tubes, block {g['block_hw']})",
                         out["ctx_enc"],
                         f"{SAMPLES_DIR}/vjepa_masks_{g['label']}.png")

    save_loss_curves(out["losses_per_group"], f"{SAMPLES_DIR}/vjepa_loss.png")
    print(f"artifacts in ./{SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
