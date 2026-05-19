"""V-JEPA 2.1 visualization utilities (masks, dense PCA features, loss curves)."""

import os
import random

import matplotlib.pyplot as plt
import torch

from vjepa2_1 import sample_vjepa_masks, train as _train

SAMPLES_DIR = "samples"


def _patches_to_pixel_mask(idx_iterable, t_grid, s_grid, t_patch, patch_size, t_frames, img_size):
    mask = torch.zeros(t_frames, img_size, img_size)
    sg2 = s_grid * s_grid
    for p in idx_iterable:
        tt = p // sg2
        r = (p % sg2) // s_grid
        c = p % s_grid
        f0 = tt * t_patch
        mask[f0 : f0 + t_patch, r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = 1.0
    return mask


def save_video_masks(video, ctx_idx, pred_idx, encoder, label, path, n_frames_show=5):
    v = (video[0, 0].detach().cpu() + 0.5).clamp(0, 1)
    t_frames, h, _ = v.shape
    shown = torch.linspace(0, t_frames - 1, n_frames_show).round().long().tolist()
    cm = _patches_to_pixel_mask(ctx_idx, encoder.t_grid_vid, encoder.s_grid, encoder.t_patch, encoder.patch_size, t_frames, h)
    pm = _patches_to_pixel_mask(pred_idx, encoder.t_grid_vid, encoder.s_grid, encoder.t_patch, encoder.patch_size, t_frames, h)
    fig, axes = plt.subplots(3, n_frames_show, figsize=(n_frames_show * 1.6, 5))
    fig.suptitle(f"V-JEPA 2.1 video mask group: {label}", fontsize=10)
    for col, fi in enumerate(shown):
        f = v[fi]
        axes[0, col].imshow(f.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow((f * cm[fi]).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow((f * pm[fi]).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"t={fi}", fontsize=9)
        for row in range(3):
            axes[row, col].axis("off")
    for row, label_ in enumerate(["original", "context", "targets"]):
        axes[row, 0].set_ylabel(label_, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def save_image_mask(image, ctx_idx, pred_idx, encoder, path):
    img = (image[0, 0, 0].detach().cpu() + 0.5).clamp(0, 1)
    h = img.size(0)
    cm = _patches_to_pixel_mask(ctx_idx, 1, encoder.s_grid, 1, encoder.patch_size, 1, h)[0]
    pm = _patches_to_pixel_mask(pred_idx, 1, encoder.s_grid, 1, encoder.patch_size, 1, h)[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, data, title in zip(axes, [img, img * cm, img * pm], ["original", "context", "targets"]):
        ax.imshow(data.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def save_loss_curves(losses, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(losses["video_pred"], label="video pred", alpha=0.9)
    axes[0].plot(losses["video_ctx"], label="video ctx", alpha=0.9)
    axes[0].plot(losses["video_total"], label="video total", alpha=0.9)
    axes[0].set_title("Video batches")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("L1 loss")
    axes[0].legend()

    axes[1].plot(losses["image_pred"], label="image pred", alpha=0.9)
    axes[1].plot(losses["image_ctx"], label="image ctx", alpha=0.9)
    axes[1].plot(losses["image_total"], label="image total", alpha=0.9)
    axes[1].set_title("Image batches")
    axes[1].set_xlabel("step")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def _token_pca_rgb(tokens):
    x = tokens.float()
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=3)
    rgb = x @ v[:, :3]
    rgb = rgb - rgb.amin(0, keepdim=True)
    rgb = rgb / rgb.amax(0, keepdim=True).clamp_min(1e-6)
    return rgb


@torch.no_grad()
def save_image_dense_map(encoder, image, device, path):
    tokens = encoder(image.to(device), return_hier=False)[0].cpu()
    rgb = _token_pca_rgb(tokens).view(encoder.s_grid, encoder.s_grid, 3)
    img = (image[0, 0, 0].cpu() + 0.5).clamp(0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    axes[0].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("input image", fontsize=10)
    axes[1].imshow(rgb.numpy(), interpolation="nearest")
    axes[1].set_title("dense PCA map", fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


@torch.no_grad()
def save_video_dense_map(encoder, video, device, path):
    tokens = encoder(video.to(device), return_hier=False)[0].cpu()
    rgb = _token_pca_rgb(tokens).view(encoder.t_grid_vid, encoder.s_grid, encoder.s_grid, 3)
    frames = (video[0, 0].cpu() + 0.5).clamp(0, 1)
    shown = list(range(encoder.t_grid_vid))
    fig, axes = plt.subplots(2, len(shown), figsize=(len(shown) * 1.8, 4.2))
    for col, ti in enumerate(shown):
        fi = min(frames.size(0) - 1, ti * encoder.t_patch)
        axes[0, col].imshow(frames[fi].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"frame {fi}", fontsize=9)
        axes[1, col].imshow(rgb[ti].numpy(), interpolation="nearest")
        axes[1, col].set_title(f"tube {ti}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("dense PCA", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main(epochs=4):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out = _train(epochs=epochs)
    encoder = out["tgt_enc"]
    video = next(iter(out["video_loader"]))
    image = next(iter(out["image_loader"]))

    groups = sample_vjepa_masks(video.size(0), encoder.t_grid_vid, encoder.s_grid, rng=random.Random(42))
    for g in groups:
        save_video_masks(
            video[0:1],
            g["ctx"][0],
            g["pred"][0],
            encoder,
            f"{g['label']} ({g['n_blocks']} tubes, block {g['block_hw']})",
            f"{SAMPLES_DIR}/vjepa2_1_video_masks_{g['label']}.png",
        )

    img_group = sample_vjepa_masks(image.size(0), 1, encoder.s_grid, rng=random.Random(7), min_ctx=4)[0]
    save_image_mask(
        image[0:1],
        img_group["ctx"][0],
        img_group["pred"][0],
        encoder,
        f"{SAMPLES_DIR}/vjepa2_1_image_masks.png",
    )
    save_loss_curves(out["losses"], f"{SAMPLES_DIR}/vjepa2_1_loss.png")
    save_image_dense_map(encoder, image[0:1], out["device"], f"{SAMPLES_DIR}/vjepa2_1_dense_image.png")
    save_video_dense_map(encoder, video[0:1], out["device"], f"{SAMPLES_DIR}/vjepa2_1_dense_video.png")
    print(f"artifacts in ./{SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
