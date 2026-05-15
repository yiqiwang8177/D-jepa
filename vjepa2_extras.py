"""V-JEPA 2 visualization utilities (phase-1 masks, AC rollout demo, loss curves)."""

import os
import random

import matplotlib.pyplot as plt
import torch

from vjepa2 import sample_vjepa_masks, main as _main

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


@torch.no_grad()
def save_rollout_demo(encoder, ac, video, actions, states, path):
    """actions/states: (B, T, 2) per-tubelet velocity and normalized position."""
    B = video.size(0)
    z = encoder(video).view(B, encoder.t_grid, encoder.s_grid ** 2, -1)
    T = encoder.t_grid
    rollout_actions = actions[:, 1:T]
    rollout_states = states[:, :T - 1]
    rolled = ac.rollout(z[:, 0], rollout_actions, rollout_states)
    rolled_zero = ac.rollout(z[:, 0], torch.zeros_like(rollout_actions), rollout_states)
    err_roll = [(rolled[:, t] - z[:, t + 1]).abs().mean().item() for t in range(T - 1)]
    err_zero = [(rolled_zero[:, t] - z[:, t + 1]).abs().mean().item() for t in range(T - 1)]
    plt.figure(figsize=(6, 3.5))
    plt.plot(range(1, T), err_roll, marker="o", label="with action")
    plt.plot(range(1, T), err_zero, marker="x", label="action=0")
    plt.xlabel("rollout step"); plt.ylabel("L1 error vs true latent")
    plt.title("V-JEPA 2-AC latent rollout error")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def save_input_animation(video, action, path, t_patch=2):
    """Animated gif of one clip with the per-tubelet action vector overlaid."""
    import matplotlib.animation as animation
    v = (video[0, 0].detach().cpu() + 0.5).clamp(0, 1)
    T_frames, H, _ = v.shape
    a = action.detach().cpu().numpy()                    # (t_latent, 2)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(v[0].numpy(), cmap="gray", vmin=0, vmax=1)
    txt = ax.text(2, 5, "", color="lime", fontsize=9,
                  bbox=dict(facecolor="black", alpha=0.6, pad=2))
    arrow = ax.annotate("", xy=(H // 2, H // 2), xytext=(H // 2, H // 2),
                        arrowprops=dict(arrowstyle="->", color="lime", lw=2))

    def update(t):
        t_lat = t // t_patch
        ax_, ay_ = a[t_lat]
        im.set_data(v[t].numpy())
        txt.set_text(f"t={t}  a=({ax_:+.2f}, {ay_:+.2f})")
        arrow.xy = (H // 2 + 18 * ax_, H // 2 + 18 * ay_)
        arrow.set_position((H // 2, H // 2))
        return im, txt, arrow

    anim = animation.FuncAnimation(fig, update, frames=T_frames, interval=200, blit=False)
    anim.save(path, writer="pillow", dpi=110)
    plt.close()


def save_loss_curves(losses, path, title):
    plt.figure(figsize=(6, 3.5))
    for k, v in losses.items():
        plt.plot(v, label=k, alpha=0.85)
    plt.xlabel("step"); plt.ylabel("L1 loss"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def main(phase1_epochs=3, phase2_epochs=4):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out = _main(phase1_epochs=phase1_epochs, phase2_epochs=phase2_epochs)

    # phase-1 mask snapshot
    videos, actions, states = next(iter(out["loader"]))
    encoder = out["encoder"]
    groups = sample_vjepa_masks(videos.size(0), encoder.t_grid, encoder.s_grid,
                                 rng=random.Random(42))
    for g in groups:
        save_phase1_masks(videos[0:1], g["ctx"][0], g["pred"][0],
                          f"{g['label']} ({g['n_blocks']} tubes, block {g['block_hw']})",
                          encoder,
                          f"{SAMPLES_DIR}/vjepa2_phase1_masks_{g['label']}.png")

    # AC rollout demo
    videos = videos.to(out["device"]); actions = actions.to(out["device"]); states = states.to(out["device"])
    save_rollout_demo(encoder, out["ac"], videos[0:1], actions[0:1], states[0:1],
                      f"{SAMPLES_DIR}/vjepa2_ac_rollout.png")
    save_input_animation(videos[0:1].cpu(), actions[0].cpu(),
                         f"{SAMPLES_DIR}/vjepa2_input.gif",
                         t_patch=encoder.t_patch)

    save_loss_curves(out["phase1_losses"],
                     f"{SAMPLES_DIR}/vjepa2_phase1_loss.png",
                     "Phase 1: V-JEPA pretraining by mask group")
    save_loss_curves(out["phase2_losses"],
                     f"{SAMPLES_DIR}/vjepa2_ac_loss.png",
                     "Phase 2: V-JEPA 2-AC training")
    print(f"artifacts in ./{SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
