"""C-JEPA visualization utilities: mask grid, loss decomposition, ablation gap."""

import os
import random

import matplotlib.pyplot as plt
import torch

from cjepa import (
    CELL, GRID, IMG_SIZE, K, T, T_HIST, T_PRED,
    sample_mask_indices, train as _train,
)

SAMPLES_DIR = "samples"


def save_mask_grid(video, slot_idx, mask_indices, path):
    """Show ground-truth video alongside what the predictor sees:
       row 1: original frames
       row 2: context (anchor row at t=0 + unmasked slots in history)
       row 3: targets (masked-slot history + ALL future)
    """
    v = (video[0, 0].detach().cpu() + 0.5).clamp(0, 1)
    fig, axes = plt.subplots(3, T, figsize=(T * 1.6, 5))
    fig.suptitle(f"C-JEPA masking  --  masked slots: {mask_indices}  "
                 f"(T_HIST={T_HIST}, T_PRED={T_PRED})", fontsize=10)
    for t in range(T):
        frame = v[t]
        # context = original frame minus the masked-slot patches (except at t=0)
        ctx_mask = torch.ones_like(frame)
        if t >= 1:
            for k in mask_indices:
                r = int(slot_idx[t, k] // GRID); c = int(slot_idx[t, k] % GRID)
                ctx_mask[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 0
            if t >= T_HIST:
                ctx_mask = torch.zeros_like(frame)               # entire future is masked
        # target = exactly what the predictor must reconstruct at this t
        tgt_mask = torch.zeros_like(frame)
        if 1 <= t < T_HIST:
            for k in mask_indices:
                r = int(slot_idx[t, k] // GRID); c = int(slot_idx[t, k] % GRID)
                tgt_mask[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 1
        elif t >= T_HIST:
            tgt_mask = torch.ones_like(frame)
        axes[0, t].imshow(frame.numpy(), cmap="gray", vmin=0, vmax=1); axes[0, t].axis("off")
        is_anchor = " *anchor*" if t == 0 else (" *future*" if t >= T_HIST else "")
        axes[0, t].set_title(f"t={t}{is_anchor}", fontsize=8)
        axes[1, t].imshow((frame * ctx_mask).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, t].axis("off")
        axes[2, t].imshow((frame * tgt_mask).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2, t].axis("off")
    for r, lab in enumerate(["original", "context", "targets"]):
        axes[r, 0].set_ylabel(lab, rotation=90, fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def save_loss_decomp(losses_hist, losses_fut, losses_total, path):
    plt.figure(figsize=(6, 3.5))
    plt.plot(losses_total, label="total", color="black")
    plt.plot(losses_hist, label="L_history", alpha=0.7)
    plt.plot(losses_fut, label="L_future", alpha=0.7)
    plt.xlabel("step"); plt.ylabel("L2 loss"); plt.title("C-JEPA loss decomposition")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def save_mask_animation(video, slot_idx, mask_indices, t_anchor, path):
    """Animated gif: original | context | targets across all T frames."""
    import matplotlib.animation as animation
    v = (video[0, 0].detach().cpu() + 0.5).clamp(0, 1)
    T_frames, H, _ = v.shape
    ctx_masks, tgt_masks = [], []
    for fi in range(T_frames):
        cm = torch.ones_like(v[0])
        if fi >= T_HIST:
            cm = torch.zeros_like(v[0])
        elif fi != t_anchor:
            for k in mask_indices:
                r = int(slot_idx[fi, k] // GRID); c = int(slot_idx[fi, k] % GRID)
                cm[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 0
        tm = torch.zeros_like(v[0])
        if 1 <= fi < T_HIST:
            for k in mask_indices:
                r = int(slot_idx[fi, k] // GRID); c = int(slot_idx[fi, k] % GRID)
                tm[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 1
        elif fi >= T_HIST:
            tm = torch.ones_like(v[0])
        ctx_masks.append(cm); tgt_masks.append(tm)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    fig.suptitle(f"C-JEPA  --  masked slots {mask_indices}  --  anchor t={t_anchor}", fontsize=9)
    for ax, title in zip(axes, ["original", "context", "targets"]):
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    im0 = axes[0].imshow(v[0].numpy(), cmap="gray", vmin=0, vmax=1)
    im1 = axes[1].imshow((v[0] * ctx_masks[0]).numpy(), cmap="gray", vmin=0, vmax=1)
    im2 = axes[2].imshow((v[0] * tgt_masks[0]).numpy(), cmap="gray", vmin=0, vmax=1)
    label = axes[0].text(1, 4, "", color="lime", fontsize=8,
                         bbox=dict(facecolor="black", alpha=0.6, pad=2))

    def update(t):
        im0.set_data(v[t].numpy())
        im1.set_data((v[t] * ctx_masks[t]).numpy())
        im2.set_data((v[t] * tgt_masks[t]).numpy())
        tag = " *anchor*" if t == t_anchor else (" *future*" if t >= T_HIST else "")
        label.set_text(f"t={t}{tag}")
        return im0, im1, im2, label

    anim = animation.FuncAnimation(fig, update, frames=T_frames, interval=350, blit=False)
    anim.save(path, writer="pillow", dpi=110)
    plt.close()


def save_interaction_gap(losses_total, losses_ablate, path):
    plt.figure(figsize=(6, 3.5))
    plt.plot(losses_total, label="full context (other slots' history visible)")
    plt.plot(losses_ablate, label="anchor only (no other-slot history)", alpha=0.7)
    plt.xlabel("step"); plt.ylabel("L2 loss")
    plt.title("C-JEPA: information gain from other-slot dynamics")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def main(epochs=4):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out = _train(epochs=epochs)

    video, slot_idx, _ = next(iter(out["loader"]))
    rng = random.Random(42)
    mask_indices = sample_mask_indices(rng)
    save_mask_grid(video[0:1], slot_idx[0], mask_indices,
                   f"{SAMPLES_DIR}/cjepa_masks.png")
    save_mask_animation(video[0:1], slot_idx[0], mask_indices, t_anchor=0,
                        path=f"{SAMPLES_DIR}/cjepa_masks.gif")

    save_loss_decomp(out["losses_hist"], out["losses_fut"], out["losses_total"],
                     f"{SAMPLES_DIR}/cjepa_loss.png")
    save_interaction_gap(out["losses_total"], out["losses_ablate"],
                         f"{SAMPLES_DIR}/cjepa_interaction_gap.png")
    print(f"artifacts in ./{SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
