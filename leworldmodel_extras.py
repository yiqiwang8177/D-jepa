"""LeWorldModel visualization utilities: loss curves, Gaussianity check, rollout, surprise gif."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from leworldmodel import ActionedMovingDigit, train as _train

SAMPLES_DIR = "samples"


def save_loss_curves(losses, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(losses["pred"], label="L_pred (MSE)", color="C0")
    ax.plot(losses["zero_act"], label="L_pred (action=0)", color="C0", linestyle="--", alpha=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("MSE", color="C0"); ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(losses["sigreg"], label="L_sigreg", color="C3", alpha=0.85)
    ax2.set_ylabel("SIGReg", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
    fig.suptitle("LeWorldModel training: prediction + SIGReg")
    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


@torch.no_grad()
def save_gaussianity(encoder, loader, device, path, n_proj=6):
    encoder.eval()
    embs = []
    for i, (frames, _) in enumerate(loader):
        if i >= 8: break
        embs.append(encoder(frames.to(device)).reshape(-1, encoder.dim))
    Z = torch.cat(embs, 0)                                   # (N, D)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)                  # standardize per-dim (we plot direction-marginals)
    D = Z.size(-1)
    rng = torch.Generator(device=device).manual_seed(7)
    A = torch.randn(D, n_proj, generator=rng, device=device)
    A = A / A.norm(p=2, dim=0, keepdim=True)
    p = (Z @ A).cpu().numpy()                                # (N, n_proj)
    xs = np.linspace(-4, 4, 256)
    gauss = np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi)
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.5))
    for k, ax in enumerate(axes.flat):
        ax.hist(p[:, k], bins=60, density=True, alpha=0.75, color="C0")
        ax.plot(xs, gauss, color="C3", lw=1.5, label="N(0,1)")
        ax.set_xlim(-4, 4); ax.set_title(f"random projection #{k}", fontsize=9)
        if k == 0: ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("LeWorldModel embedding marginals along random directions (after training)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


@torch.no_grad()
def save_rollout(encoder, predictor, loader, device, path, n_clips=64):
    encoder.eval(); predictor.eval()
    err_a, err_z, count = None, None, 0
    for frames, actions in loader:
        frames = frames.to(device); actions = actions.to(device)
        z = encoder(frames)
        roll_a = predictor.rollout(z[:, 0], actions[:, :-1])
        roll_z = predictor.rollout(z[:, 0], torch.zeros_like(actions[:, :-1]))
        ea = (roll_a - z[:, 1:]).pow(2).mean(dim=(0, 2))     # per-step error, averaged over batch+dim
        ez = (roll_z - z[:, 1:]).pow(2).mean(dim=(0, 2))
        err_a = ea if err_a is None else err_a + ea
        err_z = ez if err_z is None else err_z + ez
        count += frames.size(0)
        if count >= n_clips: break
    K = err_a.size(0)
    plt.figure(figsize=(6, 3.5))
    plt.plot(range(1, K + 1), err_a.cpu().numpy(), marker="o", label="rollout with action")
    plt.plot(range(1, K + 1), err_z.cpu().numpy(), marker="x", label="rollout with action=0")
    plt.xlabel("rollout step k"); plt.ylabel("MSE vs true latent z_k")
    plt.title("LeWorldModel: autoregressive latent rollout error")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


@torch.no_grad()
def save_surprise_gif(encoder, predictor, dataset, device, path, teleport_at=5):
    """Roll the predictor forward across a clip; at `teleport_at` swap in a different clip.
       The prediction error should spike at the teleport frame and decay as the predictor
       re-syncs. Visualized as a green->red border around the frame.
    """
    import matplotlib.animation as animation
    encoder.eval(); predictor.eval()
    f0, a0 = dataset[0]; f1, _ = dataset[1]
    frames = f0.clone()
    frames[teleport_at:] = f1[teleport_at:]                  # teleport: replace from teleport_at onward
    frames_b = frames.unsqueeze(0).to(device); actions_b = a0.unsqueeze(0).to(device)
    z = encoder(frames_b)                                    # (1, T, D)
    pred = predictor(z[:, :-1], actions_b[:, :-1])           # (1, T-1, D)
    err = (pred - z[:, 1:]).pow(2).mean(-1).squeeze(0).cpu().numpy()  # (T-1,)
    err_padded = np.concatenate([[0.0], err])                # align with frame index
    e_max = max(err_padded.max(), 1e-6)

    vis = (frames[:, 0] + 0.5).clamp(0, 1).numpy()
    T_frames, H, W = vis.shape
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    im = ax.imshow(vis[0], cmap="gray", vmin=0, vmax=1)
    border = ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, lw=4, edgecolor="lime"))
    txt = ax.text(2, 5, "", color="white", fontsize=9,
                  bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("frame (border = prediction error)", fontsize=9)
    line, = ax2.plot([], [], color="C3"); marker, = ax2.plot([], [], "o", color="C3")
    ax2.axvline(teleport_at, color="gray", linestyle=":", alpha=0.7)
    ax2.text(teleport_at + 0.1, 0.95 * e_max, "teleport", fontsize=8, color="gray")
    ax2.set_xlim(0, T_frames - 1); ax2.set_ylim(0, e_max * 1.15)
    ax2.set_xlabel("t"); ax2.set_ylabel("|z_pred - z_true|^2")
    ax2.set_title("surprise (one-step prediction error)", fontsize=9)

    def update(t):
        im.set_data(vis[t])
        c = (err_padded[t] / e_max)
        border.set_edgecolor((c, 1 - c, 0))                  # green->red as error grows
        txt.set_text(f"t={t}  err={err_padded[t]:.3f}")
        line.set_data(range(t + 1), err_padded[:t + 1])
        marker.set_data([t], [err_padded[t]])
        return im, border, txt, line, marker

    anim = animation.FuncAnimation(fig, update, frames=T_frames, interval=400, blit=False)
    anim.save(path, writer="pillow", dpi=110)
    plt.close()


def save_input_gif(dataset, path):
    """A quick reference clip with the per-frame action drawn as a green arrow."""
    import matplotlib.animation as animation
    frames, actions = dataset[0]
    v = (frames[:, 0] + 0.5).clamp(0, 1).numpy()
    a = actions.numpy()
    T_frames, H, _ = v.shape
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(v[0], cmap="gray", vmin=0, vmax=1)
    txt = ax.text(2, 5, "", color="lime", fontsize=9,
                  bbox=dict(facecolor="black", alpha=0.6, pad=2))
    arrow = ax.annotate("", xy=(H // 2, H // 2), xytext=(H // 2, H // 2),
                        arrowprops=dict(arrowstyle="->", color="lime", lw=2))

    def update(t):
        im.set_data(v[t])
        ax_, ay_ = a[t]
        txt.set_text(f"t={t}  a=({ax_:+.2f}, {ay_:+.2f})")
        arrow.xy = (H // 2 + 18 * ax_, H // 2 + 18 * ay_)
        arrow.set_position((H // 2, H // 2))
        return im, txt, arrow

    anim = animation.FuncAnimation(fig, update, frames=T_frames, interval=300, blit=False)
    anim.save(path, writer="pillow", dpi=110); plt.close()


def main(epochs=4):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out = _train(epochs=epochs)

    save_input_gif(out["loader"].dataset, f"{SAMPLES_DIR}/leworldmodel_input.gif")
    save_loss_curves(out["losses"], f"{SAMPLES_DIR}/leworldmodel_loss.png")
    save_gaussianity(out["encoder"], out["loader"], out["device"],
                     f"{SAMPLES_DIR}/leworldmodel_gaussianity.png")
    save_rollout(out["encoder"], out["predictor"], out["loader"], out["device"],
                 f"{SAMPLES_DIR}/leworldmodel_rollout.png")
    save_surprise_gif(out["encoder"], out["predictor"], out["loader"].dataset, out["device"],
                      f"{SAMPLES_DIR}/leworldmodel_surprise.gif")
    print(f"artifacts in ./{SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
