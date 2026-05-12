"""I-JEPA visualization and linear-probe utilities.

Kept separate from ijepa.py to keep the algorithm file focused. Run this file
directly to: train -> save mask grid + loss curve + PCA -> run linear probe.
"""

import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ijepa import MEAN, STD, sample_ijepa_masks, train as _train

SAMPLES_DIR = "samples"


# ---------- mask grid ----------

def _denorm(x):
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _idx_to_pixel_mask(idx_iterable, grid, patch_size, img_size):
    m = torch.zeros(img_size, img_size)
    for p in idx_iterable:
        r, c = p // grid, p % grid
        m[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = 1.0
    return m


def save_mask_grid(imgs, ctx_list, tgt_lists, grid, patch_size, path, n=8):
    imgs = _denorm(imgs[:n].detach().cpu())
    img_size = imgs.shape[-1]
    n_targets = len(tgt_lists)
    rows = 2 + n_targets
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.4, rows * 1.4))
    for i in range(n):
        cm = _idx_to_pixel_mask(ctx_list[i], grid, patch_size, img_size)
        axes[0, i].imshow(imgs[i].permute(1, 2, 0).numpy()); axes[0, i].axis("off")
        axes[1, i].imshow((imgs[i] * cm).permute(1, 2, 0).numpy()); axes[1, i].axis("off")
        for k in range(n_targets):
            tm = _idx_to_pixel_mask(tgt_lists[k][i], grid, patch_size, img_size)
            axes[2 + k, i].imshow((imgs[i] * tm).permute(1, 2, 0).numpy())
            axes[2 + k, i].axis("off")
    labels = ["original", "context"] + [f"target {k}" for k in range(n_targets)]
    for r, lab in enumerate(labels):
        axes[r, 0].set_ylabel(lab, rotation=90, fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


# ---------- loss curve ----------

def save_loss_curve(losses, path):
    plt.figure(figsize=(6, 3.5))
    plt.plot(losses); plt.xlabel("step"); plt.ylabel("smooth-L1 loss")
    plt.title("I-JEPA training loss"); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


# ---------- PCA of test features ----------

@torch.no_grad()
def compute_test_features(encoder, loader, device, max_batches=20):
    feats, labels = [], []
    for i, (imgs, y) in enumerate(loader):
        if i >= max_batches: break
        imgs = imgs.to(device)
        feats.append(encoder(imgs).mean(dim=1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def save_pca(feats, labels, path, title):
    x = feats - feats.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    proj = x @ v[:2].T
    plt.figure(figsize=(5, 5))
    plt.scatter(proj[:, 0].numpy(), proj[:, 1].numpy(), c=labels.numpy(),
                cmap="tab10", s=6, alpha=0.7)
    plt.title(title); plt.colorbar(label="class", ticks=range(10))
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


# ---------- linear probe ----------

def linear_probe(encoder, test_loader, device, epochs=3, batch_size=512):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    @torch.no_grad()
    def features(imgs):
        return encoder(imgs).mean(dim=1)

    clf = nn.Linear(encoder.dim, 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        for imgs, y in train_loader:
            imgs, y = imgs.to(device), y.to(device)
            loss = F.cross_entropy(clf(features(imgs)), y)
            opt.zero_grad(); loss.backward(); opt.step()

    correct = total = 0
    for imgs, y in test_loader:
        imgs, y = imgs.to(device), y.to(device)
        pred = clf(features(imgs)).argmax(-1)
        correct += (pred == y).sum().item(); total += y.numel()
    acc = correct / total
    print(f"linear probe test accuracy: {acc:.4f}")
    return acc


# ---------- driver ----------

def main(epochs=8):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out = _train(epochs=epochs)

    # mask snapshot from a fresh batch (using the same RNG seed for repeatability)
    imgs, _ = next(iter(out["loader"]))
    ctx_list, tgt_lists = sample_ijepa_masks(imgs.size(0), out["ctx_enc"].grid,
                                              rng=random.Random(42))
    save_mask_grid(imgs, ctx_list, tgt_lists,
                   out["ctx_enc"].grid, out["ctx_enc"].patch_size,
                   f"{SAMPLES_DIR}/ijepa_masks.png")

    save_loss_curve(out["losses"], f"{SAMPLES_DIR}/ijepa_loss.png")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=2)

    feats, labels = compute_test_features(out["tgt_enc"], test_loader, out["device"])
    save_pca(feats, labels, f"{SAMPLES_DIR}/ijepa_pca.png", "PCA after training")

    linear_probe(out["tgt_enc"], test_loader, out["device"])
    print(f"artifacts in ./{SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
