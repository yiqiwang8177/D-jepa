"""Minimal C-JEPA: oracle slots, identity anchor, object-history + future masking, no EMA."""
import math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

IMG_SIZE = 32; DIGIT_SIZE = 8; GRID = 4; CELL = IMG_SIZE // GRID
T_HIST = 5; T_PRED = 3; T = T_HIST + T_PRED; K = 3; SLOT_DIM = 128


def sincos_1d(n, dim):
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.) / dim))
    pe = torch.zeros(n, dim); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


def param_groups(modules, wd):
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BouncingTriple(Dataset):
    def __init__(self, root="./data", n_samples=8000, seed=0):
        mnist = MNIST(root=root, train=True, download=True)
        imgs = mnist.data.float() / 255.
        self.imgs = F.interpolate(imgs.unsqueeze(1), (DIGIT_SIZE, DIGIT_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(1)
        self.labels = mnist.targets; self.n_samples = n_samples
        g = torch.Generator().manual_seed(seed)
        self.choices = torch.randint(0, len(self.imgs), (n_samples, K), generator=g)
        self.init_pos, self.init_vel = [], []
        for i in range(n_samples):
            g_ = torch.Generator().manual_seed(seed * 100003 + i)
            while True:
                p = torch.randint(0, GRID, (K, 2), generator=g_)
                if len({tuple(r.tolist()) for r in p}) == K: break
            v = torch.randint(0, 2, (K, 2), generator=g_) * 2 - 1
            self.init_pos.append(p); self.init_vel.append(v)

    def __len__(self): return self.n_samples

    def _simulate(self, p, v):
        positions = [p.clone()]
        for _ in range(T - 1):
            n = p + v
            for k in range(K):
                for ax in range(2):
                    if n[k, ax] < 0 or n[k, ax] >= GRID:
                        v[k, ax] = -v[k, ax]; n[k, ax] = p[k, ax] + v[k, ax]
            for i in range(K):
                for j in range(i + 1, K):
                    if torch.equal(n[i], n[j]):
                        v[i], v[j] = v[j].clone(), v[i].clone()
                        n[i] = p[i].clone(); n[j] = p[j].clone()
            p = n; positions.append(p.clone())
        return torch.stack(positions)

    def __getitem__(self, i):
        p, v = self.init_pos[i].clone(), self.init_vel[i].clone()
        positions = self._simulate(p, v)
        slot_idx = positions[..., 0] * GRID + positions[..., 1]
        digits = self.imgs[self.choices[i]]
        canvas = torch.zeros(1, T, IMG_SIZE, IMG_SIZE)
        for t in range(T):
            for k in range(K):
                r, c = positions[t, k, 0].item(), positions[t, k, 1].item()
                y0, x0 = r * CELL, c * CELL
                reg = canvas[0, t, y0:y0 + DIGIT_SIZE, x0:x0 + DIGIT_SIZE]
                canvas[0, t, y0:y0 + DIGIT_SIZE, x0:x0 + DIGIT_SIZE] = torch.maximum(reg, digits[k])
        return canvas - 0.5, slot_idx, self.labels[self.choices[i]]


class FrozenSlotEncoder(nn.Module):
    def __init__(self, dim=SLOT_DIM, n_cells=GRID * GRID):
        super().__init__()
        self.dim = dim; self.embed = nn.Embedding(n_cells, dim)
        nn.init.normal_(self.embed.weight, std=1.0)
        for p in self.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, video, slot_idx): return self.embed(slot_idx)


class MaskedSlotPredictor(nn.Module):
    def __init__(self, dim=SLOT_DIM, depth=4, heads=4):
        super().__init__()
        self.dim = dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.id_proj = nn.Linear(dim, dim)
        self.register_buffer("time_pe", sincos_1d(T, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, slots, mask_indices, ablate_to_anchor_only=False):
        B = slots.size(0)
        anchors = self.id_proj(slots[:, 0])
        real = slots + self.time_pe[None, :, None, :]
        query = self.mask_token + self.time_pe[None, :, None, :] + anchors[:, None, :, :]
        is_q = torch.zeros(T, K, dtype=torch.bool, device=slots.device)
        is_q[1:T_HIST, mask_indices] = True; is_q[T_HIST:, :] = True
        if ablate_to_anchor_only: is_q[1:T_HIST, :] = True
        x = torch.where(is_q[None, :, :, None], query, real).reshape(B, T * K, self.dim)
        for blk in self.blocks: x = blk(x)
        return self.to_out(self.norm(x)).view(B, T, K, self.dim)


def sample_mask_indices(rng, max_mask=2):
    return sorted(rng.sample(range(K), rng.randint(1, max_mask)))


def train(epochs=4, batch_size=64, lr=5e-4, wd=0.05, device=None):
    device = device or pick_device()
    print(f"device: {device}  T_HIST={T_HIST} T_PRED={T_PRED} K={K} grid={GRID}x{GRID}")
    ds = BouncingTriple()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    encoder = FrozenSlotEncoder().to(device); pred = MaskedSlotPredictor().to(device)
    opt = torch.optim.AdamW(param_groups([pred], wd), lr=lr); rng = random.Random(0)
    lh, lf, lt, la = [], [], [], []; step = 0
    for epoch in range(epochs):
        for video, slot_idx, _ in loader:
            video = video.to(device); slot_idx = slot_idx.to(device)
            slots = encoder(video, slot_idx)
            mi = sample_mask_indices(rng); p = pred(slots, mi)
            th = slots[:, 1:T_HIST, mi].detach(); tf = slots[:, T_HIST:].detach()
            l_h = F.mse_loss(p[:, 1:T_HIST, mi], th); l_f = F.mse_loss(p[:, T_HIST:], tf)
            loss = l_h + l_f
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pa = pred(slots, mi, ablate_to_anchor_only=True)
                la_ = (F.mse_loss(pa[:, 1:T_HIST, mi], th) + F.mse_loss(pa[:, T_HIST:], tf)).item()
            lh.append(l_h.item()); lf.append(l_f.item()); lt.append(loss.item()); la.append(la_)
            if step % 25 == 0:
                print(f"ep={epoch} step={step:5d} loss={loss.item():.4f} "
                      f"(hist={l_h.item():.4f} fut={l_f.item():.4f}) "
                      f"loss(anchor only)={la_:.4f} gap={la_ - loss.item():+.4f} masked={mi}")
            step += 1
    return {"encoder": encoder, "predictor": pred,
            "losses_total": lt, "losses_hist": lh, "losses_fut": lf, "losses_ablate": la,
            "loader": loader, "device": device}


if __name__ == "__main__": train()
