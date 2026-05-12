"""V-JEPA on Moving MNIST. Viz in vjepa_extras.py.

Faithful: 3D tubelet encoder + full EMA copy; 3D sin-cos pos (1D-t + 2D-s);
two mask groups (short 8x0.15 + long 2x0.7, long capped to 0.5 on tiny grids);
tubes span full temporal axis; per-item locations, random-subsample trim;
L1 on LN'd targets; EMA 0.998->1.0; WD split.
"""
import copy, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MovingMNIST

MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7)]


def sincos_1d(n, dim):
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.) / dim))
    pe = torch.zeros(n, dim); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sincos_2d(h, w, dim):
    assert dim % 4 == 0; sub = dim // 4
    yy, xx = [t.reshape(-1).float() for t in torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")]
    div = torch.exp(torch.arange(0, sub * 2, 2).float() * (-math.log(10000.) / (sub * 2)))
    return torch.cat([torch.sin(yy[:, None] * div), torch.cos(yy[:, None] * div),
                      torch.sin(xx[:, None] * div), torch.cos(xx[:, None] * div)], dim=-1)


def sincos_3d(t, h, w, dim, t_frac=0.25):
    td = int(dim * t_frac); td += td % 2; sd = dim - td
    pe_t = sincos_1d(t, td); pe_s = sincos_2d(h, w, sd)
    pe = torch.zeros(t * h * w, dim)
    pe[:, :td] = pe_t.unsqueeze(1).expand(t, h * w, td).reshape(-1, td)
    pe[:, td:] = pe_s.unsqueeze(0).expand(t, h * w, sd).reshape(-1, sd)
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


@torch.no_grad()
def ema_update(tgt, online, m):
    for pt, po in zip(tgt.parameters(), online.parameters()): pt.mul_(m).add_(po.detach(), alpha=1 - m)


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class MovingMNISTVideos(Dataset):
    def __init__(self, root="./data", num_frames=10):
        self.base = MovingMNIST(root=root, download=True); self.n = num_frames

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        v = self.base[i]
        if isinstance(v, (tuple, list)): v = v[0]
        v = v.float() / 255.; step = max(1, v.size(0) // self.n); v = v[::step][:self.n]
        return v.permute(1, 0, 2, 3).contiguous() - 0.5


class VideoEncoder(nn.Module):
    def __init__(self, num_frames=10, t_patch=2, img_size=64, patch_size=16,
                 in_chans=1, dim=128, depth=6, heads=4):
        super().__init__()
        self.t_grid = num_frames // t_patch; self.s_grid = img_size // patch_size
        self.n_patches = self.t_grid * self.s_grid * self.s_grid
        self.t_patch = t_patch; self.patch_size = patch_size; self.dim = dim
        self.tubelet_proj = nn.Conv3d(in_chans, dim,
                                      kernel_size=(t_patch, patch_size, patch_size),
                                      stride=(t_patch, patch_size, patch_size))
        self.register_buffer("pos", sincos_3d(self.t_grid, self.s_grid, self.s_grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, videos, idx=None):
        tokens = self.tubelet_proj(videos).flatten(2).transpose(1, 2)
        B, N, D = tokens.shape
        if idx is None:
            idx = torch.arange(N, device=videos.device).expand(B, -1); x = tokens + self.pos[idx]
        else:
            x = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)) + self.pos[idx]
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


class Predictor(nn.Module):
    def __init__(self, t_grid, s_grid, enc_dim=128, dim=64, depth=4, heads=4):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, enc_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_3d(t_grid, s_grid, s_grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B, T = ctx.size(0), tgt_idx.size(1)
        x = torch.cat([self.in_proj(ctx) + self.pos[ctx_idx],
                       self.mask_token.expand(B, T, -1) + self.pos[tgt_idx]], dim=1)
        for blk in self.blocks: x = blk(x)
        return self.out_proj(self.norm(x[:, -T:]))


def _bsize(g, s, ar=1.0):
    a = s * g * g
    return (max(1, min(g, round(math.sqrt(a * ar)))), max(1, min(g, round(math.sqrt(a / ar)))))


def sample_vjepa_masks(B, t_grid, s_grid, rng=None, min_ctx=8, ar_range=(0.75, 1.5)):
    rng = rng or random; sg2 = s_grid * s_grid; all_idx = set(range(t_grid * sg2))
    groups = []
    for label, n_blocks, scale in MASK_GROUPS:
        es = min(scale, 0.5) if s_grid < 8 and scale > 0.5 else scale
        h, w = _bsize(s_grid, es, rng.uniform(*ar_range))
        cs, ps = [], []
        for _ in range(B):
            tubes = set()
            for _ in range(n_blocks):
                top, left = rng.randint(0, s_grid - h), rng.randint(0, s_grid - w)
                for t in range(t_grid):
                    for r in range(top, top + h):
                        for c in range(left, left + w):
                            tubes.add(t * sg2 + r * s_grid + c)
            ctx = all_idx - tubes
            if len(ctx) < min_ctx:
                for p in sorted(tubes)[:min_ctx - len(ctx)]: tubes.discard(p); ctx.add(p)
            cs.append(sorted(ctx)); ps.append(sorted(tubes))
        Lc, Lp = min(len(c) for c in cs), min(len(p) for p in ps)
        groups.append({"label": label, "n_blocks": n_blocks, "block_hw": (h, w),
                       "ctx": [sorted(rng.sample(c, Lc)) for c in cs],
                       "pred": [sorted(rng.sample(p, Lp)) for p in ps]})
    return groups


def train(epochs=5, batch_size=32, lr=3e-4, wd=0.05, ema_start=0.998, ema_end=1.0, device=None):
    device = device or pick_device(); print(f"device: {device}")
    ds = MovingMNISTVideos(num_frames=10)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    ctx_enc = VideoEncoder().to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = Predictor(t_grid=ctx_enc.t_grid, s_grid=ctx_enc.s_grid).to(device)
    print(f"tubelet grid: t={ctx_enc.t_grid} s={ctx_enc.s_grid} -> {ctx_enc.n_patches} patches")
    opt = torch.optim.AdamW(param_groups([ctx_enc, pred], wd), lr=lr)
    total = epochs * len(loader); rng = random.Random(0)
    losses_pg = {g[0]: [] for g in MASK_GROUPS}; step = 0; D = ctx_enc.dim
    for epoch in range(epochs):
        for videos in loader:
            videos = videos.to(device); B = videos.size(0)
            groups = sample_vjepa_masks(B, ctx_enc.t_grid, ctx_enc.s_grid, rng=rng)
            with torch.no_grad(): full = F.layer_norm(tgt_enc(videos), (D,))
            per = {}
            for g in groups:
                ci = torch.tensor(g["ctx"], device=device); pi = torch.tensor(g["pred"], device=device)
                tgt = full.gather(1, pi.unsqueeze(-1).expand(-1, -1, D))
                per[g["label"]] = (pred(ctx_enc(videos, ci), ci, pi) - tgt).abs().mean()
            loss = sum(per.values()) / len(per)
            opt.zero_grad(); loss.backward(); opt.step()
            m = ema_start + (ema_end - ema_start) * (step / max(1, total - 1))
            ema_update(tgt_enc, ctx_enc, m)
            for k, v in per.items(): losses_pg[k].append(v.item())
            if step % 25 == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in per.items())
                print(f"ep={epoch} step={step:5d} {msg} ema={m:.4f}")
            step += 1
    return {"ctx_enc": ctx_enc, "tgt_enc": tgt_enc, "predictor": pred,
            "losses_per_group": losses_pg, "loader": loader, "device": device}


if __name__ == "__main__": train()
