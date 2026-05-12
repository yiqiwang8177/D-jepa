"""V-JEPA 2 + V-JEPA 2-AC on synthetic moving-digit videos. Viz in vjepa2_extras.py.

Two phases. Phase 1: V-JEPA pretraining (EMA target, two mask groups, L1).
Phase 2: V-JEPA 2-AC -- freeze encoder, train AC predictor with teacher
forcing + rollout. Per-tubelet velocity is the action.
"""
import copy, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

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


class ActionedMovingDigits(Dataset):
    """Per-tubelet velocity = action. z_t depends on action_t."""
    V_MAX = 5.0

    def __init__(self, root="./data", n_samples=8000, n_frames=10, t_patch=2,
                 img_size=64, digit_size=20, seed=0):
        mnist = MNIST(root=root, train=True, download=True)
        self.imgs = mnist.data.float() / 255.; self.n_samples = n_samples
        self.n_frames = n_frames; self.t_patch = t_patch
        self.img_size = img_size; self.digit_size = digit_size
        self.t_latent = n_frames // t_patch
        g = torch.Generator().manual_seed(seed); s = img_size - digit_size
        self.choices = torch.randint(0, len(self.imgs), (n_samples,), generator=g)
        self.starts = torch.stack([torch.randint(0, s + 1, (n_samples,), generator=g),
                                   torch.randint(0, s + 1, (n_samples,), generator=g)], 1).float()
        self.actions = torch.rand(n_samples, self.t_latent, 2, generator=g) * 2 - 1

    def __len__(self): return self.n_samples

    def __getitem__(self, i):
        d = F.interpolate(self.imgs[self.choices[i]][None, None],
                          (self.digit_size, self.digit_size), mode="bilinear",
                          align_corners=False)[0, 0]
        p = self.starts[i].clone(); mp = self.img_size - self.digit_size
        frames = torch.zeros(self.n_frames, self.img_size, self.img_size)
        for tl in range(self.t_latent):
            v = self.actions[i, tl] * self.V_MAX
            for sub in range(self.t_patch):
                t = tl * self.t_patch + sub
                xi = max(0, min(mp, int(round(p[0].item()))))
                yi = max(0, min(mp, int(round(p[1].item()))))
                frames[t, yi:yi + self.digit_size, xi:xi + self.digit_size] = torch.maximum(
                    frames[t, yi:yi + self.digit_size, xi:xi + self.digit_size], d)
                np_ = p + v
                if np_[0] < 0 or np_[0] > mp: v[0] = -v[0]; np_[0] = p[0] + v[0]
                if np_[1] < 0 or np_[1] > mp: v[1] = -v[1]; np_[1] = p[1] + v[1]
                p = np_
        return frames.unsqueeze(0) - 0.5, self.actions[i]


class VideoEncoder(nn.Module):
    def __init__(self, num_frames=10, t_patch=2, img_size=64, patch_size=8,
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


class JEPAPredictor(nn.Module):
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


class ACPredictor(nn.Module):
    """z_{t+1} ~= step(z_t, a_t). rollout(z0, actions) chains step()."""

    def __init__(self, s_grid, enc_dim=128, dim=128, depth=4, heads=4, action_dim=2):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, enc_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer("pos", sincos_2d(s_grid, s_grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def step(self, z, a):
        x = self.in_proj(z) + self.pos.unsqueeze(0) + self.action_proj(a).unsqueeze(1)
        for blk in self.blocks: x = blk(x)
        return self.out_proj(self.norm(x))

    def rollout(self, z0, actions):
        z = z0; out = []
        for k in range(actions.size(1)): z = self.step(z, actions[:, k]); out.append(z)
        return torch.stack(out, dim=1)

    forward = step


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


def pretrain(loader, epochs, device, lr=3e-4, wd=0.05, ema_start=0.998, ema_end=1.0):
    print("=== Phase 1: V-JEPA pretraining ===")
    ctx_enc = VideoEncoder().to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = JEPAPredictor(t_grid=ctx_enc.t_grid, s_grid=ctx_enc.s_grid).to(device)
    print(f"tubelet grid: t={ctx_enc.t_grid} s={ctx_enc.s_grid} -> {ctx_enc.n_patches} patches")
    opt = torch.optim.AdamW(param_groups([ctx_enc, pred], wd), lr=lr)
    total = epochs * len(loader); rng = random.Random(0)
    losses = {g[0]: [] for g in MASK_GROUPS}; step = 0; D = ctx_enc.dim
    for epoch in range(epochs):
        for videos, _ in loader:
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
            for k, v in per.items(): losses[k].append(v.item())
            if step % 25 == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in per.items())
                print(f"[phase1] ep={epoch} step={step:5d} {msg} ema={m:.4f}")
            step += 1
    return tgt_enc, losses


def train_ac(encoder, loader, epochs, device, lr=3e-4, wd=0.05, rollout_k=4, rollout_w=0.5):
    print("=== Phase 2: V-JEPA 2-AC ===")
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    ac = ACPredictor(s_grid=encoder.s_grid).to(device)
    opt = torch.optim.AdamW(param_groups([ac], wd), lr=lr)
    T, S = encoder.t_grid, encoder.s_grid ** 2
    losses = {"tf": [], "roll": [], "a=0": []}; step = 0
    for epoch in range(epochs):
        for videos, actions in loader:
            videos = videos.to(device); actions = actions.to(device); B = videos.size(0)
            with torch.no_grad(): z = encoder(videos).view(B, T, S, -1)
            preds = torch.stack([ac.step(z[:, t], actions[:, t + 1]) for t in range(T - 1)], 1)
            tgt = z[:, 1:]; loss_tf = (preds - tgt).abs().mean()
            k = min(rollout_k, T - 1)
            rolled = ac.rollout(z[:, 0], actions[:, 1:k + 1])
            loss_roll = (rolled - z[:, 1:k + 1]).abs().mean()
            loss = loss_tf + rollout_w * loss_roll
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pz = torch.stack([ac.step(z[:, t], torch.zeros_like(actions[:, t + 1]))
                                  for t in range(T - 1)], 1)
                loss_a = (pz - tgt).abs().mean().item()
            losses["tf"].append(loss_tf.item()); losses["roll"].append(loss_roll.item())
            losses["a=0"].append(loss_a)
            if step % 25 == 0:
                gap = loss_a - loss_tf.item()
                print(f"[phase2] ep={epoch} step={step:5d} tf={loss_tf.item():.4f} "
                      f"roll={loss_roll.item():.4f} a=0={loss_a:.4f} gap={gap:+.4f}")
            step += 1
    return ac, losses


def main(phase1_epochs=3, phase2_epochs=4, batch_size=32, device=None):
    device = device or pick_device(); print(f"device: {device}")
    ds = ActionedMovingDigits()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    encoder, p1 = pretrain(loader, phase1_epochs, device)
    ac, p2 = train_ac(encoder, loader, phase2_epochs, device)
    return {"encoder": encoder, "ac": ac, "phase1_losses": p1, "phase2_losses": p2,
            "loader": loader, "device": device}


if __name__ == "__main__": main()
