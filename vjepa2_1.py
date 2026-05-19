"""Minimal V-JEPA 2.1: dense predictive loss, deep self-supervision, image/video co-training."""
import copy, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, MovingMNIST

MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7)]


def sincos_1d(n, dim):
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sincos_2d(h, w, dim):
    assert dim % 4 == 0
    sub = dim // 4
    yy, xx = [
        t.reshape(-1).float()
        for t in torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ]
    div = torch.exp(torch.arange(0, sub * 2, 2).float() * (-math.log(10000.0) / (sub * 2)))
    return torch.cat(
        [
            torch.sin(yy[:, None] * div),
            torch.cos(yy[:, None] * div),
            torch.sin(xx[:, None] * div),
            torch.cos(xx[:, None] * div),
        ],
        dim=-1,
    )


def sincos_3d(t, h, w, dim, t_frac=0.25):
    td = int(dim * t_frac)
    td += td % 2
    sd = dim - td
    pe_t = sincos_1d(t, td)
    pe_s = sincos_2d(h, w, sd)
    pe = torch.zeros(t * h * w, dim)
    pe[:, :td] = pe_t.unsqueeze(1).expand(t, h * w, td).reshape(-1, td)
    pe[:, td:] = pe_s.unsqueeze(0).expand(t, h * w, sd).reshape(-1, sd)
    return pe


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp)),
            nn.GELU(),
            nn.Linear(int(dim * mlp), dim),
        )

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


def param_groups(modules, wd):
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(tgt, online, m):
    for pt, po in zip(tgt.parameters(), online.parameters()):
        pt.mul_(m).add_(po.detach(), alpha=1 - m)


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class MovingMNISTVideos(Dataset):
    def __init__(self, root="./data", num_frames=10):
        self.base = MovingMNIST(root=root, download=True)
        self.n = num_frames

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        v = self.base[i]
        if isinstance(v, (tuple, list)):
            v = v[0]
        v = v.float() / 255.0
        step = max(1, v.size(0) // self.n)
        v = v[::step][: self.n]
        return v.permute(1, 0, 2, 3).contiguous() - 0.5


class MNISTImages(Dataset):
    def __init__(self, root="./data", img_size=64):
        mnist = MNIST(root=root, train=True, download=True)
        imgs = mnist.data.float().div_(255.0).unsqueeze(1)
        self.imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bilinear", align_corners=False)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i].unsqueeze(1) - 0.5


class MultiModeEncoder(nn.Module):
    def __init__(
        self,
        num_frames=10,
        t_patch=2,
        img_size=64,
        patch_size=8,
        in_chans=1,
        dim=96,
        depth=8,
        heads=4,
        levels=4,
    ):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.t_patch = t_patch
        self.patch_size = patch_size
        self.t_grid_vid = num_frames // t_patch
        self.t_grid_img = 1
        self.s_grid = img_size // patch_size
        self.n_patches_vid = self.t_grid_vid * self.s_grid * self.s_grid
        self.n_patches_img = self.t_grid_img * self.s_grid * self.s_grid
        self.img_proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.video_proj = nn.Conv3d(
            in_chans,
            dim,
            kernel_size=(t_patch, patch_size, patch_size),
            stride=(t_patch, patch_size, patch_size),
        )
        self.register_buffer("pos_img", sincos_3d(self.t_grid_img, self.s_grid, self.s_grid, dim))
        self.register_buffer("pos_vid", sincos_3d(self.t_grid_vid, self.s_grid, self.s_grid, dim))
        self.img_mod = nn.Parameter(torch.zeros(1, 1, dim))
        self.video_mod = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.img_mod, std=1e-6)
        nn.init.normal_(self.video_mod, std=1e-6)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_layers = [round((i + 1) * depth / levels) - 1 for i in range(levels)]
        self.out_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in self.out_layers])

    def _tokenize(self, x):
        if x.ndim != 5:
            raise ValueError(f"expected (B,C,T,H,W), got shape {tuple(x.shape)}")
        if x.size(2) == 1:
            tokens = self.img_proj(x[:, :, 0]).flatten(2).transpose(1, 2)
            pos = self.pos_img
            mod = self.img_mod
            mode = "image"
        else:
            tokens = self.video_proj(x).flatten(2).transpose(1, 2)
            pos = self.pos_vid
            mod = self.video_mod
            mode = "video"
        return tokens, pos, mod, mode

    def forward(self, x, idx=None, return_hier=True):
        tokens, pos, mod, _ = self._tokenize(x)
        B, N, D = tokens.shape
        if idx is None:
            idx = torch.arange(N, device=x.device).expand(B, -1)
        x = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)) + pos[idx] + mod
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_layers:
                j = self.out_layers.index(i)
                outs.append(self.out_norms[j](x))
        if return_hier:
            return torch.cat(outs, dim=-1)
        return self.final_norm(x)


class DensePredictor(nn.Module):
    def __init__(self, t_grid_vid, s_grid, enc_dim=96, levels=4, dim=128, depth=4, heads=4):
        super().__init__()
        self.out_dim = enc_dim * levels
        self.in_proj = nn.Linear(self.out_dim, dim)
        self.out_pred = nn.Linear(dim, self.out_dim)
        self.out_ctx = nn.Linear(dim, self.out_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_img", sincos_3d(1, s_grid, s_grid, dim))
        self.register_buffer("pos_vid", sincos_3d(t_grid_vid, s_grid, s_grid, dim))
        self.img_mod = nn.Parameter(torch.zeros(1, 1, dim))
        self.video_mod = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.img_mod, std=1e-6)
        nn.init.normal_(self.video_mod, std=1e-6)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _pos_and_mod(self, mode):
        return (self.pos_img, self.img_mod) if mode == "image" else (self.pos_vid, self.video_mod)

    def forward(self, ctx, ctx_idx, tgt_idx, mode):
        pos, mod = self._pos_and_mod(mode)
        B, N_ctx, _ = ctx.shape
        N_tgt = tgt_idx.size(1)
        x_ctx = self.in_proj(ctx) + pos[ctx_idx] + mod
        x_tgt = self.mask_token.expand(B, N_tgt, -1) + pos[tgt_idx] + mod
        x = torch.cat([x_ctx, x_tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_pred(x[:, -N_tgt:]), self.out_ctx(x[:, :N_ctx])


def _bsize(g, s, ar=1.0):
    a = s * g * g
    return (
        max(1, min(g, round(math.sqrt(a * ar)))),
        max(1, min(g, round(math.sqrt(a / ar)))),
    )


def _expand_tubes(spatial_cells, t_grid, s_grid):
    return sorted(t * s_grid * s_grid + p for t in range(t_grid) for p in spatial_cells)


def _sample_spatial_tubes(n_blocks, h, w, s_grid, rng, min_visible_cells):
    all_spatial = set(range(s_grid * s_grid))
    best = None
    for _ in range(50):
        masked = set()
        for _ in range(n_blocks):
            top = rng.randint(0, s_grid - h)
            left = rng.randint(0, s_grid - w)
            masked.update(r * s_grid + c for r in range(top, top + h) for c in range(left, left + w))
        visible = all_spatial - masked
        if best is None or len(visible) > len(best[1]):
            best = (masked, visible)
        if len(visible) >= min_visible_cells:
            return masked, visible
    return best


def sample_vjepa_masks(B, t_grid, s_grid, rng=None, min_ctx=8, ar_range=(0.75, 1.5)):
    rng = rng or random
    min_visible_cells = max(1, math.ceil(min_ctx / t_grid))
    groups = []
    for label, n_blocks, scale in MASK_GROUPS:
        h, w = _bsize(s_grid, scale, rng.uniform(*ar_range))
        ctx_spatial, pred_spatial = [], []
        for _ in range(B):
            masked, visible = _sample_spatial_tubes(n_blocks, h, w, s_grid, rng, min_visible_cells)
            ctx_spatial.append(sorted(visible))
            pred_spatial.append(sorted(masked))
        Lc = min(len(c) for c in ctx_spatial)
        Lp = min(len(p) for p in pred_spatial)
        ctx = [_expand_tubes(sorted(rng.sample(c, Lc)), t_grid, s_grid) for c in ctx_spatial]
        pred = [_expand_tubes(sorted(rng.sample(p, Lp)), t_grid, s_grid) for p in pred_spatial]
        groups.append(
            {
                "label": label,
                "n_blocks": n_blocks,
                "block_hw": (h, w),
                "ctx": ctx,
                "pred": pred,
            }
        )
    return groups


def _token_coords(idx, s_grid):
    sg2 = s_grid * s_grid
    t = idx // sg2
    rc = idx % sg2
    r = rc // s_grid
    c = rc % s_grid
    return torch.stack([t.float(), r.float(), c.float()], dim=-1)


def context_distance_weights(ctx_idx, tgt_idx, s_grid):
    d = torch.cdist(_token_coords(ctx_idx, s_grid), _token_coords(tgt_idx, s_grid))
    return 1.0 / d.min(dim=-1).values.clamp_min(1.0).sqrt()


def weighted_l1(pred, tgt, weights=None):
    err = (pred - tgt).abs()
    return err.mean() if weights is None else (err * weights.unsqueeze(-1)).mean()


def _step_loss(frames, mode, ctx_enc, tgt_enc, predictor, rng, lambda_ctx, weight_distance):
    t_grid = 1 if mode == "image" else ctx_enc.t_grid_vid
    groups = sample_vjepa_masks(frames.size(0), t_grid, ctx_enc.s_grid, rng=rng, min_ctx=4 if mode == "image" else 8)
    with torch.no_grad():
        full = tgt_enc(frames, return_hier=True)
    pred_losses, ctx_losses = {}, {}
    for g in groups:
        ci = torch.tensor(g["ctx"], device=frames.device)
        pi = torch.tensor(g["pred"], device=frames.device)
        ctx = ctx_enc(frames, ci, return_hier=True)
        pred_tgt, pred_ctx = predictor(ctx, ci, pi, mode=mode)
        tgt_tgt = full.gather(1, pi.unsqueeze(-1).expand(-1, -1, full.size(-1)))
        tgt_ctx = full.gather(1, ci.unsqueeze(-1).expand(-1, -1, full.size(-1)))
        pred_losses[g["label"]] = weighted_l1(pred_tgt, tgt_tgt)
        weights = context_distance_weights(ci, pi, ctx_enc.s_grid) if weight_distance else None
        ctx_losses[g["label"]] = weighted_l1(pred_ctx, tgt_ctx, weights)
    loss_pred = sum(pred_losses.values()) / len(pred_losses)
    loss_ctx = sum(ctx_losses.values()) / len(ctx_losses)
    return loss_pred + lambda_ctx * loss_ctx, loss_pred, loss_ctx


def train(
    epochs=4,
    batch_size=24,
    img_batch_size=64,
    lr=3e-4,
    wd=0.05,
    lambda_ctx=0.5,
    ema_start=0.998,
    ema_end=1.0,
    weight_distance=True,
    device=None,
):
    device = device or pick_device()
    print(f"device: {device}")
    video_ds = MovingMNISTVideos(num_frames=10)
    image_ds = MNISTImages(img_size=64)
    video_loader = DataLoader(video_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    image_loader = DataLoader(image_ds, batch_size=img_batch_size, shuffle=True, num_workers=2, drop_last=True)
    ctx_enc = MultiModeEncoder().to(device)
    tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    predictor = DensePredictor(ctx_enc.t_grid_vid, ctx_enc.s_grid, enc_dim=ctx_enc.dim, levels=ctx_enc.levels).to(device)
    print(
        f"video grid: t={ctx_enc.t_grid_vid} s={ctx_enc.s_grid} -> {ctx_enc.n_patches_vid} tokens | "
        f"image grid: t={ctx_enc.t_grid_img} s={ctx_enc.s_grid} -> {ctx_enc.n_patches_img} tokens"
    )
    opt = torch.optim.AdamW(param_groups([ctx_enc, predictor], wd), lr=lr)
    steps_per_epoch = min(len(video_loader), len(image_loader)) * 2
    total = epochs * steps_per_epoch
    rng = random.Random(0)
    losses = {
        "video_pred": [],
        "video_ctx": [],
        "video_total": [],
        "image_pred": [],
        "image_ctx": [],
        "image_total": [],
        "total": [],
    }
    step = 0
    for epoch in range(epochs):
        for videos, images in zip(video_loader, image_loader):
            for mode, frames in (("video", videos), ("image", images)):
                frames = frames.to(device)
                loss, lp, lc = _step_loss(
                    frames,
                    mode,
                    ctx_enc,
                    tgt_enc,
                    predictor,
                    rng,
                    lambda_ctx,
                    weight_distance,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                m = ema_start + (ema_end - ema_start) * (step / max(1, total - 1))
                ema_update(tgt_enc, ctx_enc, m)
                losses[f"{mode}_pred"].append(lp.item())
                losses[f"{mode}_ctx"].append(lc.item())
                losses[f"{mode}_total"].append(loss.item())
                losses["total"].append(loss.item())
                if step % 25 == 0:
                    print(
                        f"ep={epoch} step={step:5d} mode={mode:5s} pred={lp.item():.4f} "
                        f"ctx={lc.item():.4f} total={loss.item():.4f} ema={m:.4f}"
                    )
                step += 1
    return {
        "ctx_enc": ctx_enc,
        "tgt_enc": tgt_enc,
        "predictor": predictor,
        "losses": losses,
        "video_loader": video_loader,
        "image_loader": image_loader,
        "device": device,
    }


if __name__ == "__main__":
    train()
