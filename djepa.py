"""Minimal V-JEPA2: V-JEPA pretrain + frozen block-causal action/state AC predictor."""
import numpy as np
from PIL import Image
from tqdm import tqdm 
import os, copy, math, random, wandb
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import torchvision.transforms as T

MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7)]

def visualize_tokens_pca(tokens, h=16, w=16, normalize=True):
    """
    Visualize ViT tokens using PCA.

    Args:
        tokens: torch.Tensor of shape (h*w, d) or (B, h*w, d)
        h, w: spatial dimensions
        normalize: whether to normalize output to [0, 1]

    Returns:
        img: (h, w, 3) numpy array
    """
    # Handle batch
    if tokens.dim() == 3:
        tokens = tokens[0]  # take first sample

    assert tokens.shape[0] == h * w, "Token count must match h*w"

    # Move to CPU + numpy
    x = tokens.detach().cpu().numpy()  # (h*w, d)

    # PCA → 3 components
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x)  # (h*w, 3)

    # Normalize per channel
    if normalize:
        x_min = x_pca.min(axis=0, keepdims=True)
        x_max = x_pca.max(axis=0, keepdims=True)
        x_pca = (x_pca - x_min) / (x_max - x_min + 1e-8)

    # Reshape to image
    # img = x_pca.reshape(h, w, 3)

    img = (x_pca.reshape(h, w, 3) * 255).astype(np.uint8)

    return Image.fromarray(img)

class DummyRun:
    def log(self, *args, **kwargs): pass
    def finish(self, *args, **kwargs): pass
    def watch(self, *args, **kwargs): pass

def init_logger(use_wandb, **wandb_kwargs):
    if use_wandb:
        return wandb.init(**wandb_kwargs)
    return DummyRun()

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

class RobomimicVideos(Dataset):
    def __init__(self, robots=['panda', 'kinova3', 'ur5e'], tasks=['lift'], stats_root = "/zfsauton/scratch/yiqiw2/robomimic_datasets_stats/robomimic_multi_new", png_root="/zfsauton/scratch/yiqiw2/robomimic_datasets", num_frames=8, img_size=128, episodes=100):
        self.png_root = png_root; self.num_frames = num_frames; self.img_size = img_size
        self.stats_root = stats_root
        self.indices = []
        for robot in robots:
            for task in tasks:
                stats_path = f"{stats_root}/{task}_png/{task}_{robot}_train.npz"
                traj_lengths = np.load(stats_path)["traj_lengths"][:episodes]
                self.png_dir = f"{png_root}/{task}_{robot}/fronts"
                self.indices.extend(self.get_indices(traj_lengths))
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()]) # 0-255 --> 0-1 by T.ToTensor

    def get_indices(self, traj_lengths):
        indices = []
        frame_start = 0

        for traj_len in traj_lengths:
            traj_len = int(traj_len)
            # Slide a window of size num_frames across this episode.
            # The last valid window starts at (traj_len - num_frames).
            for offset in range(traj_len - self.num_frames + 1):
                window = [
                    os.path.join(self.png_dir, f"{frame_start + offset + i}.png" ) for i in range(self.num_frames) ]
                indices.append(window)
            frame_start += traj_len

        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        frame_paths = self.indices[idx]
        frames = []
        for path in frame_paths:
            img = Image.open(path)
            frames.append(self.transform(img)) # range [0, 1]
        # Stack → (num_frames, C, H, W)
        video = torch.stack(frames).permute(1, 0, 2, 3) - 0.5
        return video

class VideoEncoder(nn.Module):
    def __init__(self, num_frames=10, t_patch=2, img_size=64, patch_size=8,
                 in_chans=3, dim=128, depth=6, heads=4):
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


class CausalBlock(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x, attn_mask):
        h = self.n1(x); x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ACPredictor(nn.Module):
    """Block-causal action/state-conditioned predictor for next latent frame."""

    def __init__(self, s_grid, enc_dim=128, dim=128, depth=4, heads=4, action_dim=2, state_dim=2):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, enc_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.state_proj = nn.Sequential(nn.Linear(state_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer("pos", sincos_2d(s_grid, s_grid, dim))
        self.register_buffer("act_token", torch.zeros(1, 1, dim))
        self.register_buffer("state_token", torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([CausalBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    @staticmethod
    def _block_causal_mask(T, width, device):
        t = torch.arange(T, device=device).repeat_interleave(width)
        return t[:, None] < t[None, :]  # True entries are masked by MultiheadAttention.

    def forward(self, z, actions, states):
        B, T, S, _ = z.shape
        a = self.action_proj(actions).unsqueeze(2) + self.act_token
        st = self.state_proj(states).unsqueeze(2) + self.state_token
        x = self.in_proj(z) + self.pos[None, None]
        x = torch.cat([a, st, x], dim=2).reshape(B, T * (S + 2), -1)
        mask = self._block_causal_mask(T, S + 2, z.device)
        for blk in self.blocks: x = blk(x, mask)
        x = self.norm(x).view(B, T, S + 2, -1)[:, :, 2:]
        return self.out_proj(x)

    def rollout(self, z0, actions, states):
        z_seq = z0[:, None]; out = []
        for k in range(actions.size(1)):
            pred = self.forward(z_seq, actions[:, :k + 1], states[:, :k + 1])[:, -1]
            z_seq = torch.cat([z_seq, pred[:, None]], dim=1); out.append(pred)
        return torch.stack(out, dim=1)


def _bsize(g, s, ar=1.0):
    a = s * g * g
    return (max(1, min(g, round(math.sqrt(a * ar)))), max(1, min(g, round(math.sqrt(a / ar)))))


def _expand_tubes(spatial_cells, t_grid, s_grid):
    return sorted(t * s_grid * s_grid + p for t in range(t_grid) for p in spatial_cells)


def _sample_spatial_tubes(n_blocks, h, w, s_grid, rng, min_visible_cells):
    all_spatial = set(range(s_grid * s_grid))
    best = None
    for _ in range(50):
        masked = set()
        for _ in range(n_blocks):
            top, left = rng.randint(0, s_grid - h), rng.randint(0, s_grid - w)
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
            ctx_spatial.append(sorted(visible)); pred_spatial.append(sorted(masked))
        Lc, Lp = min(len(c) for c in ctx_spatial), min(len(p) for p in pred_spatial)
        ctx = [_expand_tubes(sorted(rng.sample(c, Lc)), t_grid, s_grid) for c in ctx_spatial]
        pred = [_expand_tubes(sorted(rng.sample(p, Lp)), t_grid, s_grid) for p in pred_spatial]
        groups.append({"label": label, "n_blocks": n_blocks, "block_hw": (h, w),
                       "ctx": ctx, "pred": pred})
    return groups


def pretrain(cfg, loader, epochs, device, lr=3e-4, wd=0.05, ema_start=0.99925, ema_end=0.99925, logger = None):
    print("=== Phase 1: V-JEPA pretraining ===")
    ctx_enc = VideoEncoder(num_frames=cfg.num_frames, img_size=cfg.img_size).to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = JEPAPredictor(t_grid=ctx_enc.t_grid, s_grid=ctx_enc.s_grid).to(device)
    print(f"tubelet grid: t={ctx_enc.t_grid} s={ctx_enc.s_grid} -> {ctx_enc.n_patches} patches")
    opt = torch.optim.AdamW(param_groups([ctx_enc, pred], wd), lr=lr)
    total = epochs * len(loader); rng = random.Random(0)
    losses = {g[0]: [] for g in MASK_GROUPS}; step = 0; D = ctx_enc.dim
    for epoch in range(epochs):
        pbar = tqdm(loader)
        for videos in pbar:
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
                # print(f"[phase1] ep={epoch} step={step:5d} {msg} ema={m:.4f}")
                pbar.set_postfix_str(f"ep={epoch} {msg} ema={m:.4f}")
                if logger:
                    logger.log({"epoch": epoch, "step": step, "ema": m, **{f"loss_{k}": v.item() for k, v in per.items()}})
                    
            step += 1
        if logger:
            with torch.no_grad():
                # video: B x C x T x H x W --> B x T x H x W C
                last_frame = videos[0, :, -1].permute(1, 2, 0).cpu()+0.5  # (H, W, C)
                last_frame = ((last_frame).clamp(0, 1).numpy() * 255).astype(np.uint8)
                tokens = ctx_enc(videos)[:1, -ctx_enc.s_grid * ctx_enc.s_grid:]
                pca_last_frame = visualize_tokens_pca(tokens, h=ctx_enc.s_grid, w=ctx_enc.s_grid).resize((cfg.img_size, cfg.img_size))
            # concatenate 
            visual = np.concatenate([last_frame, np.array(pca_last_frame)], axis=1)
            logger.log({"visual": wandb.Image(visual), "epoch": epoch, "step": step})
        
    return tgt_enc, losses

class cfg:
    img_size = 96
    num_frames = 8
    phase1_epochs=100
    batch_size=64

    # Logging 
    use_wandb = True

def main(cfg,  device=None):
    device = device or pick_device(); print(f"device: {device}")
    ds = RobomimicVideos(img_size=cfg.img_size, num_frames=cfg.num_frames)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True)

    logger = init_logger(
        cfg.use_wandb,
        project="djepa-playground",
        name="vjepa2-pretrain",
        config=vars(cfg)
    )

    encoder, p1 = pretrain(cfg, loader, cfg.phase1_epochs, device, logger=logger)
    return {"encoder": encoder, "loader": loader, "device": device, "losses": p1}
if __name__ == "__main__": 
    cfg = cfg()
    main(cfg)
