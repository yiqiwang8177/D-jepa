"""Minimal V-JEPA2: V-JEPA pretrain + frozen block-causal action/state AC predictor."""
import numpy as np
from PIL import Image
from tqdm import tqdm 
from einops import rearrange
import os, copy, math, random, wandb, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import torchvision.transforms as T
from utils import random_drop_mask, random_spatial_mask

MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7), ("ctr", 1, 1)]


def visualize_attnmap(map, size=(96, 96)):
    # map: (s_grid, s_grid)
    # size: (H, W)
    attn_map = (map - map.min()) / (map.max() - map.min() + 1e-8)  # normalize to [0, 1]
    attn_map = cv2.resize(attn_map, size, interpolation=cv2.INTER_CUBIC)  # upsample to image size
    attn_map = (attn_map * 255).astype(np.uint8)  # convert to uint8
    attn_map = cv2.applyColorMap(attn_map, cv2.COLORMAP_OCEAN)       # (H, W, 3) BGR
    attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)   # → RGB

    return Image.fromarray(attn_map)

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

def sample_j(i, N, gap, rng):
    """
    Sample j from [0, N) such that abs(i - j) > gap.
    """
    # valid range: [0, i-gap-1] and [i+gap+1, N-1]
    left  = list(range(0, max(0, i - gap)))
    right = list(range(min(N, i + gap + 1), N))
    valid = left + right
    # print(i, N, gap, valid)
    return rng.choice(valid)

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

class VideoRandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        """
        x: (B, C, T, H, W)

        Returns:
            (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        assert H == W

        # Merge time into batch dimension
        x = x.permute(0, 2, 1, 3, 4)      # B,T,C,H,W
        x = x.reshape(B * T, C, H, W)

        # Pad
        x = F.pad(
            x,
            (self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )

        padded_size = H + 2 * self.pad

        eps = 1.0 / padded_size

        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            padded_size,
            device=x.device,
            dtype=x.dtype,
        )[:H]

        arange = arange.unsqueeze(0).repeat(H, 1).unsqueeze(2)

        base_grid = torch.cat(
            [arange, arange.transpose(1, 0)],
            dim=2,
        )

        base_grid = base_grid.unsqueeze(0)  # (1,H,W,2)

        # ONE shift per video
        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(B, 1, 1, 2),
            device=x.device,
        ).to(x.dtype)

        shift *= 2.0 / padded_size

        grid = base_grid + shift  # (B,H,W,2)

        # Repeat same grid across all frames
        grid = (
            grid[:, None]                 # B,1,H,W,2
            .expand(-1, T, -1, -1, -1)   # B,T,H,W,2
            .reshape(B * T, H, W, 2)
        )

        x = F.grid_sample(
            x,
            grid,
            padding_mode="zeros",
            align_corners=False,
        )

        x = x.reshape(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        return x
    
class RobomimicVideos(Dataset):
    def __init__(self, cfg, rng, robots=['panda', 'kinova3', 'ur5e'], tasks=['can', 'lift', 'square'], stats_root = "/zfsauton/scratch/yiqiw2/robomimic_datasets_stats/robomimic_multi_new", png_root="/zfsauton/scratch/yiqiw2/robomimic_datasets", num_frames=8, img_size=128, episodes=100, ctr_shift=False, shift=2):
        self.cfg = cfg
        self.rng = rng
        self.png_root = png_root; self.num_frames = num_frames; self.img_size = img_size; self.ctr_shift = ctr_shift
        self.stats_root = stats_root
        self.indices = []
        for robot in robots:
            for task in tasks:
                stats_path = f"{stats_root}/{task}_png/{task}_{robot}_train.npz"
                traj_lengths = np.load(stats_path)["traj_lengths"][:episodes]
                self.png_dir = f"{png_root}/{task}_{robot}/fronts"
                self.indices.extend(self.get_indices(traj_lengths))
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()]) # 0-255 --> 0-1 by T.ToTensor
        self.shift = shift

    def get_indices(self, traj_lengths):
        indices = []
        frame_start = 0
        vid = 0
        for traj_len in traj_lengths:
            traj_len = int(traj_len)
            # Slide a window of size num_frames across this episode.
            # The last valid window starts at (traj_len - num_frames).
            for offset in range(traj_len - self.num_frames + 1):
                window = [
                    os.path.join(self.png_dir, f"{frame_start + offset + i}.png" ) for i in range(self.num_frames) ]
                indices.append([window, vid])
            frame_start += traj_len
            vid += 1
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def get_valid_neighbor(self, vid, candidates):
        valid_candidates = [candidate for candidate in candidates if candidate[-1] == vid]
        return self.rng.choice(valid_candidates)
    
    def __getitem__(self, idx):
        frame_paths, vid = self.indices[idx]
        frames, ctr_video = [], None
        for path in frame_paths:
            img = Image.open(path)
            frames.append(self.transform(img)) # range [0, 1]
        if self.ctr_shift:
            # get a shifted neighbor (left/right)
            neighbors = [self.indices[idx-i] for i in range(self.num_frames, self.num_frames*self.shift)] 
            neighbors += [self.indices[idx+j] for j in range(self.num_frames, self.num_frames*self.shift) if idx+j < len(self.indices) ]
            ctr_frame_paths, ctr_vid = self.get_valid_neighbor(vid, neighbors )
            ctr_frames = [ self.transform(Image.open(ctr_frame_path)) for ctr_frame_path in ctr_frame_paths]
            ctr_video = torch.stack(ctr_frames).permute(1, 0, 2, 3) - 0.5
        # Stack → (num_frames, C, H, W) -> (C num_frames H W)
        video = torch.stack(frames).permute(1, 0, 2, 3) - 0.5
        if ctr_video is not None:
            video = torch.cat([video, ctr_video], 0)
        return video

class VideoEncoder(nn.Module):
    def __init__(self, num_frames=10, t_patch=2, img_size=64, patch_size=8,
                 in_chans=3, dim=128, depth=6, heads=4, num_registers = 4):
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
        self.num_registers = num_registers
        # Add registers number of learnable tokens at the beginning of the sequence.
        self.registers = nn.Parameter(torch.zeros(1, num_registers, dim)) if num_registers > 0 else None
        # self.registers = nn.Parameter(torch.randn(1, num_registers, dim) * 0.02) if num_registers > 0 else None

    def forward(self, videos, idx=None):
        tokens = self.tubelet_proj(videos).flatten(2).transpose(1, 2)
        B, N, D = tokens.shape

        if idx is None:
            idx = torch.arange(N, device=videos.device).expand(B, -1); x = tokens + self.pos[idx]
        else:
            x = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)) + self.pos[idx]
        if self.num_registers > 0:
            x = torch.cat([self.registers.expand(B, -1, -1), x], dim=1)

        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        if self.num_registers > 0:
            x = x[:, self.num_registers:]
        return x

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

class DINOCentering(nn.Module):
    def __init__(self, out_dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum

        self.register_buffer(
            "center",
            torch.zeros(1, out_dim)
        )

    @torch.no_grad()
    def update_center(self, teacher_logits):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)

        self.center.mul_(self.momentum).add_(
            batch_center,
            alpha=1 - self.momentum
        )

    def forward(self, teacher_logits, teacher_temp):
        centered_logits = teacher_logits - self.center
        return F.softmax(centered_logits / teacher_temp, dim=-1)
    
class TransformerPooling(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=512,
        out_dim=256,
        num_heads=8,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.pooler = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.out = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens, masks=None):
        """
        Args:
            tokens: (B, N, D)
            masks:  (B, N), True = visible token

        Returns:
            logits: (B, out_dim)
            cls_repr: (B, D)
        """
        B, N, D = tokens.shape

        # prepend CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls_token, tokens], dim=1)     # (B,N+1,D)

        key_padding_mask = None
        if masks is not None:
            if masks.ndim == 3:
                masks = masks.squeeze(1)

            cls_mask = torch.ones(
                B,
                1,
                dtype=masks.dtype,
                device=masks.device,
            )

            masks = torch.cat([cls_mask, masks], dim=1)

            # Transformer expects:
            # True = ignore token
            key_padding_mask = ~masks

        x = self.pooler( x,  src_key_padding_mask=key_padding_mask,  )
        cls_repr = x[:, 0]  # (B,D)
        logits = self.out(cls_repr)

        return logits, cls_repr

class ACPredictor(nn.Module):
    """Block-causal action/state-conditioned predictor for next latent frame."""

    def __init__(self, s_grid, enc_dim=128, dim=128, depth=4, heads=4, action_dim=2, ):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, enc_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # self.state_proj = nn.Sequential(nn.Linear(state_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer("pos", sincos_2d(s_grid, s_grid, dim))
        self.register_buffer("act_token", torch.zeros(1, 1, dim))
        self.register_buffer("state_token", torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([CausalBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    @staticmethod
    def _block_causal_mask(T, width, device):
        t = torch.arange(T, device=device).repeat_interleave(width)
        return t[:, None] < t[None, :]  # True entries are masked by MultiheadAttention.

    def forward(self, z, actions):
        B, T, S, _ = z.shape
        a = self.action_proj(actions).unsqueeze(2) + self.act_token
        x = self.in_proj(z) + self.pos[None, None]
        x = torch.cat([a, x], dim=2).reshape(B, T * (S + 1), -1)
        mask = self._block_causal_mask(T, S + 1, z.device)
        for blk in self.blocks: x = blk(x, mask)
        x = self.norm(x).view(B, T, S + 1, -1)[:, :, 1:]
        return self.out_proj(x)

    def rollout(self, z0, actions):
        z_seq = z0[:, None]; out = []
        for k in range(actions.size(1)):
            pred = self.forward(z_seq, actions[:, :k + 1])[:, -1]
            z_seq = torch.cat([z_seq, pred[:, None]], dim=1); out.append(pred)
        return torch.stack(out, dim=1)
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    
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
        if label != "ctr":
            for _ in range(B):
                masked, visible = _sample_spatial_tubes(n_blocks, h, w, s_grid, rng, min_visible_cells)
                ctx_spatial.append(sorted(visible)); pred_spatial.append(sorted(masked))
            # all items in the batch have identical-length index lists
            Lc, Lp = min(len(c) for c in ctx_spatial), min(len(p) for p in pred_spatial)
            
            ctx = [_expand_tubes(sorted(rng.sample(c, Lc)), t_grid, s_grid) for c in ctx_spatial]
            pred = [_expand_tubes(sorted(rng.sample(p, Lp)), t_grid, s_grid) for p in pred_spatial]
        else:
            ctx, pred = [], []
            # randomly masked out all patches at frame i, and set all patches not from frame i to be visible context.
            for _ in range(B):
                i = rng.randint(0, t_grid - 1) 
                visible = [p for p in range(t_grid * s_grid * s_grid) if not (i * s_grid * s_grid <= p < (i + 1) * s_grid * s_grid)]
                ctx.append(visible)
                masked = [p for p in range(t_grid * s_grid * s_grid) if (i * s_grid * s_grid <= p < (i + 1) * s_grid * s_grid)]
                pred.append(masked)
        groups.append({"label": label, "n_blocks": n_blocks, "block_hw": (h, w),
                       "ctx": ctx, "pred": pred})
    return groups

def pretrain(cfg, loader, val_loader, rng, epochs, device, lr=3e-4, wd=0.05, ema_start=0.99925, ema_end=0.99925, logger = None, encoder_path=None):
    print("=== Phase 1: V-JEPA pretraining ===")
    if cfg.augment:
        video_aug = VideoRandomShiftsAug(cfg.patch_size)
    ctx_enc = VideoEncoder(num_frames=cfg.num_frames, img_size=cfg.img_size, patch_size=cfg.patch_size).to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    centering =  DINOCentering(256).to(device)  # for centering teacher output in contrastive distillation
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = JEPAPredictor(t_grid=ctx_enc.t_grid, s_grid=ctx_enc.s_grid).to(device)
    pool = TransformerPooling(ctx_enc.dim).to(device); tgt_pool = copy.deepcopy(pool).to(device)
    for p in tgt_pool.parameters(): p.requires_grad_(False)
    print(f"tubelet grid: t={ctx_enc.t_grid} s={ctx_enc.s_grid} -> {ctx_enc.n_patches} patches")
    params_to_opt = [ctx_enc, pred]
    if cfg.name == 'djepa':
        params_to_opt += [pool]

    opt = torch.optim.AdamW(param_groups(params_to_opt, wd), lr=lr)
    total = epochs * len(loader)
    losses = {g[0]: [] for g in MASK_GROUPS}; step = 0; D = ctx_enc.dim

    tt_start, tt_end, tt_warm = cfg.tt_min, cfg.tt_max, cfg.tt_warm
    # create teacher temp schedule for each epochss that starts from tt_start, ends at tt_end, and warms up for tt_warm epochs.
    tt_schedule = [tt_start + (tt_end - tt_start) * min(1, epoch / tt_warm) for epoch in range(epochs)]
    for epoch in range(epochs):
        pbar = tqdm(loader)
        for videos in pbar:
            if cfg.augment:
                videos = video_aug(videos)
                
            videos = videos.to(device, non_blocking=True); B = videos.size(0)
            
            groups = sample_vjepa_masks(B, ctx_enc.t_grid, ctx_enc.s_grid, rng=rng)
            with torch.no_grad(): full = F.layer_norm(tgt_enc(videos), (D,))
            per = {}
            if 'vjepa' not in cfg.name:
                ctr_loss = torch.tensor(0).to(device).float() # accumulate across groups (different masks)
            for g in groups:
                ci = torch.tensor(g["ctx"], device=device); pi = torch.tensor(g["pred"], device=device)
                tgt = full.gather(1, pi.unsqueeze(-1).expand(-1, -1, D))
                encoded_patches = ctx_enc(videos, ci) 
                pred_patches = pred(encoded_patches, ci, pi)
                # Vjepa case
                # no prediction loss for ctr frame mask
                per[g["label"]] = (pred_patches - tgt).abs().mean() if g["label"] != 'ctr' else 0
                print(g["label"], tgt.shape)
                # Djepa case
                if 'vjepa' not in cfg.name and g["label"] == 'ctr':
                    for i in range(cfg.sem_distill):
                        local_patches = pred_patches # b (t m) d where t=1
                        
                        if g["label"] == 'ctr':  # frame-to-frame comparisons, no time axis         
                            global_patches = tgt; tgt_h, tgt_w = 8, 8 # 10,10
                            if i == 0 or i < cfg.sem_distill -1:
                                st_h, st_w = 4, 4
                            else:
                                st_h, st_w = tgt_h, tgt_w
                            st_mask = random_spatial_mask(B=B, h=st_h, w=st_w, H=ctx_enc.s_grid, W=ctx_enc.s_grid, device=device) # preserves 25%=36/144 of patches
                            tgt_mask = random_spatial_mask(B=B, h=tgt_h, w=tgt_w, H=ctx_enc.s_grid, W=ctx_enc.s_grid, device=device) # preserves 70%=100/144 of patches
                        else: # combine time axis with batch to achieve frame-to-frame comparisons
                            global_patches = rearrange(full, 'B (T N) d -> (B T) N d', T=ctx_enc.t_grid)
                            local_patches = rearrange(local_patches, 'B (T M) d -> (B T) M d', T=ctx_enc.t_grid)
                            st_mask = None
                            tgt_mask = random_spatial_mask(B=B*ctx_enc.t_grid, h=10, w=10, H=ctx_enc.s_grid, W=ctx_enc.s_grid, device=device) # preserves 70%=100/144 of patches
                        # Aligned comparisons for non-dynamics
                        with torch.no_grad(): p2 = centering(tgt_pool(global_patches, masks=tgt_mask)[0], tt_schedule[epoch]).detach()
                        # print(g["label"], local_patches.shape, global_patches.shape, tgt_mask.shape)
                        p1 = F.softmax(pool(local_patches, masks=st_mask)[0] / cfg.ts, dim=-1)  
                        loss = -(p2 * (p1 + 1e-8).log()).sum(dim=-1).mean()
                        ctr_loss += loss # accumulate to the ctr loss
               
            per['ctr'] = ctr_loss

            loss = sum(per.values()) / len(per)
            opt.zero_grad(); loss.backward(); opt.step()
            m = ema_start + (ema_end - ema_start) * (step / max(1, total - 1))
            ema_update(tgt_enc, ctx_enc, m); ema_update(tgt_pool, pool, m)
            if 'ctr' in cfg.suffix:
                logits_to_update = p2 # if not cfg.ctr_distill else torch.cat([p2, aligned_p2], dim=0)
                centering.update_center(logits_to_update)
            for k, v in per.items(): losses[k].append(v.item())
            
            if step % 25 == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in per.items())
                pbar.set_postfix_str(f"ep={epoch} {msg} ema={m:.4f}")
                if logger:
                    logger.log({"epoch": epoch, "step": step, "ema": m, 'temp_teacher': tt_schedule[epoch], **{f"loss_{k}": v.item() for k, v in per.items()}})
                    
            step += 1

        if logger:
            visuals, visual_interval = [], cfg.num_frames * 2
            for i, (videos) in enumerate(val_loader):
                if i % visual_interval == 0:
                    videos = videos.to(device)

                    with torch.no_grad():
                        tokens = tgt_enc(videos) # use the teacher instead of the student

                        frame_specs = [
                            (0, slice(0, ctx_enc.s_grid * ctx_enc.s_grid)),      # first frame
                            (-1, slice(-ctx_enc.s_grid * ctx_enc.s_grid, None)), # last frame
                        ]

                        for frame_idx, token_slice in frame_specs:

                            frame = videos[0, :, frame_idx].permute(1, 2, 0).cpu()+ 0.5  
                            frame = (frame.clamp(0, 1).numpy() * 255).astype(np.uint8)
                            frame_tokens = tokens[:1, token_slice] # take the first sample
                            # attn_map = tgt_pool(frame_tokens)[1].reshape(ctx_enc.s_grid, ctx_enc.s_grid).cpu().numpy()
                            pca_frame = visualize_tokens_pca( frame_tokens, h=ctx_enc.s_grid, w=ctx_enc.s_grid,).resize((cfg.img_size, cfg.img_size))
                            # attn_visual = visualize_attnmap( attn_map, size=(cfg.img_size, cfg.img_size), )
                            to_add = [  frame, np.array(pca_frame),] # np.array(attn_visual),]
                            
                            visuals.append(np.concatenate( to_add, axis=1,) )

            visuals = np.array(visuals)
            # N x H x W x C -> (N*H) x W x C
            h, w = visuals.shape[1], visuals.shape[2]
            visuals = visuals.reshape(-1, w, 3)
            logger.log({"visual": wandb.Image(visuals), "epoch": epoch, "step": step})
        torch.save( {'epoch':epoch, 'encoder': tgt_enc.state_dict()}, encoder_path)
    return tgt_enc, losses

class cfg:
    img_size = 96
    patch_size = 8 # 8 works for robomimic
    num_frames = 8
    phase1_epochs = 50
    phase2_epochs = 0 # 3 # 30
    batch_size= 128 # 128 # 64
    episodes = 100
    ts = 0.1 # student temp 
    tt_min = 0.04 # teacher temp start
    tt_max = 0.07 # teacher temp end
    tt_warm = 3 # epochs
    attn_reg = 0 # regularize entropy
    ctr_shift = False
    ctr_distill = False
    ctr_full = False
    ctr_ctxonly = False
    ctr_invert = False
    ctr_drop = 0.7
    ctr_tgt_drop = 0.3 # drop teacher patches by x%

    # semantic consistency
    sem_distill = 4
    augment = False

    # Logging 
    save_dir = './log'
    name='djepa' # or djepa
    suffix = 'ctr'
    suffix2 = 'sem4_st4_tgt8'
    use_wandb = True
    
def main(cfg,  device=None):
    if 'vjepa' in cfg.name:
        global MASK_GROUPS
        MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7)]
        cfg.suffix = cfg.suffix2 = ''; cfg.ctr_shift = False

    device = device or pick_device(); print(f"device: {device}")
    rng = random.Random(0)
    ds = RobomimicVideos(episodes=cfg.episodes, cfg= cfg, rng=rng, img_size=cfg.img_size, num_frames=cfg.num_frames, ctr_shift=cfg.ctr_shift)
    val_ds = RobomimicVideos(robots=['panda'], episodes=1, cfg= cfg, rng=random.Random(0), img_size=cfg.img_size, num_frames=cfg.num_frames)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=8, prefetch_factor=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    config_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("__")}
    exp_name = f"{cfg.name}-pretrain-{cfg.suffix}-{cfg.suffix2}"
    logger = init_logger(
        cfg.use_wandb,
        project="djepa-playground",
        name=exp_name,
        config=config_dict
    ); save_dir = os.path.join(cfg.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = os.path.join(save_dir, 'encoder.pth'); pi_ac_path = os.path.join(save_dir, 'pi_ac.pth')
    encoder, p1 = pretrain(cfg, loader, val_loader, rng, cfg.phase1_epochs, device, logger=logger, encoder_path=encoder_path)
        # torch.save( {'epoch': cfg.phase1_epochs, 'encoder': encoder.state_dict() }, encoder_path)
    
    return {"encoder": encoder, "loader": loader, "device": device}


if __name__ == "__main__": 
    cfg = cfg()
    main(cfg)
