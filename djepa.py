"""Minimal V-JEPA2: V-JEPA pretrain + frozen block-causal action/state AC predictor."""
import numpy as np
from PIL import Image
from tqdm import tqdm 
import os, copy, math, random, wandb, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import torchvision.transforms as T

MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7), ("motion", 1, 0.15)]
MOTION_range = (0.05, 0.2) # choose top 10 ~30% of patches with movement

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
    def __init__(self, cfg, rng, robots=['panda', 'kinova3', 'ur5e'], tasks=['can', 'lift', 'square'], stats_root = "/zfsauton/scratch/yiqiw2/robomimic_datasets_stats/robomimic_multi_new", png_root="/zfsauton/scratch/yiqiw2/robomimic_datasets", num_frames=8, img_size=128, episodes=100):
        self.cfg = cfg
        self.rng = rng
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

    def compute_flow(self, img1, img2):
        """
        img1, img2:
            PIL images

        return:
            flow tensor of shape (2, H, W)
        """

        # PIL -> numpy grayscale
        img1, img2 = img1.resize((self.img_size, self.img_size)), img2.resize((self.img_size, self.img_size))
        g1 = np.array(img1.convert("L"))
        g2 = np.array(img2.convert("L"))
        patch = self.cfg.patch_size
        H, W = self.img_size, self.img_size

        flow = cv2.calcOpticalFlowFarneback(
            g1,
            g2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
       
        # OpenCV returns H x W x 2
        flow_patch = flow.reshape(
            H // patch,
            patch,
            W // patch,
            patch,
            2
        ).mean(axis=(1, 3))
 
        return flow_patch

    def _flow_center(self, flows):
        # sample a random number between motion range[0] and range[1]
        top_k = self.rng.uniform(*MOTION_range)
        t, h, w, _ = flows.shape
        top_k = int( top_k * h * w )
        mag = np.linalg.norm(flows, axis=-1)
        
        centers = []
        for i in range(t):
            # sort the magnitudes and get the top_k indices
            idx = np.argpartition(mag[i].flatten(), -top_k)[-top_k]
            row, col = divmod(idx, w)
            centers.append([int(col), int(row)])  # (x, y) format
        return centers

    def __getitem__(self, idx):
        frame_paths = self.indices[idx]
        first_img, frames, flows = None, [], []
        for path in frame_paths:
            img = Image.open(path)
            
            frames.append(self.transform(img)) # range [0, 1]
            
            if first_img is None:
                first_img = img
            else:
                flows.append(self.compute_flow(first_img, img))
        # Stack → (num_frames, C, H, W)
        video = torch.stack(frames).permute(1, 0, 2, 3) - 0.5
        flows = np.stack(flows)  # (num_frames-1, H_patch, W_patch, 2)
        flows_norm = np.linalg.norm(flows, axis=-1)  # (num_frames-1, H_patch, W_patch)
        return video, flows_norm
        # centers = self._flow_center(flows)  
        # centers = np.array( [centers[0]] + centers )  # repeat the first center for the first frame
        # return video, centers

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

def _expand_tubes_center(spatial_cells, t_grid, s_grid, center_B, min_visible_cells=2):
    cx0, cy0 = center_B[0, 0], center_B[0, 1]
    result = []

    for t in range(t_grid):
        cx_t, cy_t = center_B[t, 0], center_B[t, 1]
        dx = round(cx_t - cx0)
        dy = round(cy_t - cy0)

        shifted = []
        for p in spatial_cells:
            row, col = divmod(p, s_grid)
            new_row = row + dy
            new_col = col + dx
            if 0 <= new_row < s_grid and 0 <= new_col < s_grid:
                shifted.append(t * s_grid * s_grid + new_row * s_grid + new_col)

        if len(shifted) < min_visible_cells:
            # fall back to unshifted cells for this frame
            # shifted = [t * s_grid * s_grid + p for p in spatial_cells]
            pass

        result.extend(shifted)

    return sorted(result)

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

# def sample_vjepa_masks(B, t_grid, s_grid, rng=None, min_ctx=8, ar_range=(0.75, 1.5), center_Bs=None):
#     rng = rng or random
#     min_visible_cells = max(1, math.ceil(min_ctx / t_grid))
#     groups = []
#     for label, n_blocks, scale in MASK_GROUPS:
#         h, w = _bsize(s_grid, scale, rng.uniform(*ar_range))
#         ctx_spatial, pred_spatial = [], []
#         for _ in range(B):
#             masked, visible = _sample_spatial_tubes(n_blocks, h, w, s_grid, rng, min_visible_cells)
#             ctx_spatial.append(sorted(visible)); pred_spatial.append(sorted(masked))
#         # all items in the batch have identical-length index lists
#         Lc, Lp = min(len(c) for c in ctx_spatial), min(len(p) for p in pred_spatial)
#         if label != "motion":
#             ctx = [_expand_tubes(sorted(rng.sample(c, Lc)), t_grid, s_grid) for c in ctx_spatial]
#             pred = [_expand_tubes(sorted(rng.sample(p, Lp)), t_grid, s_grid) for p in pred_spatial]
#         else:
#             ctx_min_limit, pred_min_limit = max( Lc//10, min_visible_cells), max( Lp//10, min_visible_cells)
            
#             ctx = [_expand_tubes_center(c, t_grid, s_grid, center_B, ctx_min_limit) for c, center_B in zip(ctx_spatial, center_Bs)]
#             pred = [_expand_tubes_center(p, t_grid, s_grid, center_B, pred_min_limit) for p, center_B in zip(pred_spatial, center_Bs)]
#             min_ctx = min(len(c) for c in ctx); min_pred = min(len(p) for p in pred)
#             # sampling leads to unaligned ctx-pred, causing issues
#             ctx = [ sorted(rng.sample(c, min_ctx)) for c in ctx ]
#             pred = [ sorted(rng.sample(p, min_pred)) for p in pred ]
#         groups.append({"label": label, "n_blocks": n_blocks, "block_hw": (h, w),
#                        "ctx": ctx, "pred": pred})
#     return groups

def _sample_motion_mask(t,  flow_norm, topk, min_visible_cells=4):
    h, w = flow_norm.shape
    num_patches = h * w
    # convert local frame indices to global indices with time
    offset = t * num_patches
    k = max(min_visible_cells, int(topk * num_patches))
    idx = np.argpartition(flow_norm.flatten(), -k)[-k:] + offset
    masked = set(idx.tolist())
    visible = set(range(offset, offset + num_patches)) - masked
    
    return list(masked), list(visible)

def sample_vjepa_masks(B, t_grid, s_grid, rng=None, min_ctx=8, ar_range=(0.75, 1.5), flows=None):
    rng = rng or random
    min_visible_cells = max(1, math.ceil(min_ctx / t_grid))
    groups = []
    for label, n_blocks, scale in MASK_GROUPS:
        h, w = _bsize(s_grid, scale, rng.uniform(*ar_range))
        ctx_spatial, pred_spatial = [], []
        if label != "motion":
            for _ in range(B):
                masked, visible = _sample_spatial_tubes(n_blocks, h, w, s_grid, rng, min_visible_cells)
                ctx_spatial.append(sorted(visible)); pred_spatial.append(sorted(masked))
            # all items in the batch have identical-length index lists
            Lc, Lp = min(len(c) for c in ctx_spatial), min(len(p) for p in pred_spatial)
        
            ctx = [_expand_tubes(sorted(rng.sample(c, Lc)), t_grid, s_grid) for c in ctx_spatial]
            pred = [_expand_tubes(sorted(rng.sample(p, Lp)), t_grid, s_grid) for p in pred_spatial]
        else:
            ctx, pred = [], []
            topk = rng.uniform(*MOTION_range) # all frames across batch masked out same amount of patches
            # flows: (B, T-1, H_patch, W_patch)
            index = np.linspace(0, flows.shape[1]-1, t_grid).round().astype(int)  # select t_grid frames from T-1 frames
            for flow in flows: # looping across sample in batch
                masked_sample, visible_sample = [], []
                for j, i in enumerate(index): # looping across frame in sample (match t_grid frames)
                    masked, visible = _sample_motion_mask(j, flow[i], topk)  
                    masked_sample.extend(masked); visible_sample.extend(visible)
                ctx.append(sorted(masked_sample)); pred.append(sorted(visible_sample))
            
        groups.append({"label": label, "n_blocks": n_blocks, "block_hw": (h, w),
                       "ctx": ctx, "pred": pred})
    return groups

def pretrain(cfg, loader, val_loader, rng, epochs, device, lr=3e-4, wd=0.05, ema_start=0.99925, ema_end=0.99925, logger = None):
    print("=== Phase 1: V-JEPA pretraining ===")
    ctx_enc = VideoEncoder(num_frames=cfg.num_frames, img_size=cfg.img_size, patch_size=cfg.patch_size).to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = JEPAPredictor(t_grid=ctx_enc.t_grid, s_grid=ctx_enc.s_grid).to(device)
    print(f"tubelet grid: t={ctx_enc.t_grid} s={ctx_enc.s_grid} -> {ctx_enc.n_patches} patches")
    opt = torch.optim.AdamW(param_groups([ctx_enc, pred], wd), lr=lr)
    total = epochs * len(loader)
    losses = {g[0]: [] for g in MASK_GROUPS}; step = 0; D = ctx_enc.dim
    for epoch in range(epochs):
        pbar = tqdm(loader)
        for videos, flows in pbar:
            videos = videos.to(device, non_blocking=True); B = videos.size(0); flows = flows.numpy()
            groups = sample_vjepa_masks(B, ctx_enc.t_grid, ctx_enc.s_grid, rng=rng, flows=flows)
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
                pbar.set_postfix_str(f"ep={epoch} {msg} ema={m:.4f}")
                if logger:
                    logger.log({"epoch": epoch, "step": step, "ema": m, **{f"loss_{k}": v.item() for k, v in per.items()}})
                    
            step += 1
        if logger:
            visuals, visual_interval = [], cfg.num_frames * 2
            for i, (videos, _) in enumerate(val_loader):
                
                if i % visual_interval == 0:
                    videos = videos.to(device)
                    with torch.no_grad():
                        # video: B x C x T x H x W --> B x T x H x W C
                        first_frame = videos[0, :, 0].permute(1, 2, 0).cpu()+0.5; last_frame = videos[0, :, -1].permute(1, 2, 0).cpu()+0.5  # (H, W, C)
                        first_frame = ((first_frame).clamp(0, 1).numpy() * 255).astype(np.uint8); last_frame = ((last_frame).clamp(0, 1).numpy() * 255).astype(np.uint8)
                        first_tokens = ctx_enc(videos)[:1, :ctx_enc.s_grid * ctx_enc.s_grid];  last_tokens = ctx_enc(videos)[:1, -ctx_enc.s_grid * ctx_enc.s_grid:]
                    pca_first_frame = visualize_tokens_pca(first_tokens, h=ctx_enc.s_grid, w=ctx_enc.s_grid).resize((cfg.img_size, cfg.img_size)); pca_last_frame = visualize_tokens_pca(last_tokens, h=ctx_enc.s_grid, w=ctx_enc.s_grid).resize((cfg.img_size, cfg.img_size))
                    # concatenate 
                    visuals.append( np.concatenate([first_frame, np.array(pca_first_frame)], axis=1) )
                    visuals.append( np.concatenate([last_frame, np.array(pca_last_frame)], axis=1) )
            visuals = np.array(visuals)
            # N x H x W x C -> (N*H) x W x C
            h, w = visuals.shape[1], visuals.shape[2]
            visuals = visuals.reshape(-1, w, 3)
            logger.log({"visual": wandb.Image(visuals), "epoch": epoch, "step": step})
        
    return tgt_enc, losses

class cfg:
    img_size = 96
    patch_size = 8 # 8 works for lifting, too blury for can.
    num_frames = 8
    phase1_epochs=50
    batch_size=128 # 64
    episodes = 100

    # Logging 
    name='vjepa' # or djepa
    suffix = '' # '-motion_topk'
    use_wandb = True

def main(cfg,  device=None):
    device = device or pick_device(); print(f"device: {device}")
    rng = random.Random(0)
    ds = RobomimicVideos(episodes=cfg.episodes, cfg= cfg, rng=rng, img_size=cfg.img_size, num_frames=cfg.num_frames)
    val_ds = RobomimicVideos(robots=['panda'], episodes=1, cfg= cfg, rng=random.Random(0), img_size=cfg.img_size, num_frames=cfg.num_frames)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=8, prefetch_factor=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    logger = init_logger(
        cfg.use_wandb,
        project="djepa-playground",
        name=f"{cfg.name}-pretrain{cfg.suffix}",
        config=vars(cfg)
    )
    if "vjepa" in cfg.name:
        global MASK_GROUPS
        MASK_GROUPS = [("short", 8, 0.15), ("long", 2, 0.7)]
    
    encoder, p1 = pretrain(cfg, loader, val_loader, rng, cfg.phase1_epochs, device, logger=logger)
    return {"encoder": encoder, "loader": loader, "device": device, "losses": p1}
if __name__ == "__main__": 
    cfg = cfg()
    main(cfg)
