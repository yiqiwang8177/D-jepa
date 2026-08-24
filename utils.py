import math, random, torch, numpy as np
from einops import rearrange 

def random_drop_mask(B, S, p, device='cpu'):
    """
    Args:
        B: batch size
        S: sequence length (number of patches)
        p: drop probability, e.g. 0.3 drops 30% of patches
        device: torch device
    Returns:
        mask: (B, S) boolean tensor, True = keep, False = drop
    """
    return torch.rand(B, S, device=device) >= p

def random_spatial_mask(B, h, w, H, W, device="cpu"):
    """
    Args:
        B: batch size
        h, w: size of the kept rectangle
        H, W: full spatial dimensions
        device: torch device

    Returns:
        mask: (B, H*W) bool tensor
              Exactly h*w entries are True per sample.
    """
    assert h <= H and w <= W

    # Sample top-left corner for each batch element
    top = torch.randint(0, H - h + 1, (B,), device=device)
    left = torch.randint(0, W - w + 1, (B,), device=device)

    # Coordinates within the rectangle
    rows = torch.arange(h, device=device).view(1, h, 1)
    cols = torch.arange(w, device=device).view(1, 1, w)

    # Absolute coordinates of kept pixels
    y = top.view(B, 1, 1) + rows      # (B, h, w)
    x = left.view(B, 1, 1) + cols     # (B, h, w)

    # Convert to flattened indices
    idx = (y * W + x).reshape(B, h * w)

    # Build mask
    mask = torch.zeros(B, H * W, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)

    return mask

def get_random_h_w(rng, low=1, high=10, min_size=5, max_size=130):
    valid_pairs = [
        (h, w)
        for h in range(low, high + 1)
        for w in range(low, high + 1)
        if min_size <= h * w <= max_size
    ]

    if not valid_pairs:
        raise ValueError(
            f"No feasible (h, w) exists with "
            f"h,w in [{low}, {high}] and "
            f"area in [{min_size}, {max_size}]"
        )

    return rng.choice(valid_pairs)

def set_seed(seed: int = 42):
    random.seed(seed)                  # Python's built-in RNG
    np.random.seed(seed)               # NumPy global RNG
    torch.manual_seed(seed)            # CPU RNG (also seeds CUDA in recent versions)
    torch.cuda.manual_seed_all(seed)   # all GPUs, explicit for safety

def dynamic_spatial_mask( B, h, w, H, W,
    activate,
    N=20,
    occup=0.33,
    device="cpu",
):
    """
    Args:
        B: batch size
        h, w: size of the kept rectangle
        H, W: full spatial dimensions
        activate: (H*W,) bool tensor indicating activated locations
        N: number of candidate masks sampled per batch element
        occup: minimum occupancy ratio inside activate
        device: torch device

    Returns:
        mask: (B, H*W) bool tensor
        valid: (B,) bool tensor
            True if at least one valid candidate was found.
    """
    assert h <= H and w <= W
    assert activate.shape == (B, H * W)
    activate = activate.to(device=device, dtype=torch.bool)
    # ------------------------------------------------------------------
    # Sample N candidate rectangles for each batch element
    # ------------------------------------------------------------------
    top = torch.randint(0, H - h + 1, (B, N), device=device)
    left = torch.randint(0, W - w + 1, (B, N), device=device)
    rows = torch.arange(h, device=device).view(1, 1, h, 1)
    cols = torch.arange(w, device=device).view(1, 1, 1, w)
    y = top[:, :, None, None] + rows          # (B, N, h, w)
    x = left[:, :, None, None] + cols         # (B, N, h, w)
    idx = (y * W + x).reshape(B, N, h * w)    # (B, N, h*w)
    # ------------------------------------------------------------------
    # Compute occupancy
    # ------------------------------------------------------------------
    batch_idx = torch.arange(B, device=device)[:, None, None]  # (B,1,1
    occ_count = activate[batch_idx, idx].sum(dim=-1)  # (B,N)
    occ_ratio = occ_count.float() / (h * w)
    valid_candidates = occ_ratio >= occup     # (B, N)
    # ------------------------------------------------------------------
    rand = torch.rand(B, N, device=device)
    rand = rand.masked_fill(~valid_candidates, -1)
    chosen = rand.argmax(dim=1)
    has_valid = valid_candidates.any(dim=1)
    # ------------------------------------------------------------------
    # Gather selected indices
    # ------------------------------------------------------------------
    selected_idx = idx[
        torch.arange(B, device=device), chosen, ]                                          # (B, h*w)

    mask = torch.zeros(B, H * W, dtype=torch.bool, device=device)
    mask.scatter_(1, selected_idx, True)

    return mask, has_valid

def get_kmean_threshold(x, num_iters=15, eps=1e-6):
    """
    Given per-(timestep, token) statistics, run an independent 2-means clustering
    per batch sample and return the smallest value belonging to the
    higher-valued cluster (i.e. an adaptive high/low threshold per sample).

    Args:
        x: (B, T, N) tensor
        num_iters: max Lloyd iterations
        eps: convergence tolerance on center movement

    Returns:
        (B,) tensor of thresholds.
    """
    B = x.shape[0]
    x = x.reshape(B, -1)  # (B, M)

    # Deterministic init (min/max) instead of random, so results are reproducible.
    c_lo = x.min(dim=1).values
    c_hi = x.max(dim=1).values
    centers = torch.stack([c_lo, c_hi], dim=1)  # (B, 2)
    has_converged = False
    for _ in range(num_iters):
        dist = (x.unsqueeze(-1) - centers.unsqueeze(1)) ** 2  # (B, M, 2)
        labels = dist.argmin(dim=-1)  # (B, M) in {0, 1}

        mask0 = (labels == 0).to(x.dtype)
        mask1 = (labels == 1).to(x.dtype)
        count0 = mask0.sum(dim=1)
        count1 = mask1.sum(dim=1)

        new_c0 = (x * mask0).sum(dim=1) / count0.clamp_min(1.0)
        new_c1 = (x * mask1).sum(dim=1) / count1.clamp_min(1.0)

        # if a cluster lost all its points this iteration, keep its previous center
        new_c0 = torch.where(count0 == 0, centers[:, 0], new_c0)
        new_c1 = torch.where(count1 == 0, centers[:, 1], new_c1)

        new_centers = torch.stack([new_c0, new_c1], dim=1)
        shift = (new_centers - centers).abs().max(); ind_converged = (new_centers - centers).abs().max(dim=1).values < eps
        centers = new_centers
        if shift < eps:
            has_converged = True 
            break

    dist = (x.unsqueeze(-1) - centers.unsqueeze(1)) ** 2
    labels = dist.argmin(dim=-1)  # (B, M)

    high_cluster = centers.argmax(dim=1)  # (B,) which cluster index is higher per row
    high_mask = labels == high_cluster.unsqueeze(1)  # (B, M)

    masked_x = x.masked_fill(~high_mask, float("inf"))
    thresholds = masked_x.min(dim=1).values  # (B,)
    return thresholds, _, has_converged, ind_converged.cpu()

def get_token_diff(full_tokens, T):
    # Get median difference across time
    with torch.no_grad():
        full_tokens = rearrange(full_tokens, 'B (T N) d -> B T N d', T=T)
        B, T_, N, d = full_tokens.shape
        # Pairwise diff across timesteps: (B, T, T, N, d) -> abs -> mean over d -> (B, T, T, N)
        diff = ( full_tokens.unsqueeze(2) - full_tokens.unsqueeze(1) ).abs().mean(dim=-1)
        # Drop the diagonal (t vs itself): (B, T, T, N) -> (B, T, T-1, N)
        off_diag = ~torch.eye(T_, dtype=torch.bool, device=full_tokens.device)
        diff = diff[:, off_diag].reshape(B, T_, T_ - 1, N)
        # Median across the "other timesteps" dimension -> (B, T, N)
        diff = diff.median(dim=2).values
    # B x T x N
    return diff

def save_proposal( id2proposals, ids, proposals, multi_dyn):
    ids = ids.cpu().numpy()
    proposals = proposals.cpu()
    for id, proposal in zip(ids, proposals):
        if id not in id2proposals:
            id2proposals[id] = [  ]
        if len(id2proposals[id]) < multi_dyn:
            id2proposals[id] += [ proposal ]
    return id2proposals

def get_proposal(ids, id2proposals, multi_dyn, rng, device = 'cuda'):
    ids = ids.cpu().numpy()
    exists = [ id in id2proposals and len(id2proposals[id]) == multi_dyn for id in ids]
    all_exists = False not in exists
    if not all_exists:
        return False, np.array( [False] * len(ids)), None, None
    # only start to sample if all exists
    proposals1 = torch.cat( [ rng.choice( id2proposals[id] ).unsqueeze(0)  for id in ids], 0 )
    proposals2 = torch.cat( [ rng.choice( id2proposals[id] ).unsqueeze(0)  for id in ids], 0 )

    return all_exists, np.array( exists), proposals1.to(device), proposals2.to(device)

