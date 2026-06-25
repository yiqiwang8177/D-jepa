import torch

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