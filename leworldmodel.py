"""Minimal LeWorldModel: end-to-end next-embedding MSE + SIGReg, no EMA/stop-grad."""
import math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def param_groups(modules, wd):
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]


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


class SIGReg(nn.Module):
    """Single-GPU SIGReg / Epps-Pulley statistic with Gaussian-windowed quadrature."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32); w[[0, -1]] = dt   # trapezoid weights
        phi = torch.exp(-t.square() / 2.0)                                       # N(0,1) char fn at knots
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)

    def forward(self, proj):                                                     # proj: (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)        # fresh projections per step ("sketch")
        A = A.div_(A.norm(p=2, dim=0))                                           # unit-norm
        x_t = (proj @ A).unsqueeze(-1) * self.t                                  # (T, B, P, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()                     # mean(-3) averages over batch B


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Projector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        shape = x.shape
        y = self.linear(x.reshape(-1, shape[-1]))
        if self.training and y.size(0) == 1:
            y = F.batch_norm(y, self.bn.running_mean, self.bn.running_var,
                             self.bn.weight, self.bn.bias, training=False)
        else:
            y = self.bn(y)
        return y.view(shape)


class CondBlock(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.heads = heads; self.dh = dim // heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.ada[-1].weight, 0); nn.init.constant_(self.ada[-1].bias, 0)  # AdaLN-zero init

    def _attn(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, c):
        sa, ka, ga, sm, km, gm = self.ada(c).chunk(6, dim=-1)           # AdaLN-zero modulators from action c
        x = x + ga * self._attn(self.n1(x) * (1 + ka) + sa)
        x = x + gm * self.mlp(self.n2(x) * (1 + km) + sm)
        return x


class TinyViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, dim=128, depth=4, heads=4):
        super().__init__()
        self.s_grid = img_size // patch_size; self.dim = dim
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos", sincos_2d(self.s_grid, self.s_grid, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.projector = Projector(dim)

    def forward(self, frames):
        B, T = frames.size(0), frames.size(1)
        x = self.proj(frames.reshape(B * T, *frames.shape[2:])).flatten(2).transpose(1, 2)
        x = x + self.pos[None]
        for blk in self.blocks: x = blk(x)
        return self.projector(self.norm(x).mean(1).view(B, T, self.dim))


class ARPredictor(nn.Module):
    def __init__(self, num_frames, dim=128, depth=4, heads=4, action_dim=2):
        super().__init__()
        self.act_proj = nn.Sequential(nn.Linear(action_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer("time_pe", sincos_1d(num_frames, dim))
        self.blocks = nn.ModuleList([CondBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.projector = Projector(dim)

    def forward(self, z, a):                                          # z: (B, T, D); a: (B, T, action_dim)
        x = z + self.time_pe[None, :z.size(1)]
        c = self.act_proj(a)
        for blk in self.blocks: x = blk(x, c)
        return self.projector(self.norm(x))

    @torch.no_grad()
    def rollout(self, z0, actions):                                   # z0: (B, D); actions: (B, K, action_dim)
        seq = z0.unsqueeze(1); outs = []
        for k in range(actions.size(1)):
            pred = self.forward(seq, actions[:, :k + 1])[:, -1:]
            seq = torch.cat([seq, pred], dim=1); outs.append(pred.squeeze(1))
        return torch.stack(outs, dim=1)                               # (B, K, D)


class ActionedMovingDigit(Dataset):
    V_MAX = 5.0

    def __init__(self, root="./data", n_samples=8000, n_frames=10,
                 img_size=64, digit_size=20, seed=0):
        mnist = MNIST(root=root, train=True, download=True)
        self.imgs = mnist.data.float() / 255.; self.n_samples = n_samples
        self.n_frames = n_frames; self.img_size = img_size; self.digit_size = digit_size
        g = torch.Generator().manual_seed(seed); s = img_size - digit_size
        self.choices = torch.randint(0, len(self.imgs), (n_samples,), generator=g)
        self.starts = torch.stack([torch.randint(0, s + 1, (n_samples,), generator=g),
                                   torch.randint(0, s + 1, (n_samples,), generator=g)], 1).float()
        self.actions = torch.rand(n_samples, n_frames, 2, generator=g) * 2 - 1   # per-frame velocity

    def __len__(self): return self.n_samples

    def __getitem__(self, i):
        d = F.interpolate(self.imgs[self.choices[i]][None, None],
                          (self.digit_size, self.digit_size), mode="bilinear",
                          align_corners=False)[0, 0]
        p = self.starts[i].clone(); mp = self.img_size - self.digit_size
        frames = torch.zeros(self.n_frames, 1, self.img_size, self.img_size)
        for t in range(self.n_frames):
            v = self.actions[i, t] * self.V_MAX
            xi = max(0, min(mp, int(round(p[0].item()))))
            yi = max(0, min(mp, int(round(p[1].item()))))
            frames[t, 0, yi:yi + self.digit_size, xi:xi + self.digit_size] = d
            np_ = p + v
            if np_[0] < 0 or np_[0] > mp: v[0] = -v[0]; np_[0] = p[0] + v[0]
            if np_[1] < 0 or np_[1] > mp: v[1] = -v[1]; np_[1] = p[1] + v[1]
            p = np_
        return frames - 0.5, self.actions[i]


def train(epochs=4, batch_size=32, lr=3e-4, wd=0.05, sigreg_w=1.0, device=None):
    device = device or pick_device(); print(f"device: {device}")
    ds = ActionedMovingDigit()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    encoder = TinyViT().to(device)
    predictor = ARPredictor(num_frames=ds.n_frames, dim=encoder.dim).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(param_groups([encoder, predictor], wd), lr=lr)
    losses = {"pred": [], "sigreg": [], "total": [], "zero_act": []}; step = 0
    for epoch in range(epochs):
        for frames, actions in loader:
            frames = frames.to(device); actions = actions.to(device)
            emb = encoder(frames)                                             # z (B, T, D) -- joint with predictor
            ctx_z, ctx_a = emb[:, :-1], actions[:, :-1]                       # (z_t, a_t) -- a_t drives z_t -> z_{t+1}
            pred = predictor(ctx_z, ctx_a)                                    # predicted next-frame embeddings
            tgt = emb[:, 1:]                                                  # targets: NOT detached (end-to-end)
            pred_loss = (pred - tgt).pow(2).mean()                            # L_pred
            sr = sigreg(emb.transpose(0, 1))                                  # L_sigreg on (T, B, D)
            loss = pred_loss + sigreg_w * sr                                  # L = L_pred + lambda * L_sigreg
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():                                             # diagnostic: zero-action prediction
                pz = predictor(ctx_z, torch.zeros_like(ctx_a))
                zero_act = (pz - tgt).pow(2).mean().item()
            losses["pred"].append(pred_loss.item())
            losses["sigreg"].append(sr.item())
            losses["total"].append(loss.item())
            losses["zero_act"].append(zero_act)
            if step % 25 == 0:
                gap = zero_act - pred_loss.item()
                print(f"ep={epoch} step={step:5d} pred={pred_loss.item():.4f} "
                      f"sigreg={sr.item():.4f} a=0={zero_act:.4f} gap={gap:+.4f}")
            step += 1
    return {"encoder": encoder, "predictor": predictor, "sigreg": sigreg,
            "losses": losses, "loader": loader, "device": device}


if __name__ == "__main__": train()
