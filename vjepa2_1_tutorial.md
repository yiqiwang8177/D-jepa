# V-JEPA 2.1 from Scratch in 401 Lines of PyTorch

This post extends the [V-JEPA tutorial](./vjepa_tutorial.md) and the [V-JEPA 2 tutorial](./vjepa2_tutorial.md) to **V-JEPA 2.1** — the training recipe that shifts the focus from "just learn a good video encoder" to **learn dense, temporally consistent features**. We'll implement a teaching-sized version in a single file.

- Source: [`vjepa2_1.py`](./vjepa2_1.py)
- Official release notes: [`facebookresearch/vjepa2/README.md`](https://github.com/facebookresearch/vjepa2)

## From V-JEPA 2 to V-JEPA 2.1

V-JEPA 2 already gives you a strong video encoder: tubelet patches, two mask groups, EMA target encoder, latent prediction, and then a second action-conditioned world-model phase.

**V-JEPA 2.1** changes the *pretraining recipe* itself. The official repo highlights four ingredients:

1. **Dense predictive loss** — predict not only the masked target tokens, but also the visible context tokens.
2. **Deep self-supervision** — apply the self-supervised objective to multiple intermediate encoder layers, not just the top one.
3. **Multi-modal tokenizers** — train one encoder across images and videos.
4. **Scaling** — larger models and larger datasets.

This minimal script preserves the first three and shrinks the fourth into something you can run locally.

## The setting

We want the smallest setup that still shows the 2.1 algorithmic shape.

So we train one encoder on two toy modalities:

- **videos**: Moving MNIST clips, 10 frames, 64×64, tokenized into `(2, 8, 8)` tubelets
- **images**: single MNIST digits resized to 64×64, tokenized into 2D patches of size `8×8`

The encoder therefore sees both:

- a **video** token grid of `5 × 8 × 8 = 320` tokens
- an **image** token grid of `1 × 8 × 8 = 64` tokens

The image branch is not there for cosmetics. It is how the minimal script preserves V-JEPA 2.1's **image/video co-training shape** without importing the official repo's distributed multi-dataset pipeline.

## Multi-modal tokenization

The encoder has two patchifiers but one shared transformer:

```python
class MultiModeEncoder(nn.Module):
    def __init__(..., num_frames=10, t_patch=2, patch_size=8, dim=96, depth=8, levels=4):
        self.img_proj = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.video_proj = nn.Conv3d(
            1, dim,
            kernel_size=(t_patch, patch_size, patch_size),
            stride=(t_patch, patch_size, patch_size))
        self.register_buffer("pos_img", sincos_3d(1, s_grid, s_grid, dim))
        self.register_buffer("pos_vid", sincos_3d(t_grid_vid, s_grid, s_grid, dim))
        self.img_mod = nn.Parameter(torch.zeros(1, 1, dim))
        self.video_mod = nn.Parameter(torch.zeros(1, 1, dim))
```

If `T == 1`, we use the image patchifier; otherwise we use the video tubelet patchifier. In both cases the output is a sequence of tokens fed into the **same transformer blocks**. Tiny learned modality embeddings (`img_mod`, `video_mod`) tell the shared backbone which branch it is processing.

That is the teaching equivalent of the official 2.1 repo's "multi-modal tokenizer" idea.

## Deep self-supervision

Plain V-JEPA predicts the top-layer target tokens. V-JEPA 2.1 supervises **multiple depths** of the encoder.

In the minimal script we keep 4 hierarchical outputs from an 8-block encoder:

```python
self.out_layers = [1, 3, 5, 7]
self.out_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.out_layers])

def forward(self, x, idx=None, return_hier=True):
    ...
    outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in self.out_layers:
            j = self.out_layers.index(i)
            outs.append(self.out_norms[j](x))
    if return_hier:
        return torch.cat(outs, dim=-1)   # (B, N, 4*dim)
```

So each token is represented by the concatenation of four normalized intermediate features.

This is a simplification of the official code, but it preserves the load-bearing idea: the target is **not just the final-layer representation**. The predictor must match a stack of intermediate abstractions.

## Dense predictive loss

This is the defining change.

Older JEPA variants only ask the predictor to fill in the **masked target tokens**:

$$\mathcal{L}_{\text{pred}} = \|\hat{z}_{\text{target}} - z_{\text{target}}\|_1$$

V-JEPA 2.1 also predicts the **visible context tokens**:

$$\mathcal{L}_{\text{ctx}} = \|\hat{z}_{\text{context}} - z_{\text{context}}\|_1$$

and combines them as:

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{ctx}}$$

with `lambda_ctx = 0.5` in the minimal implementation.

The official repo also weights the context loss by the distance from each visible token to the nearest masked token. The single-file script keeps a light version of that idea:

```python
def context_distance_weights(ctx_idx, tgt_idx, s_grid):
    d = torch.cdist(_token_coords(ctx_idx, s_grid), _token_coords(tgt_idx, s_grid))
    return 1.0 / d.min(dim=-1).values.clamp_min(1.0).sqrt()
```

Tokens near the masked frontier get slightly larger weight than very far-away visible tokens.

## The predictor

The predictor takes hierarchical context features and outputs two things:

1. predictions for the masked target tokens
2. predictions for the visible context tokens

```python
class DensePredictor(nn.Module):
    def __init__(self, ..., enc_dim=96, levels=4, dim=128):
        self.in_proj = nn.Linear(enc_dim * levels, dim)
        self.out_pred = nn.Linear(dim, enc_dim * levels)
        self.out_ctx = nn.Linear(dim, enc_dim * levels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, ctx, ctx_idx, tgt_idx, mode):
        x_ctx = self.in_proj(ctx) + pos[ctx_idx] + mod
        x_tgt = self.mask_token.expand(B, N_tgt, -1) + pos[tgt_idx] + mod
        x = torch.cat([x_ctx, x_tgt], dim=1)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return self.out_pred(x[:, -N_tgt:]), self.out_ctx(x[:, :N_ctx])
```

This is the smallest way to preserve the official 2.1 training shape. The context tokens do not just serve as input; they are also supervised outputs.

## The training loop

The full loop alternates one video batch and one image batch:

```python
for epoch in range(epochs):
    for videos, images in zip(video_loader, image_loader):
        for mode, frames in (("video", videos), ("image", images)):
            loss, lp, lc = _step_loss(frames, mode, ctx_enc, tgt_enc, predictor, ...)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(tgt_enc, ctx_enc, m)
```

Each `_step_loss(...)` call does the usual V-JEPA mask sampling, but then computes **two** losses from the same forward pass:

- masked-token loss
- visible-token dense loss

The target encoder remains an EMA copy, just like earlier JEPA variants.

## Running it

```bash
python vjepa2_1.py
python vjepa2_1_extras.py
```

The extras script writes:

- video mask grids
- an image mask grid
- image/video loss curves
- dense PCA visualizations for both modalities

## Results

The most informative outputs are not rollout plots or linear probes. They are the **dense feature maps**.

For one image, we take the encoder's patch tokens, run PCA to 3 dimensions, and paint those 3 coordinates as RGB over the `8×8` patch grid. For one video, we do the same per tubelet slice.

That is a simple teaching proxy for the official 2.1 claim: the recipe learns **higher-quality dense features** rather than only a global pooled representation.

## Core insights

V-JEPA 2.1 adds three important lessons on top of V-JEPA:

1. **Predict more than the hole.** If only masked tokens matter, visible tokens can drift toward a representation that is useful as context but not useful as dense output. Predicting visible tokens too constrains the whole token field.

2. **Supervise depth, not just the top.** Earlier layers capture local and mid-level structure. By supervising multiple levels at once, the encoder is pushed toward representations that are coherent throughout the stack.

3. **Train one backbone across images and videos.** Shared tokenization and shared transformer weights encourage features that work in both static and dynamic settings.

## Hyperparameters

- Encoder: dim `96`, depth `8`, heads `4`
- Hierarchical supervision: `4` layers (`[1, 3, 5, 7]`)
- Predictor: dim `128`, depth `4`, heads `4`
- Patch sizes: image `8×8`, video `(2, 8, 8)` tubelets
- Mask groups: short `8 × 0.15`, long `2 × 0.7`
- Loss: `L1 masked + 0.5 * dense context L1`
- EMA: `0.998 → 1.0`

## What we simplified

The official V-JEPA 2.1 recipe is much larger:

- internet-scale video datasets plus image data
- ViT-B/L/g/G backbones
- distributed mixed-data training
- long schedules, cooldown stages, and extensive downstream evaluation

The teaching script keeps the algorithmic core but makes four substitutions:

- Moving MNIST + MNIST instead of large internet video/image mixtures
- tiny encoder and predictor instead of billion-parameter ViTs
- alternating image/video batches instead of the official multi-rank mixed loader
- simple dense PCA plots instead of the official dense-task evaluation suite

Those simplifications shrink the system a lot, but they still expose what makes V-JEPA 2.1 different from plain V-JEPA: **dense loss, deep supervision, and multimodal pretraining shape**.
