import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and projects each to embed_dim."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, 3, img_size, img_size)
        x = self.proj(x)                    # (batch, embed_dim, grid_size, grid_size)
        x = x.flatten(2).transpose(1, 2)     # (batch, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Standard bidirectional (non-causal) multi-head self-attention."""

    def __init__(self, embed_dim, num_heads, qkv_bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.W_query  = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_key    = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_value  = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, n, d = x.shape

        q = self.W_query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores  = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    """Pre-norm transformer block: reused by the encoder and the predictor."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Encoder(nn.Module):
    """
    ViT encoder shared by the context encoder and the target encoder
    (same architecture, separate weight instances).

    forward(x, patch_indices=None):
      - patch_indices=None -> encodes all patches (used by the target encoder,
        which needs the full image for self-attention over the whole scene).
      - patch_indices=<tensor of patch ids> -> encodes only that subset of
        patches (used by the context encoder, which only ever sees the
        visible/context patches, never the masked-out ones).
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                embed_dim=128, depth=6, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches       = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, patch_indices=None):
        x = self.patch_embed(x) + self.pos_embed   # (batch, num_patches, embed_dim)

        if patch_indices is not None:
            x = x[:, patch_indices, :]              # keep only the requested patches

        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class Predictor(nn.Module):
    """
    Lightweight, narrower transformer: predicts target-block representations
    from the context encoder's output.

    forward(context_tokens, context_indices, target_indices):
      - context_tokens:  (batch, n_context, embed_dim) — output of the context encoder.
      - context_indices: patch ids (in the full grid) the context tokens correspond to.
      - target_indices:  patch ids we must predict a representation for.
    Returns predictions of shape (batch, n_target, embed_dim) — same dim as the
    encoder output, so they can be compared directly to the target encoder's
    representations in the loss.
    """

    def __init__(self, num_patches, embed_dim=128, predictor_dim=64,
                depth=4, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.proj_in  = nn.Linear(embed_dim, predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            Block(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

    def forward(self, context_tokens, context_indices, target_indices):
        batch = context_tokens.shape[0]

        x = self.proj_in(context_tokens) + self.pos_embed[:, context_indices, :]

        mask_tokens = self.mask_token.expand(batch, len(target_indices), -1)
        mask_tokens = mask_tokens + self.pos_embed[:, target_indices, :]

        x = torch.cat([x, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        preds = x[:, -len(target_indices):, :]   # keep only the mask-token outputs
        return self.proj_out(preds)
