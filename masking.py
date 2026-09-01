"""
Block-masking strategy for I-JEPA, applied to a square grid of patches.

The same mask is shared across the whole batch (one sampled mask per training
step, not one per image) — this matches the original I-JEPA implementation and
keeps things simple: every image in a batch uses the same context/target
patch indices, so the context encoder always processes a fixed-length sequence.
"""

import random

import torch


def _sample_block(grid_size, scale_range, aspect_ratio_range):
    """Sample one rectangular block on the grid, return its flat patch indices."""
    num_patches = grid_size ** 2
    scale       = random.uniform(*scale_range)
    aspect      = random.uniform(*aspect_ratio_range)

    area = scale * num_patches
    h = max(1, min(grid_size, round((area / aspect) ** 0.5)))
    w = max(1, min(grid_size, round((area * aspect) ** 0.5)))

    top  = torch.randint(0, grid_size - h + 1, (1,)).item()
    left = torch.randint(0, grid_size - w + 1, (1,)).item()

    rows = torch.arange(top, top + h)
    cols = torch.arange(left, left + w)
    grid = torch.cartesian_prod(rows, cols)          # (h*w, 2) pairs of (row, col)
    indices = grid[:, 0] * grid_size + grid[:, 1]      # flatten to patch index
    return indices


def sample_target_blocks(grid_size, num_blocks=4,
                         scale_range=(0.15, 0.2), aspect_ratio_range=(0.75, 1.5)):
    """Sample M independent target blocks. They may overlap each other — that's fine."""
    return [_sample_block(grid_size, scale_range, aspect_ratio_range)
            for _ in range(num_blocks)]


def sample_context_indices(grid_size, target_blocks,
                           scale_range=(0.85, 1.0), aspect_ratio_range=(1.0, 1.0)):
    """
    Sample one large context block, then remove any patch that belongs to
    any of the target blocks — the context must never see what it has to predict.
    """
    context = _sample_block(grid_size, scale_range, aspect_ratio_range)

    target_union = torch.cat(target_blocks).unique()
    keep_mask    = ~torch.isin(context, target_union)
    context      = context[keep_mask]

    if context.numel() == 0:
        # Extremely unlikely, but fall back to "everything not in a target block"
        # rather than handing the encoder an empty sequence.
        all_patches = torch.arange(grid_size ** 2)
        context = all_patches[~torch.isin(all_patches, target_union)]

    return context
