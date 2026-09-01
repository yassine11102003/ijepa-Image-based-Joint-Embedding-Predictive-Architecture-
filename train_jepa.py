"""
I-JEPA training loop.

The context encoder (+ predictor) is trained by backprop to predict the
target encoder's representations of masked-out blocks. The target encoder
never receives gradients — its weights are an exponential moving average
(EMA) of the context encoder's, updated after every step.
"""

import copy

import torch
import torch.nn.functional as F

from config import MODEL_CONFIG, MASK_CONFIG, TRAIN_CONFIG
from model import Encoder, Predictor
from masking import sample_target_blocks, sample_context_indices
from dataset import get_pretrain_loader


@torch.no_grad()
def update_target_encoder(context_encoder, target_encoder, momentum):
    """target = momentum * target + (1 - momentum) * context, param by param."""
    for tp, cp in zip(target_encoder.parameters(), context_encoder.parameters()):
        tp.data.mul_(momentum).add_(cp.data, alpha=1 - momentum)


def train(model_cfg=MODEL_CONFIG, mask_cfg=MASK_CONFIG, train_cfg=TRAIN_CONFIG):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    context_encoder = Encoder(
        img_size=model_cfg["img_size"], patch_size=model_cfg["patch_size"],
        in_channels=model_cfg["in_channels"], embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"], num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
    ).to(device)

    # Target encoder starts as an exact copy, then only ever moves via EMA.
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    predictor = Predictor(
        num_patches=context_encoder.patch_embed.num_patches,
        embed_dim=model_cfg["embed_dim"], predictor_dim=model_cfg["predictor_dim"],
        depth=model_cfg["predictor_depth"], num_heads=model_cfg["predictor_num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
    ).to(device)

    # Only the context encoder + predictor are optimized by gradient descent.
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=train_cfg["lr"],
    )

    loader     = get_pretrain_loader(batch_size=train_cfg["batch_size"])
    grid_size  = context_encoder.patch_embed.grid_size

    for epoch in range(train_cfg["epochs"]):
        context_encoder.train()
        predictor.train()
        epoch_loss, n = 0.0, 0

        for images, _ in loader:   # label is unused: JEPA pretraining is self-supervised
            images = images.to(device)

            target_blocks = sample_target_blocks(
                grid_size, mask_cfg["num_target_blocks"],
                mask_cfg["target_scale_range"], mask_cfg["target_aspect_ratio_range"],
            )
            context_indices = sample_context_indices(
                grid_size, target_blocks,
                mask_cfg["context_scale_range"], mask_cfg["context_aspect_ratio_range"],
            ).to(device)
            target_indices = torch.cat(target_blocks).unique().to(device)

            with torch.no_grad():
                target_full = target_encoder(images)              # full image, all patches
                targets     = target_full[:, target_indices, :]

            context_tokens = context_encoder(images, context_indices)
            preds = predictor(context_tokens, context_indices, target_indices)

            loss = F.smooth_l1_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_target_encoder(context_encoder, target_encoder, train_cfg["ema_momentum"])

            epoch_loss += loss.item()
            n += 1

        print(f"Epoch {epoch:3d} | loss={epoch_loss / n:.4f}")

    torch.save(context_encoder.state_dict(), "context_encoder.pth")
    torch.save(predictor.state_dict(), "predictor.pth")
    print("Saved context_encoder.pth and predictor.pth")


if __name__ == "__main__":
    train()
