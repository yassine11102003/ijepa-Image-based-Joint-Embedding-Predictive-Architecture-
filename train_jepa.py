"""
I-JEPA training loop.

The context encoder (+ predictor) is trained by backprop to predict the
target encoder's representations of masked-out blocks. The target encoder
never receives gradients — its weights are an exponential moving average
(EMA) of the context encoder's, updated after every training step.

A held-out validation split (carved out of CIFAR-10's train set, independent
from the test set used later for the linear probe) is used to monitor the
JEPA pretext loss and only keep the best checkpoint, instead of blindly
saving whatever the last epoch happens to produce.
"""

import copy

import torch
import torch.nn.functional as F

from config import MODEL_CONFIG, MASK_CONFIG, TRAIN_CONFIG
from model import Encoder, Predictor
from masking import sample_target_blocks, sample_context_indices
from dataset import get_pretrain_loaders


@torch.no_grad()
def update_target_encoder(context_encoder, target_encoder, momentum):
    """target = momentum * target + (1 - momentum) * context, param by param."""
    for tp, cp in zip(target_encoder.parameters(), context_encoder.parameters()):
        tp.data.mul_(momentum).add_(cp.data, alpha=1 - momentum)


def jepa_loss(images, context_encoder, target_encoder, predictor,
              grid_size, mask_cfg, device):
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
        target_full = target_encoder(images)          # full image, all patches
        targets     = target_full[:, target_indices, :]

    context_tokens = context_encoder(images, context_indices)
    preds = predictor(context_tokens, context_indices, target_indices)

    return F.smooth_l1_loss(preds, targets)


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

    train_loader, val_loader = get_pretrain_loaders(batch_size=train_cfg["batch_size"])
    grid_size = context_encoder.patch_embed.grid_size

    best_val_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
        # --- training ---
        context_encoder.train()
        predictor.train()
        train_loss, n = 0.0, 0

        for images, _ in train_loader:   # label unused: JEPA pretraining is self-supervised
            loss = jepa_loss(images, context_encoder, target_encoder, predictor,
                             grid_size, mask_cfg, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_target_encoder(context_encoder, target_encoder, train_cfg["ema_momentum"])

            train_loss += loss.item()
            n += 1
        train_loss /= n

        # --- validation ---
        context_encoder.eval()
        predictor.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for images, _ in val_loader:
                loss = jepa_loss(images, context_encoder, target_encoder, predictor,
                                 grid_size, mask_cfg, device)
                val_loss += loss.item()
                vn += 1
        val_loss /= vn

        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(context_encoder.state_dict(), "context_encoder.pth")
            torch.save(predictor.state_dict(), "predictor.pth")
            print(f"  -> new best (val={val_loss:.4f}), checkpoint saved")

    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
