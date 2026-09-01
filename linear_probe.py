"""
Linear-probe evaluation: freeze the pretrained context encoder, average-pool its
patch representations into one vector per image, and train a single linear layer
on top with the real CIFAR-10 labels. Accuracy here is the proxy for "did JEPA
pretraining learn useful representations" — I-JEPA doesn't reconstruct pixels,
so this is the standard way papers in this family report results.

Run with:
    python linear_probe.py           # evaluate the pretrained encoder
    python linear_probe.py --random  # baseline: same architecture, random weights
"""

import argparse

import torch
import torch.nn as nn

from config import MODEL_CONFIG, PROBE_CONFIG
from model import Encoder
from dataset import get_probe_loaders


def build_encoder(weights_path=None, device="cpu"):
    encoder = Encoder(
        img_size=MODEL_CONFIG["img_size"], patch_size=MODEL_CONFIG["patch_size"],
        in_channels=MODEL_CONFIG["in_channels"], embed_dim=MODEL_CONFIG["embed_dim"],
        depth=MODEL_CONFIG["depth"], num_heads=MODEL_CONFIG["num_heads"],
        mlp_ratio=MODEL_CONFIG["mlp_ratio"],
    ).to(device)

    if weights_path is not None:
        encoder.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def encode(encoder, images):
    tokens = encoder(images)     # (batch, num_patches, embed_dim)
    return tokens.mean(dim=1)    # average-pool patches into one vector per image


def evaluate(classifier, encoder, loader, device):
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = encode(encoder, images)
            preds    = classifier(features).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def run_linear_probe(weights_path="context_encoder.pth", label="pretrained"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder    = build_encoder(weights_path, device)
    classifier = nn.Linear(MODEL_CONFIG["embed_dim"], PROBE_CONFIG["num_classes"]).to(device)

    train_loader, test_loader = get_probe_loaders(batch_size=PROBE_CONFIG["batch_size"])

    optimizer = torch.optim.Adam(classifier.parameters(), lr=PROBE_CONFIG["lr"])
    loss_fn   = nn.CrossEntropyLoss()

    for epoch in range(PROBE_CONFIG["epochs"]):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            features = encode(encoder, images)

            optimizer.zero_grad()
            loss = loss_fn(classifier(features), labels)
            loss.backward()
            optimizer.step()

        acc = evaluate(classifier, encoder, test_loader, device)
        print(f"[{label}] epoch {epoch:2d} | test acc={acc:.4f}")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true",
                        help="Evaluate a randomly-initialized encoder instead (baseline).")
    args = parser.parse_args()

    if args.random:
        run_linear_probe(weights_path=None, label="random-init baseline")
    else:
        run_linear_probe(weights_path="context_encoder.pth", label="pretrained")
