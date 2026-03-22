"""
Training script for brain stroke classification.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 40 --lr 3e-4 --batch_size 32
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config
from models.classifier import get_classifier
from utils.dataset import StrokeClassificationDataset, classification_collate_fn
from utils.transforms import get_classification_transforms
from utils.metrics import compute_classification_metrics


# ══════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════

def setup_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val data loaders.

    The training loader uses a ``WeightedRandomSampler`` to handle
    class imbalance (Normal vs Stroke).
    """
    train_transform = get_classification_transforms(
        mode="train", image_size=config.IMAGE_SIZE,
    )
    val_transform = get_classification_transforms(
        mode="val", image_size=config.IMAGE_SIZE,
    )

    train_dataset = StrokeClassificationDataset(
        root_dir=config.CLASSIFICATION_TRAIN_DIR,
        transform=train_transform,
    )
    val_dataset = StrokeClassificationDataset(
        root_dir=config.CLASSIFICATION_VAL_DIR,
        transform=val_transform,
    )

    # Weighted sampler for class imbalance
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=classification_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=classification_collate_fn,
    )

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val   samples : {len(val_dataset)}")
    print(f"  Batch size    : {batch_size}")
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════
#  Training loop (one epoch)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        ``(avg_loss, accuracy)``
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=100)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
#  Validation loop
# ══════════════════════════════════════════════════════════════════════

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Run validation and compute full classification metrics.

    Returns:
        Dictionary with ``loss``, ``accuracy``, ``f1``, ``roc_auc``.
    """
    model.eval()
    running_loss = 0.0
    total = 0

    all_labels: list = []
    all_probs: list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  val  ", leave=False, ncols=100):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]  # P(Stroke)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    metrics = compute_classification_metrics(all_labels, all_probs)

    return {
        "loss":     avg_loss,
        "accuracy": metrics["accuracy"],
        "f1":       metrics["f1_score"],
        "roc_auc":  metrics["roc_auc"],
    }


# ══════════════════════════════════════════════════════════════════════
#  Compute class weights for CrossEntropyLoss
# ══════════════════════════════════════════════════════════════════════

def _class_weights_for_ce(dataset: StrokeClassificationDataset) -> torch.Tensor:
    """Inverse-frequency weights per class for CrossEntropyLoss."""
    from collections import Counter
    counts = Counter(dataset.labels)
    total = len(dataset)
    num_classes = len(counts)
    weights = torch.tensor(
        [total / (num_classes * counts[c]) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    epochs     = args.epochs
    lr         = args.lr
    batch_size = args.batch_size
    device     = config.DEVICE
    unfreeze_epoch = 5

    print("=" * 64)
    print("  Brain Stroke Classification — Training")
    print("=" * 64)
    print(f"  Device          : {device}")
    print(f"  Epochs          : {epochs}")
    print(f"  Learning rate   : {lr}")
    print(f"  Unfreeze at     : epoch {unfreeze_epoch + 1}")

    # ── data ──────────────────────────────────────────────────────────
    train_loader, val_loader = setup_dataloaders(batch_size)

    # ── model (backbone frozen) ───────────────────────────────────────
    model = get_classifier(pretrained=True, freeze=True)

    # ── loss with class weights ───────────────────────────────────────
    train_dataset = train_loader.dataset
    class_weights = _class_weights_for_ce(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  CE class weights: {class_weights.cpu().tolist()}")

    # ── optimizer & scheduler ─────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # ── history tracker ───────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": [],
    }
    best_f1 = 0.0
    best_epoch = -1

    # ── training loop ─────────────────────────────────────────────────
    print("\n" + "-" * 64)
    for epoch in range(epochs):
        epoch_start = time.time()

        # Unfreeze backbone after `unfreeze_epoch` epochs
        if epoch == unfreeze_epoch:
            print(f"\n  >>> Unfreezing backbone at epoch {epoch + 1}")
            model.unfreeze_backbone()
            # Rebuild optimizer with all params & reduced LR
            new_lr = lr / 10.0
            optimizer = optim.AdamW(
                model.parameters(), lr=new_lr, weight_decay=1e-4,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=1e-7,
            )
            print(f"  >>> New LR: {new_lr}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["roc_auc"])

        # Print epoch summary
        print(
            f"  Epoch {epoch+1:>3}/{epochs} │ "
            f"train_loss {train_loss:.4f} │ train_acc {train_acc:.4f} │ "
            f"val_loss {val_metrics['loss']:.4f} │ val_acc {val_metrics['accuracy']:.4f} │ "
            f"val_f1 {val_metrics['f1']:.4f} │ val_auc {val_metrics['roc_auc']:.4f} │ "
            f"lr {current_lr:.2e} │ {elapsed:.1f}s"
        )

        # Save best model by F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_classifier.pth")
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved (F1={best_f1:.4f}) → {ckpt_path}")

    # ── save training history ─────────────────────────────────────────
    history_path = os.path.join(config.CHECKPOINT_DIR, "classifier_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history → {history_path}")

    # ── final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Training Complete — Summary")
    print("=" * 64)
    header = f"  {'Epoch':>5} │ {'Train Loss':>10} │ {'Train Acc':>9} │ {'Val Loss':>8} │ {'Val Acc':>7} │ {'Val F1':>6} │ {'Val AUC':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i in range(epochs):
        marker = " ◀ best" if (i + 1) == best_epoch else ""
        print(
            f"  {i+1:>5} │ "
            f"{history['train_loss'][i]:>10.4f} │ "
            f"{history['train_acc'][i]:>9.4f} │ "
            f"{history['val_loss'][i]:>8.4f} │ "
            f"{history['val_acc'][i]:>7.4f} │ "
            f"{history['val_f1'][i]:>6.4f} │ "
            f"{history['val_auc'][i]:>7.4f}"
            f"{marker}"
        )
    print(f"\n  Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print("=" * 64)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train brain stroke classifier (EfficientNet-B4).",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS_CLASSIFIER,
        help=f"Number of training epochs (default: {config.NUM_EPOCHS_CLASSIFIER})",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Initial learning rate (default: {config.LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    args = parser.parse_args()
    main(args)
