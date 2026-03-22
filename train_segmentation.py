"""
Training script for brain stroke hemorrhage segmentation (U-Net / U-Net++).

Usage:
    python train_segmentation.py
    python train_segmentation.py --epochs 80 --lr 5e-4 --model_type unet++
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.unet import get_segmentation_model
from utils.dataset import StrokeSegmentationDataset, segmentation_collate_fn
from utils.transforms import get_segmentation_transforms
from utils.metrics import (
    CombinedLoss,
    dice_coefficient,
    iou_score,
    compute_segmentation_metrics,
)


# ══════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════

def setup_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Build train and val data loaders for segmentation."""
    train_transform = get_segmentation_transforms(
        mode="train", image_size=config.IMAGE_SIZE,
    )
    val_transform = get_segmentation_transforms(
        mode="val", image_size=config.IMAGE_SIZE,
    )

    train_dataset = StrokeSegmentationDataset(
        images_dir=config.SEGMENTATION_TRAIN_IMG_DIR,
        masks_dir=config.SEGMENTATION_TRAIN_MASK_DIR,
        transform=train_transform,
    )
    val_dataset = StrokeSegmentationDataset(
        images_dir=config.SEGMENTATION_VAL_IMG_DIR,
        masks_dir=config.SEGMENTATION_VAL_MASK_DIR,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
    )

    print(f"  Train slices : {len(train_dataset)}")
    print(f"  Val   slices : {len(val_dataset)}")
    print(f"  Batch size   : {batch_size}")
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
) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns:
        ``{"loss": float, "dice": float, "iou": float}``
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=110)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)                      # (B, 1, H, W)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        # Metrics on binarised predictions
        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) >= 0.5).float()
            d = dice_coefficient(preds_bin, masks).item()
            j = iou_score(preds_bin, masks).item()

        running_loss += loss.item()
        running_dice += d
        running_iou  += j
        total_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{d:.4f}",
            iou=f"{j:.4f}",
        )

    return {
        "loss": running_loss / total_batches,
        "dice": running_dice / total_batches,
        "iou":  running_iou  / total_batches,
    }


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
    Run validation and compute segmentation metrics.

    Returns:
        ``{"loss": float, "dice": float, "iou": float}``
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  val  ", leave=False, ncols=110):
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            preds_bin = (torch.sigmoid(logits) >= 0.5).float()
            d = dice_coefficient(preds_bin, masks).item()
            j = iou_score(preds_bin, masks).item()

            running_loss += loss.item()
            running_dice += d
            running_iou  += j
            total_batches += 1

    return {
        "loss": running_loss / total_batches,
        "dice": running_dice / total_batches,
        "iou":  running_iou  / total_batches,
    }


# ══════════════════════════════════════════════════════════════════════
#  Sample visualisation
# ══════════════════════════════════════════════════════════════════════

def _save_sample_visualisation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epoch: int,
    save_dir: str,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
) -> None:
    """
    Save a PNG with 3 columns (CT scan · true mask · predicted mask)
    for up to 3 sample slices from the validation set.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    masks  = batch["mask"]

    with torch.no_grad():
        preds = model.predict(images)   # binary (B, 1, H, W)

    n_show = min(3, images.size(0))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]      # ensure 2-D indexing

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)

    for i in range(n_show):
        # De-normalise image for display
        img = images[i].cpu() * std_t + mean_t
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        true_m = masks[i, 0].cpu().numpy()
        pred_m = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("CT Scan")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_m, cmap="Reds", vmin=0, vmax=1)
        axes[i, 1].set_title("True Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_m, cmap="Reds", vmin=0, vmax=1)
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    fig.suptitle(f"Epoch {epoch}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  📷 Sample visualisation → {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    epochs     = args.epochs
    lr         = args.lr
    batch_size = args.batch_size
    model_type = args.model_type
    device     = config.DEVICE

    print("=" * 68)
    print("  Brain Stroke Segmentation — Training")
    print("=" * 68)
    print(f"  Device       : {device}")
    print(f"  Model type   : {model_type}")
    print(f"  Epochs       : {epochs}")
    print(f"  Learning rate: {lr}")

    # ── data ──────────────────────────────────────────────────────────
    train_loader, val_loader = setup_dataloaders(batch_size)

    # ── model ─────────────────────────────────────────────────────────
    model = get_segmentation_model(model_type=model_type)

    # ── loss ──────────────────────────────────────────────────────────
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)

    # ── optimiser & scheduler ─────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # ── history tracker ───────────────────────────────────────────────
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }
    best_dice = 0.0
    best_epoch = -1

    samples_dir = os.path.join(config.CHECKPOINT_DIR, "samples")

    # ── training loop ─────────────────────────────────────────────────
    print("\n" + "-" * 68)
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_m = validate(model, val_loader, criterion, device)

        # Step scheduler on val dice
        scheduler.step(val_m["dice"])
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Log history
        history["train_loss"].append(train_m["loss"])
        history["train_dice"].append(train_m["dice"])
        history["train_iou"].append(train_m["iou"])
        history["val_loss"].append(val_m["loss"])
        history["val_dice"].append(val_m["dice"])
        history["val_iou"].append(val_m["iou"])

        # Print epoch summary
        print(
            f"  Epoch {epoch:>3}/{epochs} │ "
            f"t_loss {train_m['loss']:.4f} │ t_dice {train_m['dice']:.4f} │ t_iou {train_m['iou']:.4f} │ "
            f"v_loss {val_m['loss']:.4f} │ v_dice {val_m['dice']:.4f} │ v_iou {val_m['iou']:.4f} │ "
            f"lr {current_lr:.2e} │ {elapsed:.1f}s"
        )

        # Save best model by val dice
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            best_epoch = epoch
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_segmentation.pth")
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_type": model_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved (Dice={best_dice:.4f}) → {ckpt_path}")

        # Save sample visualisation every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            _save_sample_visualisation(
                model, val_loader, device, epoch, samples_dir,
            )

    # ── save training history ─────────────────────────────────────────
    history_path = os.path.join(config.CHECKPOINT_DIR, "segmentation_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history → {history_path}")

    # ── final summary table ───────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  Training Complete — Summary")
    print("=" * 68)
    header = (
        f"  {'Ep':>3} │ {'T Loss':>7} │ {'T Dice':>7} │ {'T IoU':>6} │ "
        f"{'V Loss':>7} │ {'V Dice':>7} │ {'V IoU':>6}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i in range(epochs):
        marker = " ◀ best" if (i + 1) == best_epoch else ""
        print(
            f"  {i+1:>3} │ "
            f"{history['train_loss'][i]:>7.4f} │ "
            f"{history['train_dice'][i]:>7.4f} │ "
            f"{history['train_iou'][i]:>6.4f} │ "
            f"{history['val_loss'][i]:>7.4f} │ "
            f"{history['val_dice'][i]:>7.4f} │ "
            f"{history['val_iou'][i]:>6.4f}"
            f"{marker}"
        )
    print(f"\n  Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    print("=" * 68)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train brain stroke segmentation model (U-Net / U-Net++).",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS_SEGMENTATION,
        help=f"Number of training epochs (default: {config.NUM_EPOCHS_SEGMENTATION})",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Initial learning rate (default: {config.LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--model_type", type=str, default="unet", choices=["unet", "unet++"],
        help="Segmentation architecture: 'unet' or 'unet++' (default: unet)",
    )
    args = parser.parse_args()
    main(args)
