"""
Training script for brain stroke hemorrhage segmentation (U-Net / U-Net++).

Version 3 — Fine-tuning from checkpoint:
  - Resumes from best_segmentation.pth
  - Triple Loss: 0.4*Dice + 0.3*BCE + 0.3*Tversky(α=0.3, β=0.7)
  - Encoder: EfficientNet-B3
  - Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
  - OHEM: Online Hard Example Mining (top 50% hardest samples)
  - 5-epoch EMA of val Dice for checkpoint saving (avoids lucky spikes)
  - Default LR: 5e-5 for fine-tuning

Usage:
    python train_segmentation.py --resume --epochs 50 --lr 5e-5
    python train_segmentation.py --epochs 100 --lr 1e-4 --model_type unet
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.unet import get_segmentation_model
from utils.dataset import StrokeSegmentationDataset, segmentation_collate_fn
from utils.transforms import get_segmentation_transforms
from utils.metrics import (
    TripleLoss,
    dice_coefficient,
    iou_score,
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
#  EMA Tracker for val Dice
# ══════════════════════════════════════════════════════════════════════

class EMATracker:
    """
    5-epoch Exponential Moving Average for a scalar metric.
    Prevents saving checkpoints during lucky spikes.
    """

    def __init__(self, span: int = 5) -> None:
        self.alpha = 2.0 / (span + 1)
        self.value: float = 0.0
        self._initialized = False

    def update(self, new_val: float) -> float:
        if not self._initialized:
            self.value = new_val
            self._initialized = True
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value


# ══════════════════════════════════════════════════════════════════════
#  Training loop with OHEM
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    ohem_ratio: float = 0.5,
) -> Dict[str, float]:
    """
    Run one training epoch with Online Hard Example Mining (OHEM).

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

        # ── OHEM: keep top-k hardest samples ──────────────────────────
        with torch.no_grad():
            sample_losses = []
            for b in range(images.size(0)):
                l = criterion(logits[b:b+1], masks[b:b+1])
                sample_losses.append(l.item())

            sample_losses_t = torch.tensor(sample_losses)
            n_keep = max(1, int(len(sample_losses_t) * ohem_ratio))
            _, indices = torch.topk(sample_losses_t, n_keep)

        # Backward on hard samples only
        hard_logits = logits[indices]
        hard_masks  = masks[indices]
        loss = criterion(hard_logits, hard_masks)

        loss.backward()
        optimizer.step()

        # Metrics on full batch
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
    """Run validation and compute segmentation metrics."""
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
    """Save PNG with (Scan | True Mask | Pred Mask) columns."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    orig_masks = batch["mask"]

    with torch.no_grad():
        preds = model.predict(images)

    n_show = min(3, images.size(0))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)

    for i in range(n_show):
        img = images[i].cpu() * std_t + mean_t
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        true_m = orig_masks[i, 0].cpu().numpy()
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
    print("  Brain Stroke Segmentation — Training (v3)")
    print("=" * 68)
    print(f"  Device       : {device}")
    print(f"  Architecture : {model_type.upper()}")
    print(f"  Encoder      : EfficientNet-B3")
    print(f"  Epochs       : {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Loss         : TripleLoss (0.4*Dice + 0.3*BCE + 0.3*Tversky)")
    print(f"  Tversky      : α=0.3, β=0.7 (penalise missed lesions)")
    print(f"  Scheduler    : CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    print(f"  OHEM Ratio   : 0.5 (top 50% hardest samples)")
    print(f"  Checkpoint   : 5-epoch EMA of val Dice (no lucky spikes)")
    print(f"  Resume       : {args.resume}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = setup_dataloaders(batch_size)

    # ── Model (EfficientNet-B3 Encoder) ───────────────────────────────
    model = get_segmentation_model(
        model_type=model_type,
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    best_dice = 0.0
    best_ema_dice = 0.0
    best_epoch = -1

    resume_path = os.path.join(config.CHECKPOINT_DIR, "best_segmentation.pth")
    if args.resume and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_dice = ckpt.get("best_dice", 0.0)
        best_ema_dice = ckpt.get("best_ema_dice", best_dice)
        best_epoch = start_epoch
        print(f"  ✓ Resumed from epoch {start_epoch}")
        print(f"    Best Dice (raw)={best_dice:.4f}, EMA={best_ema_dice:.4f}")
    elif args.resume:
        print(f"  ⚠ --resume set but no checkpoint found at {resume_path}")
        print(f"    Starting from scratch.")

    # ── Triple Loss ───────────────────────────────────────────────────
    criterion = TripleLoss(
        dice_weight=0.4,
        bce_weight=0.3,
        tversky_weight=0.3,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    )

    # ── Optimiser & Scheduler ─────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7,
    )

    # ── EMA tracker for val Dice ──────────────────────────────────────
    dice_ema = EMATracker(span=5)
    # Warm up EMA with best_dice if resuming
    if args.resume and best_dice > 0:
        dice_ema.update(best_dice)

    # ── History tracker ───────────────────────────────────────────────
    # Load existing history if resuming
    history_path = os.path.join(config.CHECKPOINT_DIR, "segmentation_history.json")
    if args.resume and os.path.isfile(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        # Ensure ema_dice key exists
        if "ema_dice" not in history:
            history["ema_dice"] = []
        print(f"  ✓ Loaded existing history ({len(history['train_loss'])} epochs)")
    else:
        history = {
            "train_loss": [], "train_dice": [], "train_iou": [],
            "val_loss": [], "val_dice": [], "val_iou": [],
            "ema_dice": [],
        }

    samples_dir = os.path.join(config.CHECKPOINT_DIR, "samples")

    # ── Training loop ─────────────────────────────────────────────────
    end_epoch = start_epoch + epochs
    print(f"\n  Training epochs {start_epoch + 1} → {end_epoch}")
    print("-" * 68)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        epoch_start = time.time()

        # Train with OHEM
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ohem_ratio=0.5,
        )

        # Validate
        val_m = validate(model, val_loader, criterion, device)

        # Update EMA
        ema_val = dice_ema.update(val_m["dice"])

        # Step scheduler
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Log history
        history["train_loss"].append(train_m["loss"])
        history["train_dice"].append(train_m["dice"])
        history["train_iou"].append(train_m["iou"])
        history["val_loss"].append(val_m["loss"])
        history["val_dice"].append(val_m["dice"])
        history["val_iou"].append(val_m["iou"])
        history["ema_dice"].append(ema_val)

        # Print epoch summary
        print(
            f"  Epoch {epoch:>3}/{end_epoch} │ "
            f"t_loss {train_m['loss']:.4f} │ t_dice {train_m['dice']:.4f} │ "
            f"v_loss {val_m['loss']:.4f} │ v_dice {val_m['dice']:.4f} │ "
            f"ema_dice {ema_val:.4f} │ "
            f"lr {current_lr:.2e} │ {elapsed:.1f}s"
        )

        # Save best model by EMA val dice (prevents lucky spike saves)
        if ema_val > best_ema_dice:
            best_ema_dice = ema_val
            best_dice = val_m["dice"]
            best_epoch = epoch
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_segmentation.pth")
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_type": model_type,
                    "encoder": "efficientnet-b3",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_ema_dice": best_ema_dice,
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved (EMA Dice={best_ema_dice:.4f}, raw={best_dice:.4f}) → {ckpt_path}")

        # Save sample visualisation
        if epoch % 10 == 0 or epoch == start_epoch + 1:
            _save_sample_visualisation(
                model, val_loader, device, epoch, samples_dir,
            )

    # ── Save history ──────────────────────────────────────────────────
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history → {history_path}")

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  Best EMA Dice: {best_ema_dice:.4f} (raw: {best_dice:.4f}) at epoch {best_epoch}")
    print("=" * 68)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train brain stroke segmentation model (EfficientNet-B3).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of ADDITIONAL training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate (default: 5e-5 for fine-tuning)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--model_type", type=str, default="unet", choices=["unet", "unet++"],
        help="Segmentation architecture: 'unet' or 'unet++'",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from best_segmentation.pth checkpoint.",
    )
    args = parser.parse_args()
    main(args)
