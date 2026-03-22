"""
Comprehensive evaluation script for brain stroke classification and
hemorrhage segmentation models.

Usage:
    python evaluate.py
    python evaluate.py --cls_only
    python evaluate.py --seg_only
    python evaluate.py --seg_model_type unet++
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

import config
from models.classifier import get_classifier, StrokeClassifier
from models.unet import get_segmentation_model
from utils.dataset import (
    StrokeClassificationDataset,
    StrokeSegmentationDataset,
    classification_collate_fn,
    segmentation_collate_fn,
)
from utils.transforms import (
    get_classification_transforms,
    get_segmentation_transforms,
)
from utils.metrics import (
    compute_classification_metrics,
    dice_coefficient,
    iou_score,
)
from sklearn.metrics import f1_score as sklearn_f1_score


# ──────────────────────────────────────────────────────────────────────
#  Globals
# ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(config.BASE_DIR, "evaluate_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFICATION EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _build_cls_val_loader(batch_size: int) -> DataLoader:
    val_transform = get_classification_transforms(
        mode="val", image_size=config.IMAGE_SIZE,
    )
    val_dataset = StrokeClassificationDataset(
        root_dir=config.CLASSIFICATION_VAL_DIR,
        transform=val_transform,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=classification_collate_fn,
    )


def find_optimal_threshold(
    y_true: List[int],
    y_probs: List[float],
    low: float = 0.30,
    high: float = 0.70,
    step: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds from *low* to *high* in *step* increments and pick
    the one that maximises F1 score.

    Returns:
        ``(best_threshold, {threshold: f1, ...})``
    """
    y_true_np = np.asarray(y_true)
    y_probs_np = np.asarray(y_probs)

    results: Dict[str, float] = {}
    best_thr = 0.5
    best_f1 = 0.0

    thr = low
    while thr <= high + 1e-9:
        y_pred = (y_probs_np >= thr).astype(int)
        f1 = float(sklearn_f1_score(y_true_np, y_pred, zero_division=0))
        results[f"{thr:.2f}"] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_thr = round(thr, 2)
        thr = round(thr + step, 4)

    return best_thr, results


def evaluate_classifier(
    model_path: str,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, object]:
    """
    Load a saved classifier, run inference on the validation set, and
    produce metrics + publication-quality plots.

    Returns:
        The metrics dictionary from ``compute_classification_metrics``.
    """
    print("\n" + "=" * 64)
    print("  Classification Evaluation")
    print("=" * 64)

    # ── load model ────────────────────────────────────────────────────
    model = get_classifier(pretrained=False, freeze=False)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {model_path}  (epoch {ckpt.get('epoch', '?')})")

    # ── inference ─────────────────────────────────────────────────────
    all_labels: List[int]   = []
    all_probs:  List[float] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  infer", leave=False, ncols=100):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]

            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]  # P(Stroke)

            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # ── find optimal threshold ────────────────────────────────────────
    optimal_thr, thr_results = find_optimal_threshold(all_labels, all_probs)
    print(f"\n  ── Optimal Threshold Search (0.30–0.70, step=0.01) ──")
    print(f"  Optimal threshold : {optimal_thr:.2f}")
    # Show top-5 thresholds
    sorted_thrs = sorted(thr_results.items(), key=lambda x: x[1], reverse=True)[:5]
    for thr_str, f1_val in sorted_thrs:
        marker = " ◀ BEST" if thr_str == f"{optimal_thr:.2f}" else ""
        print(f"    threshold={thr_str}  →  F1={f1_val:.4f}{marker}")

    # ── compute metrics at optimal threshold ──────────────────────────
    metrics = compute_classification_metrics(all_labels, all_probs, threshold=optimal_thr)
    print(f"\n  ── Metrics at optimal threshold ({optimal_thr:.2f}) ──")

    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")

    # ── 1) Confusion Matrix heatmap ───────────────────────────────────
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        linewidths=0.5, linecolor="grey",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix → {cm_path}")

    # ── 2) ROC Curve ──────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"  📊 ROC curve → {roc_path}")

    # ── 3) Classification Report (text + heatmap) ─────────────────────
    report_str = metrics["classification_report"]
    report_txt_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(report_str)
    print(f"  📄 Report text → {report_txt_path}")

    # Build a numeric matrix from precision / recall / f1 per class
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = (np.asarray(all_probs) >= optimal_thr).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, y_pred, labels=[0, 1], zero_division=0,
    )
    report_matrix = np.array([prec, rec, f1]).T  # (2, 3)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(
        report_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
        xticklabels=["Precision", "Recall", "F1-Score"],
        yticklabels=config.CLASS_NAMES,
        vmin=0, vmax=1, linewidths=0.5, linecolor="grey",
        ax=ax,
    )
    ax.set_title("Classification Report Heatmap")
    fig.tight_layout()
    report_hm_path = os.path.join(OUTPUT_DIR, "classification_report_heatmap.png")
    fig.savefig(report_hm_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Report heatmap → {report_hm_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════
#  SEGMENTATION EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _build_seg_val_loader(batch_size: int) -> DataLoader:
    val_transform = get_segmentation_transforms(
        mode="val", image_size=config.IMAGE_SIZE,
    )
    val_dataset = StrokeSegmentationDataset(
        images_dir=config.SEGMENTATION_VAL_IMG_DIR,
        masks_dir=config.SEGMENTATION_VAL_MASK_DIR,
        transform=val_transform,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
    )


def evaluate_segmentation(
    model_path: str,
    val_loader: DataLoader,
    device: str,
    model_type: str = "unet",
) -> Dict[str, float]:
    """
    Load a saved segmentation model, compute metrics, and produce
    a sample-grid visualisation with overlays.

    Returns:
        ``{"mean_dice": float, "mean_iou": float}``
    """
    print("\n" + "=" * 64)
    print("  Segmentation Evaluation")
    print("=" * 64)

    # ── load model ────────────────────────────────────────────────────
    ckpt = torch.load(model_path, map_location=device)
    ckpt_model_type = ckpt.get("model_type", model_type)
    ckpt_encoder = ckpt.get("encoder", "resnet34")
    model = get_segmentation_model(
        model_type=ckpt_model_type,
        encoder_name=ckpt_encoder,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {model_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"  Encoder: {ckpt_encoder}, Architecture: {ckpt_model_type}")

    # ── inference + metrics ───────────────────────────────────────────
    dice_scores: List[float] = []
    iou_scores:  List[float] = []

    # Collect samples for visualisation
    sample_images:  List[torch.Tensor] = []
    sample_masks:   List[torch.Tensor] = []
    sample_preds:   List[torch.Tensor] = []
    max_samples = 8

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  infer", leave=False, ncols=100):
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            preds_bin = (torch.sigmoid(logits) >= 0.5).float()

            # Per-sample metrics
            for i in range(images.size(0)):
                d = dice_coefficient(preds_bin[i], masks[i]).item()
                j = iou_score(preds_bin[i], masks[i]).item()
                dice_scores.append(d)
                iou_scores.append(j)

                if len(sample_images) < max_samples:
                    sample_images.append(images[i].cpu())
                    sample_masks.append(masks[i].cpu())
                    sample_preds.append(preds_bin[i].cpu())

    mean_dice = float(np.mean(dice_scores))
    mean_iou  = float(np.mean(iou_scores))

    print(f"\n  Mean Dice : {mean_dice:.4f}")
    print(f"  Mean IoU  : {mean_iou:.4f}")
    print(f"  Samples   : {len(dice_scores)}")

    # ── sample grid: CT | true mask | predicted mask | overlay ────────
    n_show = min(max_samples, len(sample_images))
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    mean_t = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    std_t  = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)

    col_titles = ["CT Scan", "True Mask", "Predicted Mask", "Overlay (40%)"]

    for i in range(n_show):
        # De-normalise
        img = (sample_images[i] * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        true_m = sample_masks[i][0].numpy()
        pred_m = sample_preds[i][0].numpy()

        # Overlay: predicted mask in red at 40 % opacity
        overlay = img.copy()
        red_mask = np.zeros_like(img)
        red_mask[:, :, 0] = 1.0                           # red channel
        mask_region = pred_m[:, :, np.newaxis] > 0.5
        overlay = np.where(mask_region, overlay * 0.6 + red_mask * 0.4, overlay)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(true_m, cmap="Reds", vmin=0, vmax=1)
        axes[i, 2].imshow(pred_m, cmap="Reds", vmin=0, vmax=1)
        axes[i, 3].imshow(overlay)

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=12, fontweight="bold")

        # Show per-sample dice on the overlay
        axes[i, 3].text(
            5, 15, f"Dice: {dice_scores[i]:.3f}",
            fontsize=10, color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )

    fig.suptitle(
        f"Segmentation Results  —  Mean Dice: {mean_dice:.4f}  |  Mean IoU: {mean_iou:.4f}",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "segmentation_samples.png")
    fig.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Sample grid → {grid_path}")

    return {"mean_dice": mean_dice, "mean_iou": mean_iou}


# ══════════════════════════════════════════════════════════════════════
#  TRAINING HISTORY PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_training_history(
    history_json_path: str,
    task_name: str,
) -> None:
    """
    Read a training history JSON and produce side-by-side plots:
    left = loss curves, right = metric curves.
    """
    print(f"\n  Plotting training history for {task_name}...")

    if not os.path.isfile(history_json_path):
        print(f"  ⚠  History file not found: {history_json_path}")
        return

    with open(history_json_path, "r") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curves ───────────────────────────────────────────────────
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", lw=2)
    ax_loss.plot(epochs, history["val_loss"],   label="Val Loss",   lw=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{task_name} — Loss")
    ax_loss.legend()

    # ── Metric curves ─────────────────────────────────────────────────
    if task_name.lower() == "classifier":
        # Accuracy, F1, AUC
        ax_metric.plot(epochs, history["train_acc"], label="Train Acc", lw=2)
        ax_metric.plot(epochs, history["val_acc"],   label="Val Acc",   lw=2)
        ax_metric.plot(epochs, history["val_f1"],    label="Val F1",    lw=2, ls="--")
        ax_metric.plot(epochs, history["val_auc"],   label="Val AUC",   lw=2, ls=":")
        ax_metric.set_ylabel("Score")
        ax_metric.set_title(f"{task_name} — Accuracy / F1 / AUC")
    else:
        # Dice, IoU
        ax_metric.plot(epochs, history["train_dice"], label="Train Dice", lw=2)
        ax_metric.plot(epochs, history["val_dice"],   label="Val Dice",   lw=2)
        ax_metric.plot(epochs, history["train_iou"],  label="Train IoU",  lw=2, ls="--")
        ax_metric.plot(epochs, history["val_iou"],    label="Val IoU",    lw=2, ls=":")
        ax_metric.set_ylabel("Score")
        ax_metric.set_title(f"{task_name} — Dice / IoU")

    ax_metric.set_xlabel("Epoch")
    ax_metric.legend()

    fig.suptitle(f"Training History — {task_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{task_name.lower()}_training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Training curves → {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    device     = config.DEVICE
    batch_size = args.batch_size
    run_cls    = not args.seg_only
    run_seg    = not args.cls_only

    cls_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_classifier.pth")
    seg_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_segmentation.pth")
    cls_hist = os.path.join(config.CHECKPOINT_DIR, "classifier_history.json")
    seg_hist = os.path.join(config.CHECKPOINT_DIR, "segmentation_history.json")

    cls_metrics = None
    seg_metrics = None

    # ── Classification ────────────────────────────────────────────────
    if run_cls:
        if os.path.isfile(cls_ckpt):
            val_loader = _build_cls_val_loader(batch_size)
            cls_metrics = evaluate_classifier(cls_ckpt, val_loader, device)
            plot_training_history(cls_hist, "Classifier")
        else:
            print(f"\n  ⚠  Classifier checkpoint not found: {cls_ckpt}")

    # ── Segmentation ──────────────────────────────────────────────────
    if run_seg:
        if os.path.isfile(seg_ckpt):
            val_loader = _build_seg_val_loader(batch_size)
            seg_metrics = evaluate_segmentation(
                seg_ckpt, val_loader, device, model_type=args.seg_model_type,
            )
            plot_training_history(seg_hist, "Segmentation")
        else:
            print(f"\n  ⚠  Segmentation checkpoint not found: {seg_ckpt}")

    # ── Combined report ───────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  FINAL EVALUATION REPORT")
    print("═" * 64)

    if cls_metrics is not None:
        print("\n  ┌─ Classification ───────────────────────────┐")
        print(f"  │  Accuracy  : {cls_metrics['accuracy']:.4f}                      │")
        print(f"  │  F1 Score  : {cls_metrics['f1_score']:.4f}                      │")
        print(f"  │  ROC-AUC   : {cls_metrics['roc_auc']:.4f}                      │")
        print(f"  │  Precision : {cls_metrics['precision']:.4f}                      │")
        print(f"  │  Recall    : {cls_metrics['recall']:.4f}                      │")
        print("  └─────────────────────────────────────────────┘")

    if seg_metrics is not None:
        print("\n  ┌─ Segmentation ─────────────────────────────┐")
        print(f"  │  Mean Dice : {seg_metrics['mean_dice']:.4f}                      │")
        print(f"  │  Mean IoU  : {seg_metrics['mean_iou']:.4f}                      │")
        print("  └─────────────────────────────────────────────┘")

    if cls_metrics is None and seg_metrics is None:
        print("\n  No checkpoints found. Train models first.")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("═" * 64)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained classification and segmentation models.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size for inference (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--cls_only", action="store_true",
        help="Evaluate only the classifier.",
    )
    parser.add_argument(
        "--seg_only", action="store_true",
        help="Evaluate only the segmentation model.",
    )
    parser.add_argument(
        "--seg_model_type", type=str, default="unet",
        choices=["unet", "unet++"],
        help="Segmentation model type (default: unet)",
    )
    args = parser.parse_args()
    main(args)
