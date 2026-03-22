"""
Evaluation metrics and custom loss functions for brain stroke
classification and lesion segmentation.

Usage:
    from utils.metrics import (
        compute_classification_metrics,
        dice_coefficient, iou_score, compute_segmentation_metrics,
        DiceLoss, CombinedLoss,
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFICATION METRICS  (sklearn-based)
# ══════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Compute a full set of classification metrics.

    Args:
        y_true:       Ground-truth labels — array-like of shape ``(N,)``
                      with values in ``{0, 1}``.
        y_pred_proba: Predicted **probabilities for the positive class**
                      (Stroke) — array-like of shape ``(N,)``, values in
                      ``[0, 1]``.
        threshold:    Decision boundary for converting probabilities to
                      hard labels.

    Returns:
        Dictionary with keys:
        ``accuracy``, ``f1_score``, ``roc_auc``, ``precision``,
        ``recall``, ``confusion_matrix`` (np.ndarray 2×2),
        ``classification_report`` (str).
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # ROC-AUC needs at least both classes present
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_score":  f1_score(y_true, y_pred, average="binary"),
        "roc_auc":   auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Normal", "Stroke"], zero_division=0,
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  SEGMENTATION METRICS  (pure-tensor, no sklearn)
# ══════════════════════════════════════════════════════════════════════

def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Sørensen–Dice coefficient for **binary** masks.

    Args:
        pred:   Binary tensor (any shape, values 0/1).
        target: Binary tensor (same shape as *pred*).
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar tensor.
    """
    pred   = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Intersection-over-Union (Jaccard Index) for **binary** masks.

    Args:
        pred:   Binary tensor (any shape, values 0/1).
        target: Binary tensor (same shape as *pred*).
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar tensor.
    """
    pred   = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def compute_segmentation_metrics(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mean Dice and IoU over a batch of masks.

    Args:
        pred_masks: Raw logits or probabilities of shape ``(B, 1, H, W)``.
        true_masks: Binary ground-truth masks ``(B, 1, H, W)``.
        threshold:  Binarisation threshold applied after sigmoid.

    Returns:
        ``{"mean_dice": float, "mean_iou": float}``
    """
    # Binarise predictions
    preds_bin = (torch.sigmoid(pred_masks) >= threshold).float()

    batch_size = pred_masks.shape[0]
    dice_scores = []
    iou_scores  = []

    for i in range(batch_size):
        d = dice_coefficient(preds_bin[i], true_masks[i])
        j = iou_score(preds_bin[i], true_masks[i])
        dice_scores.append(d.item())
        iou_scores.append(j.item())

    return {
        "mean_dice": float(np.mean(dice_scores)),
        "mean_iou":  float(np.mean(iou_scores)),
    }


# ══════════════════════════════════════════════════════════════════════
#  CUSTOM LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft Dice loss: ``1 − dice_coeff(sigmoid(logits), target)``.

    Works with **raw logits** (sigmoid is applied internally).
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        pred   = pred.contiguous().view(-1)
        target = target.float().contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Weighted sum of :class:`DiceLoss` and ``BCEWithLogitsLoss``.

    .. math::

        L = w_{dice} \\cdot DiceLoss + w_{bce} \\cdot BCEWithLogitsLoss
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss  = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        d_loss = self.dice_loss(pred_logits, target)
        b_loss = self.bce_loss(pred_logits, target.float())
        return self.dice_weight * d_loss + self.bce_weight * b_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss — generalisation of Dice that allows asymmetric
    penalisation of false positives vs false negatives.

    With ``alpha=0.3, beta=0.7`` false negatives are penalised more
    heavily, which is useful for small hemorrhage regions.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits).contiguous().view(-1)
        target = target.float().contiguous().view(-1)

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class TripleLoss(nn.Module):
    """
    Weighted sum of DiceLoss + BCEWithLogitsLoss + TverskyLoss.

    Default: ``0.4 * Dice + 0.3 * BCE + 0.3 * Tversky``
    """

    def __init__(
        self,
        dice_weight: float = 0.4,
        bce_weight: float = 0.3,
        tversky_weight: float = 0.3,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.tversky_loss = TverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, smooth=smooth,
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        d = self.dice_loss(pred_logits, target)
        b = self.bce_loss(pred_logits, target.float())
        t = self.tversky_loss(pred_logits, target)
        return self.dice_weight * d + self.bce_weight * b + self.tversky_weight * t


class FocalLoss(nn.Module):
    """
    Focal Loss for classification — down-weights easy examples so the
    model focuses on hard, misclassified ones.

    Supports label smoothing.  Works with logits (softmax applied internally).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw logits.
            targets: (B,) integer class labels.
        """
        probs = F.softmax(logits, dim=1)

        # One-hot with optional label smoothing
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        if self.label_smoothing > 0:
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        # Focal modulating factor
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1.0 - pt) ** self.gamma

        # Alpha weighting
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device),
        )

        # CE loss per sample
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Classification metrics test")
    print("=" * 60)

    y_true  = [0, 0, 1, 1, 1, 0, 1, 0]
    y_proba = [0.1, 0.3, 0.9, 0.8, 0.7, 0.2, 0.6, 0.4]

    cls_metrics = compute_classification_metrics(y_true, y_proba)
    print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {cls_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {cls_metrics['roc_auc']:.4f}")
    print(f"  Precision: {cls_metrics['precision']:.4f}")
    print(f"  Recall:    {cls_metrics['recall']:.4f}")
    print(f"  Confusion Matrix:\n{cls_metrics['confusion_matrix']}")
    print(f"\n{cls_metrics['classification_report']}")

    print("=" * 60)
    print("  Segmentation metrics test")
    print("=" * 60)

    # Simulated batch: (B=4, C=1, H=8, W=8)
    torch.manual_seed(42)
    pred_logits = torch.randn(4, 1, 8, 8)
    true_masks  = (torch.rand(4, 1, 8, 8) > 0.6).float()

    seg_metrics = compute_segmentation_metrics(pred_logits, true_masks)
    print(f"  Mean Dice: {seg_metrics['mean_dice']:.4f}")
    print(f"  Mean IoU:  {seg_metrics['mean_iou']:.4f}")

    print("\n" + "=" * 60)
    print("  Loss functions test")
    print("=" * 60)

    dice_loss_fn     = DiceLoss()
    combined_loss_fn = CombinedLoss(dice_weight=0.5, bce_weight=0.5)

    d_loss = dice_loss_fn(pred_logits, true_masks)
    c_loss = combined_loss_fn(pred_logits, true_masks)

    print(f"  Dice Loss:     {d_loss.item():.4f}")
    print(f"  Combined Loss: {c_loss.item():.4f}")
    print("\n  ✓ All tests passed.")
