"""
Brain Stroke Classification model built on EfficientNet-B4 (timm).

Usage:
    from models.classifier import get_classifier, StrokeClassifier

    model = get_classifier()           # on config.DEVICE, backbone frozen
    model.unfreeze_backbone()          # call after epoch 5
    probs = model.predict_proba(x)     # softmax probabilities
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────────────────────────────
#  StrokeClassifier
# ──────────────────────────────────────────────────────────────────────

class StrokeClassifier(nn.Module):
    """
    EfficientNet-B4 backbone with a custom two-layer classification head.

    Head architecture::

        Linear(in_features, 512)
        → BatchNorm1d(512)
        → ReLU
        → Dropout(0.4)
        → Linear(512, num_classes)
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.4,
    ) -> None:
        super().__init__()

        # ── backbone ──────────────────────────────────────────────────
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove original head → returns features
        )
        in_features = self.backbone.num_features  # e.g. 1792 for B4

        # ── custom head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes),
        )

        # ── parameter summary ────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[StrokeClassifier] {model_name}")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    # ── forward / inference ───────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)``."""
        features = self.backbone(x)          # (B, in_features)
        logits   = self.head(features)       # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape ``(B, num_classes)``."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    # ── backbone freeze / unfreeze ────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (train only the head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Backbone FROZEN  — trainable params: {trainable:,}")

    def unfreeze_backbone(self) -> None:
        """Unfreeze the entire backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Backbone UNFROZEN — trainable params: {trainable:,}")


# ──────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────

def get_classifier(
    model_name: str = "efficientnet_b4",
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze: bool = True,
) -> StrokeClassifier:
    """
    Build a :class:`StrokeClassifier`, optionally freeze the backbone,
    and move to ``config.DEVICE``.

    Args:
        model_name:  timm model identifier.
        num_classes: number of output classes (default from config).
        pretrained:  load ImageNet weights.
        freeze:      whether to freeze backbone on creation (recommended
                     for the first 5 epochs).

    Returns:
        ``StrokeClassifier`` on ``config.DEVICE``.
    """
    model = StrokeClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    if freeze:
        model.freeze_backbone()

    model = model.to(config.DEVICE)
    return model


# ──────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = get_classifier()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)

    logits = model(dummy)
    probs  = model.predict_proba(dummy)

    print(f"\n  Input shape:   {dummy.shape}")
    print(f"  Logits shape:  {logits.shape}")
    print(f"  Probs shape:   {probs.shape}")
    print(f"  Probs sample:  {probs[0].cpu().tolist()}")
