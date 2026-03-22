"""
Segmentation models for brain stroke lesion / hemorrhage region segmentation.

Built on top of ``segmentation-models-pytorch`` (smp).

Usage:
    from models.unet import get_segmentation_model

    model = get_segmentation_model("unet")          # smp.Unet
    model = get_segmentation_model("unet++")        # smp.UnetPlusPlus

    logits = model(x)                  # (B, 1, H, W) raw logits
    masks  = model.predict(x)          # (B, 1, H, W) binary 0/1
"""

import sys
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────────────────────────────
#  Helper: count parameters for a module
# ──────────────────────────────────────────────────────────────────────

def _count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────
#  U-Net
# ──────────────────────────────────────────────────────────────────────

class StrokeSegmentationModel(nn.Module):
    """
    Standard U-Net with a ResNet-34 encoder (ImageNet pretrained).

    Returns **raw logits** from ``forward()`` — apply sigmoid + threshold
    yourself, or use the convenience :meth:`predict` method.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,            # raw logits
        )

    # ── forward / inference ───────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 1, H, W)``."""
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask ``(B, 1, H, W)`` with values 0.0 / 1.0."""
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    # ── summary ───────────────────────────────────────────────────────

    def get_model_summary(self) -> None:
        """Print encoder and decoder parameter counts separately."""
        enc_total, enc_train = _count_params(self.model.encoder)
        dec_total, dec_train = _count_params(self.model.decoder)
        seg_total, seg_train = _count_params(self.model.segmentation_head)
        total = enc_total + dec_total + seg_total
        trainable = enc_train + dec_train + seg_train

        print("[StrokeSegmentationModel — U-Net]")
        print(f"  Encoder params:          {enc_total:>12,}  (trainable: {enc_train:,})")
        print(f"  Decoder params:          {dec_total:>12,}  (trainable: {dec_train:,})")
        print(f"  Segmentation head:       {seg_total:>12,}  (trainable: {seg_train:,})")
        print(f"  ─────────────────────────────────────")
        print(f"  Total params:            {total:>12,}")
        print(f"  Total trainable:         {trainable:>12,}")


# ──────────────────────────────────────────────────────────────────────
#  U-Net++
# ──────────────────────────────────────────────────────────────────────

class StrokeSegmentationModelPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) with a ResNet-34 encoder (ImageNet pretrained).

    Shares the same API as :class:`StrokeSegmentationModel`.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    # ── forward / inference ───────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 1, H, W)``."""
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask ``(B, 1, H, W)`` with values 0.0 / 1.0."""
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    # ── summary ───────────────────────────────────────────────────────

    def get_model_summary(self) -> None:
        """Print encoder and decoder parameter counts separately."""
        enc_total, enc_train = _count_params(self.model.encoder)
        dec_total, dec_train = _count_params(self.model.decoder)
        seg_total, seg_train = _count_params(self.model.segmentation_head)
        total = enc_total + dec_total + seg_total
        trainable = enc_train + dec_train + seg_train

        print("[StrokeSegmentationModelPlusPlus — U-Net++]")
        print(f"  Encoder params:          {enc_total:>12,}  (trainable: {enc_train:,})")
        print(f"  Decoder params:          {dec_total:>12,}  (trainable: {dec_train:,})")
        print(f"  Segmentation head:       {seg_total:>12,}  (trainable: {seg_train:,})")
        print(f"  ─────────────────────────────────────")
        print(f"  Total params:            {total:>12,}")
        print(f"  Total trainable:         {trainable:>12,}")


# ──────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────

def get_segmentation_model(
    model_type: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Build a segmentation model and move it to ``config.DEVICE``.

    Args:
        model_type: ``"unet"`` or ``"unet++"`` (``"unetplusplus"``).
        encoder_name: timm / smp encoder name.
        encoder_weights: pretrained weights identifier.
        in_channels: number of input channels.
        classes: number of output mask classes.

    Returns:
        Model on ``config.DEVICE``.
    """
    model_type = model_type.lower().replace(" ", "")

    if model_type in ("unet",):
        model = StrokeSegmentationModel(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_type in ("unet++", "unetplusplus", "unet_plusplus"):
        model = StrokeSegmentationModelPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'unet' or 'unet++'."
        )

    model.get_model_summary()
    model = model.to(config.DEVICE)
    return model


# ──────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  U-Net")
    print("=" * 60)
    unet = get_segmentation_model("unet")

    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    logits = unet(dummy)
    masks  = unet.predict(dummy)

    print(f"\n  Input shape:   {dummy.shape}")
    print(f"  Logits shape:  {logits.shape}")
    print(f"  Masks shape:   {masks.shape}")
    print(f"  Mask unique:   {masks.unique().tolist()}")

    print("\n" + "=" * 60)
    print("  U-Net++")
    print("=" * 60)
    unetpp = get_segmentation_model("unet++")

    logits_pp = unetpp(dummy)
    masks_pp  = unetpp.predict(dummy)

    print(f"\n  Input shape:   {dummy.shape}")
    print(f"  Logits shape:  {logits_pp.shape}")
    print(f"  Masks shape:   {masks_pp.shape}")
    print(f"  Mask unique:   {masks_pp.unique().tolist()}")
