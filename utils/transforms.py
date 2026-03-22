"""
Brain CT scan preprocessing pipelines using Albumentations.

Usage:
    from utils.transforms import get_classification_transforms, get_segmentation_transforms

    train_cls_tf = get_classification_transforms(mode="train")
    val_seg_tf   = get_segmentation_transforms(mode="val")
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ──────────────────────────────────────────────
#  Defaults (can be overridden via arguments)
# ──────────────────────────────────────────────
_DEFAULT_SIZE = 256
_DEFAULT_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_STD  = (0.5, 0.5, 0.5)


# ──────────────────────────────────────────────────────────────────────
#  Classification Transforms
# ──────────────────────────────────────────────────────────────────────

def get_classification_transforms(
    mode: str = "train",
    image_size: int = _DEFAULT_SIZE,
    mean: tuple = _DEFAULT_MEAN,
    std: tuple = _DEFAULT_STD,
) -> A.Compose:
    """
    Return an albumentations pipeline for classification.

    Args:
        mode: One of ``"train"`` or ``"val"`` / ``"test"``.
        image_size: Target spatial size (square).
        mean: Per-channel mean for normalisation.
        std: Per-channel std  for normalisation.

    Returns:
        ``albumentations.Compose`` pipeline.
    """
    if mode == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ──────────────────────────────────────────────────────────────────────
#  Segmentation Transforms
# ──────────────────────────────────────────────────────────────────────

def get_segmentation_transforms(
    mode: str = "train",
    image_size: int = _DEFAULT_SIZE,
    mean: tuple = _DEFAULT_MEAN,
    std: tuple = _DEFAULT_STD,
) -> A.Compose:
    """
    Return an albumentations pipeline for segmentation.

    Both ``image`` and ``mask`` are transformed together thanks to
    ``additional_targets={"mask": "mask"}``.

    Args:
        mode: One of ``"train"`` or ``"val"`` / ``"test"``.
        image_size: Target spatial size (square).
        mean: Per-channel mean for normalisation.
        std: Per-channel std  for normalisation.

    Returns:
        ``albumentations.Compose`` pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    p=0.3,
                ),
                A.GridDistortion(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )
    else:  # val / test
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )
