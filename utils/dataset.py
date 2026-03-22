"""
PyTorch Dataset classes for Brain Stroke CT scan classification and
lesion segmentation.

Usage:
    from utils.dataset import StrokeClassificationDataset, StrokeSegmentationDataset

    cls_ds = StrokeClassificationDataset(root_dir="data/classification/train", transform=...)
    seg_ds = StrokeSegmentationDataset(images_dir="data/segmentation/train/images",
                                       masks_dir="data/segmentation/train/masks",
                                       transform=...)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Dict, List, Optional, Callable, Tuple

# Supported image extensions
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────
#  Classification Dataset
# ──────────────────────────────────────────────────────────────────────

class StrokeClassificationDataset(Dataset):
    """
    Expects a directory layout::

        root_dir/
        ├── Normal/
        │   ├── img001.png
        │   └── ...
        └── Stroke/
            ├── img002.png
            └── ...

    Labels: Normal → 0, Stroke → 1
    """

    CLASS_MAP = {"Normal": 0, "Stroke": 1}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for class_name, label in self.CLASS_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if os.path.splitext(fname)[1].lower() in _IMG_EXTENSIONS:
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

        assert len(self.image_paths) > 0, (
            f"No images found in {root_dir}. "
            "Expected subfolders: Normal/ and Stroke/"
        )

    # ── public helpers ────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Return inverse-frequency weights for every *sample* so that they
        can be passed directly to ``WeightedRandomSampler``.

        Weight for class c  =  total_samples / (num_classes * count_c)
        """
        counts = Counter(self.labels)
        num_classes = len(counts)
        total = len(self.labels)
        class_weights = {
            c: total / (num_classes * cnt) for c, cnt in counts.items()
        }
        # Per-sample weight vector
        sample_weights = torch.tensor(
            [class_weights[lbl] for lbl in self.labels], dtype=torch.float64
        )
        return sample_weights

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image with OpenCV and convert BGR → RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]  # Tensor after ToTensorV2

        return {"image": img, "label": label}


# ──────────────────────────────────────────────────────────────────────
#  Segmentation Dataset
# ──────────────────────────────────────────────────────────────────────

class StrokeSegmentationDataset(Dataset):
    """
    Expects a directory layout::

        images_dir/          masks_dir/
        ├── 070/             ├── 070/
        │   ├── slice001.png │   ├── slice001.png
        │   └── ...          │   └── ...
        ├── 071/             ├── 071/
        │   └── ...          │   └── ...
        └── ...              └── ...

    Each image is paired with its mask by identical relative path.
    Masks are binarised: pixel > 127 → 1.0, else 0.0
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Recursively collect relative paths (patient/slice.ext)
        self.relative_paths: List[str] = []
        for patient in sorted(os.listdir(images_dir)):
            patient_img_dir = os.path.join(images_dir, patient)
            if not os.path.isdir(patient_img_dir):
                continue
            for fname in sorted(os.listdir(patient_img_dir)):
                if os.path.splitext(fname)[1].lower() in _IMG_EXTENSIONS:
                    rel = os.path.join(patient, fname)
                    # Only add if the matching mask exists
                    mask_path = os.path.join(masks_dir, rel)
                    if os.path.isfile(mask_path):
                        self.relative_paths.append(rel)

        assert len(self.relative_paths) > 0, (
            f"No image–mask pairs found.\n"
            f"  images_dir: {images_dir}\n"
            f"  masks_dir:  {masks_dir}"
        )

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel = self.relative_paths[idx]

        img_path  = os.path.join(self.images_dir, rel)
        mask_path = os.path.join(self.masks_dir, rel)

        # Read image (BGR → RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask as grayscale and binarise
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # 0.0 or 1.0

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]   # Tensor (C, H, W)
            mask = augmented["mask"]    # Tensor (H, W)

        # Ensure mask has a channel dimension → (1, H, W)
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return {"image": img, "mask": mask}


# ──────────────────────────────────────────────────────────────────────
#  Collate Functions
# ──────────────────────────────────────────────────────────────────────

def classification_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack classification samples into batched tensors."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def segmentation_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack segmentation samples into batched tensors."""
    images = torch.stack([item["image"] for item in batch])
    masks  = torch.stack([item["mask"]  for item in batch])
    return {"image": images, "mask": masks}


# ──────────────────────────────────────────────────────────────────────
#  Quick sanity-check
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    from utils.transforms import (
        get_classification_transforms,
        get_segmentation_transforms,
    )

    # ── Classification ────────────────────────────────────────────────
    if os.path.isdir(config.CLASSIFICATION_TRAIN_DIR):
        cls_train = StrokeClassificationDataset(
            root_dir=config.CLASSIFICATION_TRAIN_DIR,
            transform=get_classification_transforms(mode="train"),
        )
        cls_val = StrokeClassificationDataset(
            root_dir=config.CLASSIFICATION_VAL_DIR,
            transform=get_classification_transforms(mode="val"),
        )
        print(f"[Classification]  train: {len(cls_train)}  |  val: {len(cls_val)}")
        print(f"  class weights shape: {cls_train.get_class_weights().shape}")
        sample = cls_train[0]
        print(f"  sample image shape:  {sample['image'].shape}  label: {sample['label']}")
    else:
        print(f"[Classification]  skipped — {config.CLASSIFICATION_TRAIN_DIR} not found")

    # ── Segmentation ─────────────────────────────────────────────────
    if os.path.isdir(config.SEGMENTATION_TRAIN_IMG_DIR):
        seg_train = StrokeSegmentationDataset(
            images_dir=config.SEGMENTATION_TRAIN_IMG_DIR,
            masks_dir=config.SEGMENTATION_TRAIN_MASK_DIR,
            transform=get_segmentation_transforms(mode="train"),
        )
        seg_val = StrokeSegmentationDataset(
            images_dir=config.SEGMENTATION_VAL_IMG_DIR,
            masks_dir=config.SEGMENTATION_VAL_MASK_DIR,
            transform=get_segmentation_transforms(mode="val"),
        )
        print(f"[Segmentation]    train: {len(seg_train)}  |  val: {len(seg_val)}")
        sample = seg_train[0]
        print(f"  sample image shape:  {sample['image'].shape}")
        print(f"  sample mask  shape:  {sample['mask'].shape}")
    else:
        print(f"[Segmentation]    skipped — {config.SEGMENTATION_TRAIN_IMG_DIR} not found")
