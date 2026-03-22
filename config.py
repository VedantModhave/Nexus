import os
import torch

# ──────────────────────────────────────────────
#  Base Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")  # ../data (Nexus/data/)

# ──────────────────────────────────────────────
#  Classification Data Paths
# ──────────────────────────────────────────────
CLASSIFICATION_TRAIN_DIR = os.path.join(DATA_DIR, "classification", "train")
CLASSIFICATION_VAL_DIR   = os.path.join(DATA_DIR, "classification", "val")

# ──────────────────────────────────────────────
#  Segmentation Data Paths
# ──────────────────────────────────────────────
SEGMENTATION_TRAIN_IMG_DIR  = os.path.join(DATA_DIR, "Segmentation", "train", "images")
SEGMENTATION_TRAIN_MASK_DIR = os.path.join(DATA_DIR, "Segmentation", "train", "masks")
SEGMENTATION_VAL_IMG_DIR    = os.path.join(DATA_DIR, "Segmentation", "val", "images")
SEGMENTATION_VAL_MASK_DIR   = os.path.join(DATA_DIR, "Segmentation", "val", "masks")

# ──────────────────────────────────────────────
#  Model Checkpoints
# ──────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASSIFIER_CHECKPOINT  = os.path.join(CHECKPOINT_DIR, "classifier_best.pth")
SEGMENTOR_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "segmentor_best.pth")

# ──────────────────────────────────────────────
#  Image / Data Settings
# ──────────────────────────────────────────────
IMAGE_SIZE  = 256          # Resize all inputs to (IMAGE_SIZE x IMAGE_SIZE)
BATCH_SIZE  = 16
NUM_WORKERS = 4

# ──────────────────────────────────────────────
#  Training Hyperparameters
# ──────────────────────────────────────────────
NUM_EPOCHS_CLASSIFIER   = 30
NUM_EPOCHS_SEGMENTATION = 50
LEARNING_RATE           = 1e-4

# ──────────────────────────────────────────────
#  Model Architecture
# ──────────────────────────────────────────────
CLASSIFIER_BACKBONE  = "resnet50"        # any timm-compatible name
SEGMENTOR_ENCODER    = "efficientnet-b0" # any smp-compatible encoder

# ──────────────────────────────────────────────
#  Class Labels
# ──────────────────────────────────────────────
CLASS_NAMES = ["Normal", "Stroke"]
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────
SEED = 42
