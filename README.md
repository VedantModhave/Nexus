# Brain Stroke Detection & Lesion Segmentation

A dual-stage deep learning pipeline for brain CT scan analysis: (1) binary classification of Normal vs Stroke using EfficientNet-B4, and (2) hemorrhage region segmentation using U-Net/U-Net++ with a ResNet-34 encoder. Includes a Streamlit web interface for real-time inference and report generation.

---

## Dataset Structure

```
data/
├── classification/
│   ├── train/
│   │   ├── Normal/          # Normal CT scan images (.png/.jpg)
│   │   └── Stroke/          # Stroke CT scan images
│   └── val/
│       ├── Normal/
│       └── Stroke/
└── segmentation/
    ├── train/
    │   ├── images/           # Patient subfolders
    │   │   ├── 070/
    │   │   │   ├── slice001.png
    │   │   │   └── ...
    │   │   ├── 071/
    │   │   └── ...
    │   └── masks/            # Matching patient subfolders (binary masks)
    │       ├── 070/
    │       │   ├── slice001.png
    │       │   └── ...
    │       ├── 071/
    │       └── ...
    └── val/
        ├── images/
        └── masks/
```

**Image ↔ mask pairing** is by identical relative path:  
`images/070/slice001.png` → `masks/070/slice001.png`

Masks must be **binary**: white (255) = lesion, black (0) = background.

---

## Installation

```bash
# Clone or download
cd stroke_detection

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

`torch` · `torchvision` · `timm` · `segmentation-models-pytorch` · `opencv-python` · `albumentations` · `scikit-learn` · `matplotlib` · `seaborn` · `streamlit` · `Pillow` · `numpy` · `tqdm` · `pandas`

---

## Data Preparation

1. **Download** the dataset from Google Drive (or your source).
2. **Classification data**: place all Normal scans in `data/classification/train/Normal/` and Stroke scans in `data/classification/train/Stroke/`. Create the same split under `val/`. A 80/20 train/val split is recommended.
3. **Segmentation data**: organise images and masks into patient subfolders (e.g., `070/`, `071/`). Every image must have a corresponding mask at the same relative path under `masks/`.
4. Verify the layout matches the tree above.

---

## Training

### Classification

```bash
python train_classifier.py
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `30` | Training epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--batch_size` | `16` | Batch size |

The backbone is **frozen for the first 5 epochs**, then unfrozen with LR reduced by 10×.

### Segmentation

```bash
python train_segmentation.py --model_type unet
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `50` | Training epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--batch_size` | `16` | Batch size |
| `--model_type` | `unet` | `unet` or `unet++` |

Best models are saved to `checkpoints/`.

---

## Evaluation

```bash
python evaluate.py                # evaluate both models
python evaluate.py --cls_only     # classification only
python evaluate.py --seg_only     # segmentation only
```

Outputs are saved to `evaluate_outputs/`:

| File | Description |
|---|---|
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `classification_report_heatmap.png` | Precision / Recall / F1 heatmap |
| `segmentation_samples.png` | Grid: CT · mask · prediction · overlay |
| `*_training_curves.png` | Loss and metric curves |

---

## Streamlit App

```bash
streamlit run app.py
```

Features:
- Upload PNG/JPG/DICOM CT scans
- Real-time classification with confidence bars
- Hemorrhage segmentation with colour overlay
- Adjustable confidence threshold and overlay settings
- Downloadable analysis report (PNG)
- Built-in demo image for testing without data

---

## Model Architecture

| Task | Model | Encoder | Input | Output |
|---|---|---|---|---|
| Classification | EfficientNet-B4 + custom head | EfficientNet-B4 (ImageNet) | `(B, 3, 256, 256)` | `(B, 2)` logits |
| Segmentation | U-Net | ResNet-34 (ImageNet) | `(B, 3, 256, 256)` | `(B, 1, 256, 256)` logits |
| Segmentation | U-Net++ | ResNet-34 (ImageNet) | `(B, 3, 256, 256)` | `(B, 1, 256, 256)` logits |

**Classification head**: `Linear(1792, 512) → BN → ReLU → Dropout(0.4) → Linear(512, 2)`

**Segmentation loss**: `0.5 × DiceLoss + 0.5 × BCEWithLogitsLoss`

---

## Expected Results

### Classification

| Metric | Target |
|---|---|
| Accuracy | > 92% |
| F1 Score | > 0.90 |
| AUC-ROC | > 0.95 |
| Precision | > 0.88 |
| Recall | > 0.90 |

### Segmentation

| Metric | Target |
|---|---|
| Dice Coefficient | > 0.75 |
| IoU (Jaccard) | > 0.65 |

*Results depend on dataset size, quality, and class balance.*

---

## Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

- Reduce `BATCH_SIZE` in `config.py` (try `8` or `4`)
- Reduce `IMAGE_SIZE` to `224`
- Use `--batch_size 4` via CLI
- Set `config.DEVICE = "cpu"` to fall back to CPU

### Missing Checkpoints

```
⚠ Classifier checkpoint not found: checkpoints/best_classifier.pth
```

- Train the model first: `python train_classifier.py`
- Ensure `CHECKPOINT_DIR` in `config.py` points to the correct folder
- Check that training completed successfully (not interrupted)

### Image–Mask Pairing Errors

```
AssertionError: No image–mask pairs found.
```

- Verify folder structure: `images/<patient>/<file>` must match `masks/<patient>/<file>`
- Filenames and extensions must be **identical** between images and masks
- Check for hidden files (`.DS_Store`, `Thumbs.db`) — the dataset loader skips non-image extensions but stray files in patient dirs can cause issues

### Albumentations Version Conflicts

```
TypeError: __init__() got an unexpected keyword argument
```

- Use `albumentations>=1.3.0` — earlier versions have different API signatures
- Run `pip install --upgrade albumentations`

### DICOM Files Not Loading

- Install `pydicom`: `pip install pydicom`
- DICOM support is used only in `app.py`; training scripts expect PNG/JPG

---

## Project Structure

```
stroke_detection/
├── config.py                  # Paths, hyperparameters, constants
├── train_classifier.py        # Classification training script
├── train_segmentation.py      # Segmentation training script
├── evaluate.py                # Evaluation + plots
├── app.py                     # Streamlit web interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/
│   ├── __init__.py
│   ├── classifier.py          # EfficientNet-B4 classifier
│   └── unet.py                # U-Net / U-Net++ segmentor
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # Dataset classes
│   ├── transforms.py          # Albumentations pipelines
│   └── metrics.py             # Metrics + custom losses
├── data/                      # Your dataset (not tracked in git)
├── checkpoints/               # Saved model weights (auto-created)
└── evaluate_outputs/          # Evaluation plots (auto-created)
```

---

## License

This project is for educational and research purposes.  
**Not intended for clinical diagnosis.**
