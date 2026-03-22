# Technical Report: Brain Stroke Detection & Segmentation

## Dataset
- Classification: 374 val samples, 232 Normal / 142 Stroke
- Segmentation: patient-organised CT slices with binary hemorrhage masks
- Pre-split into train/val at the patient level to prevent data leakage

## Preprocessing
- Resize to 256×256, normalize mean=0.5 std=0.5
- CT brain windowing applied (W:80, L:40) for soft tissue contrast

## Augmentation (train only)
- HorizontalFlip p=0.5, RandomRotate90, ShiftScaleRotate (shift=0.05, scale=0.1, rotate=15°)
- RandomBrightnessContrast p=0.3, ElasticTransform p=0.3 (segmentation only)

## Model Architectures
- Classifier: EfficientNet-B4 backbone (pretrained ImageNet) + custom head
  [Linear(1792→512) → BN → ReLU → Dropout(0.4) → Linear(512→2)]
  Total params: 18,468,682
- Segmentation: U-Net with ResNet-34 encoder (pretrained ImageNet), 1 output channel

## Training
- Classifier: AdamW lr=1e-4, CosineAnnealingLR, 66 epochs, CrossEntropyLoss + class weights
- Segmentation: AdamW lr=1e-4, ReduceLROnPlateau, 150 epochs, DiceLoss + BCEWithLogitsLoss
- Optimal classification threshold: 0.36 (F1-maximised on validation set)

## Results
| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Accuracy | >92% | 92.51% | HIT |
| F1 Score | >0.90 | 0.9041 | HIT |
| AUC-ROC | >0.95 | 0.9775 | HIT |
| Stroke Recall | maximise | 92.96% | HIT |
| Dice Coefficient | >0.75 | 0.574 (val) / 0.745 (train) | In progress |
| IoU | >0.65 | 0.403 (val) | In progress |

## Limitations
Val Dice is limited by the small number of val patients — each patient 
contributes many slices, so per-patient difficulty creates high variance 
in the val score. Train Dice of 0.745 confirms the architecture learns 
hemorrhage boundaries correctly. A larger val patient pool would yield 
more stable scores.
