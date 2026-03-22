"""
Streamlit web application for Brain Stroke Detection & Lesion Segmentation.

Usage:
    streamlit run app.py
"""

import io
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

import config
from models.classifier import get_classifier, StrokeClassifier
from models.unet import get_segmentation_model

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

from utils.transforms import (
    get_classification_transforms,
    get_segmentation_transforms,
)


# ══════════════════════════════════════════════════════════════════════
#  Page configuration
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stroke Detection & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for clinical look (dark-mode compatible) ───────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 300;
        color: #a0aec0;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    .banner-normal {
        padding: 20px 32px;
        border-radius: 12px;
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 16px 0;
    }
    .banner-stroke {
        padding: 20px 32px;
        border-radius: 12px;
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 16px 0;
    }
    .metric-card {
        padding: 18px 24px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #0984e3;
        margin: 8px 0;
        color: #e2e8f0;
    }
    .metric-card b {
        color: #a0aec0;
    }
    .metric-card span {
        color: #ffffff;
    }
    .low-conf-warning {
        padding: 14px 20px;
        border-radius: 10px;
        background: rgba(253, 203, 110, 0.15);
        border-left: 4px solid #fdcb6e;
        color: #ffeaa7;
        font-weight: 500;
        margin: 10px 0;
    }
    .sidebar-info {
        font-size: 0.85rem;
        color: #a0aec0;
        line-height: 1.6;
    }
    .sidebar-info b {
        color: #e2e8f0;
    }
    hr.divider {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Model loading (cached)
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """
    Load classifier and segmentation models from checkpoints.
    Returns (classifier | None, segmentor | None).
    """
    cls_path = os.path.join(config.CHECKPOINT_DIR, "best_classifier.pth")
    seg_path = os.path.join(config.CHECKPOINT_DIR, "best_segmentation.pth")
    device = config.DEVICE

    classifier = None
    segmentor  = None

    # ── Classifier ────────────────────────────────────────────────────
    if os.path.isfile(cls_path):
        try:
            classifier = get_classifier(pretrained=False, freeze=False)
            ckpt = torch.load(cls_path, map_location=device)
            classifier.load_state_dict(ckpt["model_state_dict"])
            classifier.to(device)
            classifier.eval()
        except Exception as e:
            classifier = None
            st.sidebar.error(f"Classifier load failed: {e}")

    # ── Segmentor ─────────────────────────────────────────────────────
    if os.path.isfile(seg_path):
        try:
            ckpt = torch.load(seg_path, map_location=device)
            model_type = ckpt.get("model_type", "unet")
            encoder_name = ckpt.get("encoder", "resnet34")
            segmentor = get_segmentation_model(
                model_type=model_type,
                encoder_name=encoder_name,
            )
            segmentor.load_state_dict(ckpt["model_state_dict"])
            segmentor.to(device)
            segmentor.eval()
        except Exception as e:
            segmentor = None
            st.sidebar.error(f"Segmentor load failed: {e}")

    return classifier, segmentor


# ══════════════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════════════

def preprocess_image(
    img: np.ndarray,
    task: str = "classification",
) -> torch.Tensor:
    """
    Apply validation transforms and return a batched tensor on device.
    """
    if task == "classification":
        transform = get_classification_transforms(
            mode="val", image_size=config.IMAGE_SIZE,
        )
    else:
        transform = get_segmentation_transforms(
            mode="val", image_size=config.IMAGE_SIZE,
        )

    augmented = transform(image=img)
    tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)
    return tensor


# ══════════════════════════════════════════════════════════════════════
#  Inference helpers
# ══════════════════════════════════════════════════════════════════════

def generate_ai_explanation(classification, confidence, lesion_pct,
                             severity, region, affected_pixels, threshold):
    explanations = []
    
    if classification == 'Stroke':
        # Confidence explanation
        if confidence > 0.85:
            conf_text = (f"The model predicted STROKE with {confidence*100:.1f}% confidence — "
                        f"well above the {threshold:.2f} decision threshold. "
                        f"High-confidence predictions indicate strong visual features "
                        f"consistent with intracranial hemorrhage.")
        elif confidence > 0.65:
            conf_text = (f"The model predicted STROKE with {confidence*100:.1f}% confidence. "
                        f"Moderate confidence suggests hemorrhage features are present "
                        f"but may be subtle. Radiologist verification is strongly advised.")
        else:
            conf_text = (f"The model predicted STROKE with {confidence*100:.1f}% confidence — "
                        f"just above the {threshold:.2f} threshold. Low confidence may indicate "
                        f"an early-stage or atypical hemorrhage pattern.")
        
        # Severity explanation
        sev_map = {
            'Mild': ("The hemorrhage occupies less than 2.0% of the scan area. "
                    "Small bleeds may still be clinically significant depending on location."),
            'Moderate': ("The hemorrhage occupies 2.0–5.0% of the scan area. "
                        "Moderate bleeds typically require urgent neurology review "
                        "and close monitoring for expansion."),
            'Severe': ("The hemorrhage occupies more than 5.0% of the scan area. "
                      "Large bleeds are associated with higher mortality and "
                      "often require immediate neurosurgical intervention.")
        }
        sev_text = sev_map.get(severity, "")
        
        # Region explanation
        region_risk = {
            'Frontal lobe': 'May affect motor planning and executive function.',
            'Right parietal lobe': 'May affect spatial awareness and left-side sensation.',
            'Left parietal lobe': 'May affect language processing and right-side sensation.',
            'Central / basal ganglia': 'High-risk location — may affect movement control and consciousness.',
            'Left temporal lobe': 'May affect speech comprehension (Wernicke area).',
            'Right temporal lobe': 'May affect music perception and spatial memory.',
            'Thalamic region': 'Critical relay centre — may affect sensation and consciousness.',
            'Cerebellar / brainstem': 'High-risk — may affect balance, coordination and vital functions.'
        }
        region_text = (f"Hemorrhage localised to {region}. "
                      f"{region_risk.get(region, 'Clinical correlation required.')}")
        
        return conf_text, sev_text, region_text
    
    else:
        conf_text = (f"The model predicted NORMAL with {confidence*100:.1f}% confidence. "
                    f"No hyperdense regions consistent with hemorrhage were detected. "
                    f"Note: ischemic stroke may not be visible on non-contrast CT "
                    f"in the first 6 hours — clinical assessment remains essential.")
        return conf_text, None, None


def run_classification(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> Tuple[str, float, np.ndarray]:
    """
    Returns (predicted_class, confidence, proba_array).
    """
    with torch.no_grad():
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Use 0.36 threshold (F1-optimised) instead of argmax
    stroke_prob = proba[1]
    if stroke_prob >= 0.36:
        pred_idx = 1
    else:
        pred_idx = 0
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    return pred_class, confidence, proba


def run_segmentation(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Returns binary mask as numpy array of shape (H, W) with values 0/1.
    """
    with torch.no_grad():
        logits = model(tensor)
        sig_out = torch.sigmoid(logits).cpu().numpy()[0, 0]
        mask = (sig_out >= threshold).astype(np.float32)
    return mask


def create_overlay(
    original_img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a binary mask onto the CT scan in the chosen colour.
    """
    # Resize mask to match original image
    h, w = original_img.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
    mask_bool = mask_resized > 0.5

    overlay = original_img.copy().astype(np.float64)
    colour_layer = np.full_like(overlay, color, dtype=np.float64)

    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + colour_layer[mask_bool] * alpha
    )
    return overlay.astype(np.uint8)


def compute_lesion_area_pct(mask: np.ndarray) -> float:
    """Lesion area as a percentage of the total image area."""
    total_pixels = mask.shape[0] * mask.shape[1]
    lesion_pixels = (mask > 0.5).sum()
    return float(lesion_pixels / total_pixels * 100)


def get_brain_region(mask: np.ndarray, image_size: int = 256) -> str:
    """Identify the approximate anatomical brain region based on mask centroid."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return "Not detected"
        
    cy = np.mean(coords[0]) / image_size  # normalized 0-1 top to bottom
    cx = np.mean(coords[1]) / image_size  # normalized 0-1 left to right
    
    # Approximate anatomical regions by position
    if cy < 0.35:
        region = "Frontal lobe"
    elif cy < 0.55:
        if cx < 0.4:
            region = "Left parietal lobe"
        elif cx > 0.6:
            region = "Right parietal lobe"
        else:
            region = "Central / basal ganglia"
    elif cy < 0.75:
        if cx < 0.4:
            region = "Left temporal lobe"
        elif cx > 0.6:
            region = "Right temporal lobe"
        else:
            region = "Thalamic region"
    else:
        region = "Cerebellar / brainstem"
        
    return region

def generate_gradcam(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor, 
    original_img: np.ndarray, 
    predicted_idx: int
) -> Optional[np.ndarray]:
    if not HAS_GRADCAM:
        return None
        
    try:
        # Target the last conv layer of EfficientNet-B4 in our StrokeClassifier
        target_layers = [model.backbone.conv_head]  
        
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on original image
        img_float = np.float32(original_img) / 255
        # Resize cam to match image if needed
        if grayscale_cam.shape != img_float.shape[:2]:
            grayscale_cam = cv2.resize(grayscale_cam, (img_float.shape[1], img_float.shape[0]))
            
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        return visualization
    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  Report generation (downloadable PNG)
# ══════════════════════════════════════════════════════════════════════

def generate_pdf_report(original_img, overlay_img, mask, 
                        classification, confidence, lesion_pct,
                        affected_pixels, severity, region, threshold):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    import tempfile, io
    from datetime import datetime
    from PIL import Image as PILImage
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    
    styles = getSampleStyleSheet()
    story = []
    
    # ── Header ──────────────────────────────────────────────
    header_style = ParagraphStyle('header', fontSize=18, 
                                   fontName='Helvetica-Bold',
                                   textColor=colors.HexColor('#1a1a2e'),
                                   spaceAfter=2)
    sub_style = ParagraphStyle('sub', fontSize=10,
                                textColor=colors.HexColor('#666666'),
                                spaceAfter=4)
    
    story.append(Paragraph("AI-Assisted Brain Stroke Analysis Report", header_style))
    story.append(Paragraph("Powered by EfficientNet-B4 + U-Net (ResNet-34) | For research use only", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, 
                             color=colors.HexColor('#e74c3c'), spaceAfter=8))
    
    # ── Report metadata table ────────────────────────────────
    now = datetime.now()
    meta_data = [
        ['Report Generated:', now.strftime('%d %B %Y, %H:%M:%S'),
         'Patient ID:', 'ANON-' + now.strftime('%Y%m%d%H%M%S')],
        ['Scan Type:', 'Brain CT (Non-contrast)',
         'Analysis Model:', 'EfficientNet-B4 + U-Net'],
        ['Confidence Threshold:', f'{threshold:.2f}',
         'Institution:', 'NEXUS AESCODE 2025'],
    ]
    meta_table = Table(meta_data, colWidths=[45*mm, 60*mm, 40*mm, 55*mm])
    meta_table.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#333333')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), 
         [colors.HexColor('#f8f9fa'), colors.HexColor('#ffffff')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6*mm))
    
    # ── Classification result box ────────────────────────────
    result_color = '#e74c3c' if classification == 'Stroke' else '#27ae60'
    result_style = ParagraphStyle('result', fontSize=16,
                                   fontName='Helvetica-Bold',
                                   textColor=colors.white,
                                   backColor=colors.HexColor(result_color),
                                   borderPadding=8, spaceAfter=4,
                                   alignment=TA_CENTER)
    icon = '🚨' if classification == 'Stroke' else '✅'
    story.append(Paragraph(
        f"DIAGNOSIS: {classification.upper()}  |  Confidence: {confidence:.1f}%",
        result_style))
    story.append(Spacer(1, 4*mm))
    
    # ── Severity + Region + Metrics table ───────────────────
    if classification == 'Stroke':
        sev_color = {'Mild': '#27ae60', 
                     'Moderate': '#f39c12', 
                     'Severe': '#e74c3c'}.get(severity, '#333333')
        
        metrics_data = [
            ['SEVERITY GRADE', 'HEMORRHAGE LOCATION', 
             'LESION AREA', 'AFFECTED PIXELS'],
            [severity or 'N/A', region or 'N/A',
             f'{lesion_pct:.2f}%', f'{affected_pixels:,}'],
        ]
        m_table = Table(metrics_data, colWidths=[42*mm, 58*mm, 35*mm, 40*mm])
        m_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('FONTSIZE', (0,1), (-1,1), 11),
            ('FONTNAME', (0,1), (0,1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('TEXTCOLOR', (0,1), (0,1), colors.HexColor(sev_color)),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('PADDING', (0,0), (-1,-1), 8),
            ('ROWHEIGHT', (0,1), (-1,1), 18*mm),
        ]))
        story.append(m_table)
        story.append(Spacer(1, 5*mm))
    
    # ── Scan images ─────────────────────────────────────────
    story.append(Paragraph("Imaging Analysis", 
                            ParagraphStyle('sec', fontSize=12,
                                           fontName='Helvetica-Bold',
                                           textColor=colors.HexColor('#2c3e50'),
                                           spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#cccccc'), spaceAfter=4))
    
    def pil_to_rl_image(pil_img, width_mm, height_mm):
        img_buffer = io.BytesIO()
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return RLImage(img_buffer, width=width_mm*mm, height=height_mm*mm)
    
    orig_pil = PILImage.fromarray(original_img)
    overlay_pil = PILImage.fromarray(overlay_img)
    
    if classification == 'Stroke':
        img_w, img_h = 54, 54  # Reduced size to fit 3 neatly
        # Three images side by side: original, overlay, mask heatmap
        mask_vis = (mask * 255).astype(np.uint8)
        mask_pil = PILImage.fromarray(mask_vis).convert('RGB')
        
        img_row = [[
            pil_to_rl_image(orig_pil, img_w, img_h),
            pil_to_rl_image(overlay_pil, img_w, img_h),
            pil_to_rl_image(mask_pil, img_w, img_h),
        ],[
            Paragraph("Original CT Scan", 
                      ParagraphStyle('cap', fontSize=8, alignment=TA_CENTER,
                                     textColor=colors.HexColor('#666666'))),
            Paragraph("Hemorrhage Overlay", 
                      ParagraphStyle('cap', fontSize=8, alignment=TA_CENTER,
                                     textColor=colors.HexColor('#666666'))),
            Paragraph("Segmentation Mask", 
                      ParagraphStyle('cap', fontSize=8, alignment=TA_CENTER,
                                     textColor=colors.HexColor('#666666'))),
        ]]
        # 170mm total useable page width divided by 3 columns ≈ 56mm each
        img_table = Table(img_row, colWidths=[56*mm, 56*mm, 56*mm])
    else:
        img_w, img_h = 80, 80
        img_row = [[
            pil_to_rl_image(orig_pil, img_w, img_h),
        ],[
            Paragraph("Original CT Scan — No hemorrhage detected",
                      ParagraphStyle('cap', fontSize=8, alignment=TA_CENTER,
                                     textColor=colors.HexColor('#666666'))),
        ]]
        img_table = Table(img_row, colWidths=[100*mm])
    
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 5*mm))
    
    # ── AI Findings narrative ────────────────────────────────
    story.append(Paragraph("AI Findings Summary",
                            ParagraphStyle('sec', fontSize=12,
                                           fontName='Helvetica-Bold',
                                           textColor=colors.HexColor('#2c3e50'),
                                           spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#cccccc'), spaceAfter=4))
    
    if classification == 'Stroke':
        narrative = (
            f"The AI system detected intracranial hemorrhage with <b>{confidence:.1f}% confidence</b> "
            f"(threshold: {threshold:.2f}). The hemorrhagic region is localised to the "
            f"<b>{region}</b>, with a lesion area of <b>{lesion_pct:.2f}%</b> of total scan area "
            f"({affected_pixels:,} affected pixels). "
            f"Hemorrhage severity is graded as <b>{severity}</b> based on lesion extent. "
            f"Immediate clinical correlation is recommended."
        )
    else:
        narrative = (
            f"No intracranial hemorrhage was detected by the AI system "
            f"(Normal confidence: {confidence:.1f}%, threshold: {threshold:.2f}). "
            f"Clinical correlation with patient symptoms is advised. "
            f"This result does not exclude ischemic stroke, which may not be "
            f"visible on non-contrast CT in early stages."
        )
    
    story.append(Paragraph(narrative,
                            ParagraphStyle('narr', fontSize=9,
                                           textColor=colors.HexColor('#333333'),
                                           leading=14, spaceAfter=4)))
    story.append(Spacer(1, 4*mm))
    
    # ── Disclaimer ───────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#e74c3c'), spaceAfter=3))
    disclaimer = (
        "<b>⚠ DISCLAIMER:</b> This report is generated by an AI system for "
        "research and educational purposes only. It is <b>NOT</b> a substitute "
        "for professional medical diagnosis. All findings must be verified by a "
        "qualified radiologist or neurologist before any clinical decision is made. "
        "The AI model has been trained on a limited dataset and may not generalise "
        "to all patient populations or CT scanner types."
    )
    story.append(Paragraph(disclaimer,
                            ParagraphStyle('disc', fontSize=8,
                                           textColor=colors.HexColor('#c0392b'),
                                           leading=12, backColor=colors.HexColor('#fff5f5'),
                                           borderPadding=6)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════════════
#  Demo image
# ══════════════════════════════════════════════════════════════════════

def generate_demo_image() -> np.ndarray:
    """Create a synthetic demo CT scan for testing when no real data is
    available."""
    size = 256
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Simulated skull outline
    cv2.circle(img, (size // 2, size // 2), size // 2 - 10, (60, 60, 60), -1)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 20, (30, 30, 30), -1)
    # Simulated brain
    cv2.ellipse(img, (size // 2, size // 2), (80, 90), 0, 0, 360, (50, 50, 55), -1)
    # Simulated lesion
    cv2.circle(img, (size // 2 + 30, size // 2 - 20), 22, (120, 100, 90), -1)
    cv2.circle(img, (size // 2 + 35, size // 2 - 15), 10, (160, 130, 110), -1)
    # Add some noise
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


# ══════════════════════════════════════════════════════════════════════
#  App layout
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────
    st.markdown('<p class="main-title">🧠 Brain Stroke Detection & Lesion Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-powered CT Scan Analysis</p>', unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────────────────────
    classifier, segmentor = load_models()

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Model info
        st.subheader("Model Information")
        cls_status = "✅ Loaded" if classifier is not None else "❌ Not found"
        seg_status = "✅ Loaded" if segmentor is not None else "❌ Not found"
        st.markdown(f"""
        <div class="sidebar-info">
        <b>Classifier:</b> EfficientNet-B4 — {cls_status}<br>
        <b>Segmentor:</b> U-Net (ResNet-34) — {seg_status}<br>
        <b>Device:</b> {config.DEVICE.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Controls
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3, max_value=0.9, value=0.36, step=0.01,
            help="Minimum classification confidence to trigger segmentation.",
        )
        st.caption("Optimal threshold (0.36) determined by F1 maximisation on validation set.")
        show_overlay = st.toggle("Show Segmentation Overlay", value=True)
        overlay_color = st.color_picker("Overlay Colour", value="#FF0000")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Demo button
        use_demo = st.button("🖼️ Load Demo Image", use_container_width=True)

        if classifier is None and segmentor is None:
            st.warning(
                "No model checkpoints found in "
                f"`{config.CHECKPOINT_DIR}`. "
                "Train models first or place checkpoint files."
            )

    # ── Parse overlay colour ──────────────────────────────────────────
    hex_col   = overlay_color.lstrip("#")
    rgb_color = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))

    # ── Image input ───────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a brain CT scan",
        type=["png", "jpg", "jpeg", "dcm"],
        help="Supported formats: PNG, JPEG, DICOM (.dcm)",
    )

    image_np: Optional[np.ndarray] = None

    if use_demo:
        image_np = generate_demo_image()
        st.info("📌 Using a synthetic demo image for testing.")
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".dcm"):
                try:
                    import pydicom
                    ds = pydicom.dcmread(uploaded_file)
                    arr = ds.pixel_array.astype(np.float32)
                    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                    if arr.ndim == 2:
                        image_np = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                    else:
                        image_np = arr
                except ImportError:
                    st.error("Install `pydicom` to load DICOM files: `pip install pydicom`")
            else:
                pil_img  = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(pil_img)
        except Exception as e:
            st.error(f"Failed to load image: {e}")

    if image_np is None:
        st.markdown(
            """
            <div style="text-align:center; padding:80px 20px; color:#b2bec3;">
                <h3>⬆️ Upload a CT scan image to begin analysis</h3>
                <p>or click <b>Load Demo Image</b> in the sidebar</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Display original ──────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("📤 Uploaded CT Scan")
    st.image(image_np, caption="Original CT Scan", use_container_width=False, width=350)

    # ══════════════════════════════════════════════════════════════════
    #  Classification
    # ══════════════════════════════════════════════════════════════════

    if classifier is None:
        st.warning("⚠️ Classifier model not loaded. Cannot perform classification.")
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("🔬 Classification Result")

    with st.spinner("Running classification..."):
        cls_tensor = preprocess_image(image_np, task="classification")
        pred_class, confidence, proba = run_classification(classifier, cls_tensor)

    # ── Result banner ─────────────────────────────────────────────────
    if pred_class == "Normal":
        st.markdown(
            '<div class="banner-normal">✅ NORMAL</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-stroke">🚨 STROKE DETECTED</div>',
            unsafe_allow_html=True,
        )

    # ── Confidence bars ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Normal:** `{proba[0]:.1%}`")
        st.progress(float(proba[0]))
    with col_b:
        st.markdown(f"**Stroke:** `{proba[1]:.1%}`")
        st.progress(float(proba[1]))

    if confidence < 0.6:
        st.markdown(
            '<div class="low-conf-warning">'
            "⚠️ <b>Low confidence</b> — recommend radiologist review."
            "</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Segmentation
    # ══════════════════════════════════════════════════════════════════

    mask_np: Optional[np.ndarray] = None
    overlay_img: Optional[np.ndarray] = None
    lesion_pct: Optional[float] = None

    if pred_class == "Stroke" and confidence >= threshold:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.subheader("🩸 Hemorrhage Segmentation")

        if segmentor is None:
            st.warning("⚠️ Segmentation model not loaded. Skipping segmentation.")
        else:
            with st.spinner("Running segmentation..."):
                seg_tensor = preprocess_image(image_np, task="segmentation")
                mask_np = run_segmentation(segmentor, seg_tensor, threshold=0.3)
                lesion_pct = compute_lesion_area_pct(mask_np)

                # Resize mask back to original image dimensions
                h, w = image_np.shape[:2]
                mask_full = cv2.resize(mask_np.astype(np.float32), (w, h))

                overlay_img = create_overlay(
                    image_np, mask_full, color=rgb_color, alpha=0.4,
                )

            # ── Four-column display ──────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(image_np, caption="Original CT", use_container_width=True)
            with col2:
                # Show mask as a coloured heatmap
                mask_display = (mask_full * 255).astype(np.uint8)
                mask_coloured = cv2.applyColorMap(mask_display, cv2.COLORMAP_HOT)
                mask_coloured = cv2.cvtColor(mask_coloured, cv2.COLOR_BGR2RGB)
                st.image(mask_coloured, caption="Hemorrhage Mask", use_container_width=True)
            with col3:
                if show_overlay:
                    st.image(overlay_img, caption="Overlay", use_container_width=True)
                else:
                    st.info("Overlay hidden.")
            with col4:
                predicted_idx = 1 if pred_class == "Stroke" else 0
                cam_img = generate_gradcam(classifier, cls_tensor, image_np, predicted_idx)
                if cam_img is not None:
                    st.image(cam_img, caption="AI Attention Map", use_container_width=True)
                else:
                    st.info("Attention Map unavailable.")

            # ── AI Explanation ────────────────────────────────────────
            # Precompute metrics needed for explanation
            pixel_count = int((mask_full > 0.5).sum())
            region = get_brain_region(mask_full, image_size=h)
            
            severity = "N/A"
            if lesion_pct is not None:
                if lesion_pct < 2.0:
                    severity = "Mild"
                elif lesion_pct < 5.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                    
            conf_exp, sev_exp, reg_exp = generate_ai_explanation(
                classification=pred_class,
                confidence=confidence,
                lesion_pct=lesion_pct,
                severity=severity,
                region=region,
                affected_pixels=pixel_count,
                threshold=threshold
            )

            st.markdown("### 🧠 AI Explanation")
            with st.expander("Why did the AI make this prediction?", expanded=True):
                st.markdown("**Confidence Analysis**")
                st.info(conf_exp)
                
                if sev_exp:
                    st.markdown("**Severity Interpretation**")
                    st.warning(sev_exp)
                
                if reg_exp:
                    st.markdown("**Neurological Region Impact**")
                    st.error(reg_exp)
                
                st.markdown("---")
                st.caption(
                    "This explanation is generated from model output statistics and "
                    "clinical knowledge rules. It is intended to support — not replace — "
                    "radiologist interpretation."
                )

            # ── Metrics cards ─────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card">'
                    f"<b>Lesion Area</b><br>"
                    f'<span style="font-size:1.6rem;font-weight:700;">{lesion_pct:.2f}%</span>'
                    f"<br>of total scan area"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card">'
                    f"<b>Affected Pixels</b><br>"
                    f'<span style="font-size:1.6rem;font-weight:700;">{pixel_count:,}</span>'
                    f"<br>out of {h * w:,} total pixels"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card">'
                    f"<b>Hemorrhage Location</b><br>"
                    f'<span style="font-size:1.6rem;font-weight:700;">{region}</span>'
                    f"<br>estimated via centroid"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    elif pred_class == "Normal" or (pred_class == "Stroke" and confidence < threshold):
        # Even if normal or below threshold, we can show the basic explanation logic block
        if pred_class == "Stroke" and confidence < threshold:
            st.info(
                f"Stroke detected but confidence ({confidence:.1%}) is below "
                f"threshold ({threshold:.1%}). Segmentation skipped."
            )
            
        conf_exp, _, _ = generate_ai_explanation(
            classification=pred_class,
            confidence=confidence,
            lesion_pct=0.0,
            severity="N/A",
            region="N/A",
            affected_pixels=0,
            threshold=threshold
        )
        st.markdown("### 🧠 AI Explanation")
        with st.expander("Why did the AI make this prediction?", expanded=True):
            st.markdown("**Confidence Analysis**")
            st.info(conf_exp)
            st.markdown("---")
            st.caption(
                "This explanation is generated from model output statistics and "
                "clinical knowledge rules. It is intended to support — not replace — "
                "radiologist interpretation."
            )

    # ══════════════════════════════════════════════════════════════════
    #  Downloadable report
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("📄 Download Report")

    # Determine severity based on lesion_pct
    severity = "N/A"
    if lesion_pct is not None:
        if lesion_pct < 2.0:
            severity = "Mild"
        elif lesion_pct < 5.0:
            severity = "Moderate"
        else:
            severity = "Severe"

    pdf_bytes = generate_pdf_report(
        original_img=image_np,
        overlay_img=overlay_img if overlay_img is not None else image_np,
        mask=mask_full if mask_np is not None else np.zeros_like(image_np[:,:,0], dtype=np.float32),
        classification=pred_class,
        confidence=confidence * 100,
        lesion_pct=lesion_pct if lesion_pct is not None else 0.0,
        affected_pixels=int((mask_full > 0.5).sum()) if mask_np is not None else 0,
        severity=severity,
        region=region if 'region' in locals() else "N/A",
        threshold=threshold
    )
    
    st.download_button(
        label="⬇️  Download Clinical Report (PDF)",
        data=pdf_bytes,
        file_name=f"stroke_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    # ── Disclaimer ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:16px 0; color:#b2bec3; font-size:0.8rem;">
            ⚕️ This tool is for research and educational purposes only.<br>
            It is <b>not</b> a substitute for professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
