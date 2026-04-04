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
    .ci-bar-container {
        position: relative;
        height: 28px;
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        margin: 8px 0 4px 0;
        overflow: hidden;
    }
    .ci-bar-range {
        position: absolute;
        top: 0; bottom: 0;
        border-radius: 14px;
        opacity: 0.35;
    }
    .ci-bar-mean {
        position: absolute;
        top: 0; bottom: 0;
        width: 3px;
        border-radius: 2px;
        transform: translateX(-1px);
    }
    .ci-bar-label {
        position: absolute;
        top: 50%; transform: translateY(-50%);
        font-size: 0.72rem;
        font-weight: 600;
        color: #e2e8f0;
        pointer-events: none;
    }
    .uncertainty-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    .uncertainty-low  { background: rgba(0,184,148,0.2); color: #00b894; }
    .uncertainty-mid  { background: rgba(9,132,227,0.2); color: #74b9ff; }
    .uncertainty-high { background: rgba(225,112,85,0.2); color: #e17055; }
    .nihss-card {
        padding: 22px 28px;
        border-radius: 14px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin: 16px 0;
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }
    .nihss-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.08;
        background: radial-gradient(circle at 30% 50%, white, transparent 70%);
        pointer-events: none;
    }
    .nihss-mild {
        background: linear-gradient(135deg, #00b894, #00cec9);
        border-left: 5px solid #00d2d3;
        box-shadow: 0 4px 20px rgba(0, 184, 148, 0.25);
    }
    .nihss-moderate {
        background: linear-gradient(135deg, #f39c12, #e17055);
        border-left: 5px solid #fdcb6e;
        box-shadow: 0 4px 20px rgba(243, 156, 18, 0.25);
    }
    .nihss-severe {
        background: linear-gradient(135deg, #d63031, #e74c3c);
        border-left: 5px solid #ff7675;
        box-shadow: 0 4px 20px rgba(214, 48, 49, 0.30);
    }
    .nihss-card .nihss-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .nihss-card .nihss-band {
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .nihss-card .nihss-score {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
        margin: 4px 0;
    }
    .nihss-card .nihss-note {
        font-size: 0.88rem;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 8px;
        line-height: 1.5;
    }
    .nihss-disclaimer {
        font-size: 0.78rem;
        color: #a0aec0;
        font-style: italic;
        padding: 8px 0 0 0;
    }
    .midline-card {
        padding: 22px 28px;
        border-radius: 14px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin: 16px 0;
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }
    .midline-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.08;
        background: radial-gradient(circle at 70% 50%, white, transparent 70%);
        pointer-events: none;
    }
    .midline-minimal {
        background: linear-gradient(135deg, #00b894, #00cec9);
        border-left: 5px solid #00d2d3;
        box-shadow: 0 4px 20px rgba(0, 184, 148, 0.25);
    }
    .midline-mild {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        border-left: 5px solid #74b9ff;
        box-shadow: 0 4px 20px rgba(9, 132, 227, 0.25);
    }
    .midline-moderate {
        background: linear-gradient(135deg, #f39c12, #e17055);
        border-left: 5px solid #fdcb6e;
        box-shadow: 0 4px 20px rgba(243, 156, 18, 0.25);
    }
    .midline-severe {
        background: linear-gradient(135deg, #d63031, #e74c3c);
        border-left: 5px solid #ff7675;
        box-shadow: 0 4px 20px rgba(214, 48, 49, 0.30);
    }
    .midline-card .midline-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .midline-card .midline-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
        margin: 4px 0;
    }
    .midline-card .midline-sev {
        font-size: 1.2rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .midline-card .midline-note {
        font-size: 0.85rem;
        font-weight: 400;
        opacity: 0.85;
        margin-top: 8px;
        line-height: 1.4;
    }
    .triage-banner {
        padding: 24px 32px;
        border-radius: 14px;
        margin: 20px 0 12px 0;
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }
    .triage-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.06;
        background: radial-gradient(circle at 20% 50%, white, transparent 60%);
        pointer-events: none;
    }
    .triage-watch {
        background: linear-gradient(135deg, #0984e3, #74b9ff);
        border-left: 6px solid #74b9ff;
        box-shadow: 0 6px 24px rgba(9, 132, 227, 0.30);
    }
    .triage-urgent {
        background: linear-gradient(135deg, #e17055, #f39c12);
        border-left: 6px solid #fdcb6e;
        box-shadow: 0 6px 24px rgba(243, 156, 18, 0.35);
    }
    .triage-critical {
        background: linear-gradient(135deg, #d63031, #c0392b);
        border-left: 6px solid #ff7675;
        box-shadow: 0 6px 24px rgba(214, 48, 49, 0.40);
        animation: triage-pulse 2s ease-in-out infinite;
    }
    @keyframes triage-pulse {
        0%, 100% { box-shadow: 0 6px 24px rgba(214, 48, 49, 0.40); }
        50% { box-shadow: 0 6px 36px rgba(214, 48, 49, 0.60); }
    }
    .triage-banner .triage-level {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .triage-banner .triage-action {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    .triage-banner .triage-details {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.92;
        line-height: 1.7;
    }
    .triage-banner .triage-time {
        display: inline-block;
        margin-top: 10px;
        padding: 6px 16px;
        border-radius: 20px;
        background: rgba(255,255,255,0.18);
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .triage-disclaimer {
        font-size: 0.82rem;
        color: #e17055;
        font-weight: 600;
        padding: 10px 0 0 0;
        line-height: 1.5;
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


def compute_nihss_proxy(
    lesion_pct: float,
    brain_region: str,
) -> dict:
    """
    Estimate an NIHSS-like severity proxy from lesion percentage and brain
    region.  Returns a dict with:
      - nihss_band   : "Mild (0\u20135)", "Moderate (6\u201315)", or "Severe (16+)"
      - nihss_estimate : representative integer in that band
      - clinical_note  : one-sentence expected-deficit description
    """

    # ── Canonical region mapping (handles both naming conventions) ────
    _REGION_ALIASES = {
        "Frontal":               "Frontal",
        "Frontal lobe":          "Frontal",
        "Parietal L":            "Parietal L",
        "Left parietal lobe":    "Parietal L",
        "Parietal R":            "Parietal R",
        "Right parietal lobe":   "Parietal R",
        "Temporal L":            "Temporal L",
        "Left temporal lobe":    "Temporal L",
        "Temporal R":            "Temporal R",
        "Right temporal lobe":   "Temporal R",
        "Central/Basal Ganglia": "Central/Basal Ganglia",
        "Central / basal ganglia": "Central/Basal Ganglia",
        "Thalamic":              "Thalamic",
        "Thalamic region":       "Thalamic",
        "Cerebellar/Brainstem":  "Cerebellar/Brainstem",
        "Cerebellar / brainstem": "Cerebellar/Brainstem",
    }

    canon = _REGION_ALIASES.get(brain_region, brain_region)

    # ── Base severity from lesion percentage ──────────────────────────
    MILD, MOD, SEV = 0, 1, 2
    if lesion_pct < 0.5:
        base = MILD
    elif lesion_pct <= 2.0:
        base = MOD
    else:
        base = SEV

    # ── Region modifiers ──────────────────────────────────────────────
    if canon in ("Cerebellar/Brainstem",):
        severity_level = SEV  # always severe
    elif canon in ("Thalamic", "Central/Basal Ganglia"):
        severity_level = min(base + 1, SEV)  # bump up one band
    else:
        severity_level = base  # Frontal, Parietal, Temporal — no modifier

    # ── Map to NIHSS band + representative score ──────────────────────
    BAND_MAP = {
        MILD: ("Mild (0\u20135)",   3),
        MOD:  ("Moderate (6\u201315)", 10),
        SEV:  ("Severe (16+)",  20),
    }
    nihss_band, nihss_estimate = BAND_MAP[severity_level]

    # ── Clinical note per region + severity ───────────────────────────
    _NOTES = {
        ("Frontal", MILD):               "Small frontal hemorrhage; minor executive-function or motor-planning deficits possible.",
        ("Frontal", MOD):                "Moderate frontal bleed; expect contralateral weakness with possible expressive aphasia if dominant hemisphere.",
        ("Frontal", SEV):                "Large frontal hemorrhage; significant motor deficit, personality changes, and raised intracranial pressure likely.",
        ("Parietal L", MILD):            "Small left parietal hemorrhage; mild right-sided sensory loss or difficulty with language processing.",
        ("Parietal L", MOD):             "Moderate left parietal bleed; right hemibody sensory loss and receptive language impairment expected.",
        ("Parietal L", SEV):             "Large left parietal hemorrhage; dense right hemiparesis, aphasia, and spatial disorientation likely.",
        ("Parietal R", MILD):            "Small right parietal hemorrhage; mild left-sided neglect or sensory change possible.",
        ("Parietal R", MOD):             "Moderate right parietal bleed; left hemispatial neglect and sensory loss expected.",
        ("Parietal R", SEV):             "Large right parietal hemorrhage; dense left neglect, sensory loss, and visuospatial deficits likely.",
        ("Temporal L", MILD):            "Small left temporal hemorrhage; mild speech comprehension difficulty (Wernicke-type) possible.",
        ("Temporal L", MOD):             "Moderate left temporal bleed; significant receptive aphasia and verbal memory impairment expected.",
        ("Temporal L", SEV):             "Large left temporal hemorrhage; severe Wernicke aphasia and risk of uncal herniation.",
        ("Temporal R", MILD):            "Small right temporal hemorrhage; subtle spatial-memory or prosody perception deficits possible.",
        ("Temporal R", MOD):             "Moderate right temporal bleed; impaired spatial memory and emotional prosody recognition expected.",
        ("Temporal R", SEV):             "Large right temporal hemorrhage; major non-verbal cognitive deficits with herniation risk.",
        ("Central/Basal Ganglia", MILD): "Small basal-ganglia hemorrhage (bumped to Moderate); contralateral motor deficit and dysarthria possible.",
        ("Central/Basal Ganglia", MOD):  "Moderate basal-ganglia bleed; hemiparesis, sensory loss, and potential speech impairment expected.",
        ("Central/Basal Ganglia", SEV):  "Large basal-ganglia hemorrhage; dense hemiplegia, decreased consciousness, and midline shift risk.",
        ("Thalamic", MILD):              "Small thalamic hemorrhage (bumped to Moderate); contralateral sensory loss and possible drowsiness.",
        ("Thalamic", MOD):               "Moderate thalamic bleed; significant sensory deficit, gaze palsy, and altered consciousness expected.",
        ("Thalamic", SEV):               "Large thalamic hemorrhage; severe sensory loss, coma risk, and possible intraventricular extension.",
        ("Cerebellar/Brainstem", MILD):  "Cerebellar/brainstem hemorrhage (always Severe); ataxia, cranial nerve palsies, and respiratory compromise possible.",
        ("Cerebellar/Brainstem", MOD):   "Cerebellar/brainstem hemorrhage (always Severe); significant balance loss, vertigo, and bulbar dysfunction expected.",
        ("Cerebellar/Brainstem", SEV):   "Cerebellar/brainstem hemorrhage (always Severe); high risk of obstructive hydrocephalus and cardiorespiratory arrest.",
    }

    clinical_note = _NOTES.get(
        (canon, severity_level),
        f"{brain_region} hemorrhage detected; clinical correlation required.",
    )

    return {
        "nihss_band":     nihss_band,
        "nihss_estimate":  nihss_estimate,
        "clinical_note":   clinical_note,
    }


def get_treatment_urgency(
    nihss_band: str,
    brain_region: str,
    lesion_pct: float,
) -> dict:
    """
    Determine clinical triage urgency from NIHSS band, brain region, and
    lesion percentage.  Designed for **hemorrhagic** stroke — thrombolysis
    (tPA) is always flagged as contraindicated.

    Returns
    -------
    dict with keys:
        urgency_level    : "WATCH", "URGENT", or "CRITICAL"
        primary_action   : short recommended action string
        thrombolysis_note: note about tPA applicability
        surgical_note    : note about surgical referral
        time_message     : window-of-action reminder
    """
    # Canonical region
    _CRIT_REGIONS = {
        "Cerebellar/Brainstem", "Cerebellar / brainstem",
        "Thalamic", "Thalamic region",
    }
    _HIGH_REGIONS = {
        "Central/Basal Ganglia", "Central / basal ganglia",
    }

    is_critical_region = brain_region in _CRIT_REGIONS
    is_high_region = brain_region in _HIGH_REGIONS

    # Constant: hemorrhagic stroke
    thrombolysis_note = (
        "⛔ tPA / thrombolysis is CONTRAINDICATED. "
        "This system detects hemorrhagic stroke; thrombolytic therapy "
        "would worsen bleeding."
    )

    # ── Decision logic ────────────────────────────────────────────────
    if (
        nihss_band == "Severe (16+)"
        or is_critical_region
        or lesion_pct > 5.0
    ):
        urgency_level = "CRITICAL"
        primary_action = "Activate Stroke Code — neurosurgical consult STAT"
        surgical_note = (
            "Immediate neurosurgical evaluation required. "
            "Consider craniotomy / decompressive hemicraniectomy "
            "or external ventricular drain (EVD) based on imaging."
        )
        time_message = "⏰ Immediate intervention — every minute counts"

    elif nihss_band == "Moderate (6\u201315)" or is_high_region:
        urgency_level = "URGENT"
        primary_action = "Immediate neurology consult — admit to Stroke Unit"
        surgical_note = (
            "Surgical referral if hemorrhage expands or midline shift "
            "exceeds 5 mm on repeat imaging."
        )
        time_message = "⏰ Act within 1 hour — repeat CT in 6 hrs to monitor expansion"

    else:  # Mild + non-critical region
        urgency_level = "WATCH"
        primary_action = "Neurology consult within 6 hours — close monitoring"
        surgical_note = (
            "Surgical intervention unlikely at this stage. "
            "Monitor for clinical deterioration and repeat imaging in 12–24 hrs."
        )
        time_message = "⏰ Monitor closely — repeat CT if symptoms change"

    return {
        "urgency_level":    urgency_level,
        "primary_action":   primary_action,
        "thrombolysis_note": thrombolysis_note,
        "surgical_note":    surgical_note,
        "time_message":     time_message,
    }


def compute_asymmetry(
    original_img: np.ndarray,
    mask: np.ndarray,
) -> tuple:
    """
    Hemispheric asymmetry analysis.

    Flips the CT image horizontally, computes the absolute pixel
    difference, and returns a hot-coloured heatmap plus a scalar score.

    Parameters
    ----------
    original_img : np.ndarray   (H, W, 3) uint8
    mask         : np.ndarray   (H, W)    binary, not currently used
                   but accepted for future mask-weighted scoring.

    Returns
    -------
    asymmetry_heatmap : np.ndarray (H, W, 3) uint8  – hot-coloured map
    asymmetry_score   : float – mean intensity of top-5% diff pixels
                        (range roughly 0–255; higher = more asymmetric)
    """
    # 1. Flip horizontally
    flipped = cv2.flip(original_img, 1)

    # 2. Absolute difference
    diff = cv2.absdiff(original_img, flipped)

    # 3. Convert to single-channel grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # 4. Normalise to 0–255
    dmin, dmax = float(diff_gray.min()), float(diff_gray.max())
    if dmax - dmin > 0:
        diff_norm = ((diff_gray - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        diff_norm = np.zeros_like(diff_gray, dtype=np.uint8)

    # 5. Apply hot colormap
    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 6. Score: mean of top 5% brightest diff pixels
    flat = diff_norm.flatten().astype(np.float64)
    k = max(1, int(len(flat) * 0.05))
    top_k = np.partition(flat, -k)[-k:]
    asymmetry_score = float(np.mean(top_k))

    return heatmap_rgb, asymmetry_score


def run_classification(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> Tuple[str, float, np.ndarray]:
    """
    Returns (predicted_class, confidence, proba_array).
    """
    model.eval()  # ensure eval mode (safety against prior crash)
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


def mc_dropout_predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    n_passes: int = 15,
) -> dict:
    """
    Monte Carlo Dropout prediction.

    Enables dropout at inference time and runs *n_passes* stochastic
    forward passes to estimate epistemic uncertainty.

    Parameters
    ----------
    model  : classifier with Dropout layers
    tensor : (1, 3, H, W) preprocessed input
    n_passes : number of stochastic forward passes

    Returns
    -------
    dict with:
        mean_prob  (float) – mean P(Stroke) across passes
        std_prob   (float) – std  P(Stroke)
        ci_lower   (float) – lower 95% CI, clipped to [0, 1]
        ci_upper   (float) – upper 95% CI, clipped to [0, 1]
        all_probs  (list[float]) – individual pass probabilities
    """
    # Keep model in eval mode (BatchNorm needs it for batch_size=1)
    # but selectively enable only Dropout layers for stochastic passes
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    stroke_probs = []

    try:
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(tensor)
                proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
                stroke_probs.append(float(proba[1]))
    finally:
        # Always restore full eval mode, even on error
        model.eval()

    mean_prob = float(np.mean(stroke_probs))
    std_prob  = float(np.std(stroke_probs))
    ci_lower  = float(np.clip(mean_prob - 1.96 * std_prob, 0.0, 1.0))
    ci_upper  = float(np.clip(mean_prob + 1.96 * std_prob, 0.0, 1.0))

    return {
        "mean_prob": mean_prob,
        "std_prob":  std_prob,
        "ci_lower":  ci_lower,
        "ci_upper":  ci_upper,
        "all_probs": stroke_probs,
    }


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


def compute_midline_shift(
    mask: np.ndarray,
    px_per_mm: float = 2.0,
) -> dict:
    """
    Estimate midline shift from a binary hemorrhage mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary mask (values 0/1 or 0.0/1.0).
    px_per_mm : float
        Pixels-per-mm conversion factor.  Default 2.0 corresponds to the
        commonly used 0.5 mm/pixel assumption for standard head-CT.

    Returns
    -------
    dict with keys:
        shift_pixels   (int)   – absolute horizontal displacement of the
                                  lesion centroid from the image midline.
        shift_mm       (float) – same displacement converted to mm.
        shift_severity (str)   – categorical severity label.
    """
    h, w = mask.shape[:2]
    midline_x = w / 2.0

    # Centroid via image moments
    coords = np.where(mask > 0.5)
    if len(coords[0]) == 0:
        return {
            "shift_pixels": 0,
            "shift_mm": 0.0,
            "shift_severity": "N/A (no lesion)",
        }

    cx = float(np.mean(coords[1]))  # mean column index
    shift_pixels = int(round(abs(cx - midline_x)))
    shift_mm = round(shift_pixels / px_per_mm, 1)

    # Severity classification
    if shift_mm < 3:
        severity = "Minimal (<3 mm)"
    elif shift_mm <= 5:
        severity = "Mild (3\u20135 mm)"
    elif shift_mm <= 10:
        severity = "Moderate (5\u201310 mm)"
    else:
        severity = "Severe (>10 mm) \u2014 herniation risk"

    return {
        "shift_pixels": shift_pixels,
        "shift_mm": shift_mm,
        "shift_severity": severity,
    }


def draw_midline_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Draw a red dashed vertical midline and a yellow dot at the lesion
    centroid on a copy of *image*.
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    mid_x = w // 2

    # Red dashed centre line
    dash_len, gap_len = 8, 6
    y = 0
    while y < h:
        y_end = min(y + dash_len, h)
        cv2.line(vis, (mid_x, y), (mid_x, y_end), (255, 60, 60), 2)
        y += dash_len + gap_len

    # Centroid of mask
    coords = np.where(mask > 0.5)
    if len(coords[0]) > 0:
        cy = int(np.mean(coords[0]))
        cx = int(np.mean(coords[1]))
        # Yellow filled circle with dark outline
        cv2.circle(vis, (cx, cy), 7, (0, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 5, (255, 255, 0), -1)
        # Thin line from centroid to midline
        cv2.line(vis, (cx, cy), (mid_x, cy), (255, 255, 0), 1)

    return vis


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

def generate_pdf_report(
    original_img, overlay_img, mask, classification, confidence,
    lesion_pct=0.0, affected_pixels=0, severity='N/A', region='N/A', 
    threshold=0.3, nihss=None, midline=None, triage=None, mc_dropout=None
):
    """Generate a clinical PDF report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                   TableStyle, Image as RLImage, HRFlowable, 
                                   PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    import io
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
                                   spaceAfter=6,
                                   leading=22)
    sub_style = ParagraphStyle('sub', fontSize=10,
                                textColor=colors.HexColor('#666666'),
                                spaceAfter=4,
                                leading=14)
    
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
    meta_table = Table(meta_data, colWidths=[35*mm, 50*mm, 35*mm, 50*mm])
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
    
    # ── Diagnosis Summary Box ────────────────────────────────
    story.append(Spacer(1, 2*mm))
    diag_color = colors.HexColor('#e74c3c') if classification == 'Stroke' else colors.HexColor('#27ae60')
    
    # Diagnosis Header Table (Solid color block)
    diag_text = f"DIAGNOSIS: {classification.upper()}"
    if mc_dropout:
        m = mc_dropout['mean_prob'] * 100
        diag_text += f" ({m:.1f}% ± {mc_dropout['std_prob']*100:.1f}%)"
    else:
        diag_text += f" | {confidence:.1f}% Confidence"

    d_header_style = ParagraphStyle('dh', fontSize=16, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER)
    d_table = Table([[Paragraph(diag_text, d_header_style)]], colWidths=[170*mm])
    d_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), diag_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    story.append(d_table)
    
    # Subtext below diagnosis in a bordered box
    if mc_dropout:
        lo, hi = mc_dropout['ci_lower']*100, mc_dropout['ci_upper']*100
        conf_subtext = f"<b>AI Uncertainty Analytics:</b> 95% Confidence Interval (MC Dropout) is <b>{lo:.1f}% \u2013 {hi:.1f}%</b>"
    else:
        conf_subtext = "Diagnostic confidence based on single-pass model response architecture."

    story.append(Spacer(1, 2*mm))
    u_style = ParagraphStyle('confsub', fontSize=9, textColor=colors.HexColor('#2c3e50'), alignment=TA_CENTER)
    u_table = Table([[Paragraph(conf_subtext, u_style)]], colWidths=[150*mm])
    u_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fcfcfc')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    # Center the table on page by putting it in another table if needed, or just align
    story.append(u_table)
    story.append(Spacer(1, 6*mm))
    
    # ── Severity + Region + Metrics table ───────────────────
    if classification == 'Stroke':
        sev_color = {'Mild': '#27ae60', 
                     'Moderate': '#f39c12', 
                     'Severe': '#e74c3c'}.get(severity, '#333333')
        
        m_header_style = ParagraphStyle('mheader', fontSize=8, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER)
        m_row_style = ParagraphStyle('mrow', fontSize=11, textColor=colors.HexColor(sev_color), fontName='Helvetica-Bold', alignment=TA_CENTER)
        m_row_gen_style = ParagraphStyle('mrowgen', fontSize=10, alignment=TA_CENTER, leading=12)

        metrics_data = [
            [Paragraph('SEVERITY GRADE', m_header_style), 
             Paragraph('HEMORRHAGE LOCATION', m_header_style), 
             Paragraph('LESION AREA', m_header_style), 
             Paragraph('AFFECTED PIXELS', m_header_style)],
            [Paragraph(severity or 'N/A', m_row_style), 
             Paragraph(region or 'N/A', m_row_gen_style),
             Paragraph(f'{lesion_pct:.2f}%', m_row_gen_style), 
             Paragraph(f'{affected_pixels:,}', m_row_gen_style)],
        ]
        m_table = Table(metrics_data, colWidths=[35*mm, 60*mm, 35*mm, 40*mm])
        m_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('PADDING', (0,0), (-1,-1), 8),
            ('ROWHEIGHT', (0,1), (-1,1), 18*mm),
        ]))
        story.append(m_table)
        story.append(Spacer(1, 5*mm))

        # ── NIHSS Section ─────────────────────────────────────
        if nihss:
            story.append(Paragraph("Neurological Prognosis (Estimated NIHSS)", 
                                    ParagraphStyle('sec', fontSize=12,
                                                   fontName='Helvetica-Bold',
                                                   textColor=colors.HexColor('#2c3e50'),
                                                   spaceAfter=4)))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor('#cccccc'), spaceAfter=4))
            
            nihss_header_style = ParagraphStyle('nheader', fontSize=8, fontName='Helvetica-Bold', textColor=colors.white)
            n_table_style = ParagraphStyle('ntable', fontSize=9, leading=12)
            
            nihss_data = [
                [Paragraph('NIHSS CATEGORY', nihss_header_style), 
                 Paragraph('ESTIMATED SCORE', nihss_header_style), 
                 Paragraph('CLINICAL DEFICIT PROXY', nihss_header_style)],
                [Paragraph(nihss['nihss_band'], n_table_style), 
                 Paragraph(str(nihss['nihss_estimate']), n_table_style), 
                 Paragraph(nihss['clinical_note'], n_table_style)]
            ]
            n_table = Table(nihss_data, colWidths=[35*mm, 30*mm, 105*mm])
            
            n_sev_color = {
                "Mild (0\u20135)":   '#27ae60',
                "Moderate (6\u201315)": '#f39c12',
                "Severe (16+)":  '#e74c3c',
            }.get(nihss['nihss_band'], '#333333')

            n_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 8),
                ('FONTSIZE', (0,1), (-1,1), 9),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('TEXTCOLOR', (0,1), (0,1), colors.HexColor(n_sev_color)),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#34495e')),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(n_table)
            story.append(Spacer(1, 4*mm))

        # ── Midline & Triage Section ───────────────────────────
        if midline or triage:
            triage_bg = {
                'WATCH': '#3498db',
                'URGENT': '#f39c12',
                'CRITICAL': '#e74c3c'
            }.get(triage['urgency_level'] if triage else 'WATCH', '#34495e')

            story.append(Paragraph("Clinical Triage & Structural Analysis", 
                                    ParagraphStyle('sec', fontSize=12,
                                                   fontName='Helvetica-Bold',
                                                   textColor=colors.HexColor('#2c3e50'),
                                                   spaceAfter=4)))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor('#cccccc'), spaceAfter=4))

            t_header_style = ParagraphStyle('theader', fontSize=8, fontName='Helvetica-Bold', textColor=colors.white)
            t_data_style = ParagraphStyle('tdata', fontSize=9, leading=11)
            t_urgency_style = ParagraphStyle('turg', fontSize=11, fontName='Helvetica-Bold', 
                                              textColor=colors.white, alignment=TA_CENTER)

            # Table for Triage and Midline
            t_headers = [Paragraph('TRIAGE PRIORITY', t_header_style), 
                         Paragraph('MIDLINE ANALYSIS', t_header_style), 
                         Paragraph('PRIMARY CLINICAL ACTION', t_header_style)]
            
            t_row = [
                Paragraph(triage['urgency_level'] if triage else 'N/A', t_urgency_style),
                Paragraph(f"<b>{midline['shift_mm']:.1f} mm shift</b><br/>{midline['shift_severity']}" if midline else 'N/A', t_data_style),
                Paragraph(triage['primary_action'] if triage else 'N/A', t_data_style)
            ]

            t_table = Table([t_headers, t_row], colWidths=[35*mm, 45*mm, 90*mm])
            t_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
                ('BACKGROUND', (0,1), (0,1), colors.HexColor(triage_bg)),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
                ('PADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,1), (-1,1), 10),
                ('TOPPADDING', (0,1), (-1,1), 10),
            ]))
            story.append(t_table)
            
            if triage:
                story.append(Spacer(1, 3*mm))
                note_style = ParagraphStyle('note', fontSize=9, leading=12, leftIndent=4)
                story.append(Paragraph(f"<b>\u2022 Urgency Info:</b> {triage['time_message']}", note_style))
                story.append(Paragraph(f"<b>\u2022 Thrombolysis:</b> {triage['thrombolysis_note']}", note_style))
                story.append(Paragraph(f"<b>\u2022 Surgical Plan:</b> {triage['surgical_note']}", note_style))
            
            story.append(Spacer(1, 6*mm))
    
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
    
    # ── AI Findings narrative (on NEW PAGE) ──────────────────
    story.append(PageBreak())
    
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
    st.image(image_np, caption="Original CT Scan", width=350)

    # ══════════════════════════════════════════════════════════════════
    #  Classification
    # ══════════════════════════════════════════════════════════════════

    if classifier is None:
        st.warning("⚠️ Classifier model not loaded. Cannot perform classification.")
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("🔬 Classification Result")

    with st.spinner("Running classification (MC Dropout × 15)..."):
        cls_tensor = preprocess_image(image_np, task="classification")
        pred_class, confidence, proba = run_classification(classifier, cls_tensor)
        mc = mc_dropout_predict(classifier, cls_tensor, n_passes=15)

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

    # ── MC Dropout confidence display ─────────────────────────────────
    mean_prob = mc["mean_prob"]
    normal_prob = 1.0 - mean_prob
    
    mean_pct = mean_prob * 100
    std_pct  = mc["std_prob"]  * 100
    ci_lo    = mc["ci_lower"]  * 100
    ci_hi    = mc["ci_upper"]  * 100

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Normal:** `{normal_prob:.1%}`")
        st.progress(float(normal_prob))
    with col_b:
        st.markdown(
             f"**Stroke (AI Mean):** `{mean_pct:.1f}% ± {std_pct:.1f}%` "
        )
        st.progress(float(mean_prob))
    
    st.markdown(
        f'<div style="text-align:right; font-size:0.8rem; color:#a0aec0; margin-top:-10px;">'
        f'95% CI: {ci_lo:.1f}% – {ci_hi:.1f}%'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Visual CI range bar ───────────────────────────────────────────
    bar_color = "#e74c3c" if pred_class == "Stroke" else "#00b894"
    st.markdown(
        f'<div class="ci-bar-container">'
        f'<div class="ci-bar-range" style="'
        f'left:{ci_lo}%; width:{ci_hi - ci_lo}%; '
        f'background:{bar_color};"></div>'
        f'<div class="ci-bar-mean" style="'
        f'left:{mean_pct}%; background:{bar_color};"></div>'
        f'<div class="ci-bar-label" style="left:4px;">0%</div>'
        f'<div class="ci-bar-label" style="right:4px;">100%</div>'
        f'</div>'
        f'<div style="text-align:center;font-size:0.75rem;color:#a0aec0;">'
        f'95% confidence interval from {len(mc["all_probs"])} stochastic passes'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Uncertainty badge ─────────────────────────────────────────────
    if mc["std_prob"] > 0.15:
        st.markdown(
            '<span class="uncertainty-badge uncertainty-high">'
            '⚠️ High uncertainty</span>',
            unsafe_allow_html=True,
        )
        st.warning(
            f"Prediction uncertainty is high (σ = {std_pct:.1f}%). "
            f"The model is not confident — radiologist review is strongly recommended."
        )
    elif mc["std_prob"] < 0.05:
        st.markdown(
            '<span class="uncertainty-badge uncertainty-low">'
            '✅ High confidence</span>',
            unsafe_allow_html=True,
        )
        st.success(
            f"High-confidence prediction (σ = {std_pct:.1f}%). "
            f"Dropout-enabled passes show consistent results."
        )
    else:
        st.markdown(
            '<span class="uncertainty-badge uncertainty-mid">'
            'ℹ️ Moderate confidence</span>',
            unsafe_allow_html=True,
        )

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

                # Midline shift visualisation
                midline_vis = draw_midline_overlay(overlay_img, mask_full)

                # Hemispheric asymmetry
                asym_heatmap, asym_score = compute_asymmetry(image_np, mask_full)

            # ── Six-column display ──────────────────────────────────
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.image(image_np, caption="Original CT", width="stretch")
            with col2:
                # Show mask as a coloured heatmap
                mask_display = (mask_full * 255).astype(np.uint8)
                mask_coloured = cv2.applyColorMap(mask_display, cv2.COLORMAP_HOT)
                mask_coloured = cv2.cvtColor(mask_coloured, cv2.COLOR_BGR2RGB)
                st.image(mask_coloured, caption="Hemorrhage Mask", width="stretch")
            with col3:
                if show_overlay:
                    st.image(overlay_img, caption="Overlay", width="stretch")
                else:
                    st.info("Overlay hidden.")
            with col4:
                predicted_idx = 1 if pred_class == "Stroke" else 0
                cam_img = generate_gradcam(classifier, cls_tensor, image_np, predicted_idx)
                if cam_img is not None:
                    st.image(cam_img, caption="AI Attention Map", width="stretch")
                else:
                    st.info("Attention Map unavailable.")
            with col5:
                st.image(midline_vis, caption="Midline Shift", width="stretch")
            with col6:
                st.image(asym_heatmap, caption="Hemispheric Asymmetry", width="stretch")
                st.caption(f"Score: **{asym_score:.1f}** / 255")

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

            # ── NIHSS Proxy Card ──────────────────────────────────────
            nihss = compute_nihss_proxy(lesion_pct, region)
            _sev_css = {
                "Mild (0\u20135)":   "nihss-mild",
                "Moderate (6\u201315)": "nihss-moderate",
                "Severe (16+)":  "nihss-severe",
            }
            css_class = _sev_css.get(nihss["nihss_band"], "nihss-moderate")

            st.markdown("### 📊 Estimated NIHSS Proxy")
            st.markdown(
                f'<div class="nihss-card {css_class}">'
                f'<div class="nihss-label">Estimated NIHSS Band</div>'
                f'<div class="nihss-band">{nihss["nihss_band"]}</div>'
                f'<div class="nihss-score">{nihss["nihss_estimate"]}</div>'
                f'<div class="nihss-note">{nihss["clinical_note"]}</div>'
                f'</div>'
                f'<div class="nihss-disclaimer">'
                f'⚕️ NIHSS estimate is AI-derived and must be confirmed by a neurologist.'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Midline Shift Card ────────────────────────────────────
            midline = compute_midline_shift(mask_full)
            _ml_css = {
                "Minimal (<3 mm)":  "midline-minimal",
                "Mild (3\u20135 mm)":    "midline-mild",
                "Moderate (5\u201310 mm)": "midline-moderate",
                "Severe (>10 mm) \u2014 herniation risk": "midline-severe",
            }
            ml_css_class = _ml_css.get(midline["shift_severity"], "midline-minimal")

            st.markdown("### ↔️ Estimated Midline Shift")
            st.markdown(
                f'<div class="midline-card {ml_css_class}">'
                f'<div class="midline-label">Midline Displacement</div>'
                f'<div class="midline-value">{midline["shift_mm"]:.1f} mm</div>'
                f'<div class="midline-sev">{midline["shift_severity"]}</div>'
                f'<div class="midline-note">'
                f'Centroid offset: {midline["shift_pixels"]} px from image centre'
                f'</div>'
                f'</div>'
                f'<div class="nihss-disclaimer">'
                f'⚕️ Estimated midline shift is a heuristic and requires '
                f'radiological confirmation.'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Treatment Urgency / Triage Banner ─────────────────────
            triage = get_treatment_urgency(
                nihss["nihss_band"], region, lesion_pct
            )
            _triage_css = {
                "WATCH":    "triage-watch",
                "URGENT":   "triage-urgent",
                "CRITICAL": "triage-critical",
            }
            _triage_emoji = {
                "WATCH":    "⏱️",
                "URGENT":   "🚨",
                "CRITICAL": "🔴",
            }
            t_css = _triage_css.get(triage["urgency_level"], "triage-watch")
            t_emoji = _triage_emoji.get(triage["urgency_level"], "")

            st.markdown("### 🏥 Clinical Triage Recommendation")
            st.markdown(
                f'<div class="triage-banner {t_css}">'
                f'<div class="triage-level">{t_emoji} {triage["urgency_level"]}</div>'
                f'<div class="triage-action">{triage["primary_action"]}</div>'
                f'<div class="triage-details">'
                f'💉 <b>Thrombolysis:</b> {triage["thrombolysis_note"]}<br>'
                f'🩺 <b>Surgical:</b> {triage["surgical_note"]}'
                f'</div>'
                f'<div class="triage-time">{triage["time_message"]}</div>'
                f'</div>'
                f'<div class="triage-disclaimer">'
                f'⚠️ <b>This is an AI-generated triage aid. '
                f'All clinical decisions must be made by a licensed physician.</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Metrics cards ─────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
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
            with m4:
                st.markdown(
                    f'<div class="metric-card" style="border-left-color: #e17055;">'
                    f"<b>Midline Shift</b><br>"
                    f'<span style="font-size:1.6rem;font-weight:700;">{midline["shift_mm"]:.1f} mm</span>'
                    f"<br>{midline['shift_severity']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m5:
                asym_label = "Low" if asym_score < 80 else ("Moderate" if asym_score < 160 else "High")
                asym_color = "#00b894" if asym_score < 80 else ("#f39c12" if asym_score < 160 else "#e74c3c")
                st.markdown(
                    f'<div class="metric-card" style="border-left-color: {asym_color};">'
                    f"<b>Hemispheric Asymmetry</b><br>"
                    f'<span style="font-size:1.6rem;font-weight:700;">{asym_score:.1f}</span>'
                    f"<br>{asym_label} — higher = more abnormal"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.caption("🔍 High asymmetry may indicate mass effect or midline displacement.")

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

    # Precompute all metrics for the report
    severity = "N/A"
    if lesion_pct is not None:
        if lesion_pct < 2.0:
            severity = "Mild"
        elif lesion_pct < 5.0:
            severity = "Moderate"
        else:
            severity = "Severe"

    nihss_for_report = None
    midline_for_report = None
    triage_for_report = None
    mc_for_report = None

    if pred_class == "Stroke" and lesion_pct is not None:
        actual_region = region if 'region' in locals() else "N/A"
        nihss_for_report = compute_nihss_proxy(lesion_pct, actual_region)
        midline_for_report = compute_midline_shift(mask_full)
        triage_for_report = get_treatment_urgency(nihss_for_report['nihss_band'], actual_region, lesion_pct)
        if 'mc' in locals():
            mc_for_report = mc

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
        threshold=threshold,
        nihss=nihss_for_report,
        midline=midline_for_report,
        triage=triage_for_report,
        mc_dropout=mc_for_report
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
