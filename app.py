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
            segmentor = get_segmentation_model(model_type=model_type)
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

    pred_idx   = int(np.argmax(proba))
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    return pred_class, confidence, proba


def run_segmentation(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Returns binary mask as numpy array of shape (H, W) with values 0/1.
    """
    with torch.no_grad():
        logits = model(tensor)
        mask = (torch.sigmoid(logits) >= threshold).float()
        mask = mask.cpu().numpy()[0, 0]  # (H, W)
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


# ══════════════════════════════════════════════════════════════════════
#  Report generation (downloadable PNG)
# ══════════════════════════════════════════════════════════════════════

def generate_report_image(
    original_img: np.ndarray,
    pred_class: str,
    confidence: float,
    overlay_img: Optional[np.ndarray] = None,
    lesion_pct: Optional[float] = None,
) -> bytes:
    """Create a summary report image and return as PNG bytes."""
    n_cols = 3 if overlay_img is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(original_img)
    axes[0].set_title("CT Scan", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    if overlay_img is not None:
        axes[1].imshow(
            (cv2.resize((create_overlay(original_img, np.zeros_like(original_img[:,:,0]), (0,0,0), 0.0) * 0 + 1).astype(np.uint8) * 0, (original_img.shape[1], original_img.shape[0])) if False else
             cv2.resize(
                 np.stack([
                     cv2.resize((overlay_img[:,:,0] if overlay_img.ndim == 3
                                 else overlay_img).astype(np.float32),
                                (original_img.shape[1], original_img.shape[0]))
                 ] * 3, axis=-1) if overlay_img.ndim == 2 else overlay_img,
                 (original_img.shape[1], original_img.shape[0])
             )),
            cmap="Reds" if overlay_img.ndim == 2 else None,
        )
        # Simpler approach — just show the mask
        h, w = original_img.shape[:2]
        mask_display = cv2.resize(overlay_img.astype(np.float32) if overlay_img.ndim == 2 else overlay_img[:,:,0].astype(np.float32), (w, h))
        axes[1].clear()
        axes[1].imshow(mask_display, cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title("Hemorrhage Mask", fontsize=13, fontweight="bold")
        axes[1].axis("off")

        # Overlay on original
        overlay_display = create_overlay(original_img, cv2.resize(overlay_img.astype(np.float32) if overlay_img.ndim == 2 else overlay_img[:,:,0].astype(np.float32), (w, h)), (255, 0, 0), 0.4)
        axes[2].imshow(overlay_display)
        axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
        axes[2].axis("off")

    colour = "#00b894" if pred_class == "Normal" else "#d63031"
    fig.suptitle(
        f"Diagnosis: {pred_class}  (Confidence: {confidence:.1%})"
        + (f"  |  Lesion Area: {lesion_pct:.2f}%" if lesion_pct else ""),
        fontsize=14, fontweight="bold", color=colour, y=1.02,
    )
    fig.text(
        0.5, -0.02,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  AI-Assisted — Not a substitute for clinical diagnosis",
        ha="center", fontsize=9, color="#636e72",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
            min_value=0.3, max_value=0.9, value=0.5, step=0.05,
            help="Minimum classification confidence to trigger segmentation.",
        )
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
                mask_np = run_segmentation(segmentor, seg_tensor, threshold=0.5)
                lesion_pct = compute_lesion_area_pct(mask_np)

                # Resize mask back to original image dimensions
                h, w = image_np.shape[:2]
                mask_full = cv2.resize(mask_np.astype(np.float32), (w, h))

                overlay_img = create_overlay(
                    image_np, mask_full, color=rgb_color, alpha=0.4,
                )

            # ── Three-column display ──────────────────────────────────
            col1, col2, col3 = st.columns(3)
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
                    st.info("Overlay hidden. Toggle in sidebar.")

            # ── Metrics cards ─────────────────────────────────────────
            m1, m2 = st.columns(2)
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
                pixel_count = int((mask_full > 0.5).sum())
                st.markdown(
                    f'<div class="metric-card">'
                    f"<b>Affected Pixels</b><br>"
                    f'<span style="font-size:1.6rem;font-weight:700;">{pixel_count:,}</span>'
                    f"<br>out of {h * w:,} total pixels"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    elif pred_class == "Stroke" and confidence < threshold:
        st.info(
            f"Stroke detected but confidence ({confidence:.1%}) is below "
            f"threshold ({threshold:.1%}). Segmentation skipped."
        )

    # ══════════════════════════════════════════════════════════════════
    #  Downloadable report
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("📄 Download Report")

    report_bytes = generate_report_image(
        original_img=image_np,
        pred_class=pred_class,
        confidence=confidence,
        overlay_img=mask_np if mask_np is not None else None,
        lesion_pct=lesion_pct,
    )
    st.download_button(
        label="⬇️  Download Analysis Report (PNG)",
        data=report_bytes,
        file_name=f"stroke_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
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
