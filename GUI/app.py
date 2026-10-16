"""Streamlit Interactive Ensemble Dashboard for CXR Lung Disease Classification.

Provides real-time multi-backbone inference, dynamic user-controlled ensemble
weighting, and Explainable AI (XAI) Occlusion Sensitivity Heatmaps.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure repository root is in Python path for src module imports
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainability import apply_heatmap_overlay, compute_occlusion_map_fast
from src.models import (
    FCNNClassifier,
    MultiBackboneFeatureExtractor,
    load_ensemble_models,
)
from src.preprocessing import load_class_names, preprocess_image

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="🩻 Lung Disease CXR Classifier",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished medical research interface
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-card {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Model & Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    """Detects and returns the optimal available PyTorch hardware acceleration device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner=True)
def load_resources(
    weights_dir: str,
) -> Tuple[Dict[str, Tuple[MultiBackboneFeatureExtractor, FCNNClassifier]], List[str], List[str]]:
    """Loads all ensemble model checkpoints and class name metadata once per session."""
    dir_path = Path(weights_dir).resolve()
    class_names_path = dir_path / "class_names.json"
    class_names = load_class_names(class_names_path)
    device = get_device()
    models_dict, warnings = load_ensemble_models(
        weights_dir=dir_path,
        device=device,
        backbone_names=["vgg16", "resnet50", "densenet121"],
        num_classes=len(class_names),
    )
    return models_dict, class_names, warnings


@torch.no_grad()
def predict_probabilities(
    image_tensor: torch.Tensor,
    extractor: MultiBackboneFeatureExtractor,
    classifier: FCNNClassifier,
    device: torch.device,
) -> np.ndarray:
    """Calculates softmax probabilities for a single backbone."""
    extractor.eval()
    classifier.eval()
    image_tensor = image_tensor.to(device)
    features = extractor(image_tensor)
    logits = classifier(features)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy()


# ==============================================================================
# Application Header & Disclaimer
# ==============================================================================
st.markdown('<div class="main-header">🩻 Lung Disease X-Ray Classifier & Ensemble Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Backbone Deep Learning (VGG-16 • ResNet-50 • DenseNet-121) with Explainable AI (XAI)</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-card">
    ⚠️ <strong>Research & Educational Use Only:</strong> This application is a research prototype developed for academic 
    investigation. It is not an FDA/CE-cleared medical diagnostic device and must not replace professional clinical evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load cached models
with st.spinner("Loading deep learning model checkpoints..."):
    models_dict, class_names, load_warnings = load_resources(str(CURRENT_DIR))

# Surface any non-fatal model warnings (e.g. missing optional weights)
for warn in load_warnings:
    st.toast(warn, icon="ℹ️")

# ==============================================================================
# Sidebar: Controls & Settings
# ==============================================================================
st.sidebar.header("⚖️ Ensemble Weights")
st.sidebar.caption("Tune the influence of each deep learning backbone in the final consensus decision:")

DEFAULT_WEIGHTS = {"vgg16": 0.33, "resnet50": 0.33, "densenet121": 0.34}

if "weights" not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()

if st.sidebar.button("🔄 Reset to Default Weights", use_container_width=True):
    st.session_state.weights = DEFAULT_WEIGHTS.copy()

w_vgg = st.sidebar.slider("VGG-16 Weight", 0.0, 1.0, float(st.session_state.weights["vgg16"]), 0.05)
w_res = st.sidebar.slider("ResNet-50 Weight", 0.0, 1.0, float(st.session_state.weights["resnet50"]), 0.05)
w_dense = st.sidebar.slider("DenseNet-121 Weight", 0.0, 1.0, float(st.session_state.weights["densenet121"]), 0.05)

total_w = w_vgg + w_res + w_dense
if total_w > 0:
    weights = {
        "vgg16": w_vgg / total_w,
        "resnet50": w_res / total_w,
        "densenet121": w_dense / total_w,
    }
else:
    weights = {"vgg16": 1 / 3, "resnet50": 1 / 3, "densenet121": 1 / 3}

st.session_state.weights = weights

# Weight distribution visualization
fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.5))
ax_pie.pie(
    list(weights.values()),
    labels=[k.upper() for k in weights.keys()],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#3B82F6", "#10B981", "#F59E0B"],
    wedgeprops=dict(width=0.6, edgecolor="w"),
)
ax_pie.set_title("Voting Power Distribution", fontsize=10, fontweight="bold")
st.sidebar.pyplot(fig_pie, use_container_width=True)
plt.close(fig_pie)

st.sidebar.divider()
st.sidebar.header("⚙️ Preprocessing Options")
enhance_image = st.sidebar.checkbox(
    "Enable CLAHE & Sharpening",
    value=True,
    help="Applies Contrast-Limited Adaptive Histogram Equalization and a Laplacian sharpening kernel.",
)
noise_stress_test = st.sidebar.checkbox(
    "Noise Robustness Test",
    value=False,
    help="Injects synthetic Gaussian noise to test model stability against degraded radiograph quality.",
)

# Hardware status badge in sidebar
device = get_device()
device_label = "NVIDIA CUDA GPU" if device.type == "cuda" else ("Apple Silicon (MPS)" if device.type == "mps" else "Host CPU")
st.sidebar.info(f"⚡ Acceleration: **{device_label}**")

# ==============================================================================
# Disease Taxonomy Guide (Expander)
# ==============================================================================
with st.expander("📚 View 9-Class Pathological Taxonomy & Clinical Definitions", expanded=False):
    st.markdown(
        """
        1. **00 Normal Anatomy**: Healthy lung fields without acute consolidations, infiltrates, or effusions.
        2. **01 Pulmonary Inflammatory Processes**: Active parenchymal infections including Lobar & Bronchopneumonia.
        3. **02 Increased Density**: Opacification due to Pleural Effusion, Atelectatic Consolidation, Hydrothorax, Empyema.
        4. **03 Decreased Density**: Hyperlucency conditions such as Pneumothorax, Pneumomediastinum, or Emphysema.
        5. **04 Obstructive Pulmonary Diseases**: Chronic obstructive conditions including Bronchiectasis and Pulmonary Embolism.
        6. **05 Degenerative Infectious Diseases**: Chronic granulomatous infections including Tuberculosis (TB) and Pulmonary Fibrosis.
        7. **06 Encapsulated Lesions**: Well-circumscribed lesions including Lung Abscesses, Solitary Pulmonary Nodules, Cysts, and Malignant Masses.
        8. **07 Mediastinal Changes**: Structural mediastinal abnormalities, Pericarditis, and Lymphadenopathy.
        9. **08 Thoracic Changes**: Chest wall deformities, congenital hypoplasia, agenesis, and volume loss.
        """
    )

# ==============================================================================
# Main Section: Upload & Inference
# ==============================================================================
uploaded_file = st.file_uploader(
    "Select or drop a Chest X-ray (CXR) image (JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    col_img, col_pred = st.columns([1, 1.2])

    with col_img:
        st.subheader("🖼️ Input Radiograph")
        st.image(uploaded_file, caption="Uploaded Radiograph", use_container_width=True)

    # Process image
    image_tensor = preprocess_image(
        uploaded_file,
        enhance=enhance_image,
        add_noise=noise_stress_test,
    )

    # Compute predictions for all models
    model_outputs: Dict[str, np.ndarray] = {}
    weighted_probs_list = []

    for name, (extractor, classifier) in models_dict.items():
        probs = predict_probabilities(image_tensor, extractor, classifier, device)
        model_outputs[name] = probs
        weighted_probs_list.append(probs * weights[name])

    # Ensemble combination
    ensemble_probs = np.sum(np.stack(weighted_probs_list), axis=0)
    top_class_idx = int(np.argmax(ensemble_probs))
    pred_class = class_names[top_class_idx]
    confidence = float(ensemble_probs[top_class_idx])

    with col_pred:
        st.subheader("🎯 Diagnostic Consensus")
        if confidence >= 0.70:
            st.success(f"### Predicted: **{pred_class}**\n**Consensus Confidence: {confidence:.2%}**")
        elif confidence >= 0.45:
            st.warning(f"### Predicted: **{pred_class}**\n**Moderate Confidence: {confidence:.2%}** (Radiologist review recommended)")
        else:
            st.error(f"### Inconclusive: **{pred_class}**\n**Low Confidence: {confidence:.2%}** (Secondary diagnostic imaging required)")

        st.caption(
            f"Active Weights: VGG-16: `{weights['vgg16']:.2f}` | "
            f"ResNet-50: `{weights['resnet50']:.2f}` | "
            f"DenseNet-121: `{weights['densenet121']:.2f}`"
        )

        # Quick Top 3 Ensemble Preview
        st.markdown("**Top 3 Differential Hypotheses:**")
        top3_indices = np.argsort(ensemble_probs)[::-1][:3]
        for rank, idx in enumerate(top3_indices, 1):
            st.write(f"{rank}. **{class_names[idx]}** — `{ensemble_probs[idx]:.2%}`")

    st.divider()

    # ==============================================================================
    # Individual Model Predictions
    # ==============================================================================
    st.subheader("🤖 Individual Backbone Analysis")
    st.caption("Inspect the independent assessment of each model architecture before weighted fusion:")

    cols = st.columns(3)
    for i, (name, probs) in enumerate(model_outputs.items()):
        m_idx = int(np.argmax(probs))
        m_pred = class_names[m_idx]
        m_conf = float(probs[m_idx])

        with cols[i]:
            st.markdown(f"#### {name.upper()}")
            st.metric(label="Predicted Class", value=m_pred.split(" (")[0])
            st.metric(label="Model Confidence", value=f"{m_conf:.2%}")

    st.divider()

    # ==============================================================================
    # Visual Explanation: Explainable AI (XAI)
    # ==============================================================================
    st.subheader("🔍 Explainable AI: Occlusion Sensitivity Attribution")
    st.markdown(
        """
        Occlusion sensitivity reveals the spatial anatomical regions that most heavily dictated each model's prediction.
        A sliding occluder patch ($30\\times30$ px) measures the drop in target class probability when specific regions are obscured.
        """
    )

    if st.button("🚀 Generate Occlusion Explanation Heatmaps", use_container_width=True):
        with st.spinner("Computing fast occlusion sensitivity maps across all backbones..."):
            heatmaps = {}
            for name, (extractor, classifier) in models_dict.items():
                heatmaps[name] = compute_occlusion_map_fast(
                    image_tensor=image_tensor,
                    extractor=extractor,
                    classifier=classifier,
                    target_class_idx=top_class_idx,
                    device=device,
                    patch_size=30,
                    stride=30,
                    downsample=112,
                )

            # Load original image for overlay
            orig_pil = Image.open(uploaded_file).convert("RGB").resize((224, 224))
            orig_np = np.array(orig_pil)

            xai_cols = st.columns(3)
            for i, (name, heatmap) in enumerate(heatmaps.items()):
                overlay = apply_heatmap_overlay(orig_np, heatmap, alpha=0.6)
                with xai_cols[i]:
                    st.image(
                        overlay,
                        caption=f"{name.upper()} Sensitivity Map",
                        use_container_width=True,
                    )

            st.markdown(
                """
                **Interpretation Legend:**
                - 🔴 **Red / Yellow**: **Critical anatomical evidence**. Occlusion caused a significant drop in diagnostic confidence.
                - 🟢 **Green**: Moderately influential lung parenchyma or border features.
                - 🔵 **Blue / Dark**: Minimal influence; occluding these regions did not alter model prediction.
                """
            )

    st.divider()

    # ==============================================================================
    # Full Probability Distribution Breakdown
    # ==============================================================================
    st.subheader("📊 Class Probability Distribution")
    selected_view = st.radio(
        "Select Distribution to Visualize:",
        ["Ensemble Consensus", "VGG-16", "ResNet-50", "DenseNet-121"],
        horizontal=True,
    )

    if selected_view == "Ensemble Consensus":
        display_probs = ensemble_probs
    else:
        display_probs = model_outputs[selected_view.lower().replace("-", "")]

    prob_df = pd.DataFrame({
        "Pathology Class": class_names,
        "Probability": display_probs,
    }).sort_values("Probability", ascending=True)

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5))
    bars = ax_bar.barh(prob_df["Pathology Class"], prob_df["Probability"], color="#2563EB")
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Predicted Probability", fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.6)

    # Highlight top prediction bar
    bars[-1].set_color("#DC2626")

    # Add text labels on bars
    for bar in bars:
        width = bar.get_width()
        if width > 0.03:
            ax_bar.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1%}",
                va="center",
                fontsize=9,
            )

    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)
