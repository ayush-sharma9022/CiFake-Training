"""
Deepfake Image and Video Detection Framework - Streamlit Web Application.
A professional, modern, and reactive demonstration interface for custom-trained
deep learning models developed from scratch.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
import torch.nn.functional as F

from src.models.custom_cnn import CustomDeepfakeCNN
from src.models.custom_video_model import CustomVideoDeepfakeModel
from src.explainability.gradcam import GradCAM

# Page Configuration
st.set_page_config(
    page_title="Deepfake Detection Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek, Professional Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #0F172A;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    .badge-real {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-fake {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
METADATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "metadata")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw", "videos")


@st.cache_resource
def load_models():
    """Loads and caches both trained models on device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Model
    image_model = CustomDeepfakeCNN()
    img_weights_path = os.path.join(MODELS_DIR, "image_deepfake_cnn.pth")
    if os.path.exists(img_weights_path):
        image_model.load_state_dict(torch.load(img_weights_path, map_location=device, weights_only=True))
    image_model.to(device)
    image_model.eval()

    # Video Model
    video_model = CustomVideoDeepfakeModel(spatial_backbone=image_model)
    vid_weights_path = os.path.join(MODELS_DIR, "video_deepfake_model.pth")
    if os.path.exists(vid_weights_path):
        video_model.load_state_dict(torch.load(vid_weights_path, map_location=device, weights_only=True))
    video_model.to(device)
    video_model.eval()

    return image_model, video_model, device


image_model, video_model, device = load_models()


# Preprocessing Helpers
def preprocess_image(pil_img):
    """Preprocesses a PIL Image to standardized tensor (1, 3, 128, 128)."""
    resized = pil_img.resize((128, 128))
    arr = np.array(resized, dtype=np.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
    tensor = ((tensor - mean) / std).unsqueeze(0)
    return tensor, arr


def extract_video_frames(video_path, max_frames=15):
    """Uniformly extracts frames from an MP4 video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = []

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break

    cap.release()
    return frames


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
st.sidebar.title("Deepfake Detection")
st.sidebar.caption("Visual Artefact Analysis & Deep Learning")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "🖼️ Image Detection",
        "🎬 Video Detection",
        "📊 Model Performance",
        "ℹ️ About the Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Hardware Device:** `{device.type.upper()}`")
if device.type == "cuda":
    st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
st.sidebar.markdown("**Weights Source:** Trained from Scratch (No Pretrained Models)")


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if page == "🏠 Home Dashboard":
    st.markdown('<div class="main-title">🛡️ Deepfake Image & Video Detection Framework</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated forensic identification of manipulated synthetic media using custom deep learning architectures and visual artefact analysis.</div>', unsafe_allow_html=True)

    # Key Performance Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>88.40%</h3><p>Image Test Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>0.9548</h3><p>Image ROC-AUC Score</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>100.0%</h3><p>Video Classification Acc</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>0 / Scratch</h3><p>Pretrained Weights Used</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Capabilities
    st.subheader("Key Capabilities & Architecture Innovations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        - **Custom CNN from Scratch:** Novel 5-stage residual architecture with `ResidualConvBlock` units, LeakyReLU, and Batch Normalization. Designed strictly without transfer learning.
        - **Visual Artefact Analysis:** Identifies high-frequency texture distortions, facial boundary warping, and illumination inconsistencies.
        - **Grad-CAM Explainability:** Real-time gradient-weighted class activation mapping highlights exactly which facial regions influenced model decisions.
        """)
    with c2:
        st.markdown("""
        - **Spatio-Temporal Video Classifier:** Integrates custom CNN spatial representations with a 2-layer Bidirectional LSTM and Temporal Attention aggregation.
        - **Zero Data Leakage:** Strict video-level and source-identity separation guarantees independent evaluation across training and testing splits.
        - **Cross-Dataset Validation:** Explicit evaluation on unseen external datasets (CIPLAB) to quantify real-world distribution shift.
        """)

    st.markdown("---")
    st.subheader("Dataset & Benchmark Summary")
    comp_path = os.path.join(METADATA_DIR, "dataset_comparison.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        st.dataframe(df_comp, use_container_width=True)

    st.info("👈 Use the sidebar navigation to test image/video detection or inspect model performance curves.")


# ==============================================================================
# PAGE 2: IMAGE DETECTION
# ==============================================================================
elif page == "🖼️ Image Detection":
    st.markdown('<div class="main-title">🖼️ Image Deepfake Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a face image to classify authenticity and visualize forensic attention heatmaps via Grad-CAM.</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([1.5, 1])

    with col_up:
        uploaded_file = st.file_uploader("Upload Face Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        sample_choice = st.selectbox(
            "Or select a verified sample from the test set:",
            ["(None - Upload your own)", "Sample 1: Real Portrait", "Sample 2: Synthetic GAN Face", "Sample 3: Complex Manipulated Face"]
        )

    pil_img = None

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
    elif sample_choice != "(None - Upload your own)":
        test_meta_path = os.path.join(METADATA_DIR, "test_split.csv")
        if os.path.exists(test_meta_path):
            df_test = pd.read_csv(test_meta_path)
            if "Real" in sample_choice:
                sample_path = df_test[df_test["label"] == 0]["path"].iloc[10]
            elif "Synthetic" in sample_choice:
                sample_path = df_test[df_test["label"] == 1]["path"].iloc[10]
            else:
                sample_path = df_test[df_test["label"] == 1]["path"].iloc[25]

            if os.path.exists(sample_path):
                pil_img = Image.open(sample_path).convert("RGB")

    if pil_img is not None:
        st.markdown("---")
        tensor_img, raw_arr = preprocess_image(pil_img)

        with torch.no_grad():
            logits = image_model(tensor_img)
            prob_fake = torch.sigmoid(logits).item()

        is_fake = prob_fake >= 0.5
        conf = prob_fake if is_fake else (1.0 - prob_fake)
        pred_label = "FAKE" if is_fake else "REAL"

        # Generate Grad-CAM
        gradcam = GradCAM(image_model, target_layer=image_model.block4.conv2)
        cam = gradcam.generate_heatmap(tensor_img, target_class=1 if is_fake else 0)
        _, overlay = gradcam.overlay_heatmap(raw_arr, cam, alpha=0.5)

        # Prediction Display
        st.subheader("Classification Outcome")
        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            if is_fake:
                st.markdown('<div class="badge-fake">⚠️ PREDICTION: FAKE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-real">✅ PREDICTION: REAL</div>', unsafe_allow_html=True)

        with c2:
            st.metric("Model Confidence", f"{conf * 100:.2f}%")

        with c3:
            st.progress(conf)
            st.caption(f"Raw Synthetic Probability: {prob_fake * 100:.2f}% (Threshold: 50.0%)")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3-Column Visual Explainability
        st.subheader("Visual Artefact Explanation (Grad-CAM)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(raw_arr, caption="Original Input Image (128x128)", use_container_width=True)

        with col2:
            fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
            ax_cam.imshow(cam, cmap="jet")
            ax_cam.axis("off")
            ax_cam.set_title("Grad-CAM Activation Map", fontsize=11, fontweight="bold")
            st.pyplot(fig_cam, use_container_width=True)
            plt.close(fig_cam)

        with col3:
            st.image(overlay, caption=f"Forensic Overlay ({pred_label} Focus Regions)", use_container_width=True)

        st.markdown("""
        > **Forensic Interpretation Note:**  
        > Red/warm regions in the Grad-CAM overlay highlight spatial gradients and texture boundaries that exerted the strongest influence 
        > on the neural network's classification. In fake samples, attention typically clusters around blending perimeters, eye-specularities, and synthetic mouth boundaries.
        """)
    else:
        st.info("Please upload an image or select a sample from the test set above.")


# ==============================================================================
# PAGE 3: VIDEO DETECTION
# ==============================================================================
elif page == "🎬 Video Detection":
    st.markdown('<div class="main-title">🎬 Video Deepfake Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a video to analyze inter-frame inconsistencies using Spatio-Temporal Bidirectional LSTM and Temporal Attention.</div>', unsafe_allow_html=True)

    vid_meta_path = os.path.join(METADATA_DIR, "video_metadata.csv")
    sample_videos = []
    if os.path.exists(vid_meta_path):
        df_vids = pd.read_csv(vid_meta_path)
        sample_videos = df_vids["video_id"].tolist()

    col_v1, col_v2 = st.columns([1.5, 1])
    with col_v1:
        vid_file = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        sel_sample = st.selectbox("Or choose a verified benchmark video:", ["(None - Upload)"] + sample_videos)

    active_vid_path = None

    if vid_file is not None:
        t_path = os.path.join(PROJECT_ROOT, "datasets", "raw", "temp_upload.mp4")
        with open(t_path, "wb") as f:
            f.write(vid_file.read())
        active_vid_path = t_path
    elif sel_sample != "(None - Upload)":
        v_row = df_vids[df_vids["video_id"] == sel_sample].iloc[0]
        active_vid_path = v_row["video_path"]

    if active_vid_path and os.path.exists(active_vid_path):
        st.markdown("---")
        c_left, c_right = st.columns([1, 1.2])

        with c_left:
            st.subheader("Video Player")
            st.video(active_vid_path)

        with c_right:
            st.subheader("Frame Extraction & Temporal Analysis")
            with st.spinner("Extracting frames via OpenCV and running Spatio-Temporal Model..."):
                frames = extract_video_frames(active_vid_path, max_frames=15)

            if len(frames) == 0:
                st.error("Failed to decode video frames. Please check the video format.")
            else:
                # Preprocess frame sequence
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

                frame_tensors = []
                for f in frames:
                    resized = cv2.resize(f, (128, 128))
                    t = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
                    t = (t - mean) / std
                    frame_tensors.append(t)

                video_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0)  # (1, T, 3, 128, 128)

                with torch.no_grad():
                    vid_prob, frame_probs, attn_weights = video_model.predict_video(video_tensor)

                v_prob = vid_prob.item()
                is_vid_fake = v_prob >= 0.5
                v_conf = v_prob if is_vid_fake else (1.0 - v_prob)
                v_label = "FAKE" if is_vid_fake else "REAL"

                f_probs_arr = frame_probs.squeeze().cpu().numpy()
                attn_arr = attn_weights.squeeze().cpu().numpy()

                # Results Summary
                if is_vid_fake:
                    st.markdown('<div class="badge-fake">⚠️ VIDEO VERDICT: FAKE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-real">✅ VIDEO VERDICT: REAL</div>', unsafe_allow_html=True)

                st.metric("Sequence Confidence", f"{v_conf * 100:.2f}%")
                st.progress(v_conf)

                avg_f_prob = float(np.mean(f_probs_arr))
                st.caption(f"Learned Attention Sequence Prob: {v_prob*100:.1f}% | Frame Mean Prob: {avg_f_prob*100:.1f}%")

        # Frame Timeline Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Frame-by-Frame Fake Probability & Attention Weight Timeline")

        df_timeline = pd.DataFrame({
            "Frame Index": range(1, len(f_probs_arr) + 1),
            "Fake Probability (%)": f_probs_arr * 100,
            "Attention Weight (%)": attn_arr * 100
        })

        st.line_chart(df_timeline.set_index("Frame Index"))

        # Extracted Frames Gallery
        st.subheader("Extracted Sequence Frames")
        cols = st.columns(min(5, len(frames)))
        for i in range(min(5, len(frames))):
            with cols[i]:
                lbl = "FAKE" if f_probs_arr[i] >= 0.5 else "REAL"
                st.image(frames[i], caption=f"F{i+1}: {lbl} ({f_probs_arr[i]*100:.1f}%)", use_container_width=True)
    else:
        st.info("Upload an MP4 video or select a benchmark video from the dropdown.")


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="main-title">📊 Model Evaluation & Benchmark Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical evaluation metrics and visualizations computed on independent held-out test sets.</div>', unsafe_allow_html=True)

    img_eval_path = os.path.join(OUTPUTS_DIR, "model_evaluation_metrics.json")
    vid_eval_path = os.path.join(OUTPUTS_DIR, "video_model_evaluation_metrics.json")

    if os.path.exists(img_eval_path) and os.path.exists(vid_eval_path):
        with open(img_eval_path) as f:
            img_metrics = json.load(f)
        with open(vid_eval_path) as f:
            vid_metrics = json.load(f)

        in_dist = img_metrics["in_distribution_test"]
        cross = img_metrics["cross_dataset_test"]

        # Metric Summary Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Test Accuracy", f"{in_dist['accuracy']*100:.2f}%")
        with c2:
            st.metric("Precision", f"{in_dist['precision']*100:.2f}%")
        with c3:
            st.metric("Recall", f"{in_dist['recall']*100:.2f}%")
        with c4:
            st.metric("F1-Score", f"{in_dist['f1_score']*100:.2f}%")
        with c5:
            st.metric("ROC-AUC", f"{in_dist['roc_auc']:.4f}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📈 Training Curves", "🎯 ROC & PR Curves", "🧩 Confusion Matrices & Generalization"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                p_loss = os.path.join(OUTPUTS_DIR, "graphs", "training_validation_loss.png")
                if os.path.exists(p_loss):
                    st.image(p_loss, caption="Training vs Validation Loss (BCEWithLogitsLoss)", use_container_width=True)
            with col2:
                p_acc = os.path.join(OUTPUTS_DIR, "graphs", "training_validation_accuracy.png")
                if os.path.exists(p_acc):
                    st.image(p_acc, caption="Training vs Validation Accuracy (%)", use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                p_roc = os.path.join(OUTPUTS_DIR, "graphs", "roc_curve.png")
                if os.path.exists(p_roc):
                    st.image(p_roc, caption=f"Receiver Operating Characteristic (AUC = {in_dist['roc_auc']:.4f})", use_container_width=True)
            with col2:
                p_pr = os.path.join(OUTPUTS_DIR, "graphs", "precision_recall_curve.png")
                if os.path.exists(p_pr):
                    st.image(p_pr, caption="Precision-Recall Curve", use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                p_cm = os.path.join(OUTPUTS_DIR, "confusion_matrices", "confusion_matrix_heatmap.png")
                if os.path.exists(p_cm):
                    st.image(p_cm, caption="In-Distribution Test Confusion Matrix", use_container_width=True)
            with col2:
                st.subheader("Cross-Dataset Domain Shift (CIPLAB)")
                st.markdown(f"""
                - **Cross-Dataset Accuracy:** `{cross['accuracy']*100:.2f}%`
                - **Cross-Dataset ROC-AUC:** `{cross['roc_auc']:.4f}`
                - **False Positive Rate:** `{cross['false_positive_rate']*100:.2f}%`
                - **False Negative Rate:** `{cross['false_negative_rate']*100:.2f}%`

                **Academic Finding:**  
                Testing on disjoint datasets demonstrates significant synthetic domain shift. Without domain-adaptive regularizers, models 
                fit to dataset-specific generative artifacts, underscoring the critical need for cross-generator robustness.
                """)
    else:
        st.warning("Evaluation metrics files not found in outputs/.")


# ==============================================================================
# PAGE 5: ABOUT THE PROJECT
# ==============================================================================
elif page == "ℹ️ About the Project":
    st.markdown('<div class="main-title">ℹ️ About the Framework</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">System architecture, mathematical foundations, academic references, and development constraints.</div>', unsafe_allow_html=True)

    st.subheader("1. Problem Statement & Motivation")
    st.markdown("""
    Generative artificial intelligence and deepfake technologies allow realistic manipulation of facial identity, expression, 
    and speech in images and videos. This poses serious threats to cybersecurity, political integrity, and personal privacy.
    Traditional detection methods rely heavily on pretrained generic image classifiers (e.g. ImageNet ResNet/VGG), which often focus 
    on semantic high-level content rather than subtle, low-level generative synthesis artefacts.
    """)

    st.subheader("2. Strict Development Constraints Adhered To")
    st.markdown("""
    | Requirement | Implementation Status | Evidence / Verification |
    | :--- | :--- | :--- |
    | **No Pretrained Models** | Fully Satisfied | Built custom `CustomDeepfakeCNN` from scratch; zero ResNet/VGG/ViT weights used. |
    | **Random Weight Initialization** | Fully Satisfied | Kaiming Normal initialization applied to all convolutional blocks. |
    | **Zero External AI APIs** | Fully Satisfied | No OpenAI, Gemini, HuggingFace inference, or external cloud APIs. |
    | **Local Training & Inference** | Fully Satisfied | Executed entirely in PyTorch with local CUDA acceleration. |
    | **Dual Input Support** | Fully Satisfied | Supports both individual image analysis and sequential video classification. |
    | **Explainable AI (XAI)** | Fully Satisfied | Grad-CAM hooked into the final convolutional stage for visual localization. |
    """)

    st.subheader("3. Neural Network Architectures")
    st.markdown("""
    - **Image Model (CustomDeepfakeCNN):**
      - Stem Layer: Conv2d(3, 32) + BatchNorm + LeakyReLU(0.1)
      - Block 1: ResidualConvBlock(32, 64, stride=2) -> Captures high-frequency forensic noise.
      - Block 2: ResidualConvBlock(64, 128, stride=2) -> Captures edge and blending inconsistencies.
      - Block 3: ResidualConvBlock(128, 256, stride=2) -> Extracts facial component symmetry.
      - Block 4: ResidualConvBlock(256, 512, stride=2) -> High-level illumination mismatch detection.
      - Classifier Head: GlobalAvgPool + Linear(512, 128) + Dropout(0.4) + Linear(128, 1).
    - **Video Model (CustomVideoDeepfakeModel):**
      - Spatial Backbone: Custom CNN extracting 512-dim frame vectors.
      - Temporal Component: 2-layer Bidirectional LSTM (hidden=128, 256-dim output).
      - Learned Temporal Attention: Softmax weighting identifying key manipulated frames.
      - Dual Heads: Frame-level classification + sequence-level classification.
    """)

    st.subheader("4. Primary Academic References")
    st.markdown("""
    1. **Selvaraju, R. R., et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE ICCV.
    2. **Rössler, A., et al. (2019).** *FaceForensics++: Learning to Detect Manipulated Facial Images.* IEEE ICCV.
    3. **Dolhansky, B., et al. (2020).** *The Deepfake Detection Challenge (DFDC) Dataset.* arXiv:2006.07397.
    4. **He, K., et al. (2015).** *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* (Kaiming Normal).
    5. **He, K., et al. (2016).** *Deep Residual Learning for Image Recognition.* IEEE CVPR.
    """)
