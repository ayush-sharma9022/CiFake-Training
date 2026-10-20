# Deepfake Image and Video Detection Framework Using Deep Learning and Visual Artefact Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1%20Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture: Scratch](https://img.shields.io/badge/Architecture-100%25%20From%20Scratch-red.svg)](#strict-technology-constraints)
[![Streamlit App](https://img.shields.io/badge/Deployment-Streamlit%20Web%20App-ff4b4b.svg)](webapp/app.py)

An end-to-end, academic and practical deep learning framework engineered from first principles to automatically detect, classify, and visually explain digital media manipulations (deepfakes) across both static images and sequential video files.

---

## 📌 Key Architectural Highlights

* **100% Custom Neural Architectures:** Developed entirely from first principles without transfer learning, pretrained weights, or external detection APIs.
* **Random Initialization (Kaiming Normal):** All convolutional and linear weights are initialized randomly and learned exclusively through forensic gradient updates.
* **Residual Spatial Artefact Extraction:** `CustomDeepfakeCNN` employs five hierarchical residual blocks equipped with projection shortcuts, LeakyReLU, and Batch Normalization to detect subtle generative noise and blending seams.
* **Spatio-Temporal Video Modeling:** `CustomVideoDeepfakeModel` integrates the spatial CNN backbone with a 2-layer Bidirectional LSTM and learned Temporal Self-Attention to capture inter-frame flickering and micro-expression discontinuity.
* **Forensic Explainability (Grad-CAM):** Real-time Gradient-Weighted Class Activation Mapping hooked into the final convolutional stage to localize the precise facial regions influencing predictions.
* **Cross-Dataset Rigor:** Evaluated on both in-distribution test splits and external unseen benchmarks (CIPLAB) to quantify real-world synthetic domain shift.
* **Interactive Web Application:** Lightweight, reactive 5-page Streamlit deployment interface for image/video testing and metric visualization.

---

## 🔒 Strict Technology Constraints Compliance

| Requirement / Constraint | Compliance Status | Implementation Verification |
| :--- | :--- | :--- |
| **No Pretrained Architectures** | **STRICTLY ENFORCED** | No ResNet, VGG, MobileNet, EfficientNet, Xception, or Vision Transformers used. |
| **No Transfer Learning / Fine-Tuning** | **STRICTLY ENFORCED** | Zero downloaded weights or fine-tuned model backbones. |
| **Random Weight Initialization** | **STRICTLY ENFORCED** | Kaiming Normal initialization applied to all convolutional filters. |
| **No External Cloud / AI APIs** | **STRICTLY ENFORCED** | Zero calls to OpenAI, Gemini, HuggingFace Inference, or cloud APIs. |
| **Dual Media Support** | **STRICTLY ENFORCED** | Supports single-image (.jpg, .png) and multi-frame video (.mp4) classification. |
| **Local GPU Execution** | **STRICTLY ENFORCED** | Optimized with Automatic Mixed Precision (AMP) on local NVIDIA RTX 3050 GPU. |

---

## 📊 Empirical Performance Summary

All metrics reported below were computed from actual experimental runs on held-out test sets:

| Evaluation Metric | Image CNN (In-Distribution Test) | Image CNN (Cross-Dataset CIPLAB) | Video Model (Frame-Level) | Video Model (Sequence-Level) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **88.40%** | 49.30% | **100.0%** | **100.0%** |
| **Precision** | **87.02%** | 49.54% | **100.0%** | **100.0%** |
| **Recall (TPR)** | **90.27%** | 76.20% | **100.0%** | **100.0%** |
| **F1-Score** | **88.61%** | 60.05% | **100.0%** | **100.0%** |
| **ROC-AUC** | **0.9548** | 0.5098 | **1.0000** | **1.0000** |
| **False Positive Rate (FPR)** | **13.47%** | 77.60% | **0.00%** | **0.00%** |
| **False Negative Rate (FNR)** | **9.73%** | 23.80% | **0.00%** | **0.00%** |

---

## 🧠 Neural Network Architectures

### 1. `CustomDeepfakeCNN` (Image Classifier)
```
Input (3, 128, 128)
  │
  ├─► Stem Conv: Conv2d(3 -> 32, k=3, s=1, p=1) + BatchNorm + LeakyReLU(0.1)
  │
  ├─► Block 1 (Residual): 32 -> 64 ch, stride=2  (64x64) -> High-frequency noise capture
  │
  ├─► Block 2 (Residual): 64 -> 128 ch, stride=2 (32x32) -> Edge & blending seam extraction
  │
  ├─► Block 3 (Residual): 128 -> 256 ch, stride=2 (16x16) -> Facial symmetry & component alignment
  │
  ├─► Block 4 (Residual): 256 -> 512 ch, stride=2 (8x8)   -> Chromatic aberration & lighting (Grad-CAM Target)
  │
  ├─► Global Average Pooling: AdaptiveAvgPool2d((1, 1)) -> 512-dim embedding
  │
  └─► Classification Head:
        Flatten -> Linear(512, 128) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.4) -> Linear(128, 1)
```

### 2. `CustomVideoDeepfakeModel` (Spatio-Temporal Sequence Classifier)
```
Input Sequence: (Batch, Num_Frames=15, 3, 128, 128)
  │
  ├─► Spatial Extractor: Parallel CustomDeepfakeCNN (512-dim per frame)
  │     Output: (Batch, Num_Frames, 512)
  │
  ├─► Temporal Sequence Learner: 2-Layer Bidirectional LSTM (hidden=128 per dir)
  │     Output: (Batch, Num_Frames, 256)
  │
  ├─► Temporal Self-Attention:
  │     Learns dynamic attention weights across frames to focus on transient glitches
  │     Output: Context vector (Batch, 256) & Attention weights (Batch, Num_Frames, 1)
  │
  ├─► Frame Classifier: Linear(256 -> 64) -> ReLU -> Dropout(0.3) -> Linear(64, 1)
  │
  └─► Video Classifier: Linear(256 -> 128) -> BatchNorm1d -> Dropout(0.4) -> Linear(128, 1)
```

---

## 📂 Project Directory Structure

```text
Deepfake-Detection-Framework/
│
├── datasets/
│   ├── raw/                  # Original downloaded images and generated MP4 videos
│   ├── processed/            # Standardized 128x128 arrays & extracted video frames
│   └── metadata/             # Partition CSVs, frame manifests, and dataset comparison tables
│
├── notebooks/
│   ├── master_deepfake_detection_framework.ipynb   # Master notebook (all 28 sections)
│   ├── 01_dataset_analysis.ipynb                  # EDA and class distribution analysis
│   ├── 02_image_preprocessing.ipynb               # Cleaning, MD5 deduplication, and scaling
│   ├── 03_video_preprocessing.ipynb               # OpenCV frame extraction and sequence indexing
│   ├── 04_image_cnn_training.ipynb                # Scratch CNN training and validation loops
│   ├── 05_video_model_training.ipynb              # BiLSTM + Attention video model training
│   └── 06_model_evaluation.ipynb                 # Confusion matrices, ROC/PR, and Grad-CAM
│
├── models/
│   ├── image_deepfake_cnn.pth                     # Best Image CNN checkpoint (~19.8 MB)
│   └── video_deepfake_model.pth                   # Best Video Model checkpoint (~24.4 MB)
│
├── outputs/
│   ├── graphs/               # Loss curves, accuracy curves, ROC, PR, and attention plots
│   ├── heatmaps/             # Grad-CAM original/heatmap/overlay comparison images
│   ├── confusion_matrices/   # Image and video confusion matrix heatmaps
│   ├── model_evaluation_metrics.json              # Full image evaluation report
│   ├── video_model_evaluation_metrics.json        # Full video evaluation report
│   └── training_history.json                      # Epoch-by-epoch loss and accuracy records
│
├── webapp/
│   ├── app.py                # 5-page Streamlit demonstration portal
│   └── requirements.txt      # Web application runtime dependencies
│
├── documentation/
│   ├── Deepfake_Detection_Project_Report.docx     # Formal 10-chapter academic report (DOCX)
│   └── Deepfake_Detection_Project_Report.pdf      # Formal 10-chapter academic report (PDF)
│
├── scripts/
│   ├── prepare_data.py       # Dataset acquisition, MD5 deduplication, and video generation
│   ├── cache_dataset.py      # Zero-overhead memory-mapped numpy array generator
│   ├── train_image_cnn.py    # GPU training, evaluation, and Grad-CAM generation
│   ├── train_video_model.py  # Spatio-temporal video training and sequence evaluation
│   ├── build_notebooks.py    # Automated Jupyter Notebook constructor
│   └── generate_documentation.py # Automated academic DOCX/PDF documentation builder
│
├── src/
│   ├── models/               # PyTorch custom model definitions
│   └── explainability/       # Grad-CAM implementation hooked into convolutional layers
│
├── README.md                 # Complete project documentation and execution guide
└── requirements.txt          # Complete Python virtual environment dependencies
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Clone the repository and create an isolated Python 3.11 virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Verify GPU Acceleration
Verify that CUDA is recognized by PyTorch:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Run Training Pipelines (Trained from Scratch)
Execute the modular training scripts:
```bash
# 1. Train Custom Image CNN from scratch on GPU
python scripts/train_image_cnn.py

# 2. Train Spatio-Temporal Video Model (BiLSTM + Attention)
python scripts/train_video_model.py
```

### 4. Launch Streamlit Web Application
Run the interactive demonstration interface:
```bash
streamlit run webapp/app.py
```
Open your browser at `http://localhost:8501` to access:
- **Page 1: Home Dashboard** — Architecture summary and high-level performance cards.
- **Page 2: Image Detection** — Upload face images, compute authenticity confidence, and view Grad-CAM heatmaps.
- **Page 3: Video Detection** — Upload MP4 videos, extract frames dynamically, and view frame probability timelines.
- **Page 4: Model Performance** — Interactive inspection of loss curves, ROC/PR curves, and confusion matrices.
- **Page 5: About the Project** — Mathematical formulations, architecture diagrams, and academic citations.

### 5. Re-generate Academic Documentation & Notebooks
To rebuild the master Jupyter Notebooks and formal DOCX/PDF reports:
```bash
# Generate all 7 Jupyter Notebooks (Master + 6 Modular)
python scripts/build_notebooks.py

# Generate DOCX and PDF Academic Reports
python scripts/generate_documentation.py
```

---

## 🔬 Explainable AI (Grad-CAM) Visualizations

Grad-CAM computes the gradient of the predicted class with respect to the final convolutional feature maps (`block4.conv2`):

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

Generated visualizations demonstrate that the model attends directly to:
1. **Facial Blending Boundaries:** Warping seams along the jawline and hairline caused by face-swapping masks.
2. **Eye Specularities:** Asymmetric corneal reflections characteristic of GAN-generated faces.
3. **Mouth & Teeth Inconsistencies:** Texture blurring and unnatural tooth geometry in synthetic expressions.

---

## 📚 Academic References & Citations

1. **Selvaraju, R. R., et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE International Conference on Computer Vision (ICCV), pp. 618-626.
2. **Rössler, A., et al. (2019).** *FaceForensics++: Learning to Detect Manipulated Facial Images.* IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1-11.
3. **Dolhansky, B., et al. (2020).** *The Deepfake Detection Challenge (DFDC) Dataset.* arXiv:2006.07397.
4. **Li, L., et al. (2020).** *Face X-ray for More General Face Forgery Detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5001-5010.
5. **He, K., et al. (2015).** *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* IEEE International Conference on Computer Vision (ICCV).
6. **He, K., et al. (2016).** *Deep Residual Learning for Image Recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.
7. **Güera, D., & Delp, E. J. (2018).** *Deepfake Video Detection Using Recurrent Neural Networks.* IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS).
8. **Goodfellow, I., et al. (2014).** *Generative Adversarial Nets.* Advances in Neural Information Processing Systems (NeurIPS), 27.
9. **Karki, M. (2021).** *Deepfake and Real Images Dataset.* Kaggle. Available at: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
10. **CIPLAB (2020).** *Real and Fake Face Detection Dataset.* Kaggle. Available at: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

---

## 📄 License and Academic Attribution
This project was developed for academic, forensic, and scientific research into synthetic media detection. All models were developed and trained independently from scratch.
