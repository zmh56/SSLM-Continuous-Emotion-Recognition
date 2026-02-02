# 🎭 Emotion Recognition with Self-Supervised Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-ee4c2c.svg)](https://pytorch.org/)

> A two-stage approach for physiological signal-based emotion recognition using self-supervised learning and multi-instance learning with mixture of experts (MIL-MoE).

---

## 📋 Overview

This repository implements a robust framework for emotion recognition:
- **Stage 1 (Pre-training)**: Self-supervised learning using contrastive learning with **MultiModalResNet1D**.
- **Stage 2 (Downstream)**: Emotion classification using **MIL-MoE** (Multi-Instance Learning with Mixture of Experts) classifier with a frozen pretrained encoder.

## 📦 Installation

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

### Requirements
- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- tqdm >= 4.62.0

## 📂 Project Structure

```
emotion_recognition_ssl_github/
├── 🧠 models/             # Model definitions
│   ├── multimodal_resnet.py  # MultiModalResNet1D encoder
│   └── mil_moe.py           # MIL-MoE classifier
├── 🛠️ utils/              # Utility functions
│   ├── data_utils.py    # Data loading utilities
│   └── moco.py          # MoCo implementation
├── ⚙️ configs/             # Configuration files
│   ├── ssl_example.cfg  # Configuration for SSL training
│   └── train_example.cfg # Configuration for downstream training
├── 🚀 train_ssl.py         # Stage 1: Self-supervised training
└── 🚀 train_downstream.py  # Stage 2: Downstream training
```

## 🚀 Usage

### Stage 1: Self-Supervised Pre-training

Train the encoder using MoCo contrastive learning:

```bash
python train_ssl.py --cfg configs/ssl_example.cfg --epochs 300 --batch_size 32 --lr 0.03 --moco-dim 128 --moco-k 4096
```

**Key Arguments:**

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--cfg` | `configs/ssl_example.cfg` | Path to configuration file |
| `--epochs` | `300` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.03` | Learning rate |
| `--moco-dim` | `128` | Feature dimension |
| `--moco-k` | `4096` | Queue size for negative samples |
| `--moco-m` | `0.99` | Momentum coefficient |
| `--moco-t` | `0.08` | Temperature parameter |
| `--fold-idx` | `15` | Fold index for data list file |

### Stage 2: Downstream Training

Train the MIL-MoE classifier using the pretrained encoder:

```bash
python train_downstream.py --cfg configs/train_example.cfg --pretrained path/to/checkpoint_epoch_300.pth --fold-idx 15
```

**Key Arguments:**

| Argument | Description |
|----------|-------------|
| `--cfg` | Path to configuration file (default: `configs/train_example.cfg`) |
| `--pretrained` | **Required**. Path to pretrained model checkpoint |
| `--fold-idx` | Fold index for data list file (default: `15`) |

## ⚙️ Configuration

### SSL Configuration (`configs/ssl_example.cfg`)
Define your pre-training setup:
- **Data**: Dataset name, folder path, training lists.
- **Windowing**: Sampling rate (`fs`), window length (`cw_len`), channels.
- **CNN**: Architecture type, filter specs.
- **Optimization**: LR, batch size.

### Downstream Configuration (`configs/train_example.cfg`)
Define your classification setup:
- **Data**: Inputs and output directories.
- **Windowing**: Must match SSL settings.
- **CNN**: Architecture `arch` **must** match SSL training; `num_classes`.
- **Optimization**: Training parameters.

> **Note**: Downstream training automatically sets segment length to `wlen // 6`. Ensure `num_classes` in the encoder matches feature dimension (128).

## 🧠 Model Architecture

### MultiModalResNet1D
- **SincConv**: Feature extraction tailored for time-series.
- **ResNet-18 Backbone**: Deep feature learning per modality.
- **Cross-Modal Attention**: Fuses information across modalities.
- **Output**: 128-dimensional features.

### MoEEmotionClassifier
- **Frozen Encoder**: Uses the pre-trained weights (frozen during training).
- **Mixture of Experts (MoE)**: Segment-level processing.
- **MIL Aggregation**: Aggregates segment features for trial-level classification.
- **Flexibility**: Supports both single-label and multi-label (e.g., V-A-D for DEAP) tasks.

## 💾 Data Format

The pipeline expects `.mat` files:
- **DEAP**: Expects `data['data']` for signals. File pattern: `{sub}_{trial}_{val}_{aro}_{dom}.mat`.
- **SEED**: Expects `data['eeg']` for signals.

## 🔄 Training Workflow

1.  **Self-Supervised Pre-training**:
    *   Learn representations on unlabeled/weakly labeled data using MoCo.
2.  **Downstream Fine-tuning**:
    *   Freeze encoder weights.
    *   Train MIL-MoE classifier on labeled data.
    *   *Note: Only matching layers (by name and shape) are transferred from the pretrained model.*

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.
