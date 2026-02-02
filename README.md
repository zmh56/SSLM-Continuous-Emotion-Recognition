# Emotion Recognition with Self-Supervised Learning

A two-stage approach for physiological signal-based emotion recognition using self-supervised learning and multi-instance learning with mixture of experts (MIL-MoE).

## Overview

This repository implements:
- **Stage 1**: Self-supervised pre-training using contrastive learning with MultiModalResNet1D
- **Stage 2**: Downstream emotion classification using MIL-MoE classifier with frozen pretrained encoder

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- tqdm >= 4.62.0

## Project Structure

```
emotion_recognition_ssl_github/
├── models/              # Model definitions
│   ├── multimodal_resnet.py  # MultiModalResNet1D encoder
│   └── mil_moe.py           # MIL-MoE classifier
├── utils/               # Utility functions
│   ├── data_utils.py    # Data loading utilities
│   └── moco.py          # MoCo implementation
├── configs/             # Configuration files
│   ├── ssl_example.cfg  # Configuration for SSL training
│   └── train_example.cfg # Configuration for downstream training
├── train_ssl.py         # Stage 1: Self-supervised training
└── train_downstream.py  # Stage 2: Downstream training
```

## Usage

### Stage 1: Self-Supervised Pre-training

Train the encoder using MoCo contrastive learning:

```bash
python train_ssl.py --cfg configs/ssl_example.cfg --epochs 300 --batch_size 32 --lr 0.03 --moco-dim 128 --moco-k 4096
```

Arguments:
- `--cfg`: Path to configuration file (default: `configs/ssl_example.cfg`)
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.03)
- `--moco-dim`: Feature dimension (default: 128)
- `--moco-k`: Queue size for negative samples (default: 4096)
- `--moco-m`: Momentum coefficient (default: 0.99)
- `--moco-t`: Temperature parameter (default: 0.08)
- `--fold-idx`: Fold index for data list file (default: 15)
- `--multi-epoch-repeat`: Repeat dataloader iterations per epoch (default: 30)
- `--resume`: Path to checkpoint to resume from (optional)
- `--results-dir`: Output directory for checkpoints (optional)

### Stage 2: Downstream Training

Train the MIL-MoE classifier using the pretrained encoder:

```bash
python train_downstream.py --cfg configs/train_example.cfg --pretrained path/to/checkpoint_epoch_300.pth --fold-idx 15
```

Arguments:
- `--cfg`: Path to configuration file (default: `configs/train_example.cfg`)
- `--pretrained`: Path to pretrained model checkpoint (required)
- `--fold-idx`: Fold index for data list file (default: 15)

## Configuration

### SSL Configuration (`configs/ssl_example.cfg`)

Edit the configuration file to set:
- **Data section**: Dataset name, data folder, training list
- **Windowing section**: Sampling rate (fs), window length (cw_len), channel configuration
- **CNN section**: Architecture type, filter configurations
- **Optimization section**: Learning rate, batch size, etc.

### Downstream Configuration (`configs/train_example.cfg`)

Edit the configuration file to set:
- **Data section**: Dataset name, data folder, training list, output folder
- **Windowing section**: Sampling rate, window length, channel configuration
- **CNN section**: Architecture type (must match SSL training), number of classes
- **Optimization section**: Learning rate, batch size, number of epochs

**Important Notes**:
- The downstream training uses `wlen // 6` as segment length, where `wlen` is calculated from `cw_len` in the config
- The encoder's `num_classes` should match the feature dimension (128) to load pretrained weights correctly
- Make sure the architecture (`arch`) matches between SSL and downstream training

## Model Architecture

### MultiModalResNet1D
- SincConv-based feature extraction
- ResNet-18 backbone per modality
- Cross-modal attention mechanism
- Outputs 128-dimensional features

### MoEEmotionClassifier
- Uses pretrained encoder (frozen during training)
- Mixture of Experts (MoE) for segment-level processing
- Aggregates segment features for trial-level classification
- Supports both single-label and multi-label classification

## Data Format

The code expects data files in `.mat` format:
- For DEAP dataset: `data['data']` should contain the EEG signals
- For SEED dataset: `data['eeg']` should contain the EEG signals
- File naming convention for DEAP: `{subject_id}_{trial_id}_{valence}_{arousal}_{dominance}.mat`

## Training Process

1. **Self-supervised training**: Train the encoder using MoCo contrastive learning on unlabeled or weakly labeled data
2. **Downstream training**: Freeze the encoder and train the MIL-MoE classifier on labeled emotion data
3. The pretrained encoder weights are loaded and only matching layers are transferred

## Notes

- The encoder is frozen during downstream training
- Only matching layers (by name and shape) are loaded from the pretrained model
- The model supports both single-label and multi-label classification (e.g., V-A-D for DEAP)
- Segment length in downstream training is automatically set to `wlen // 6` to process longer trials

## License

MIT License - see LICENSE file for details

