# Limit Order Book Prediction with Transformers

Bachelor Thesis Project - Deep Learning for Financial Market Prediction

## Overview

This repository implements and compares different deep learning architectures for predicting mid-price movements in limit order books (LOB), with a focus on Transformer-based models.

## Project Structure

```
thesis/
├── src/
│   ├── preprocessing/         # Data preprocessing pipeline (steps 01-04)
│   ├── models/               # Model architectures
│   └── training/             # Training scripts
├── data/                     # Data directories (raw & processed)
└── lightning_logs/           # Training logs and checkpoints
```

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Pipeline

The preprocessing pipeline consists of four sequential steps:

1. **Step 01: Basic Transformation** (`01_basic_transformation.py`)
   - Transform raw LOBSTER data to preprocessed features and labels

2. **Step 02: Temporal Split** (`02_temporal_split.py`)
   - Split data into train/validation/test sets

3. **Step 03: Normalization** (`03_fit_and_apply_normalization.py`)
   - Fit and apply z-score normalization

4. **Step 04: Build Windows** (`04_build_model_windows.py`)
   - Create fixed-length sliding windows for sequence models

Run each step sequentially:
```bash
python src/preprocessing/01_basic_transformation.py
python src/preprocessing/02_temporal_split.py
python src/preprocessing/03_fit_and_apply_normalization.py
python src/preprocessing/04_build_model_windows.py
```

## Models

Three architectures are implemented and compared:

- **Baseline MLP**: Simple multi-layer perceptron
- **DeepLOB**: CNN-based architecture for LOB prediction
- **LiT (Lightweight Transformer)**: Custom Transformer architecture

## Training

Train models using the scripts in `src/training/`:

```bash
python src/training/train_baseline.py
python src/training/train_deeplob.py
python src/training/train_lit.py
```

Training logs and checkpoints are saved in `lightning_logs/`.

## Data

The project uses LOBSTER (Limit Order Book System - The Efficient Reconstructor) data for NASDAQ stocks:
- AAPL (Apple)
- CSCO (Cisco)
- GOOG (Google)
- INTC (Intel)

Additional benchmark: FI-2010 dataset

## License

This is a bachelor thesis project. All rights reserved.
