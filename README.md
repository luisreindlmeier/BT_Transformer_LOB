# Transformer-Based Prediction of Limit Order Book Dynamics for Market Making

## Overview

This repository implements and compares different deep learning architectures for predicting mid-price movements in limit order books (LOB), with a focus on a pure Transformer-based model.

## Project Structure

```
thesis/
├── src/
│   ├── preprocessing/        # Data preprocessing pipeline (steps 01-04)
│   ├── models/               # Model architectures
│   └── training/             # Training scripts
└── data/                     # Data directories (raw & processed)
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

The project is bsaed on an academic license for LOBSTER (Limit Order Book System - The Efficient Reconstructor) Level 2 LOB data for four NASDAQ stocks covering 20 trading days from July 2023:
- AAPL (Apple)
- CSCO (Cisco)
- GOOG (Google)
- INTC (Intel)

Additionaly, the models were also trained and evaluated on the FI-2010 benchmark dataset given in a slightly altered structure.

## License

This is a bachelor thesis project. All rights reserved. For any questions or remarks, please contact `luis.reindlmeier@fs-students.de`
