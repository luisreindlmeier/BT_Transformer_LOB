# DeepLOB Architecture (Concise)

## Overview
CNN + LSTM hybrid tailored for Limit Order Books. Captures spatial depth structure with convolutions and temporal dynamics with LSTM, followed by attention-style pooling.

- Location: src/training/train_deeplob.py
- Framework: PyTorch (plain)
- Input: (B, 100, 40) LOB depth/volume features (no aggregated features)

## Architecture
- CNN Block 1: Conv2d(in=1, out=16, kernel=(1,2)) + ReLU + MaxPool
- CNN Block 2: Conv2d(out=16→32, kernel=(4,2)) + ReLU + MaxPool
- Reshape to sequence
- LSTM: 1 layer, hidden=64, batch_first
- Attention-like temporal weighting → context vector
- Head: Linear 64 → 3 logits
- Params: ~130K (compact)

## Training Setup
- Epochs: 10, Batch: 64, Patience: 3
- Optimizer: AdamW, max LR 2e-3, OneCycleLR
- Loss: CrossEntropy with label_smoothing=0.1
- DataLoader: shuffle, num_workers=0, drop_last=True
- Device: CPU; threads capped

## Notes
- Strong inductive bias for LOB microstructure (price levels × volume)
- Much smaller than LiT; trains fastest
- Data path fallback: data/…/CSCO or ~/thesis_output/…/CSCO
