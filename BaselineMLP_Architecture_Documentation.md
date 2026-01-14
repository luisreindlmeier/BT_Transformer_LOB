# Baseline MLP Architecture (Concise)

## Overview
Simple fully-connected baseline for LOB mid-price direction (down/stationary/up). Serves as a non-temporal reference.

- Location: src/training/train_baseline.py
- Framework: PyTorch (plain)
- Input: (B, 100, 49) → flatten to (B, 4900)

## Architecture
- Flatten 100×49 sequence
- FC1: 4900 → 1024, ReLU, Dropout 0.2
- FC2: 1024 → 512, ReLU, Dropout 0.2
- FC3: 512 → 256, ReLU, Dropout 0.2
- Head: 256 → 3 logits
- Params: ~5.8M

## Training Setup
- Epochs: 10, Batch: 64, Patience: 3
- Optimizer: AdamW, max LR 2e-3, OneCycleLR
- Loss: CrossEntropy with label_smoothing=0.1
- DataLoader: shuffle, num_workers=0, drop_last=True
- Device: CPU (VM/Mac); threads capped for stability

## Notes
- No temporal modeling (flatten loses order)
- Fastest to train, baseline accuracy benchmark
- Data path fallback: data/…/CSCO or ~/thesis_output/…/CSCO
