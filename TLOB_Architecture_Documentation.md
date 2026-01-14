# TLOB Architecture (Concise)

## Overview
Transformer-style dual-attention model for LOB. Uses two attention stacks to capture both short-term micro-movements and longer horizons, then fuses for classification.

- Location: src/training/train_tlob.py
- Framework: PyTorch (plain)
- Input: (B, 100, 49) unified features

## Architecture (typical configuration)
- Input Linear: 49 → 128
- Positional encoding: learnable or sinusoidal (per script)
- Encoder stack A: 4 layers, 4 heads (short-horizon focus)
- Encoder stack B: 4 layers, 4 heads (broader context)
- Pool/fuse: concat or add the two CLS outputs (see script)
- Head: MLP 128/256 → 3 logits
- Params: ~2.8M (mid-size)

## Training Setup
- Epochs: 10, Batch: 64, Patience: 3
- Optimizer: AdamW, max LR 2e-3, OneCycleLR
- Loss: CrossEntropy with label_smoothing=0.1
- DataLoader: shuffle, num_workers=0, drop_last=True
- Device: CPU; threads capped

## Notes
- Balances capacity and speed between DeepLOB (small) and LiT (larger)
- Dual-attention design targets multi-scale temporal patterns
- Data path fallback: data/…/CSCO or ~/thesis_output/…/CSCO
