# LiT Transformer Architecture Documentation

## Overview

The LiT (Limit Order Book Transformer) model is a pure Transformer-based architecture for high-frequency LOB classification. It processes unified feature sequences using self-attention mechanisms and predicts mid-price movement direction (down/stationary/up) at horizon H=10 events.

**Implementation:** Plain PyTorch (not PyTorch Lightning)  
**Location:** `src/training/train_lit.py` (lines 168-210)  
**Status:** Active production model  

---

## Architecture Components

### 1. Input Projection
```python
self.input_proj = nn.Linear(input_dim, d_model)  # 49 → 256
```
- Transforms raw 49-dimensional features into d_model=256 dimensional embeddings
- Applied to entire sequence: (B, T=100, 49) → (B, 100, 256)
- No CNN/LSTM preprocessing (unlike DeepLOB)

### 2. CLS Token
```python
self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, 256)
```
- **Learnable parameter** prepended to sequence
- Serves as aggregation point for sequence-level classification
- After prepending: (B, 100, 256) → (B, 101, 256)
- **Design Choice:** CLS token vs mean pooling (discussed below)

### 3. Positional Embeddings
```python
self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))  # (1, 101, 256)
```
- **Learnable positional encodings** (not fixed sinusoidal)
- Added element-wise to input sequence + CLS token
- Encodes temporal position information (critical for time-series)
- **Design Choice:** Learnable vs sinusoidal (discussed below)

### 4. Transformer Encoder
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,  # 4 × d_model
    dropout=0.1,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-LN: LayerNorm before attention/FFN
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
```
- **6 layers** of multi-head self-attention + feed-forward networks
- **8 attention heads** (d_model / nhead = 32 dimensions per head)
- **Pre-LN normalization** (norm_first=True): More stable for deep networks
- **GELU activation**: Smoother than ReLU, better gradients
- **Dropout 0.1**: Regularization at attention and FFN layers
- Processes full sequence: (B, 101, 256) → (B, 101, 256)

### 5. Classification Head
```python
self.head = nn.Sequential(
    nn.Linear(d_model, d_model),  # 256 → 256
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(d_model, num_classes)  # 256 → 3
)
```
- Two-layer MLP with GELU activation
- Processes CLS token output: (B, 256) → (B, 3)
- Outputs raw logits (pre-softmax) for 3 classes: [down, stationary, up]

---

## Hyperparameters

### Architecture
```python
D_MODEL = 256          # Hidden dimension
N_HEADS = 8            # Attention heads
N_LAYERS = 6           # Transformer layers
D_FF = 1024            # FFN dimension (4 × D_MODEL)
DROPOUT = 0.1          # All dropout layers
MAX_SEQ_LEN = 100      # Input sequence length
INPUT_DIM = 49         # Unified feature dimension
NUM_CLASSES = 3        # Down/Stationary/Up
```

### Training Configuration
```python
EPOCHS = 10
BATCH_SIZE = 64        # Reduced from 256 for CPU stability
LR_MAX = 2e-3          # Peak learning rate for OneCycleLR
WEIGHT_DECAY = 1e-4    # L2 regularization
LABEL_SMOOTHING = 0.1  # CrossEntropyLoss smoothing
PATIENCE = 3           # Early stopping patience
DEVICE = 'cpu'         # Training on CPU (VM/Mac)
NUM_WORKERS = 0        # Single-threaded data loading
```

### Optimizer & Scheduler
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR_MAX,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR_MAX,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.3,      # 30% warmup
    anneal_strategy='cos',
    div_factor=25.0,    # Initial LR = LR_MAX / 25 = 8e-5
    final_div_factor=1e4  # Final LR = LR_MAX / 1e4 = 2e-7
)
```

---

## Forward Pass (Step-by-Step)

```python
def forward(self, x):
    # Input: x shape (B, T=100, F=49)
    B = x.size(0)
    
    # Step 1: Project features to d_model
    x = self.input_proj(x)  # (B, 100, 49) → (B, 100, 256)
    
    # Step 2: Prepend CLS token
    cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, 256) → (B, 1, 256)
    x = torch.cat([cls_tokens, x], dim=1)  # (B, 101, 256)
    
    # Step 3: Add positional embeddings
    x = x + self.pos_emb  # (B, 101, 256) + (1, 101, 256) [broadcast]
    
    # Step 4: Transformer encoding
    x = self.encoder(x)  # (B, 101, 256) → (B, 101, 256)
    
    # Step 5: Extract CLS token
    cls_output = x[:, 0]  # (B, 256) - first token contains aggregated info
    
    # Step 6: Classification
    logits = self.head(cls_output)  # (B, 256) → (B, 3)
    
    return logits  # Shape: (B, 3)
```

---

## Design Decisions & Rationale

### 1. CLS Token vs Mean Pooling
**Choice:** CLS token (BERT-style)

**Alternatives:**
- Mean pooling over all 100 timesteps
- Max pooling over temporal dimension
- Last timestep only (like RNN)

**Rationale:**
- CLS token learns task-specific aggregation via backpropagation
- More flexible than fixed pooling (mean/max)
- Proven in NLP (BERT) and Vision Transformers (ViT)
- Allows model to weight informative timesteps dynamically

### 2. Learnable vs Sinusoidal Positional Embeddings
**Choice:** Learnable `nn.Parameter`

**Alternatives:**
- Fixed sinusoidal (original Transformer paper)
- Relative positional encodings (T5/DeBERTa)

**Rationale:**
- LOB sequences are **non-periodic** (unlike language/images)
- Learnable embeddings adapt to domain-specific temporal patterns
- Slight memory overhead (101×256 = 25,856 params) is negligible
- Empirically better for Vision Transformers (ViT paper)

### 3. Pre-LN vs Post-LN
**Choice:** Pre-LN (`norm_first=True`)

**Alternatives:**
- Post-LN (original Transformer paper)
- RMSNorm (GPT-3)

**Rationale:**
- **More stable gradients** for deep networks (6+ layers)
- Original Transformer used Post-LN but struggled beyond 12 layers
- Pre-LN avoids gradient explosion in residual connections
- Standard in modern Transformers (GPT-2+, ViT, BERT successors)

### 4. Pure Transformer vs Hybrid (CNN/LSTM + Transformer)
**Choice:** Pure Transformer (no CNN/LSTM preprocessing)

**Alternatives:**
- CNN feature extraction + Transformer (DeepLOB-style)
- LSTM encoding + Transformer attention

**Rationale:**
- Self-attention can capture both **local and global patterns**
- Simpler architecture (fewer hyperparameters)
- Direct comparison to DeepLOB (which uses CNN+LSTM)
- Modern trend: ViT, BERT, GPT all use pure attention

---

## Comparison to Other Models

### vs Legacy LiT (PyTorch Lightning, deleted)
**Old:** Dual-stream architecture (LOB + aggregated features)
- Separate `lob_proj` (40→256) and `agg_proj` (9→256)
- Concatenated along sequence: (B, 200, 256) total length
- More complex, harder to interpret

**New (Current):** Unified input stream
- Single `input_proj` (49→256)
- Sequence length 100 (not 200)
- Cleaner, consistent with Vision Transformer design

### vs DeepLOB
**DeepLOB:** CNN (spatial) → LSTM (temporal) → Attention → FC
- 2-layer CNN: 16/32 filters, kernel (1,2)/(4,2)
- 1-layer LSTM: 64 hidden units
- Attention mechanism over LSTM outputs

**LiT:** Direct sequence → Transformer → CLS → FC
- No CNN preprocessing (Transformer handles spatial patterns)
- No LSTM (Transformer handles temporal dependencies)
- CLS token instead of attention-weighted pooling

**Advantage LiT:** Fewer inductive biases, more flexible
**Advantage DeepLOB:** LOB-specific CNN design, fewer parameters

### vs SimpleMLP Baseline
**Baseline:** Flatten sequence → 3-layer MLP
- Input: 100×49 = 4900 dimensions (flattened)
- Hidden layers: [1024, 512, 256]
- No temporal modeling (all timesteps concatenated)

**LiT:** Temporal modeling via self-attention
- Captures dependencies between timesteps
- Positional embeddings encode order
- ~3× more parameters but theoretically more expressive

---

## Parameter Count

### Breakdown
```
Input Projection:       49 × 256 + 256 bias        = 12,800
CLS Token:              1 × 256                    = 256
Positional Embeddings:  101 × 256                  = 25,856

Transformer Encoder (6 layers × per-layer):
  - Multi-Head Attention: 4 × (256 × 256) + biases ≈ 263,168 per layer
  - Feed-Forward:         2 × (256 × 1024) + biases ≈ 525,312 per layer
  - LayerNorms:           2 × (256 × 2)              ≈ 1,024 per layer
  - Total per layer:                                 ≈ 789,504
  - 6 layers:                                        ≈ 4,737,024

Classification Head:
  - Linear 1:             256 × 256 + 256           = 65,792
  - Linear 2:             256 × 3 + 3               = 771

Total:                                              ≈ 4,841,499 parameters
```

**Comparison:**
- SimpleMLP Baseline: ~5.8M parameters
- DeepLOB: ~130K parameters (much smaller, domain-specific)
- TLOB: ~2.8M parameters (dual-attention)

**LiT is moderately large** but justified by Transformer expressiveness.

---

## Training Details

### Data Pipeline
```python
# Dataset
train_dataset = UnifiedLOBDataset(
    X_train,  # (5,380,000, 100, 49)
    y_train,  # (5,380,000,)
    device='cpu'
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=True,  # Prevents incomplete batch issues
    pin_memory=False  # CPU training
)
```

### Loss Function
```python
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,
    weight=None  # No class weighting (removed WeightedRandomSampler)
)
```
- **Label smoothing 0.1:** Prevents overconfident predictions
- **No sampler:** Removed to fix memory issues on VM

### Metrics Logged
Per epoch CSV (`metrics_lit.csv`):
```
epoch, train_loss, train_acc, val_loss, val_acc
1, 0.8234, 0.6123, 0.7891, 0.6345
2, 0.7456, 0.6523, 0.7234, 0.6678
...
```

### Early Stopping
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_lit.pt')
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Graceful Shutdown
```python
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
```
- Ctrl+C saves checkpoint and metrics before exit
- Safe for long VM training runs

---

## Implementation Notes

### Multi-Threading Control
```python
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
```
- Prevents CPU oversubscription on VM
- Critical for stable training on shared machines

### Path Fallback Logic
```python
_local_data = Path("data/03_normalized_NEW/CSCO")
_vm_data = Path.home() / "thesis_output" / "03_normalized_NEW" / "CSCO"
DATA_ROOT = _local_data if _local_data.exists() else _vm_data
```
- Works on both local Mac and Google Cloud VM
- No manual path configuration needed

### Batch Progress Logging
```python
if (batch_idx + 1) % 500 == 0:
    print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
          f"Loss: {running_loss/(batch_idx+1):.4f} | "
          f"Acc: {100*running_correct/running_total:.2f}%")
```
- Every 500 batches during training (visibility on VM)
- Helps detect hangs (elapsed time should keep incrementing)

---

## Known Limitations

1. **CPU-Only Training:** No GPU support in current setup
   - VM has no GPU quota
   - Mac M2 MPS not used (compatibility issues)
   - Training takes ~10-15 hours for 10 epochs

2. **No Test Evaluation:** Test split exists but not evaluated yet
   - Requires separate `evaluate_on_test.py` script
   - Step 04 windows not built for test split

3. **No Hyperparameter Tuning:** All hyperparameters manually chosen
   - D_MODEL, N_HEADS, N_LAYERS not grid-searched
   - Learning rate schedule not optimized
   - BATCH_SIZE constrained by CPU memory

4. **Single Stock (CSCO):** Model trained only on Cisco data
   - Not tested on AAPL, GOOG, INTC
   - Cross-stock generalization unknown

---

## Future Work

### Potential Improvements
1. **Relative Positional Encodings:** Like T5/DeBERTa (better extrapolation)
2. **Multi-Scale Attention:** Different heads for different time horizons
3. **Causal Masking:** Prevent attention to future timesteps (not needed for classification but theoretically cleaner)
4. **Knowledge Distillation:** Train smaller model from LiT predictions

### Ablation Studies
- CLS vs mean pooling
- Learnable vs sinusoidal positional embeddings
- Pre-LN vs Post-LN
- Number of layers (3 vs 6 vs 12)
- Number of heads (4 vs 8 vs 16)

### Cross-Model Ensemble
- Average predictions from SimpleMLP, DeepLOB, LiT, TLOB
- Potential accuracy boost from diverse architectures

---

## References

**Vision Transformer (ViT):**
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
- CLS token design, learnable positional embeddings

**BERT:**
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
- CLS token for classification tasks

**Pre-LN Transformers:**
- Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020
- Stability benefits of norm_first=True

**DeepLOB:**
- Zhang et al., "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", IEEE TIM 2019
- CNN-LSTM architecture for LOB prediction

**OneCycleLR:**
- Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks", AAAI 2019
- Cyclic learning rate schedule

---

## File Locations

- **Model Implementation:** `src/training/train_lit.py` (lines 168-210)
- **Training Script:** `src/training/train_lit.py` (full script)
- **Preprocessed Data:** `data/03_normalized_NEW/CSCO/` (local) or `~/thesis_output/03_normalized_NEW/CSCO/` (VM)
- **Output Metrics:** `metrics_lit.csv` (written to current directory)
- **Best Model Checkpoint:** `best_lit.pt` (saved after training)
- **Legacy Code (Deleted):** `src/models/lit_model.py` (PyTorch Lightning dual-stream implementation)

---

## Summary

The LiT Transformer is a **Vision Transformer-inspired architecture** for limit order book classification. It uses:
- **Pure self-attention** (no CNN/LSTM)
- **CLS token aggregation** (learnable sequence summary)
- **Learnable positional embeddings** (adapted to non-periodic timeseries)
- **Pre-LN normalization** (stable training for 6 layers)
- **256-dimensional hidden space, 8 heads, 6 layers** (~4.8M parameters)

It represents a **modern deep learning approach** to LOB prediction, contrasting with:
- **DeepLOB** (domain-specific CNN+LSTM)
- **SimpleMLP** (no temporal modeling)
- **TLOB** (dual-attention mechanism)

Training is stable on CPU-only VMs with proper multi-threading control and graceful shutdown handling. The model achieves competitive validation accuracy (~65-67%) and will be evaluated on held-out test data for final thesis results.
