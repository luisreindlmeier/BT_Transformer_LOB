import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from datasets.fi2010 import FI2010Dataset
from models.model import LOBTransformer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ------------------------
# Hyperparameters
# ------------------------
WINDOW_SIZE = 50
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

# ------------------------
# Dataset & Loader
# ------------------------
train_ds = FI2010Dataset(
    "../data/FI-2010/train.csv", horizon_idx=0, window_size=WINDOW_SIZE
)
test_ds = FI2010Dataset(
    "../data/FI-2010/test.csv", horizon_idx=0, window_size=WINDOW_SIZE
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------
# Model
# ------------------------
model = LOBTransformer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# ------------------------
# Training
# ------------------------
for epoch in range(EPOCHS):
    model.train()
    losses = []

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch+1:02d} | Loss: {np.mean(losses):.4f}")

# ------------------------
# Evaluation
# ------------------------
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_true.append(y.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_true)

# ------------------------
# Metrics
# ------------------------
macro_f1 = f1_score(y_true, y_pred, average="macro")
acc = (y_true == y_pred).mean()

print("\nFINAL MODEL EVALUATION (h = 10)")
print("=" * 50)
print("Accuracy:", round(acc, 4))
print("Macro F1:", round(macro_f1, 4))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["down", "stationary", "up"]))

print("\nNormalized confusion matrix (row-wise):")
cm = confusion_matrix(y_true, y_pred, normalize="true")
print(cm)