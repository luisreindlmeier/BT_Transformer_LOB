import torch
from sklearn.metrics import accuracy_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        targets.extend(y.numpy())

    return accuracy_score(targets, preds)