import os
from dataclasses import dataclass

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import CONFIG

torch.backends.cudnn.benchmark = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["face", "building", "landscape"]


@dataclass
class TrainConfig:
    img_size: int = 128
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    filters: tuple = (16, 32, 64)
    dense: int = 128
    dropout: float = 0.3
    patience: int = 5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform_train, transform_val


def get_loaders(batch_size, num_workers):
    tt, tv = get_transforms()
    train_ds = datasets.ImageFolder(CONFIG["train_classifier"], transform=tt)
    val_ds = datasets.ImageFolder(CONFIG["val_classifier"], transform=tv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


class MiniCNN(nn.Module):
    def __init__(self, num_classes=3, img_size=128, filters=(16, 32, 64), dense=128, dropout=0.3):
        super().__init__()
        c1, c2, c3 = filters
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        feat_dim = c3 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, dense), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def fit(model, train_loader, val_loader, cfg: TrainConfig, verbose=True):
    device = cfg.device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    best_state = None
    wait = 0

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        if verbose:
            print(f"Epoch {epoch + 1}/{cfg.epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                if verbose:
                    print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return best_val_acc


def search():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial: optuna.Trial):
        filters1 = trial.suggest_categorical("filters1", [16, 32])
        filters2 = trial.suggest_categorical("filters2", [32, 64])
        filters3 = trial.suggest_categorical("filters3", [64, 128])
        dense = trial.suggest_categorical("dense", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch = trial.suggest_categorical("batch", [32, 64, 128])

        train_loader, val_loader, _ = get_loaders(batch, num_workers=4)
        cfg = TrainConfig(
            epochs=5, batch_size=batch,
            lr=lr, filters=(filters1, filters2, filters3),
            dense=dense, dropout=dropout, patience=3,
            device=device
        )
        model = MiniCNN(num_classes=3,
                        filters=cfg.filters, dense=cfg.dense, dropout=cfg.dropout)
        val_acc = fit(model, train_loader, val_loader, cfg, verbose=False)
        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, 20)
    print("Best params:", study.best_params)
    print("Best val acc:", study.best_value)
    return study.best_params


def final_train_and_save(params, batch, lr, patience):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, classes = get_loaders(batch, num_workers=4)
    cfg = TrainConfig(
        batch_size=batch, lr=lr,
        filters=(params["filters1"], params["filters2"], params["filters3"]),
        dense=params["dense"], dropout=params["dropout"], patience=patience,
        device=device
    )
    model = MiniCNN(num_classes=len(classes),
                    filters=cfg.filters, dense=cfg.dense, dropout=cfg.dropout)
    best_val_acc = fit(model, train_loader, val_loader, cfg, verbose=True)

    os.makedirs(os.path.dirname(CONFIG["checkpoint_classifier_dir"]), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "img_size": 128,
        "filters": cfg.filters,
        "dense": cfg.dense,
        "dropout": cfg.dropout
    }, CONFIG["checkpoint_classifier_dir"])
    print(f"Saved model to {CONFIG["checkpoint_classifier_dir"]} (best val acc {best_val_acc:.4f})")


if __name__ == "__main__":
    best = search()
    final_train_and_save(
        params=best,
        batch=best.get("batch", 64),
        lr=best.get("lr", 1e-3),
        patience=5
    )
