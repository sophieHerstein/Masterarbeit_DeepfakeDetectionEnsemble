import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from utils.model_loader import get_model
from utils.config import CONFIG, TRAININGS_VARIANTEN, MODELS
import os
import time
import csv
import itertools

def train_model(config, model_name, variante, grid_search=False):
    device = torch.device("cuda")
    print(f"Starte Training auf Gerät: {device}")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dataset = datasets.ImageFolder(os.path.join(config["train_dir"], variante), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(config["val_dir"], variante), transform=transform)

    print(f"Train class_to_idx: {train_dataset.class_to_idx}")
    print(f"Val class_to_idx: {val_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = get_model(model_name)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_acc = 0.0
    no_improve_epochs = 0
    start_time = time.time()

    if grid_search:
        epoch_log_path = os.path.join(config["train_log_path"], variante, "gridsearch.csv")
        os.makedirs(os.path.dirname(epoch_log_path), exist_ok=True)
    else:
        epoch_log_path = os.path.join(config["train_log_path"], variante, f"{model_name}.csv")
        os.makedirs(os.path.dirname(epoch_log_path), exist_ok=True)

    with open(epoch_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoche", "Train-Acc", "Val-Acc", "Loss", "Epoch-Zeit (s)"])

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        epoch_time = time.time() - epoch_start

        with open(epoch_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_acc:.2f}", f"{val_acc:.2f}", f"{running_loss:.4f}", f"{epoch_time:.2f}"])

        eta = (time.time() - start_time) / (epoch + 1) * (config["epochs"] - epoch - 1)
        print(f"Epoche {epoch + 1}/{config['epochs']} - Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, ETA: {eta:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            os.makedirs(os.path.join(config["checkpoint_dir"], variante), exist_ok=True)
            checkpoint_path = os.path.join(config["checkpoint_dir"], variante, f"{model_name}_finetuned.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Modell gespeichert unter: {checkpoint_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config["early_stopping_patience"]:
                print(f"Early stopping ausgelöst nach {epoch + 1} Epochen.")
                break

    if not grid_search:
        log_dir = os.path.dirname(config[f"train_{variante}_result_log"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_exists = os.path.isfile(config[f"train_{variante}_result_log"])
        with open(config[f"train_{variante}_result_log"], "a", newline="") as logfile:
            writer = csv.writer(logfile)
            if not log_exists:
                writer.writerow(["Modell", "Train-Acc", "Val-Acc", "Loss", "Trainzeit (s)", "last epoch"])
            writer.writerow([
                model_name,
                f"{train_acc:.2f}",
                f"{val_acc:.2f}",
                f"{running_loss:.4f}",
                f"{int(time.time() - start_time)}",
                epoch+1
            ])

    return val_acc

def parameter_grid_search(config, grid, variante, test_model):
    print("Starte Parameter-Test mit Grid Search")
    best_acc = 0.0
    best_config = {}

    for lr, bs in itertools.product(grid["learning_rate"], grid["batch_size"]):
        config["learning_rate"] = lr
        config["batch_size"] = bs

        print(f"Test: LR={lr}, Batch={bs}")
        acc = train_model(config, test_model, variante, True)
        print(f"Ergebnis: Val Acc = {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_config = {"learning_rate": lr, "batch_size": bs}

    print(f"Beste Parameterkombination: LR={best_config['learning_rate']} | Batch={best_config['batch_size']} | Acc={best_acc:.2f}%")
    return best_config


if __name__ == '__main__':
    # Parameter definieren
    param_grid = {
        "learning_rate": [1e-4, 5e-5],
        "batch_size": [16, 32]
    }
    for variante in TRAININGS_VARIANTEN:
        for m in MODELS:
            CONFIG["epochs"] = 3
            optimal_params = parameter_grid_search(CONFIG, param_grid, variante, m)
            CONFIG["learning_rate"] = optimal_params["learning_rate"]
            CONFIG["batch_size"] = optimal_params["batch_size"]
            CONFIG["epochs"] = 40
            print(f"\n Starte Training für Modell: {m}")
            train_model(CONFIG, m, variante)
            print(f"Training für Modell {m} abgeschlossen.\n")
