# TODO: an Masterarbeit anpassen

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from model_loader import CONFIG, get_model, MODEL_NAMES
import os
import time
import csv
import itertools
import logging


def setup_logger(model_name, log_dir, variante):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    out = os.path.join(log_dir, 'train', variante, f"{model_name}.log")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = logging.FileHandler(out)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train_model(config, variante, grid_search=False):
    logger = setup_logger(config["model_name"], config.get("log_dir", "logs"), variante)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starte Training auf Gerät: {device}")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dataset = datasets.ImageFolder(config["train_dir"], transform=transform)
    val_dataset = datasets.ImageFolder(config["val_dir"], transform=transform)

    logger.info(f"Train class_to_idx: {train_dataset.class_to_idx}")
    logger.info(f"Val class_to_idx: {val_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = get_model(config["model_name"], config["num_classes"], config["use_pretrained"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_acc = 0.0
    no_improve_epochs = 0
    start_time = time.time()

    if grid_search:
        epoch_log_path = os.path.join(config["log_dir"], 'train', variante, "gridsearch.csv")
        os.makedirs(os.path.dirname(epoch_log_path), exist_ok=True)
    else:
        epoch_log_path = os.path.join(config["log_dir"], 'train', variante, f"{config['model_name']}.csv")
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
        logger.info(f"Epoche {epoch + 1}/{config['epochs']} - Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, ETA: {eta:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config["early_stopping_patience"]:
                logger.warning(f"Early stopping ausgelöst nach {epoch + 1} Epochen.")
                break

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{config['model_name']}_{variante}_finetuned.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Modell gespeichert unter: {checkpoint_path}")

    if not grid_search:
        log_dir = os.path.dirname(config["train_result"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_exists = os.path.isfile(config["train_result"])
        with open(config["train_result"], "a", newline="") as logfile:
            writer = csv.writer(logfile)
            if not log_exists:
                writer.writerow(["Modell", "Variante", "Train-Acc", "Val-Acc", "Loss", "Trainzeit (s)", "last epoch"])
            writer.writerow([
                config["model_name"],
                variante,
                f"{train_acc:.2f}",
                f"{val_acc:.2f}",
                f"{running_loss:.4f}",
                f"{int(time.time() - start_time)}",
                epoch+1
            ])

    return val_acc

def parameter_grid_search(config, param_grid, variante, test_model="mobilenet_v2",):
    logger = setup_logger("grid_search", config.get("log_dir", "logs"), variante)
    logger.info("Starte Parameter-Test mit Grid Search")
    best_acc = 0.0
    best_config = {}

    for lr, bs in itertools.product(param_grid["learning_rate"], param_grid["batch_size"]):
        config["model_name"] = test_model
        config["learning_rate"] = lr
        config["batch_size"] = bs
        config["epochs"] = 3

        logger.info(f"Test: LR={lr}, Batch={bs}")
        acc = train_model(config, variante, True)
        logger.info(f"Ergebnis: Val Acc = {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_config = {"learning_rate": lr, "batch_size": bs}

    logger.info(f"Beste Parameterkombination: LR={best_config['learning_rate']} | Batch={best_config['batch_size']} | Acc={best_acc:.2f}%")
    return best_config

# Parameter definieren
param_grid = {
    "learning_rate": [1e-4, 5e-5],
    "batch_size": [16, 32]
}

optimal_params = parameter_grid_search(CONFIG, param_grid, "celebdf_only")
CONFIG["learning_rate"] = optimal_params["learning_rate"]
CONFIG["batch_size"] = optimal_params["batch_size"]
CONFIG["epochs"] = 20

for model_name in MODEL_NAMES:
    print(f"\n Starte Training für Modell: {model_name}")
    CONFIG["model_name"] = model_name
    train_model(CONFIG, "celebdf_only")


CONFIG["train_dir"] = "data/celebdf_ff/train"
CONFIG["val_dir"] = "data/celebdf_ff/test"

optimal_params = parameter_grid_search(CONFIG, param_grid, "celebdf_ff")
CONFIG["learning_rate"] = optimal_params["learning_rate"]
CONFIG["batch_size"] = optimal_params["batch_size"]
CONFIG["epochs"] = 20

for model_name in MODEL_NAMES:
    print(f"\n Starte Training für Modell: {model_name}")
    CONFIG["model_name"] = model_name
    train_model(CONFIG, "celebdf_ff")
