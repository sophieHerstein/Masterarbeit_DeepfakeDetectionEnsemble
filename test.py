import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import os
import csv
import time
import numpy as np

from utils.model_loader import get_model
from utils.config import CONFIG, MODELS
from ensemble import Ensemble


def get_model_size(path):
    return round(os.path.getsize(path) / (1024 ** 2), 2)  # MB


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_model(model_name, config,test_dir):
    print(f"Starte Evaluation für Modell: {model_name} ({test_dir})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerät: {device}")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = datasets.ImageFolder(config[test_dir], transform=transform)

    if len(dataset) == 0:
        print(f"Keine Bilder gefunden in: {config[test_dir]}")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Modell laden
    if model_name in ["ensemble", "unweighted_ensemble"]:
        log_csv_path = os.path.join(
            "logs", "test", "ensemble",
            f"{model_name}_{test_dir}_details.csv"
        )
        ensemble = Ensemble(weighted=(model_name == "ensemble"), log_csv_path=log_csv_path)
        model = ensemble
    else:
        model = get_model(model_name)
        checkpoint_path = os.path.join(config["checkpoint_dir"], "single_models", f"{model_name}_finetuned.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        checkpoint_path = checkpoint_path

    # Vorhersagen sammeln
    y_true, y_pred, y_prob = [], [], []
    total_time, num_images = 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            if model_name in ["ensemble", "unweighted_ensemble"]:
                for i, path in enumerate([s[0] for s in loader.dataset.samples[num_images:num_images+len(labels)]]):
                    pred, prob = model.predict(path, label=labels[i].item(), verbose=False)
                    y_true.append(labels[i].item())
                    y_pred.append(pred)
                    y_prob.append(prob)
                num_images += len(labels)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
                start_time = time.time()
                outputs = model(inputs)
                total_time += time.time() - start_time
                num_images += inputs.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

    # Ressourcen
    allocated = reserved = 0.0
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2

    avg_time_per_image = total_time / num_images if num_images > 0 else 0

    # Metriken
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Ergebnisse ins CSV schreiben
    metrics_csv = os.path.join("logs", "test", f"{model_name}_metrics.csv")
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    write_header = not os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Modell", "TestVariante",
                "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
                "TP", "TN", "FP", "FN",
                "Avg-Time/Bild (s)", "GPU_Belegt_MB", "GPU_Reserviert_MB",
                "Size_MB", "Params"
            ])

        model_size = get_model_size(checkpoint_path) if model_name not in ["ensemble", "unweighted_ensemble"] else "-"
        num_params = get_num_parameters(model) if model_name not in ["ensemble", "unweighted_ensemble"] else "-"

        writer.writerow([
            model_name, test_dir,
            f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}",
            tp, tn, fp, fn,
            f"{avg_time_per_image:.6f}", f"{allocated:.2f}", f"{reserved:.2f}",
            model_size, num_params
        ])


if __name__ == "__main__":
    for name in MODELS + ["ensemble", "unweighted_ensemble"]:
        for testdir in [
            "known_test_dir",
            "unknown_test_dir",
            "known_test_jpeg_dir",
            "unknown_test_jpeg_dir",
            "known_test_noisy_dir",
            "unknown_test_noisy_dir",
            "known_test_scaled_dir",
            "unknown_test_scaled_dir"
        ]:
            evaluate_model(name, CONFIG, testdir)
