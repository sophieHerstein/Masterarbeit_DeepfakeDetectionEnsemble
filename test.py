"""Testpipeline"""
import csv
import os
import time

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ensemble import Ensemble
from utils.config import CONFIG, TEST_VARIANTEN, ALL_MODELS, ENSEMBLE_VARIANTEN
from utils.ensemble.model_loader import get_model


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path


def get_model_size(path):
    return round(os.path.getsize(path) / (1024 ** 2), 2)


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_model(model_name, config, test_dir, meta_train=False):
    print(f"Starte Evaluation für Modell: {model_name} ({test_dir})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerät: {device}")

    transform = transforms.Compose([
        transforms.Resize(int(config["image_size"] * 1.1)),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = ImageFolderWithPaths(config[test_dir], transform=transform)

    if len(dataset) == 0:
        print(f"Keine Bilder gefunden in: {config[test_dir]}")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    if model_name in ENSEMBLE_VARIANTEN:
        if meta_train:
            log_csv_path = os.path.join(
                "logs", "meta_train",
                f"{model_name}_{test_dir}_details.csv"
            )
        else:
            log_csv_path = os.path.join(
                "logs", "test", "ensemble",
                f"{model_name}_{test_dir}_details.csv"
            )
        weighted = model_name.startswith("weighted")
        meta = model_name == "unweighted_meta_classifier_ensemble" or model_name == "weighted_meta_classifier_ensemble" or model_name == "unweighted_meta_classifier_ensemble_diverse" or model_name == "weighted_meta_classifier_ensemble_diverse" or model_name == "not_specialized_meta_classifier_ensemble"
        diverse = model_name.endswith("_diverse")
        specialized = not model_name.startswith("not_specialized")
        ensemble = Ensemble(weighted=weighted, meta=meta, diverse=diverse, log_csv_path=log_csv_path,
                            specialized=specialized)
        model = ensemble
    else:
        model = get_model(model_name)
        checkpoint_path = os.path.join(config["checkpoint_dir"], "single_models", f"{model_name}_finetuned.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

    y_true, y_pred, y_prob = [], [], []
    total_time, num_images = 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"{model_name} - {test_dir}", unit="batch")

        for inputs, labels, paths in pbar:
            if model_name in ENSEMBLE_VARIANTEN:
                start_time = time.time()

                for i, path in enumerate(paths):
                    pred, prob = model.predict(path, label=labels[i].item(), verbose=False)
                    y_true.append(labels[i].item())
                    y_pred.append(pred)
                    y_prob.append(prob)

                total_time += time.time() - start_time
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

            pbar.set_postfix({
                "images": num_images,
                "avg_time/img": f"{(total_time / max(1, num_images)):.4f}s"
            })
    if not meta_train:
        avg_time_per_image = total_time / num_images if num_images > 0 else 0

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
                    "Avg-Time/Bild (s)"
                ])

            writer.writerow([
                model_name, test_dir,
                f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}",
                tp, tn, fp, fn,
                f"{avg_time_per_image:.6f}"
            ])

    print(f"Evaluation für {model_name} abgeschlossen.\n\n")


if __name__ == "__main__":
    for name in ALL_MODELS:
        for testdir in TEST_VARIANTEN:
            evaluate_model(name, CONFIG, testdir)
        for testdir in ["meta_train_dir", "meta_train_jpeg_dir", "meta_train_noisy_dir", "meta_train_scaled_dir"]:
            evaluate_model(name, CONFIG, testdir, True)
