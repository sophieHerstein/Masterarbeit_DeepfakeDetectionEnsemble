# TODO: an Masterarbeit anpassen
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import csv
import logging
import time
from model_loader import get_model, MODEL_NAMES, CONFIG
from PIL import Image
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import pandas as pd
import random


def setup_logger(name, log_dir, variante, augmentierung):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    out = os.path.join(log_dir, 'test', variante, augmentierung, f"{name}.log")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = logging.FileHandler(out)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_model_size(path):
    return round(os.path.getsize(path) / (1024 ** 2), 2)  # in MB

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate_model(model_name, config, variante):
    logger = setup_logger(model_name, config["log_dir"], variante, config["variant"])
    logger.info(f"Starte Evaluation für Modell: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Gerät: {device}")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = datasets.ImageFolder(config["test_dir"], transform=transform)
    if len(dataset) == 0:
        logger.warning(f"Keine Bilder gefunden in: {config['test_dir']}")
        return
    logger.info(f"Testbilder: {len(dataset)} Bilder aus {config['test_dir']}")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    if not os.path.exists(config["checkpoint_dir"]):
        logger.warning(f"Checkpoint nicht gefunden: {config['checkpoint_dir']}")
        return

    model = get_model(model_name, config["num_classes"], pretrained=False)
    if variante == "celebdf_train_ff_test":
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"{model_name}_celebdf_only_finetuned.pth")
    else:
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"{model_name}_{variante}_finetuned.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    cam_dir = os.path.join(config["gradcam"], variante, model_name, config["variant"])
    os.makedirs(cam_dir, exist_ok=True)

    class_to_idx = loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    logger.info(f"Label-Mapping class_to_idx: {class_to_idx}")
    logger.info(f"Label-Mapping idx_to_class: {idx_to_class}")
    logger.info(f"Label-Mapping: 0 = {idx_to_class[0]}, 1 = {idx_to_class[1]}")
    y_true, y_pred, all_paths = [], [], []
    total_time = 0
    num_images = 0
    image_index = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(inputs)
            total_time += time.time() - start_time
            num_images += inputs.size(0)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch_paths = [loader.dataset.samples[i][0] for i in range(image_index, image_index + len(labels))]
            all_paths.extend(batch_paths)
            image_index += len(labels)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # in MB
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2  # in MB
        logger.info(f"GPU Speicher belegt: {allocated:.2f} MB, reserviert: {reserved:.2f} MB")

    avg_time_per_image = total_time / num_images
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"Durchschnittliche Vorhersagezeit pro Bild: {avg_time_per_image:.4f} Sekunden")

    os.makedirs(os.path.dirname(config["result_csv"]), exist_ok=True)
    write_header = not os.path.exists(config["result_csv"])
    with open(config["result_csv"], "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Modell", "Variante-Training", "Variante-Test", "Accuracy", "Precision", "Recall", "F1-Score", "TP", "TN", "FP", "FN", "Avg-Time/Bild (s)"])
        writer.writerow([model_name, variante, config["variant"] ,f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", tp, tn, fp, fn, f"{avg_time_per_image:.4f}"])

    model_size = get_model_size(checkpoint_path)
    num_params = get_num_parameters(model)
    logger.info(f"Modellgröße: {model_size} MB")
    logger.info(f"Anzahl der Parameter: {num_params}")

    os.makedirs(os.path.dirname(config["resources_csv"]), exist_ok=True)
    write_header = not os.path.exists(config["resources_csv"])
    with open(config["resources_csv"], "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                    "Modell", "TestVariante", "Variation",
                    "InferenceTime", "GPU_Belegt_MB", "GPU_Reserviert_MB",
                    "Size_MB", "Params"
                ])
        writer.writerow([
            model_name, variante,config["variant"],
            f"{avg_time_per_image:.6f}", f"{allocated:.2f}", f"{reserved:.2f}",
            f"{model_size:.2f}", f"{num_params}"
        ])

    if "vit" in model_name:
        target_layer = model.patch_embed.proj
    elif "swin" in model_name:
        target_layer = model.norm
    elif hasattr(model, 'features'):
        target_layer = model.features[-1]
    elif "efficientnet" in model_name:
        target_layer = model.conv_head
    elif "mobilenet" in model_name:
        target_layer = model.features[-1]
    elif "xception" in model_name or (hasattr(model, "blocks") and isinstance(model.blocks, torch.nn.Sequential)):
        target_layer = model.blocks[-1]
    else:
        target_layer = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]

    logger.info(f"Target-Layer für Grad-CAM: {target_layer}")

    if "vit" in model_name:
        cam = ScoreCAM(model=model, target_layers=[target_layer])
        logger.info("ScoreCAM wird für ViT verwendet.")
    else:
        cam = GradCAM(model=model, target_layers=[target_layer])

    transform_rgb = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"]))
    ])

    cam_df = pd.DataFrame({
        "path": all_paths,
        "true": y_true,
        "pred": y_pred
    })
    real_label = class_to_idx["0_real"] if "0_real" in class_to_idx else class_to_idx["real"]
    fake_label = class_to_idx["1_fake"] if "1_fake" in class_to_idx else class_to_idx["fake"]

    cam_df["outcome"] = cam_df.apply(lambda row:
                                     "TP" if row["true"] == fake_label and row["pred"] == fake_label else
                                     "TN" if row["true"] == real_label and row["pred"] == real_label else
                                     "FP" if row["true"] == real_label and row["pred"] == fake_label else
                                     "FN", axis=1)
    cam_df["class"] = cam_df["true"].map(idx_to_class)

    cam_df["modell"] = model_name
    cam_df["train_variante"] = variante
    cam_df["test_variante"] = config["variant"]

    sampled = cam_df.groupby(["outcome", "class"], group_keys=False).apply(
        lambda g: g.sample(n=min(2, len(g)), random_state=42)
    )

    selection_path = os.path.join(cam_dir, "cam_selection.csv")
    sampled.to_csv(selection_path, index=False)
    logger.info(f"CAM-Auswahl gespeichert: {selection_path}")

    for _, row in sampled.iterrows():
        path = row["path"]
        label = row["true"]
        outcome = row["outcome"]
        true_class = row["class"]
        key = f"{outcome}_{true_class}"

        img = Image.open(path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        input_rgb = np.array(transform_rgb(img)) / 255.0
        input_rgb = np.float32(input_rgb)

        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(input_rgb, grayscale_cam, use_rgb=True)

        out_path = os.path.join(cam_dir, f"{key}_{sampled[sampled['path'] == path].index[0] + 1}.jpg")
        plt.imsave(out_path, visualization)
        sampled.loc[sampled["path"] == path, "gradcam_path"] = out_path

        logger.info(f"Grad-CAM gespeichert: {out_path}")

    logger.info(f"Grad-CAM: {len(sampled)} Visualisierungen gespeichert.")
    plot_single_run(model_name, config["variant"], variante)


if __name__ == "__main__":

    for variante in ["celebdf_only", "celebdf_ff" , "celebdf_train_ff_test"]:
        for name in MODEL_NAMES:

            for variant, folder in [
                ("standard", "test"),
                ("jpeg", "test_jpeg"),
                ("noisy", "test_noisy"),
                ("scaled", "test_scaled")
            ]:
                if variante == "celebdf_train_ff_test":
                    CONFIG["test_dir"] = f"data/celebdf_ff/{folder}"
                else:
                    CONFIG["test_dir"] = f"data/{variante}/{folder}"
                CONFIG["result_csv"] = f"results/test/{variante}/{name}/{variant}/{name}_{variant}_{variante}_results.csv"
                CONFIG["variant"] = variant
                evaluate_model(name, CONFIG, variante)

            plot_model_overview(name, variante)
