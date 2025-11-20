import json

import numpy as np
from PIL import Image, ImageFilter
from scipy.special import expit
import cv2
from torchvision import transforms
import torch
from utils.model_loader import get_model
from classifier import MiniCNN
from utils.config import CONFIG
import os
import csv
import pickle

class Ensemble:

    def __init__(self, weighted, meta, log_csv_path=None):
        self.weighted = weighted
        self.meta = meta
        if self.meta:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "meta_classifier_for_ensemble.pkl")
            with open(ckpt_path, 'rb') as file:
                self.meta_classifier = pickle.load(file)
        self.models = {
            "grayscale": self._load_model("convnext_small", "grayscaling"),
            "edges": self._load_model("xception71", "edges"),
            "frequency": self._load_model("convnext_small", "frequencies"),
            "human": self._load_model("convnext_small", "human"),
            "building": self._load_model("convnext_small", "building"),
            "landscape": self._load_model("densenet121", "landscape")
        }

        self.transform_gray = transforms.Compose([
                transforms.Resize(int(CONFIG["image_size"] * 1.1)),
                transforms.CenterCrop(CONFIG["image_size"]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.transform = transforms.Compose([
                transforms.Resize(int(CONFIG["image_size"] * 1.1)),
                transforms.CenterCrop(CONFIG["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.log_csv_path = log_csv_path
        if self.log_csv_path is not None:
            os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
            if not os.path.exists(self.log_csv_path):
                with open(self.log_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "img", "label", "prediction", "final_prob",
                        "p_human", "p_landscape", "p_building",
                        "p_edges", "p_frequency", "p_grayscale",
                        "w_human", "w_landscape", "w_building",
                        "w_edges", "w_frequency", "w_grayscale"
                    ])

    def _load_model(self, model_name, variante):
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], variante, f"{model_name}_finetuned.pth")

        model = get_model(model_name)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        return model

    def _is_deepfake_grayscale(self, img):
        image = Image.open(img).convert("L")
        img_tensor = self.transform_gray(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.models['grayscale'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_edges(self, img):

        image = Image.open(img).convert("L").filter(ImageFilter.FIND_EDGES)
        img_tensor = self.transform_gray(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.models['edges'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_frequence(self, img):
        image = Image.open(img).convert("L")

        img_array = np.array(image)

        # 2D Fourier-Transformation
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)  # Nullfrequenzen in die Mitte

        # Betrag (Magnitude) und logarithmische Skalierung
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # Auf [0,255] normalisieren
        magnitude_spectrum = (magnitude_spectrum / np.max(magnitude_spectrum) * 255).astype(np.uint8)

        img_tensor = self.transform_gray(Image.fromarray(magnitude_spectrum)).unsqueeze(0)

        with torch.no_grad():
            logits = self.models['frequency'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_human(self, img):

        image = Image.open(img).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.models['human'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_landscape(self, img):
        image = Image.open(img).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.models['landscape'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_building(self, img):

        image = Image.open(img).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.models['building'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _get_quality_weight(self, img):
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # Beispielhafte Statistikwerte (mu, sigma, tau) – normalerweise aus Datensatz berechnet
        # JSON mit Stats laden (muss vorher mit compute_stats.py erzeugt worden sein)
        with open(CONFIG['quality_stats_path'], "r") as f:
            stats = json.load(f)

        # Qualitätsvektor berechnen
        q = self._quality_vector(gray, stats)

        # Gewichte berechnen
        w = self._ensemble_weights(q)
        return w

    def _get_category_weights(self, img):
        ckpt_path = CONFIG["checkpoint_classifier_dir"]

        # Laden des Checkpoints
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Modell erstellen
        model = MiniCNN(
            num_classes=len(ckpt["classes"]),
            img_size=ckpt["img_size"],
            filters=tuple(ckpt["filters"]),
            dense=ckpt["dense"],
            dropout=ckpt["dropout"]
        )

        # Gewichte laden
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
            transforms.ToTensor(),
        ])

        # Bild laden
        image = cv2.imread(img)[:, :, ::-1]  # BGR -> RGB
        img_tensor = transform(Image.fromarray(image)).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)  # [1,num_classes]
            pred_class = probs.argmax(1).item()
            pred_label = ckpt["classes"][pred_class]

        classes = ckpt["classes"]

        # Batch-Dimension wegnehmen
        return {cls: float(p) for cls, p in zip(classes, probs[0].tolist())}

    # --------------------------------------------------------
    # 1) QUALITÄTSMERKMALE BERECHNEN
    # --------------------------------------------------------
    # Jede Funktion liefert einen Rohwert auf der nativen Skala der Metrik
    # (z. B. Laplacian-Varianz kann sehr groß sein, Dynamikumfang max. 255).

    def _laplacian_var(self, gray):
        """Kantenqualität: Schärfemaß (Varianz des Laplacians)"""
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _highfreq_ratio(self, gray, frac=0.25):
        """Frequenzqualität: Anteil der Hochfrequenzen im Fourier-Spektrum"""
        g = gray.astype(np.float32) / 255.0
        F = np.fft.fftshift(np.fft.fft2(g))
        mag = np.abs(F)
        H, W = g.shape
        yy, xx = np.ogrid[-H // 2:H // 2, -W // 2:W // 2]
        r = np.sqrt(yy ** 2 + xx ** 2) / (0.5 * min(H, W))  # Normierter Radius
        hf = mag[r >= (1 - frac)].sum()
        tot = mag.sum() + 1e-8
        return hf / tot  # Verhältnis HF zu Gesamtenergie

    def _dynamic_range(self, gray):
        """Graustufenqualität: Dynamikumfang (95. - 5. Perzentil)"""
        p5, p95 = np.percentile(gray, [5, 95])
        return p95 - p5

    def _clipping_fraction(self, gray):
        """Graustufenqualität: Anteil über- oder unterbelichteter Pixel"""
        return ((gray < 5).sum() + (gray > 250).sum()) / gray.size

    # --------------------------------------------------------
    # 2) NORMIERUNG & MAPPING
    # --------------------------------------------------------
    # Rohwerte sind nicht vergleichbar → deshalb Normierung (z-Score)
    # und Mapping (Sigmoid) auf [0,1].
    # mean/std stammen aus deinem Trainings- oder Validierungssplit.

    def _to_quality(self, raw_value, mu, sigma, tau=2.0):
        """
        Normiert einen Rohwert (z-Score) und mappt ihn per Sigmoid auf [0,1].
        tau = Skalenparameter, steuert wie "weich" die Abbildung ist.
        """
        z = (raw_value - mu) / (sigma + 1e-8)
        return float(expit(z / tau))

    # --------------------------------------------------------
    # 3) QUALITÄTSVEKTOR BILDEN
    # --------------------------------------------------------
    # Liefert (q_edge, q_freq, q_gray), alle Werte in [0,1].

    def _quality_vector(self, gray, stats):
        """
        stats = Dictionary mit (mu, sigma, tau) pro Qualitätsmerkmal,
                berechnet aus deinem Datensatz.
        """
        # Kanten (Schärfe)
        edge_raw = self._laplacian_var(gray)
        q_edge = self._to_quality(edge_raw, *stats['edge'])

        # Frequenz (HF-Anteil)
        freq_raw = self._highfreq_ratio(gray)
        q_freq = self._to_quality(freq_raw, *stats['freq'])

        # Graustufen (Dynamikumfang - Clipping)
        gray_raw = self._dynamic_range(gray) * (1 - self._clipping_fraction(gray))
        q_gray = self._to_quality(gray_raw, *stats['gray'])

        return np.array([q_edge, q_freq, q_gray])

    # --------------------------------------------------------
    # 4) GEWICHTSFORMEL
    # --------------------------------------------------------
    # Macht aus den [0,1]-Qualitätswerten gültige Modellgewichte,
    # die zusammen 1 ergeben und die Modelle gemäß Qualität betonen.

    def _ensemble_weights(self, q, eps=0.02, alpha=2.0):
        """
        q = Qualitätsvektor (zwischen 0 und 1)
        eps = kleiner Offset, damit kein Modell komplett 0 wird
        alpha = Exponent, steuert die "Schärfe" der Gewichtung
        """
        w = (q + eps) ** alpha
        return w / w.sum()  # Normierung: Summe = 1

    def predict(self, img, label=None, verbose: bool = True, log: bool = True):
        # Einzelwahrscheinlichkeiten
        probs = {
            "human": self._is_deepfake_human(img),
            "landscape": self._is_deepfake_landscape(img),
            "building": self._is_deepfake_building(img),
            "frequency": self._is_deepfake_frequence(img),
            "grayscale": self._is_deepfake_grayscale(img),
            "edges": self._is_deepfake_edges(img),
        }

        weights = {k: None for k in probs.keys()}  # default None
        if self.weighted:
            category_weights = self._get_category_weights(img)
            quality_weights = self._get_quality_weight(img)

            deepfake_prob_based_on_category = (
                    quality_weights[0] * probs["edges"] +
                    quality_weights[1] * probs["frequency"] +
                    quality_weights[2] * probs["grayscale"]
            )
            deepfake_prob_based_on_quality = (
                    category_weights.get("human", 0.0) * probs["human"] +
                    category_weights.get("landscape", 0.0) * probs["landscape"] +
                    category_weights.get("building", 0.0) * probs["building"]
            )
            denom = sum(category_weights.values())
            if denom > 0:
                deepfake_prob_based_on_quality /= denom
            else:
                deepfake_prob_based_on_quality = 0.0

            weights = {
                "human": category_weights.get("human", 0.0),
                "landscape": category_weights.get("landscape", 0.0),
                "building": category_weights.get("building", 0.0),
                "edges": quality_weights[0],
                "frequency": quality_weights[1],
                "grayscale": quality_weights[2],
            }
        elif self.meta:

            meta_features = np.array([[
                probs["human"],
                probs["landscape"],
                probs["building"],
                probs["edges"],
                probs["frequency"],
                probs["grayscale"]
            ]])

            predictions = self.meta_classifier.predict(meta_features)
        else:
            deepfake_prob_based_on_category = (
                                                      probs["edges"] + probs["frequency"] + probs["grayscale"]
                                              ) / 3.0
            deepfake_prob_based_on_quality = (
                                                     probs["human"] + probs["landscape"] + probs["building"]
                                             ) / 3.0



        if self.meta:
            prediction = int(predictions[0])
            final_prob = ""
        else:
            # Finale Wahrscheinlichkeit & Entscheidung
            final_prob = (deepfake_prob_based_on_category + deepfake_prob_based_on_quality) / 2
            prediction = int(final_prob > 0.5)

        if verbose:
            mode = "weighted" if self.weighted else "unweighted"
            print(f"Prediction for {img} ({mode}): {final_prob:.3f}")

        # ins CSV loggen
        if log and self.log_csv_path is not None:
            with open(self.log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    img, label if label is not None else "",
                    prediction, f"{final_prob:.4f}",
                    f"{probs['human']:.4f}", f"{probs['landscape']:.4f}", f"{probs['building']:.4f}",
                    f"{probs['edges']:.4f}", f"{probs['frequency']:.4f}", f"{probs['grayscale']:.4f}",
                    f"{weights['human']:.4f}" if weights['human'] is not None else "",
                    f"{weights['landscape']:.4f}" if weights['landscape'] is not None else "",
                    f"{weights['building']:.4f}" if weights['building'] is not None else "",
                    f"{weights['edges']:.4f}" if weights['edges'] is not None else "",
                    f"{weights['frequency']:.4f}" if weights['frequency'] is not None else "",
                    f"{weights['grayscale']:.4f}" if weights['grayscale'] is not None else "",
                ])

        return prediction, final_prob

