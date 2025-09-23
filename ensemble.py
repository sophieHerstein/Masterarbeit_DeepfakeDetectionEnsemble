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


class Ensemble:

    def __init__(self):
        self.models = {
            "grayscale": self._load_model("resnet50d", "grayscale"),
            "edges": self._load_model("convnext_small", "edges"),
            "frequency": self._load_model("convnext_small", "frequency"),
            "human": self._load_model("convnext_small", "human"),
            "building": self._load_model("convnext_small", "building"),
            "landscape": self._load_model("resnet50d", "landscape")
        }

        self.transform = transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])


    def _load_model(self, model_name, variante):
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], variante, f"{model_name}_finetuned.pth")

        model = get_model(model_name)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        return model

    def _is_deepfake_grayscale(self, img):
        image = Image.open(img).convert("L")
        img_tensor = self.transform_gray(image).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            logits = self.models['grayscale'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_edges(self, img):

        image = Image.open(img).convert("L").filter(ImageFilter.FIND_EDGES)
        img_tensor = self.transform_gray(image).unsqueeze(0)  # [1,3,H,W]

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

        img_tensor = self.transform_gray(Image.fromarray(magnitude_spectrum)).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            logits = self.models['frequency'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_human(self, img):

        image = Image.open(img)
        img_tensor = self.transform(image).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            logits = self.models['human'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_landscape(self, img):
        image = Image.open(img)
        img_tensor = self.transform(image).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            logits = self.models['landscape'](img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return probs[1].item()


    def _is_deepfake_building(self, img):

        image = Image.open(img)
        img_tensor = self.transform(image).unsqueeze(0)  # [1,3,H,W]

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
        print("Qualitätsvektor:", q)

        # Gewichte berechnen
        w = self._ensemble_weights(q)
        print("Gewichte:", w)
        return w


    def _get_category_weights(self, img):
        ckpt_path = CONFIG["checkpoint_classifier_dir"]

        # Laden des Checkpoints
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Modell mit den gespeicherten Parametern erstellen
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

        # Beispiel: ein einzelnes Bild laden
        image = cv2.imread(img)[:, :, ::-1]  # BGR -> RGB
        img_tensor = transform(Image.fromarray(image)).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_class = probs.argmax(1).item()
            pred_label = ckpt["classes"][pred_class]

        classes = ckpt["classes"]

        return {cls: float(p) for cls, p in zip(classes, probs.tolist())}


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


    def predict(self, img):
        category_weights = self._get_category_weights(img)
        quality_weights = self._get_quality_weight(img)

        is_deepfake_human = self._is_deepfake_human(img)
        is_deepfake_landscape = self._is_deepfake_landscape(img)
        is_deepfake_building = self._is_deepfake_building(img)
        is_deepfake_frequence = self._is_deepfake_frequence(img)
        is_deepfake_grayscale = self._is_deepfake_grayscale(img)
        is_deepfake_edges = self._is_deepfake_edges(img)

        deepfake_prob_based_on_category = quality_weights[0]*is_deepfake_edges + quality_weights[1]*is_deepfake_frequence + quality_weights[2]*is_deepfake_grayscale
        deepfake_prob_based_on_quality = category_weights['human']*is_deepfake_human + category_weights['landscape']*is_deepfake_landscape + category_weights['building']*is_deepfake_building

        deepfake_prob_based_on_quality /= sum(category_weights.values())

        return ((deepfake_prob_based_on_category + deepfake_prob_based_on_quality) / 2 ) > 0.5