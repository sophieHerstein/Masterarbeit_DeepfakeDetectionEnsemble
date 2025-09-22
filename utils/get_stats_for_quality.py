"""
Skript zur Berechnung von Qualitäts-Statistiken (mu, sigma) für Deepfake-Ensemble
---------------------------------------------------------------------------------
Dieses Skript durchläuft alle Bilder in einem angegebenen Ordner,
berechnet für jedes Bild die drei Qualitätsmerkmale:
    - Kantenqualität (Laplacian-Varianz)
    - Frequenzqualität (Hochfrequenz-Anteil)
    - Graustufenqualität (Dynamikumfang * (1 - Clipping))
und speichert anschließend den Mittelwert (mu) und die Standardabweichung (sigma)
für jedes Merkmal in einer JSON-Datei.

Diese Werte werden später bei der Normierung benötigt, um die Rohwerte
vergleichbar zu machen und mit einer Sigmoid-Funktion auf [0,1] abzubilden.
"""

import cv2
import numpy as np
import glob
import json
import os
from pathlib import Path

from utils.config import CONFIG

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------
# 1) ROHFUNKTIONEN FÜR DIE QUALITÄTSMERKMALE
# --------------------------------------------------------

def laplacian_var(gray):
    """Kantenqualität: Schärfemaß (Varianz des Laplacians).
    -> Unscharfe Bilder haben kleine Werte, scharfe große Werte.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def highfreq_ratio(gray, frac=0.25):
    """Frequenzqualität: Anteil der Hochfrequenzen im Fourier-Spektrum.
    -> Gute Qualität = genügend Hochfrequenzanteile vorhanden.
    -> Zu viele Verluste durch Kompression/Blur senken diesen Wert.
    """
    g = gray.astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)

    H, W = g.shape
    yy, xx = np.ogrid[-H//2:H//2, -W//2:W//2]
    r = np.sqrt(yy**2 + xx**2) / (0.5 * min(H, W))  # normierter Radius

    hf = mag[r >= (1-frac)].sum()    # Energie in Hochfrequenzen
    tot = mag.sum() + 1e-8           # Gesamtenergie
    return hf / tot


def dynamic_range(gray):
    """Graustufenqualität (Teil 1): Dynamikumfang zwischen 5. und 95. Perzentil.
    -> Hoher Wert = guter Kontrastumfang.
    """
    p5, p95 = np.percentile(gray, [5, 95])
    return p95 - p5


def clipping_fraction(gray):
    """Graustufenqualität (Teil 2): Anteil unter- oder überbelichteter Pixel.
    -> Je höher dieser Wert, desto schlechter (zu viele Pixel bei 0 oder 255).
    """
    return ((gray < 5).sum() + (gray > 250).sum()) / gray.size


# --------------------------------------------------------
# 2) HAUPTFUNKTION ZUR STATISTIKBERECHNUNG
# --------------------------------------------------------

def compute_stats(image_paths):
    """Berechnet mu und sigma für die drei Qualitätsmerkmale über alle Bilder."""

    edge_vals, freq_vals, gray_vals = [], [], []

    for path in image_paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue  # Falls ein Bild fehlerhaft ist

        # --- Kantenqualität (Laplacian-Varianz) ---
        edge_vals.append(laplacian_var(gray))

        # --- Frequenzqualität (HF-Anteil) ---
        freq_vals.append(highfreq_ratio(gray))

        # --- Graustufenqualität (Dynamikumfang * (1 - Clipping)) ---
        gval = dynamic_range(gray) * (1 - clipping_fraction(gray))
        gray_vals.append(gval)

    # mu = Mittelwert, sigma = Standardabweichung
    stats = {
        'edge': (float(np.mean(edge_vals)), float(np.std(edge_vals)), 2.0),
        'freq': (float(np.mean(freq_vals)), float(np.std(freq_vals)), 2.0),
        'gray': (float(np.mean(gray_vals)), float(np.std(gray_vals)), 2.0),
    }
    return stats


# --------------------------------------------------------
# 3) HAUPTTEIL: DATENSATZ DURCHLAUFEN & STATS SPEICHERN
# --------------------------------------------------------

if __name__ == "__main__":
    image_folder = Path(os.path.join(PROJECT_ROOT, CONFIG['train_classifier']))

    # Rekursiv alle Bilder mit den angegebenen Endungen finden
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(image_folder.rglob(ext))

    print(f"Gefundene Bilder: {len(image_paths)}")

    # Stats berechnen
    stats = compute_stats(image_paths)

    print("Berechnete Statistiken:")
    for k, v in stats.items():
        mu, sigma, tau = v
        print(f"{k}: mu={mu:.3f}, sigma={sigma:.3f}, tau={tau}")

    json_path = os.path.join(PROJECT_ROOT, CONFIG['quality_stats_path'])
    quality_dir = os.path.dirname(json_path)
    if quality_dir:
        os.makedirs(quality_dir, exist_ok=True)
    # Als JSON speichern
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nStatistiken wurden in '{CONFIG['quality_stats_path']}' gespeichert.")
