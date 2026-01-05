import json
import os
from pathlib import Path

import cv2
import numpy as np

from utils.config import CONFIG

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def highfreq_ratio(gray, frac=0.25):
    g = gray.astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)

    H, W = g.shape
    yy, xx = np.ogrid[-H // 2:H // 2, -W // 2:W // 2]
    r = np.sqrt(yy ** 2 + xx ** 2) / (0.5 * min(H, W))

    hf = mag[r >= (1 - frac)].sum()
    tot = mag.sum() + 1e-8
    return hf / tot


def dynamic_range(gray):
    p5, p95 = np.percentile(gray, [5, 95])
    return p95 - p5


def clipping_fraction(gray):
    return ((gray < 5).sum() + (gray > 250).sum()) / gray.size


def compute_stats(image_paths):
    edge_vals, freq_vals, gray_vals = [], [], []

    for path in image_paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        edge_vals.append(laplacian_var(gray))

        freq_vals.append(highfreq_ratio(gray))

        gval = dynamic_range(gray) * (1 - clipping_fraction(gray))
        gray_vals.append(gval)

    stats = {
        'edge': (float(np.mean(edge_vals)), float(np.std(edge_vals)), 2.0),
        'freq': (float(np.mean(freq_vals)), float(np.std(freq_vals)), 2.0),
        'gray': (float(np.mean(gray_vals)), float(np.std(gray_vals)), 2.0),
    }
    return stats


if __name__ == "__main__":
    image_folder = Path(os.path.join(PROJECT_ROOT, CONFIG['train_classifier']))

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(image_folder.rglob(ext))

    print(f"Gefundene Bilder: {len(image_paths)}")

    stats = compute_stats(image_paths)

    print("Berechnete Statistiken:")
    for k, v in stats.items():
        mu, sigma, tau = v
        print(f"{k}: mu={mu:.3f}, sigma={sigma:.3f}, tau={tau}")

    json_path = os.path.join(PROJECT_ROOT, CONFIG['quality_stats_path'])
    quality_dir = os.path.dirname(json_path)
    if quality_dir:
        os.makedirs(quality_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nStatistiken wurden in '{CONFIG['quality_stats_path']}' gespeichert.")
