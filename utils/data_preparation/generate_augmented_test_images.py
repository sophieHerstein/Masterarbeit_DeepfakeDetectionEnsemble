"""Bildaugmentierungen ausführen für die Tests"""
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.config import CONFIG

jpeg_quality = 50
gaussian_noise_stddev = 25
scaling_factor = 0.5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    print("\nApplying JPEG compression...")
    with open("temp.jpg", "wb") as f:
        image.save(f, "JPEG", quality=quality)
    return Image.open("temp.jpg")


def apply_gaussian_noise(image: Image.Image, stddev: float) -> Image.Image:
    print("\nApplying Gaussian noise...")
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, stddev, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_scaling(image: Image.Image, factor: float) -> Image.Image:
    print("\nApplying scaling...")
    original_size = image.size
    scaled_size = (int(original_size[0] * factor), int(original_size[1] * factor))
    scaled_down = image.resize(scaled_size, Image.BICUBIC)
    scaled_up = scaled_down.resize(original_size, Image.BICUBIC)
    return scaled_up


def process_images(original_root, output_root_jpeg, output_root_noisy, output_root_scaled):
    for root, _, files in os.walk(original_root):
        for file in tqdm(files, desc="Verarbeite Bilder"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            print("\nProcessing file:", file)
            in_path = os.path.join(root, file)
            rel_path = os.path.relpath(in_path, original_root)
            out_dir_jpeg = os.path.join(output_root_jpeg, os.path.dirname(rel_path))
            out_dir_noisy = os.path.join(output_root_noisy, os.path.dirname(rel_path))
            out_dir_scaled = os.path.join(output_root_scaled, os.path.dirname(rel_path))
            os.makedirs(out_dir_jpeg, exist_ok=True)
            os.makedirs(out_dir_noisy, exist_ok=True)
            os.makedirs(out_dir_scaled, exist_ok=True)

            try:
                image = Image.open(in_path).convert("RGB")

                base_name = os.path.splitext(file)[0]

                jpeg_image = apply_jpeg_compression(image, jpeg_quality)
                jpeg_image.save(os.path.join(out_dir_jpeg, f"{base_name}_jpeg.jpg"))

                noisy_image = apply_gaussian_noise(image, gaussian_noise_stddev)
                noisy_image.save(os.path.join(out_dir_noisy, f"{base_name}_noisy.jpg"))

                scaled_image = apply_scaling(image, scaling_factor)
                scaled_image.save(os.path.join(out_dir_scaled, f"{base_name}_scaled.jpg"))

            except Exception as e:
                print(f"Fehler bei Datei {in_path}: {e}")


if __name__ == "__main__":
    original_known_root = os.path.join(PROJECT_ROOT, CONFIG['known_test_dir'])
    output_known_root_jpeg = os.path.join(PROJECT_ROOT, CONFIG['known_test_jpeg_dir'])
    output_known_root_noisy = os.path.join(PROJECT_ROOT, CONFIG['known_test_noisy_dir'])
    output_known_root_scaled = os.path.join(PROJECT_ROOT, CONFIG['known_test_scaled_dir'])
    process_images(original_known_root, output_known_root_jpeg, output_known_root_noisy, output_known_root_scaled)
    original_unknown_root = os.path.join(PROJECT_ROOT, CONFIG['unknown_test_dir'])
    output_unknown_root_jpeg = os.path.join(PROJECT_ROOT, CONFIG['unknown_test_jpeg_dir'])
    output_unknown_root_noisy = os.path.join(PROJECT_ROOT, CONFIG['unknown_test_noisy_dir'])
    output_unknown_root_scaled = os.path.join(PROJECT_ROOT, CONFIG['unknown_test_scaled_dir'])
    process_images(original_unknown_root, output_unknown_root_jpeg, output_unknown_root_noisy,
                   output_unknown_root_scaled)
