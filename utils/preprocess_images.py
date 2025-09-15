import os
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np

from utils.config import CONFIG, CATEGORIES  # falls genutzt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _open_gray(image_or_path):
    with Image.open(image_or_path) as img:
        return img.convert("L")

def get_grayscale(image_path):
    print("[INFO] Getting grayscale image...")
    return _open_gray(image_path)

def get_frequency_spectrum(image_path):
    print("[INFO] Getting frequency spectrum...")
    with Image.open(image_path) as img:
        img = img.convert("L")
        img_array = np.array(img)

    # 2D Fourier-Transformation
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)  # Nullfrequenzen in die Mitte

    # Betrag (Magnitude) und logarithmische Skalierung
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Auf [0,255] normalisieren
    magnitude_spectrum = (magnitude_spectrum / np.max(magnitude_spectrum) * 255).astype(np.uint8)

    return Image.fromarray(magnitude_spectrum)

def get_edges(image_path):
    print("[INFO] Getting edges...")
    with Image.open(image_path) as img:
        img = img.convert("L")
        return img.filter(ImageFilter.FIND_EDGES)

def process_images(input_root, output_root_grayscale, output_root_edges, output_root_frequency):
    for root, _, files in os.walk(input_root):
        for file in tqdm(files, desc="Verarbeite Bilder"):
            print("\nProcessing file:", file)
            in_path = os.path.join(root, file)
            rel_path = os.path.relpath(in_path, input_root)

            out_dir_grayscale = os.path.join(output_root_grayscale, os.path.dirname(rel_path))
            out_dir_edges = os.path.join(output_root_edges, os.path.dirname(rel_path))
            out_dir_frequency = os.path.join(output_root_frequency, os.path.dirname(rel_path))
            os.makedirs(out_dir_grayscale, exist_ok=True)
            os.makedirs(out_dir_edges, exist_ok=True)
            os.makedirs(out_dir_frequency, exist_ok=True)

            try:
                base_name, _ = os.path.splitext(file)

                grayscale_image = get_grayscale(in_path)
                grayscale_image.save(os.path.join(out_dir_grayscale, f"{base_name}_grayscale.jpg"))

                edges_image = get_edges(in_path)
                edges_image.save(os.path.join(out_dir_edges, f"{base_name}_edges.jpg"))

                frequency_image = get_frequency_spectrum(in_path)
                frequency_image.save(os.path.join(out_dir_frequency, f"{base_name}_frequency.jpg"))

            except Exception as e:
                print(f"Fehler bei Datei {in_path}: {e}")

if __name__ == "__main__":
    for p in ["building_train_dir", "landscape_train_dir", "human_train_dir"]:
        original_root = os.path.join(PROJECT_ROOT, CONFIG[p])
        output_root_gray = os.path.join(PROJECT_ROOT, CONFIG['grayscale_train_dir'])
        output_root_edges = os.path.join(PROJECT_ROOT, CONFIG['edges_train_dir'])
        output_root_freq = os.path.join(PROJECT_ROOT, CONFIG['frequency_train_dir'])
        process_images(original_root, output_root_gray, output_root_edges, output_root_freq)

    for p in ["building_val_dir", "landscape_val_dir", "human_val_dir"]:
        original_root = os.path.join(PROJECT_ROOT, CONFIG[p])
        output_root_gray = os.path.join(PROJECT_ROOT, CONFIG['grayscale_val_dir'])
        output_root_edges = os.path.join(PROJECT_ROOT, CONFIG['edges_val_dir'])
        output_root_freq = os.path.join(PROJECT_ROOT, CONFIG['frequency_val_dir'])
        process_images(original_root, output_root_gray, output_root_edges, output_root_freq)
