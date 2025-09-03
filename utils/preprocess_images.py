import os
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np

from utils.config import CONFIG, CATEGORIES


def get_grayscale(image_to_be_processed):
    return Image.open(image_to_be_processed).convert('L')

def get_frequency_spectrum(image_to_be_processed):
    # Bild öffnen und in Graustufen umwandeln
    img = Image.open(image_to_be_processed).convert("L")
    img_array = np.array(img)

    # 2D Fourier-Transformation
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)  # Nullfrequenzen in die Mitte verschieben

    # Betrag (Magnitude) und logarithmische Skalierung
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Auf [0,255] normalisieren für Anzeige als Bild
    magnitude_spectrum = (magnitude_spectrum / np.max(magnitude_spectrum) * 255).astype(np.uint8)

    return Image.fromarray(magnitude_spectrum)

def get_edges(image_to_be_processed):
    img = Image.open(image_to_be_processed)
    img = img.convert("L")
    return img.filter(ImageFilter.FIND_EDGES)

def process_images(input_roots, output_root):
    for input_root in input_roots:
        for root, _, files in os.walk(input_root):
            for file in tqdm(files, desc="Verarbeite Bilder"):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                print("\nProcessing file:", file)
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_root)
                out_dir = os.path.join(output_root, os.path.dirname(rel_path))
                os.makedirs(out_dir, exist_ok=True)

                try:
                    image = Image.open(in_path).convert("RGB")

                    base_name = os.path.splitext(file)[0]

                    grayscale_image = get_grayscale(image)
                    grayscale_image.save(os.path.join(out_dir, 'grayscale', f"{base_name}_grayscale.jpg"))

                    edges_image = get_edges(image)
                    edges_image.save(os.path.join(out_dir, 'edges', f"{base_name}_edges.jpg"))

                    frequence_image = get_frequency_spectrum(image)
                    frequence_image.save(os.path.join(out_dir, 'frequence', f"{base_name}_scaled.jpg"))

                except Exception as e:
                    print(f"Fehler bei Datei {in_path}: {e}")

#TODO: Pfade setzen
#ggf. vorher datensplit
if __name__ == "__main__":
    for category in CATEGORIES:
        known_input_roots = ""
        known_output_root = ""
        process_images(known_input_roots, known_output_root)
        unknown_input_roots = ""
        unknown_output_root = ""
        process_images(unknown_input_roots, unknown_output_root)
