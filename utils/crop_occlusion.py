import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# -----------------------------
# Bildveränderungen
# -----------------------------

def random_crop(img):
    h, w, _ = img.shape
    # Crop zwischen 80% und 95% der Größe
    crop_ratio = np.random.uniform(0.80, 0.95)
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    cropped = img[y:y+new_h, x:x+new_w]
    resized = cv2.resize(cropped, (w, h))  # zurück zur Originalgröße
    return resized


def random_occlusion(img):
    h, w, _ = img.shape
    # Rechteck zwischen 10% und 25% der Bildfläche
    occ_w = np.random.randint(int(w*0.10), int(w*0.25))
    occ_h = np.random.randint(int(h*0.10), int(h*0.25))

    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)

    result = img.copy()

    # Farbe: schwarz, dunkelgrau oder hellgrau
    color = random.choice([(0,0,0), (50,50,50), (120,120,120)])

    cv2.rectangle(result, (x, y), (x+occ_w, y+occ_h), color, -1)
    return result


# -----------------------------
# Einen Ordner verarbeiten
# -----------------------------

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder), desc=f"Processing {input_folder}"):
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)

        if not os.path.isfile(in_path):
            continue

        img = cv2.imread(in_path)
        if img is None:
            continue

        # 50/50 Zufall: Cropping oder Occlusion
        if np.random.rand() < 0.5:
            img = random_crop(img)
        else:
            img = random_occlusion(img)

        cv2.imwrite(out_path, img)


# -----------------------------
# Hauptfunktion
# -----------------------------

def process_testset(base_input, base_output):
    for cls in ["0_real", "1_fake"]:
        input_path = os.path.join(base_input, cls)
        output_path = os.path.join(base_output, cls)
        process_folder(input_path, output_path)


if __name__ == "__main__":
    known_input = "data/test/known_test"
    unknown_input = "data/test/unknown_test"

    known_output = "known_test_crop_occlusion"
    unknown_output = "unknown_test_crop_occlusion"

    process_testset(known_input, known_output)
    process_testset(unknown_input, unknown_output)

    print("Cropping & Occlusion Testsets erzeugt!")