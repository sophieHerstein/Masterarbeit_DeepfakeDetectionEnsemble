import os
import cv2
from tqdm import tqdm

# -----------------------------
# Bild speichern in Format X
# -----------------------------

def save_as_format(img, path, fmt):
    # fmt = "png", "jpg90", "jpg60", "webp"
    if fmt == "png":
        cv2.imwrite(path + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    elif fmt == "jpg90":
        cv2.imwrite(path + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    elif fmt == "jpg60":
        cv2.imwrite(path + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])

    elif fmt == "webp":
        cv2.imwrite(path + ".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 90])


# -----------------------------
# Einen Ordner verarbeiten
# -----------------------------

def process_folder(input_folder, output_folder, fmt):
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder), desc=f"{fmt}: {input_folder}"):
        in_path = os.path.join(input_folder, filename)

        if not os.path.isfile(in_path):
            continue

        # Dateiname ohne Extension
        name, _ = os.path.splitext(filename)
        out_path = os.path.join(output_folder, name)

        img = cv2.imread(in_path)
        if img is None:
            continue

        save_as_format(img, out_path, fmt)


# -----------------------------
# Hauptlogik
# -----------------------------

def process_testset(base_input, base_output, fmt):
    for cls in ["0_real", "1_fake"]:
        inp = os.path.join(base_input, cls)
        out = os.path.join(base_output, cls)
        process_folder(inp, out, fmt)


if __name__ == "__main__":
    # Ursprungsordner
    known_input = "data/test/known_test"
    unknown_input = "data/test/unknown_test"

    # Formate, die erzeugt werden sollen
    formats = ["png", "jpg90", "jpg60", "webp"]

    for fmt in formats:
        known_output = f"known_test_format_{fmt}"
        unknown_output = f"unknown_test_format_{fmt}"

        print(f"Erzeuge {fmt} für KNOWN-Testset…")
        process_testset(known_input, known_output, fmt)

        print(f"Erzeuge {fmt} für UNKNOWN-Testset…")
        process_testset(unknown_input, unknown_output, fmt)

    print("Alle Format-Testsets erzeugt!")