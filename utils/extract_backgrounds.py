import os
import cv2
import shutil

# ---------------------------------------------------------
# KONFIGURATION – ggf. anpassen
# ---------------------------------------------------------

# Ordner aus denen Real-Bilder extrahiert werden sollen
INPUT_DIRS = [
    "../data/test/known_test/0_real",
    "../data/test/unknown_test/0_real"
]

# Zielordner für Hintergründe
OUTPUT_BACKGROUND_DIRs = ["../data/backgrounds/known", "../data/backgrounds/unknown"]


# ---------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def contains_face(image, detector):
    """Prüft, ob ein Gesicht im Bild erkannt wird."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0


# ---------------------------------------------------------
# HAUPTFUNKTION
# ---------------------------------------------------------

def extract_backgrounds(input_dir, output_dir):
    ensure_dir(output_dir)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    total = 0
    copied = 0

    print("=== Extrahiere Hintergrundbilder (ohne Gesichter) ===\n")

    if not os.path.exists(input_dir):
        print(f"[WARN] Ordner nicht gefunden: {input_dir}")

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f" Durchsuche {input_dir} ({len(files)} Bilder)")

    for file in files:
        total += 1
        path = os.path.join(input_dir, file)
        img = cv2.imread(path)

        if img is None:
            print(f"[WARN] Konnte Bild nicht laden: {file}")
            continue

         # Prüfen ob Gesicht enthalten ist
        if contains_face(img, face_detector):
            continue  # Kein Hintergrund

        # Hintergrund → kopieren
        out_path = os.path.join(output_dir, file)
        shutil.copy(path, out_path)
        copied += 1

    print("\n=== Fertig ===")
    print(f" Gesamtbilder durchsucht: {total}")
    print(f" Davon als Hintergründe geeignet: {copied}")
    print(f" Ergebnisse gespeichert in: {output_dir}\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    extract_backgrounds(input_dir=INPUT_DIRS[0], output_dir=OUTPUT_BACKGROUND_DIRs[0])
    extract_backgrounds(input_dir=INPUT_DIRS[1], output_dir=OUTPUT_BACKGROUND_DIRs[1])