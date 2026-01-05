import os
import random

import cv2

INPUT_FAKE_DIR_KNOWN = "../data/test/known_test/1_fake"
INPUT_FAKE_DIR_UNKNOWN = "../data/test/unknown_test/1_fake"

BACKGROUND_DIR_KNOWN = "../data/backgrounds/known"
BACKGROUND_DIR_UNKNOWN = "../data/backgrounds/unknown"

OUTPUT_KNOWN = "../data/test/known_test_insertion/1_fake"
OUTPUT_UNKNOWN = "../data/test/unknown_test_insertion/1_fake"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_face_bbox(image, face_detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return (x, y, w, h)


def insert_face_into_background(fake_face, background):
    h_bg, w_bg = background.shape[:2]
    h_f, w_f = fake_face.shape[:2]

    target_width = random.randint(int(w_bg * 0.15), int(w_bg * 0.30))
    scale = target_width / w_f

    fake_face_resized = cv2.resize(fake_face, None, fx=scale, fy=scale)
    h_f, w_f = fake_face_resized.shape[:2]

    if w_f >= w_bg or h_f >= h_bg:
        return None

    max_x = w_bg - w_f - 1
    max_y = h_bg - h_f - 1

    if max_x <= 0 or max_y <= 0:
        return None

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    roi = background[y:y + h_f, x:x + w_f]

    blended = cv2.addWeighted(roi, 0.5, fake_face_resized, 0.5, 0)

    background[y:y + h_f, x:x + w_f] = blended

    return background


def contains_face(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0


def run_insertion(input_dir, bg_dir, output_dir):
    ensure_dir(output_dir)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    bg_images = [
        os.path.join(bg_dir, f)
        for f in os.listdir(bg_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f" Starte Insertion Test für {input_dir}")
    print(f" Anzahl Fake-Bilder: {len(files)}")
    print(f" Anzahl Hintergrundbilder: {len(bg_images)}")

    for file in files:
        fake_path = os.path.join(input_dir, file)
        fake_img = cv2.imread(fake_path)

        if fake_img is None:
            print(f"[WARN] Konnte {file} nicht laden.")
            continue

        bbox = get_face_bbox(fake_img, face_detector)
        if bbox is None:
            print(f"[WARN] Kein Gesicht erkannt: {file}")
            continue

        x, y, w, h = bbox
        fake_face = fake_img[y:y + h, x:x + w]

        bg_path = random.choice(bg_images)
        background = cv2.imread(bg_path)

        if background is None:
            print(f"[WARN] Hintergrundbild defekt: {bg_path}")
            continue

        output_img = insert_face_into_background(fake_face, background)
        if output_img is None:
            print(f"[WARN] Insertion fehlgeschlagen für {file}")
            continue

        cv2.imwrite(os.path.join(output_dir, file), output_img)

    print(f" Fertig! Ergebnisse gespeichert in: {output_dir}\n")


def main():
    print("=== Fake Insertion Test läuft ===\n")

    run_insertion(INPUT_FAKE_DIR_KNOWN, BACKGROUND_DIR_KNOWN, OUTPUT_KNOWN)
    run_insertion(INPUT_FAKE_DIR_UNKNOWN, BACKGROUND_DIR_UNKNOWN, OUTPUT_UNKNOWN)

    print("=== Alle Insertion-Tests abgeschlossen ===")


if __name__ == "__main__":
    main()
