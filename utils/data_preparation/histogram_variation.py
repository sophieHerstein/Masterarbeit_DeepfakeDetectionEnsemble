"""Histogrammänderungen an Bildern ausführen für Tests"""

import os

import cv2
import numpy as np
from tqdm import tqdm


def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    img = np.int16(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)


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

        gamma = np.random.uniform(0.6, 1.4)
        img = adjust_gamma(img, gamma)

        brightness = np.random.randint(-40, 40)
        contrast = np.random.randint(-40, 40)
        img = adjust_brightness_contrast(img, brightness, contrast)

        cv2.imwrite(out_path, img)


def process_testset(base_input, base_output):
    for cls in ["0_real", "1_fake"]:
        inp = os.path.join(base_input, cls)
        out = os.path.join(base_output, cls)
        process_folder(inp, out)


if __name__ == "__main__":
    known_input = "../data/test/known_test"
    unknown_input = "../data/test/unknown_test"

    known_output = "../data/test/known_test_histogram"
    unknown_output = "../data/test/unknown_test_histogram"

    process_testset(known_input, known_output)
    process_testset(unknown_input, unknown_output)

    print("Histogram Testsets erzeugt!")
