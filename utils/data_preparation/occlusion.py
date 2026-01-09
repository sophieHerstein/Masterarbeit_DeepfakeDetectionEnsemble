"""Verdeckungen in den Bildern einfügen für Okklusionstests"""
import os
import random

import cv2
import numpy as np
from tqdm import tqdm


def random_occlusion(img):
    h, w, _ = img.shape

    occ_w = np.random.randint(int(w * 0.10), int(w * 0.3))
    occ_h = np.random.randint(int(h * 0.10), int(h * 0.3))

    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)

    result = img.copy()

    color = random.choice([(0, 0, 0), (50, 50, 50), (120, 120, 120)])

    cv2.rectangle(result, (x, y), (x + occ_w, y + occ_h), color, -1)
    return result


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

        img = random_occlusion(img)

        cv2.imwrite(out_path, img)


def process_testset(base_input, base_output):
    for cls in ["0_real", "1_fake"]:
        input_path = os.path.join(base_input, cls)
        output_path = os.path.join(base_output, cls)
        process_folder(input_path, output_path)


if __name__ == "__main__":
    known_input = "../data/test/known_test"
    unknown_input = "../data/test/unknown_test"

    known_output = "../data/test/known_test_occlusion"
    unknown_output = "../data/test/unknown_test_occlusion"

    process_testset(known_input, known_output)
    process_testset(unknown_input, unknown_output)

    print("Occlusion Testsets erzeugt!")
