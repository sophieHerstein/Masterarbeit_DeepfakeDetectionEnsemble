# 3 Modelle, 3 Kategorien
# 1050 Bilder pro Kategorie, also 350 je Modell je Kategorie
# zusätzlich 150 Bilder je Kategorie von einem unbekannten Modell

import os
import csv
from config import PROMPTS, CONFIG, CATEGORIES, VARIANTEN_BEKANNT, VARIANTEN_UNBEKANNT

# todo: methoden für alle modelle umsetzen

# models:
# https://huggingface.co/stabilityai/stable-diffusion-3.5-large
# https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
# https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0
# https://huggingface.co/stepfun-ai/NextStep-1-Large

# Todo evlt. refactoring
def create_image(text_prompt, image_category, index):
    print(f"Create image for prompt '{text_prompt}' with category '{image_category}'")
    model = "PLACEHOLDER"
    image_output = os.path.join(CONFIG["images_path"], image_category, "synthetic", model, f"{image_category}_synthetic_{model}_{prompt.replace(" ", "-")}_{index}.jpg")
    os.makedirs(os.path.dirname(image_output), exist_ok=True)

    write_csv_row("")

def create_images_unbekannt(text_prompt, image_category, index):
    print(f"Create images for unbekannt")

def write_csv_row(csv_row):
    os.makedirs(CONFIG["synthetic_images_log_path"], exist_ok=True)
    write_header = not os.path.exists(CONFIG["synthetic_images_log_path"])
    #TODO
    with open(CONFIG["synthetic_images_log_path"], "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [""])
        writer.writerow(
            [""])

#TODO
def generate_image_with_stable_diffusion_35():
    print("Generating synthetic images with stable diffusion 35")

#TODO
def generate_image_with_juggernaut_xl_v9():
    print("Generating synthetic images with Juggernaut XL v9")

#TODO
def generate_image_with_dreamlike_photoreal_20():
    print("Generating synthetic images with Dreamlike PhotoReal 20")

#TODO
def generate_image_with_nextstep_1_large():
    print("Generating synthetic images with nextstep 1 large")


if __name__ == "__main__":
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for i in range(0, VARIANTEN_BEKANNT):
                create_image(prompt, category, i)
            for j in range(0, VARIANTEN_UNBEKANNT):
                create_images_unbekannt(prompt, category, j)