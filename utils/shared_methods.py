"""Methoden & Funktionen, die in mehreren Files verwendet werden → soll Duplikate vermeiden"""
import csv
import os
import re

import torch

from utils.config import RNG, CONFIG


def slugify(text):
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9\-._]", "", text)
    return text[:60] if len(text) > 60 else text

def make_generator():
    seed = RNG.randint(1, 1000000000)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g, seed

def write_csv_row(csv_path, header, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

def get_image_output(prompt, category, model_name, seed, root, known_or_unknown, typ):
    p = slugify(prompt)
    name = f"{category}_manipulated_{model_name}_{p}_{seed}.jpg"
    image_out = os.path.join(
        root,
        CONFIG["images_path"],
        known_or_unknown,
        category,
        typ,
        model_name,
        name
    )
    os.makedirs(os.path.dirname(image_out), exist_ok=True)
    return image_out, name