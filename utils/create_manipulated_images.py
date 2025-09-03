import os
import re
import csv
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image, ImageDraw
import torch

from dotenv import load_dotenv
from huggingface_hub import login

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
)

# ------------------------------------------------------------
# Config / Globals
# ------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ersetze das bei Bedarf durch deine zentrale CONFIG
CONFIG = {
    "images_path": "images",  # relativer Pfad unter PROJECT_ROOT
    "manipulated_images_log_path": "logs/manipulated_images.csv",
    "image_size_sd15": 512,
    "image_size_sdxl": 1024,
    "steps": 30,
    "guidance": 5.0,
    "rng_seed": 42,  # fixe Seedauswahl -> reproduzierbar für Bildliste
}

# Kategorien wie bei dir (anpassen!)
CATEGORIES = ["landscape", "building", "face"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Pipelines werden gecacht
_PIPE_CACHE = {}

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _slugify(text: str) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9\-._]", "", text)
    return text[:80]

def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def _read_images_from(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files

def _choose_half(files: List[str], rng: random.Random) -> List[str]:
    n = len(files) // 2
    return rng.sample(files, n) if n > 0 else []

def _save_and_log(csv_path: str, row: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["KnownOrUnknown","Kategorie","OriginalPath","Modell","EditType","InstructionOrPrompt","Seed","OutputPath"])
        w.writerow(row)

def _output_path(original_path: str, known_or_unknown: str, category: str, model_name: str, seed: int) -> str:
    images_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"])
    base = os.path.splitext(os.path.basename(original_path))[0]
    out_dir = os.path.join(images_root, known_or_unknown, category, "manipulated", model_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_seed{seed}.jpg")

def _resize_for_model(img: Image.Image, is_sdxl: bool) -> Image.Image:
    target = CONFIG["image_size_sdxl"] if is_sdxl else CONFIG["image_size_sd15"]
    return img.resize((target, target), Image.BICUBIC)

def _torch_generator() -> Tuple[torch.Generator, int]:
    seed = random.randint(1, 1_000_000)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g, seed

# ------------------------------------------------------------
# Edit Prompts / Instructions (einfach erweiterbar)
# ------------------------------------------------------------
EDIT_LIBRARY = {
    "landscape": {
        "img2img": [
            "a moody night scene with stars and moonlight",
            "a foggy, rainy atmosphere with overcast sky",
            "golden hour lighting, warm tones, soft sun rays",
            "winter scene with fresh snow on the ground and trees",
        ],
        "inpaint_add": [
            "add a small wooden cabin",
            "add a winding dirt path",
            "add a red car on a distant road",
            "add a flock of birds in the sky",
        ],
        "instruction": [
            "turn the scene into a snowy winter landscape",
            "make it nighttime with a starry sky",
            "add a small wooden cabin near the center",
            "add light rain and fog",
        ],
    },
    "building": {
        "img2img": [
            "modern minimal style, concrete and glass, daytime",
            "nocturnal city ambiance with neon signs",
            "classic brick facade with ivy climbing the walls",
            "sunset lighting, warm orange sky, long shadows",
        ],
        "inpaint_add": [
            "add ivy on the facade",
            "add a street lamp next to the entrance",
            "add a small balcony",
            "add graffiti art on the side wall",
        ],
        "instruction": [
            "make it night with neon reflections",
            "add ivy covering parts of the facade",
            "turn the building into a warm sunset scene",
            "add subtle graffiti on the side wall",
        ],
    },
    "face": {
        "img2img": [
            "studio portrait lighting, soft light, shallow depth of field",
            "outdoor lighting, overcast sky, soft diffuse light",
            "cinematic portrait, warm tones, film grain",
            "cool tones portrait, soft rim light",
        ],
        "inpaint_add": [
            "add sunglasses",
            "add a baseball cap",
            "add freckles on the cheeks",
            "add small hoop earrings",
        ],
        "instruction": [
            "add sunglasses",
            "change hair color to blonde",
            "add freckles",
            "add a subtle beard",
        ],
    },
}

def _pick_edit(category: str, kind: str, rng: random.Random) -> str:
    options = EDIT_LIBRARY[category][kind]
    return rng.choice(options)

# ------------------------------------------------------------
# Random Mask (für Inpainting)
# ------------------------------------------------------------
def _random_mask(img_size: Tuple[int,int], rng: random.Random) -> Image.Image:
    w, h = img_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # 2-4 zufällige Ellipsen/Polygone
    count = rng.randint(2, 4)
    for _ in range(count):
        x1 = rng.randint(0, int(0.6*w))
        y1 = rng.randint(0, int(0.6*h))
        x2 = rng.randint(int(0.4*w), w)
        y2 = rng.randint(int(0.4*h), h)
        if rng.random() < 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=255)
        else:
            # grobes Polygon
            pts = [(rng.randint(0, w), rng.randint(0, h)) for _ in range(rng.randint(3,6))]
            draw.polygon(pts, fill=255)

    return mask

# ------------------------------------------------------------
# Pipeline Loader
# ------------------------------------------------------------
def _get_pipe(name: str):
    if name in _PIPE_CACHE:
        return _PIPE_CACHE[name]

    if name == "sd15_img2img":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE
        ).to(DEVICE)

    elif name == "sd15_inpaint":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=DTYPE
        ).to(DEVICE)

    elif name == "sdxl_inpaint":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=DTYPE
        ).to(DEVICE)

    elif name == "instruct_pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", torch_dtype=DTYPE
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown pipe '{name}'")

    _PIPE_CACHE[name] = pipe
    return pipe

# ------------------------------------------------------------
# Core Manipulations
# ------------------------------------------------------------
def manipulate_known(known_or_unknown: str = "known"):
    rng = random.Random(CONFIG["rng_seed"])

    for category in CATEGORIES:
        # Eingangsordner realer Bilder
        in_dir = os.path.join(PROJECT_ROOT, CONFIG["images_path"], known_or_unknown, category, "real")
        files = _read_images_from(in_dir)
        if not files:
            print(f"[{known_or_unknown}/{category}] keine realen Bilder gefunden.")
            continue

        selected = _choose_half(files, rng)
        print(f"[{known_or_unknown}/{category}] {len(selected)} von {len(files)} Bildern werden manipuliert.")

        # Pipelines für known: drei Modelle
        pipes = {
            "sd15_img2img": _get_pipe("sd15_img2img"),
            "sd15_inpaint": _get_pipe("sd15_inpaint"),
            "sdxl_inpaint": _get_pipe("sdxl_inpaint"),
        }

        for idx, path in enumerate(selected):
            img = _ensure_rgb(Image.open(path))
            # Abwechselndes Routing über die drei Modelle
            order = ["sd15_img2img", "sd15_inpaint", "sdxl_inpaint"]
            model_name = order[idx % len(order)]

            if model_name == "sd15_img2img":
                prompt = _pick_edit(category, "img2img", rng)
                gen, seed = _torch_generator()
                img_resized = _resize_for_model(img, is_sdxl=False)
                out = pipes[model_name](
                    prompt=prompt,
                    image=img_resized,
                    strength=0.35,            # moderate Änderung
                    guidance_scale=CONFIG["guidance"],
                    num_inference_steps=CONFIG["steps"],
                    generator=gen
                ).images[0]

                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log(
                    os.path.join(PROJECT_ROOT, CONFIG["manipulated_images_log_path"]),
                    [known_or_unknown, category, path, model_name, "img2img", prompt, seed, out_path]
                )

            elif model_name == "sd15_inpaint":
                prompt = _pick_edit(category, "inpaint_add", rng)
                gen, seed = _torch_generator()
                img_resized = _resize_for_model(img, is_sdxl=False)
                mask = _random_mask(img_resized.size, rng)

                out = pipes[model_name](
                    prompt=prompt,
                    image=img_resized,
                    mask_image=mask,
                    guidance_scale=CONFIG["guidance"],
                    num_inference_steps=CONFIG["steps"],
                    generator=gen
                ).images[0]

                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log(
                    os.path.join(PROJECT_ROOT, CONFIG["manipulated_images_log_path"]),
                    [known_or_unknown, category, path, model_name, "inpaint_random_mask", prompt, seed, out_path]
                )

            elif model_name == "sdxl_inpaint":
                prompt = _pick_edit(category, "inpaint_add", rng)
                gen, seed = _torch_generator()
                img_resized = _resize_for_model(img, is_sdxl=True)
                mask = _random_mask(img_resized.size, rng)

                out = pipes[model_name](
                    prompt=prompt,
                    image=img_resized,
                    mask_image=mask,
                    guidance_scale=CONFIG["guidance"],
                    num_inference_steps=CONFIG["steps"],
                    generator=gen
                ).images[0]

                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log(
                    os.path.join(PROJECT_ROOT, CONFIG["manipulated_images_log_path"]),
                    [known_or_unknown, category, path, model_name, "inpaint_random_mask", prompt, seed, out_path]
                )


def manipulate_unknown_with_instruct_pix2pix():
    known_or_unknown = "unknown"
    rng = random.Random(CONFIG["rng_seed"] + 999)

    pipe = _get_pipe("instruct_pix2pix")

    for category in CATEGORIES:
        in_dir = os.path.join(PROJECT_ROOT, CONFIG["images_path"], known_or_unknown, category, "real")
        files = _read_images_from(in_dir)
        if not files:
            print(f"[{known_or_unknown}/{category}] keine realen Bilder gefunden.")
            continue

        selected = _choose_half(files, rng)
        print(f"[{known_or_unknown}/{category}] {len(selected)} von {len(files)} Bildern werden manipuliert (InstructPix2Pix).")

        for path in selected:
            img = _ensure_rgb(Image.open(path))
            instruction = _pick_edit(category, "instruction", rng)
            gen, seed = _torch_generator()

            # Bild für SD15-Größe normalisieren (das Modell ist SD15-basiert)
            img_resized = _resize_for_model(img, is_sdxl=False)

            out = pipe(
                image=img_resized,
                prompt=instruction,
                num_inference_steps=CONFIG["steps"],
                guidance_scale=CONFIG["guidance"],
                image_guidance_scale=1.6,  # hält mehr von der Originalstruktur
                generator=gen
            ).images[0]

            out_path = _output_path(path, known_or_unknown, category, "instruct_pix2pix", seed)
            out.save(out_path, quality=95)
            _save_and_log(
                os.path.join(PROJECT_ROOT, CONFIG["manipulated_images_log_path"]),
                [known_or_unknown, category, path, "instruct_pix2pix", "instruction", instruction, seed, out_path]
            )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Starting manipulated image generation ...")
    if HF_TOKEN is None:
        raise EnvironmentError("HF_TOKEN fehlt in .env")
    login(token=HF_TOKEN, add_to_git_credential=True)

    if DEVICE != "cuda":
        print("WARN: CUDA nicht verfügbar – das wird sehr langsam.")

    # Known: 3 Modelle (sd15 img2img, sd15 inpaint, sdxl inpaint)
    manipulate_known("known")

    # Unknown: 1 Modell (InstructPix2Pix)
    manipulate_unknown_with_instruct_pix2pix()

    print("DONE.")