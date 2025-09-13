import os
import re
import csv
import random
import sys
from PIL import Image, ImageDraw

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
)

from utils.config import EDIT_LIBRARY, CATEGORIES, CONFIG, MANIPULATED_VARIANTEN_BEKANNT, \
    MANIPULATED_HUMAN_VARIANTEN_BEKANNT, MANIPULATED_VARIANTEN_UNBEKANNT

# ------------------------------------------------------------
# Config / Globals
# ------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_15_IMG2IMG = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_INPAINTING = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_2_INPAINTING = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_15_IMG2IMG = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_INPAINTING = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_2_INPAINTING = []
FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ = []
FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_15_IMG2IMG = []
FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_INPAINTING = []
FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_2_INPAINTING = []
FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE = []
FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_15_IMG2IMG = []
FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_INPAINTING = []
FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_2_INPAINTING = []

FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ = []
FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE = []
FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET = []

CATEGORIES_FOR_MANIPULATION = [*CATEGORIES, "human_2"]

RNG = random.Random(42)

STEPS = 30
GUIDANCE_SCALE = 4.5

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _slugify(text):
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9\-._]", "", text)
    return text[:60] if len(text) > 60 else text

def _read_images_from(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files

def _choose_half(files):
    n = len(files) // 2
    return RNG.sample(files, n) if n > 0 else []

def write_csv_row(row):
    csv_path = os.path.join(PROJECT_ROOT,CONFIG["manipulated_images_log_path"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Kategorie","OriginalPath","Modell","EditType","InstructionOrPrompt","Seed","OutputPath"])
        w.writerow(row)

def get_image_output(known_or_unknown, category, model_name, seed, prompt):
    p = _slugify(prompt)
    name = f"{category}_manipulated_{model_name}_{p}_{seed}.jpg"
    image_out = os.path.join(
        PROJECT_ROOT,
        CONFIG["images_path"],
        known_or_unknown,
        category,
        "manipulated",
        model_name,
        name
    )
    os.makedirs(os.path.dirname(image_out), exist_ok=True)
    return image_out, name


def make_generator():
    seed = RNG.randint(1, 1000000000)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g, seed

def _disable_safety_checker(pipe):
    # Behalte API-Signatur bei: gibt (images, [False]*N) zurück
    def dummy_checker(images, clip_input):
        return images, [False] * len(images)
    pipe.safety_checker = dummy_checker
    return pipe

def _resize(img, max_side=768):
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)  # niemals hochskalieren
    new_w = int((w * scale) // 8 * 8) or 8
    new_h = int((h * scale) // 8 * 8) or 8
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

# ------------------------------------------------------------
# Random Mask (für Inpainting)
# ------------------------------------------------------------
def _random_mask(img_size):
    w, h = img_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Mindestgröße eines Elements (proportional, aber geklammert)
    min_side = max(8, min(w, h) // 12)  # z.B. ~8–100px, je nach Bildgröße

    def _sorted_coords(ax, ay, bx, by):
        x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
        y0, y1 = (ay, by) if ay <= by else (by, ay)
        # Mindestabstand erzwingen
        if x1 - x0 < min_side:
            x1 = min(w - 1, x0 + min_side)
        if y1 - y0 < min_side:
            y1 = min(h - 1, y0 + min_side)
        return x0, y0, x1, y1

    count = RNG.randint(2, 4)
    for _ in range(count):
        try:
            # Zwei zufällige Punkte
            ax = RNG.randint(0, max(0, w - 1))
            ay = RNG.randint(0, max(0, h - 1))
            bx = RNG.randint(0, max(0, w - 1))
            by = RNG.randint(0, max(0, h - 1))
            x0, y0, x1, y1 = _sorted_coords(ax, ay, bx, by)

            if RNG.random() < 0.5:
                # Ellipse
                draw.ellipse([x0, y0, x1, y1], fill=255)
            else:
                # Polygon mit 3–6 Punkten, sauber in Bounds
                n = RNG.randint(3, 6)
                pts = []
                for _i in range(n):
                    px = RNG.randint(x0, x1)
                    py = RNG.randint(y0, y1)
                    pts.append((px, py))
                draw.polygon(pts, fill=255)
        except Exception:
            # Fallback: Sicheres Rechteck in Bildmitte
            cx, cy = w // 2, h // 2
            x0 = max(0, cx - min_side)
            y0 = max(0, cy - min_side)
            x1 = min(w - 1, cx + min_side)
            y1 = min(h - 1, cy + min_side)
            draw.rectangle([x0, y0, x1, y1], fill=255)

    return mask

# ------------------------------------------------------------------------------
# Generatoren
# ------------------------------------------------------------------------------
def manipulate_image_with_stable_diffusion_15_img2img():
    print("Manipulating images with Stable Diffusion 1.5 img2img")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    pipe.to("cuda")
    pipe = _disable_safety_checker(pipe)
    for category in CATEGORIES_FOR_MANIPULATION:
        selected = _selected_known_for_category_and_manipulation(category, 'sd_img2img')
        cat = ('human' if category == "human_2" else category)
        used_selected = {}
        for prompt in EDIT_LIBRARY[cat]['img2img']:
            wiederholungen = (MANIPULATED_HUMAN_VARIANTEN_BEKANNT if cat == "human" else MANIPULATED_VARIANTEN_BEKANNT)
            for i in range(wiederholungen):
                path = RNG.choice(selected)
                while used_selected.get(path, 0) > wiederholungen:
                    path = RNG.choice(selected)
                used_selected[path] = used_selected.get(path, 0) + 1
                print(f"[Stable Diffusion 1.5 img2img] Manipulating {path} with {prompt}")
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = _resize(img)
                    gen, seed = make_generator()
                    out = pipe(
                        prompt=prompt,
                        image=img,
                        strength=0.35,
                        guidance_scale=GUIDANCE_SCALE,
                        num_inference_steps=STEPS,
                        generator=gen
                    ).images[0]
                    out_path, image_name = get_image_output("known", category, "stable_diffusion_15_imag2img", seed, prompt)
                    out.save(out_path)
                    write_csv_row([category, path, "stable_diffusion_15_imag2img", "img2img", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()


def manipulate_image_with_stable_diffusion_inpainting():
    print("Manipulating images with Stable Diffusion Inpainting")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    pipe.to("cuda")
    pipe = _disable_safety_checker(pipe)
    for category in CATEGORIES_FOR_MANIPULATION:
        selected = _selected_known_for_category_and_manipulation(category, 'sd_inpaint')
        cat = ('human' if category == "human_2" else category)
        used_selected = {}
        for prompt in EDIT_LIBRARY[cat]['inpaint']:
            wiederholungen = (MANIPULATED_HUMAN_VARIANTEN_BEKANNT if cat == "human" else MANIPULATED_VARIANTEN_BEKANNT)
            for i in range(wiederholungen):
                path = RNG.choice(selected)
                while used_selected.get(path, 0) > wiederholungen:
                    path = RNG.choice(selected)
                used_selected[path] = used_selected.get(path, 0) + 1
                print(f"[Stable Diffusion Inpainting] Manipulating {path} with {prompt}")
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = _resize(img)
                    gen, seed = make_generator()
                    mask = _random_mask(img.size)
                    out = pipe(
                        prompt=prompt,
                        image=img,
                        mask_image=mask,
                        guidance_scale=GUIDANCE_SCALE,
                        num_inference_steps=STEPS,
                        generator=gen
                    ).images[0]
                    out_path, image_name = get_image_output("known", category, "stable_diffusion_inpainting", seed, prompt)
                    out.save(out_path)
                    write_csv_row([category, path, "stable_diffusion_inpainting", "inpaint", prompt, seed, image_name])


    del pipe
    torch.cuda.empty_cache()


def manipulate_image_with_stable_diffusion_2_inpainting():
    print("Manipulating images with Stable Diffusion 2 Inpainting")
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                          torch_dtype=torch.float16)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    pipe.to("cuda")
    pipe = _disable_safety_checker(pipe)
    for category in CATEGORIES_FOR_MANIPULATION:
        selected = _selected_known_for_category_and_manipulation(category, 'sd_2_inpaint')
        cat = ('human' if category == "human_2" else category)
        used_selected = {}
        for prompt in EDIT_LIBRARY[cat]['inpaint']:
            wiederholungen = (MANIPULATED_HUMAN_VARIANTEN_BEKANNT if cat == "human" else MANIPULATED_VARIANTEN_BEKANNT)
            for i in range(wiederholungen):
                path = RNG.choice(selected)
                while used_selected.get(path, 0) > wiederholungen:
                    path = RNG.choice(selected)
                used_selected[path] = used_selected.get(path, 0) + 1
                print(f"[Stable Diffusion 2 Inpainting] Manipulating {path} with {prompt}")
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = _resize(img)
                    gen, seed = make_generator()
                    mask = _random_mask(img.size)
                    out = pipe(
                        prompt=prompt,
                        image=img,
                        mask_image=mask,
                        guidance_scale=GUIDANCE_SCALE,
                        num_inference_steps=STEPS,
                        generator=gen
                    ).images[0]
                    out_path, image_name = get_image_output("known", category, "stable_diffusion_2_inpainting", seed, prompt)
                    out.save(out_path)
                    write_csv_row([category, path, "stable_diffusion_2_inpainting", "inpaint", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()


def manipulate_image_with_instruct_pix2pix():
    print("Manipulating images with Instruct Pix2Pix")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    ).to("cuda")
    pipe = _disable_safety_checker(pipe)
    for category in CATEGORIES:
        selected = _selected_unknown_for_category(category)
        used_selected = {}
        for prompt in EDIT_LIBRARY[category]["instruction"]:
            for i in range(MANIPULATED_VARIANTEN_UNBEKANNT):
                path = RNG.choice(selected)
                while used_selected.get(path, 0) > MANIPULATED_VARIANTEN_UNBEKANNT:
                    path = RNG.choice(selected)
                used_selected[path] = used_selected.get(path, 0) + 1
                print(f"[Instruct Pix2Pix] Manipulating {path} with {prompt}")
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    gen, seed = make_generator()
                    out = pipe(
                        image=img,
                        prompt=prompt,
                        num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        image_guidance_scale=1.6,
                        generator=gen
                    ).images[0]
                    out_path, image_name = get_image_output("unknown", category, "instruct_pix2pix", seed, prompt)
                    out.save(out_path)
                    write_csv_row([category, path, "instruct_pix2pix", "instruction", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Core Manipulations
# ------------------------------------------------------------
def _selected_known_for_category_and_manipulation(category, manipulation):
    if category == "human":
        if manipulation == 'sd_img2img':
            return FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_15_IMG2IMG
        elif manipulation == 'sd_inpaint':
            return FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_INPAINTING
        else:
            return FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_2_INPAINTING
    elif category == "human_2":
        if manipulation == 'sd_img2img':
            return FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_15_IMG2IMG
        elif manipulation == 'sd_inpaint':
            return FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_INPAINTING
        else:
            return FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_2_INPAINTING
    elif category == "building":
        if manipulation == 'sd_img2img':
            return FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_15_IMG2IMG
        elif manipulation == 'sd_inpaint':
            return FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_INPAINTING
        else:
            return FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_2_INPAINTING
    else:  # "landscape"
        if manipulation == 'sd_img2img':
            return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_15_IMG2IMG
        elif manipulation == 'sd_inpaint':
            return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_INPAINTING
        else:
            return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_2_INPAINTING

def _selected_unknown_for_category(category):
    if category == "human":
        return FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ
    elif category == "building":
        return FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET
    else:
        return FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE


def get_images_for_manipulation():
    global FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS, FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA, FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ, FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE, FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET, FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ, FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE
    for category in CATEGORIES:
        if category == "human":
            known_faceforensics = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "human", "realistic",
                                               "faceforensics")
            known_celeba = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "human", "realistic", "celeba")
            unknown_ffhq = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "unknown", "human", "realistic", "ffqh")

            k_ff_files = _read_images_from(known_faceforensics)
            k_cb_files = _read_images_from(known_celeba)
            u_ffhq = _read_images_from(unknown_ffhq)

            FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS = _choose_half(k_ff_files)
            for i in range(0, len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS)):
                if i % 3 == 0:
                    FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_15_IMG2IMG.append(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS[i])
                elif i % 3 == 1:
                    FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_INPAINTING.append(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS[i])
                elif i % 3 == 2:
                    FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_2_INPAINTING.append(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS[i])

            FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA = _choose_half(k_cb_files)
            for i in range(0, len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA)):
                if i % 3 == 0:
                    FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_15_IMG2IMG.append(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA[i])
                elif i % 3 == 1:
                    FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_INPAINTING.append(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA[i])
                elif i % 3 == 2:
                    FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_2_INPAINTING.append(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA[i])

            FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ = _choose_half(u_ffhq)

            print(f"[known/human] FaceForensics: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS)} von {len(k_ff_files)}")
            print(f"[img2img] FaceForensics: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_15_IMG2IMG)} von {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS)}")
            print(f"[sd_inpaint] FaceForensics: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_INPAINTING)} von {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS)}")
            print(f"[sd2_inpaint] FaceForensics: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_2_INPAINTING)} von {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS)}")
            print("")
            print(f"[known/human] CelebA: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA)} von {len(k_cb_files)}")
            print(f"[img2img] CelebA: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_15_IMG2IMG)} von {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA)}")
            print(f"[sd_inpaint] CelebA: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_INPAINTING)} von {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA)}")
            print(f"[sd2_inpaint] CelebA: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_2_INPAINTING)} von {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA)}")
            print("")
            print(f"[unknown/human] FFHQ: {len(FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ)} von {len(u_ffhq)}")
            print("")

        elif category == "building":
            known_arch = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "building", "realistic",
                                      "architecture")
            unknown_imn = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "unknown", "building", "realistic",
                                       "imagenet")

            k_files = _read_images_from(known_arch)
            u_files = _read_images_from(unknown_imn)

            FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE = _choose_half(k_files)
            for i in range(0, len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE)):
                if i % 3 == 0:
                    FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_15_IMG2IMG.append(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE[i])
                elif i % 3 == 1:
                    FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_INPAINTING.append(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE[i])
                elif i % 3 == 2:
                    FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_2_INPAINTING.append(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE[i])

            FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET = _choose_half(u_files)

            print(f"[known/building] architecture: {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE)} von {len(k_files)}")
            print(f"[img2img] architecture: {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_15_IMG2IMG)} von {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE)}")
            print(f"[sd_inpaint] architecture: {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_INPAINTING)} von {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE)}")
            print(f"[sd2_inpaint] architecture: {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_2_INPAINTING)} von {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE)}")
            print("")
            print(f"[unknown/building] imagenet: {len(FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET)} von {len(u_files)}")
            print("")

        elif category == "landscape":
            known_lhq = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "landscape", "realistic", "lhq")
            unknown_ls = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "unknown", "landscape", "realistic",
                                      "landscape")

            k_files = _read_images_from(known_lhq)
            u_files = _read_images_from(unknown_ls)

            FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ = _choose_half(k_files)
            for i in range(0, len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ)):
                if i % 3 == 0:
                    FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_15_IMG2IMG.append(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ[i])
                elif i % 3 == 1:
                    FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_INPAINTING.append(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ[i])
                elif i % 3 == 2:
                    FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_2_INPAINTING.append(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ[i])

            FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE = _choose_half(u_files)

            print(f"[known/landscape] LHQ: {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ)} von {len(k_files)}")
            print(f"[img2img] LHQ: {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_15_IMG2IMG)} von {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ)}")
            print(f"[sd_inpaint] LHQ: {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_INPAINTING)} von {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ)}")
            print(f"[sd2_inpaint] LHQ: {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_2_INPAINTING)} von {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ)}")
            print("")
            print(f"[unknown/landscape] LANDSCAPE: {len(FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE)} von {len(u_files)}")


# ------------------------------------------------------------------------------
# High-level Wrapper
# ------------------------------------------------------------------------------
def manipulate_images():
    manipulate_image_with_stable_diffusion_15_img2img()
    manipulate_image_with_stable_diffusion_inpainting()
    manipulate_image_with_stable_diffusion_2_inpainting()


def manipulate_images_unbekannt():
    manipulate_image_with_instruct_pix2pix()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Starting manipulated image generation ...")

    if not torch.cuda.is_available():
        print("WARN: CUDA nicht verfügbar – Ausführung auf CPU wird sehr langsam sein.", file=sys.stderr)

    login(token=HF_TOKEN)

    get_images_for_manipulation()


    manipulate_images()
    manipulate_images_unbekannt()

    print("DONE.")

