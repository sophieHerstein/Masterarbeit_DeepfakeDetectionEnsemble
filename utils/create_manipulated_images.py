import os
import re
import csv
import random
import sys
from PIL import Image, ImageDraw
import torch
import time
import argparse
from dotenv import load_dotenv
from huggingface_hub import login

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
)

from utils.config import EDIT_LIBRARY, CATEGORIES, CONFIG
from utils.prompt_balancer import PromptBalancer

# ------------------------------------------------------------
# Config / Globals
# ------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS = []
FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA = []
FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ = []
FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE = []

FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ = []
FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE = []
FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET = []

RNG = random.Random(42)

BALANCER = PromptBalancer(
    edit_library=EDIT_LIBRARY,
    rng=RNG,
    config=CONFIG,
    project_root=PROJECT_ROOT,
)


STEPS = 40
GUIDANCE_SCALE = 4.5

_PIPE_CACHE = {}

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _slugify(text):
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9\-._]", "", text)
    return text[:80]

def _ensure_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

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

def _save_and_log(row):
    csv_path = CONFIG["manipulated_images_log_path"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["KnownOrUnknown","Kategorie","OriginalPath","Modell","EditType","InstructionOrPrompt","Seed","OutputPath"])
        w.writerow(row)

def _output_path(original_path, known_or_unknown, category, model_name, seed):
    images_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"])
    base = os.path.splitext(os.path.basename(original_path))[0]
    out_dir = os.path.join(images_root, known_or_unknown, category, "manipulated", model_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_manipulated_{seed}.jpg")

def _torch_generator():
    seed = random.randint(1, 1_000_000)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g, seed

# ------------------------------------------------------------
# Random Mask (für Inpainting)
# ------------------------------------------------------------
def _random_mask(img_size):
    """
    Erzeugt eine binäre Maske ("L") gleicher Größe wie das Bild.
    Robuste Koordinaten, min. Größe je Shape, zufällig 2–4 Patches.
    """
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

# ------------------------------------------------------------
# Pipeline Loader
# ------------------------------------------------------------
def _get_pipe(name: str):
    if name in _PIPE_CACHE:
        return _PIPE_CACHE[name]

    if name == "sd15_img2img":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to("cuda")

    elif name == "sd15_inpaint":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        ).to("cuda")

    elif name == "sdxl_inpaint":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
        ).to("cuda")

    elif name == "instruct_pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ).to("cuda")
    else:
        raise ValueError(f"Unknown pipe '{name}'")

    _PIPE_CACHE[name] = pipe
    return pipe

def _free_pipe(name: str):
    if name in _PIPE_CACHE:
        try:
            del _PIPE_CACHE[name]
        except Exception:
            pass

    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Core Manipulations
# ------------------------------------------------------------
def _selected_known_for_category(category):
    if category == "human":
        return FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS + FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA
    elif category == "building":
        return FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE
    else:  # "landscape"
        return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ

def _selected_unknown_for_category():
    if category == "human":
        return FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ
    elif category == "building":
        return FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET
    else:
        return FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE

def manipulate_known():
    known_or_unknown = "known"

    for category in CATEGORIES:
        selected = _selected_known_for_category(category)
        if not selected:
            print(f"[known/{category}] keine Bilder ausgewählt.")
            continue

        print(f"[known/{category}] Starte Manipulation für {len(selected)} Bilder.")

        # Modell 1: sd15_img2img
        model_name = "sd15_img2img"
        pipe = _get_pipe(model_name)
        for path in selected:
            with Image.open(path) as im:
                img = _ensure_rgb(im)
                prompt = BALANCER.next("known", category, "img2img")
                gen, seed = _torch_generator()
                out = pipe(
                    prompt=prompt,
                    image=img,
                    strength=0.35,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=STEPS,
                    generator=gen
                ).images[0]
                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log([known_or_unknown, category, path, model_name, "img2img", prompt, seed, out_path])
        _free_pipe(model_name)

        # Modell 2: sd15_inpaint
        model_name = "sd15_inpaint"
        pipe = _get_pipe(model_name)
        for path in selected:
            with Image.open(path) as im:
                img = _ensure_rgb(im)
                prompt = BALANCER.next("known", category, "inpaint_add")
                gen, seed = _torch_generator()
                mask = _random_mask(img.size)
                out = pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=STEPS,
                    generator=gen
                ).images[0]
                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log([known_or_unknown, category, path, model_name, "inpaint_random_mask", prompt, seed, out_path])
        _free_pipe(model_name)

        # Modell 3: sdxl_inpaint
        model_name = "sdxl_inpaint"
        pipe = _get_pipe(model_name)
        for path in selected:
            with Image.open(path) as im:
                img = _ensure_rgb(im)
                prompt = BALANCER.next("known", category, "inpaint_add")
                gen, seed = _torch_generator()
                mask = _random_mask(img.size)
                out = pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=STEPS,
                    generator=gen
                ).images[0]
                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log([known_or_unknown, category, path, model_name, "inpaint_random_mask", prompt, seed, out_path])
        _free_pipe(model_name)

def manipulate_unknown_with_instruct_pix2pix():
    known_or_unknown = "unknown"
    model_name = "instruct_pix2pix"
    pipe = _get_pipe(model_name)

    for category in CATEGORIES:
        selected = _selected_unknown_for_category(category)
        if not selected:
            print(f"[unknown/{category}] keine Bilder ausgewählt.")
            continue

        print(f"[unknown/{category}] {len(selected)} Bilder werden manipuliert ({model_name}).")
        for path in selected:
            with Image.open(path) as im:
                img = _ensure_rgb(im)
                instruction = BALANCER.next("unknown", category, "instruction")
                gen, seed = _torch_generator()
                out = pipe(
                    image=img,
                    prompt=instruction,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    image_guidance_scale=1.6,
                    generator=gen
                ).images[0]
                out_path = _output_path(path, known_or_unknown, category, model_name, seed)
                out.save(out_path, quality=95)
                _save_and_log([known_or_unknown, category, path, model_name, "instruction", instruction, seed, out_path])

    _free_pipe(model_name)




# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

def _pick_any_sample_for(category: str, known_or_unknown: str) -> str | None:
    """Nimmt ein Beispielbild für die Kategorie + Split aus den bereits vorgewählten Listen."""
    if known_or_unknown == "known":
        if category == "human":
            pool = FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS + FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA
        elif category == "building":
            pool = FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE
        elif category == "landscape":
            pool = FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ
        else:
            pool = []
    else:
        if category == "human":
            pool = FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ
        elif category == "building":
            pool = FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET
        elif category == "landscape":
            pool = FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE
        else:
            pool = []
    return pool[0] if pool else None

def smoke_test():
    """
    Führt je Modell einen einzelnen Lauf aus:
      - known/sd15_img2img  (img2img)
      - known/sd15_inpaint  (inpaint_add, random mask)
      - known/sdxl_inpaint  (inpaint_add, random mask)
      - unknown/instruct_pix2pix (instruction)
    Nimmt jeweils das erste verfügbare Beispielbild aus deinen vorgewählten Listen.
    Loggt normal in die CSV (mit „_SMOKE“ im Output-Filename).
    """
    print("\n=== SMOKE TEST START ===")
    results = []

    # --- 1) SD15 Img2Img (known) ---
    cat_order = ["human", "building", "landscape"]  # Priorität für Beispielbild
    path = None
    cat_used = None
    for cat in cat_order:
        p = _pick_any_sample_for(cat, "known")
        if p:
            path, cat_used = p, cat
            break
    if path:
        model_name = "sd15_img2img"
        pipe = _get_pipe(model_name)
        prompt = BALANCER.next("known", cat_used, "img2img")
        gen, seed = _torch_generator()
        img = _ensure_rgb(Image.open(path))
        # optional: Resize aktivieren
        # img = _resize_for_model(img, is_sdxl=False)
        t0 = time.perf_counter()
        out = pipe(prompt=prompt, image=img, strength=0.35,
                   guidance_scale=GUIDANCE_SCALE, num_inference_steps=STEPS,
                   generator=gen).images[0]
        dt = time.perf_counter() - t0
        out_path = _output_path(path, "known", cat_used, model_name, f"{seed}_SMOKE")
        out.save(out_path, quality=95)
        _save_and_log(["known", cat_used, path, model_name, "img2img", prompt, seed, out_path])
        _free_pipe(model_name)
        results.append((model_name, cat_used, dt, out_path))
        print(f"[SMOKE] {model_name} ({cat_used}) OK in {dt:.2f}s → {out_path}")
    else:
        print("[SMOKE] Kein known-Beispielbild gefunden (sd15_img2img übersprungen).")

    # --- 2) SD15 Inpaint (known) ---
    path = None; cat_used = None
    for cat in cat_order:
        p = _pick_any_sample_for(cat, "known")
        if p:
            path, cat_used = p, cat
            break
    if path:
        model_name = "sd15_inpaint"
        pipe = _get_pipe(model_name)
        prompt = BALANCER.next("known", cat_used, "inpaint_add")
        gen, seed = _torch_generator()
        img = _ensure_rgb(Image.open(path))
        # img = _resize_for_model(img, is_sdxl=False)
        mask = _random_mask(img.size)
        t0 = time.perf_counter()
        out = pipe(prompt=prompt, image=img, mask_image=mask,
                   guidance_scale=GUIDANCE_SCALE, num_inference_steps=STEPS,
                   generator=gen).images[0]
        dt = time.perf_counter() - t0
        out_path = _output_path(path, "known", cat_used, model_name, f"{seed}_SMOKE")
        out.save(out_path, quality=95)
        _save_and_log(["known", cat_used, path, model_name, "inpaint_random_mask", prompt, seed, out_path])
        _free_pipe(model_name)
        results.append((model_name, cat_used, dt, out_path))
        print(f"[SMOKE] {model_name} ({cat_used}) OK in {dt:.2f}s → {out_path}")
    else:
        print("[SMOKE] Kein known-Beispielbild gefunden (sd15_inpaint übersprungen).")

    # --- 3) SDXL Inpaint (known) ---
    path = None; cat_used = None
    for cat in cat_order:
        p = _pick_any_sample_for(cat, "known")
        if p:
            path, cat_used = p, cat
            break
    if path:
        model_name = "sdxl_inpaint"
        pipe = _get_pipe(model_name)
        prompt = BALANCER.next("known", cat_used, "inpaint_add")
        gen, seed = _torch_generator()
        img = _ensure_rgb(Image.open(path))
        # img = _resize_for_model(img, is_sdxl=True)
        mask = _random_mask(img.size)
        t0 = time.perf_counter()
        out = pipe(prompt=prompt, image=img, mask_image=mask,
                   guidance_scale=GUIDANCE_SCALE, num_inference_steps=STEPS,
                   generator=gen).images[0]
        dt = time.perf_counter() - t0
        out_path = _output_path(path, "known", cat_used, model_name, f"{seed}_SMOKE")
        out.save(out_path, quality=95)
        _save_and_log(["known", cat_used, path, model_name, "inpaint_random_mask", prompt, seed, out_path])
        _free_pipe(model_name)
        results.append((model_name, cat_used, dt, out_path))
        print(f"[SMOKE] {model_name} ({cat_used}) OK in {dt:.2f}s → {out_path}")
    else:
        print("[SMOKE] Kein known-Beispielbild gefunden (sdxl_inpaint übersprungen).")

    # --- 4) InstructPix2Pix (unknown) ---
    path = None; cat_used = None
    for cat in cat_order:
        p = _pick_any_sample_for(cat, "unknown")
        if p:
            path, cat_used = p, cat
            break
    if path:
        model_name = "instruct_pix2pix"
        pipe = _get_pipe(model_name)
        instruction = BALANCER.next("unknown", cat_used, "instruction")
        gen, seed = _torch_generator()
        img = _ensure_rgb(Image.open(path))
        # img = _resize_for_model(img, is_sdxl=False)
        t0 = time.perf_counter()
        out = pipe(image=img, prompt=instruction,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE,
                   image_guidance_scale=1.6, generator=gen).images[0]
        dt = time.perf_counter() - t0
        out_path = _output_path(path, "unknown", cat_used, model_name, f"{seed}_SMOKE")
        out.save(out_path, quality=95)
        _save_and_log(["unknown", cat_used, path, model_name, "instruction", instruction, seed, out_path])
        _free_pipe(model_name)
        results.append((model_name, cat_used, dt, out_path))
        print(f"[SMOKE] {model_name} ({cat_used}) OK in {dt:.2f}s → {out_path}")
    else:
        print("[SMOKE] Kein unknown-Beispielbild gefunden (instruct_pix2pix übersprungen).")

    # --- Summary ---
    print("\n=== SMOKE SUMMARY ===")
    if not results:
        print("Keine Läufe durchgeführt.")
    else:
        for m, c, dt, outp in results:
            print(f"{m:22s} | {c:10s} | {dt:6.2f}s | {outp}")
    print("=====================\n")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Starting manipulated image generation ...")

    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN fehlt in .env")
    login(token=HF_TOKEN)

    if not torch.cuda.is_available():
        print("WARN: CUDA nicht verfügbar – Ausführung auf CPU wird sehr langsam sein.", file=sys.stderr)

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
            FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA = _choose_half(k_cb_files)
            FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ = _choose_half(u_ffhq)

            print(
                f"[known/human] FaceForensics: {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS)} von {len(k_ff_files)}")
            print(
                f"[known/human] CelebA:        {len(FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA)} von {len(k_cb_files)}")
            print(f"[unknown/human] FFHQ:         {len(FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ)} von {len(u_ffhq)}")

        elif category == "building":
            known_arch = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "building", "realistic",
                                      "architecture")
            unknown_imn = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "unknown", "building", "realistic",
                                       "imagenet")

            k_files = _read_images_from(known_arch)
            u_files = _read_images_from(unknown_imn)

            FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE = _choose_half(k_files)
            FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET = _choose_half(u_files)

            print(
                f"[known/building] architecture: {len(FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE)} von {len(k_files)}")
            print(
                f"[unknown/building] imagenet:   {len(FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET)} von {len(u_files)}")

        elif category == "landscape":
            known_lhq = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "landscape", "realistic", "lhq")
            unknown_ls = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "unknown", "landscape", "realistic",
                                      "landscape")

            k_files = _read_images_from(known_lhq)
            u_files = _read_images_from(unknown_ls)

            FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ = _choose_half(k_files)
            FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE = _choose_half(u_files)

            print(
                f"[known/landscape] LHQ:         {len(FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ)} von {len(k_files)}")
            print(
                f"[unknown/landscape] LANDSCAPE: {len(FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE)} von {len(u_files)}")

    smoke_test()

    # manipulate_known()
    # manipulate_unknown_with_instruct_pix2pix()

    print("DONE.")

