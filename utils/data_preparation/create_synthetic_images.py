"""Synthetischen Anteil des Fake-Datensatzes erstellen"""
import os
import re
import sys

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, \
    DPMSolverMultistepScheduler
from huggingface_hub import login, hf_hub_download, list_repo_files

from utils.config import CONFIG, CATEGORIES, PROMPTS, SYNTHETIC_VARIANTEN_BEKANNT, SYNTHETIC_VARIANTEN_UNBEKANNT, STEPS, \
    GUIDANCE_SCALE, HF_TOKEN
from utils.shared_methods import make_generator, write_csv_row, get_image_output

# Konstanten
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_HEADER = ["Kategorie", "Prompt", "Modell", "Path", "Seed"]
CSV_PATH = os.path.join(PROJECT_ROOT, CONFIG["synthetic_images_log_path"])

_HF_CACHE = {}

def _write_csv(row):
    """zum Loggen der Informationen der synthetischen Bilder"""
    write_csv_row(CSV_PATH, CSV_HEADER, row)

def _image_output(prompt, category, model_name, used_seed, known_or_unknown):
    """Vorbereiten des Speicherns eines manipulierten Bildes"""
    return get_image_output(prompt, category, model_name, used_seed, PROJECT_ROOT, known_or_unknown, "synthetic")

def _generate_image_with_stable_diffusion_15():
    """Bild mit Stable Diffusion 1.5 erzeugen"""
    print("Generating synthetic images with Stable Diffusion 1.5")
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for _ in range(SYNTHETIC_VARIANTEN_BEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_out, name = _image_output(prompt, category, "stable_diffusion_15", used_seed, "known")
                image = pipe(
                    prompt,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen
                ).images[0]
                image.save(image_out)
                _write_csv([category, prompt, "Stable Diffusion 1.5", name, used_seed])

    del pipe
    torch.cuda.empty_cache()


def _generate_image_with_juggernaut_xl_v9():
    """Bild mit Juggernaut XL v9 erzeugen"""
    print("Generating synthetic images with Juggernaut XL v9")
    files = list_repo_files("RunDiffusion/Juggernaut-XL-v9")
    safes = [f for f in files if f.lower().endswith(".safetensors")]
    if not safes:
        raise FileNotFoundError("Keine *.safetensors-Datei in RunDiffusion/Juggernaut-XL-v9 gefunden.")
    sdxl_candidates = [f for f in safes if re.search(r"(sdxl|xl)", f, re.IGNORECASE)]
    cand = sdxl_candidates[0] if sdxl_candidates else safes[0]
    if "jug_ckpt" not in _HF_CACHE:
        _HF_CACHE["jug_ckpt"] = hf_hub_download("RunDiffusion/Juggernaut-XL-v9", cand)
    pipe = StableDiffusionXLPipeline.from_single_file(
        _HF_CACHE["jug_ckpt"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    ).to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for _ in range(SYNTHETIC_VARIANTEN_BEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_out, name = _image_output(prompt, category, "juggernaut_xl_v9", used_seed, "known")
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen
                ).images[0]
                image.save(image_out)
                _write_csv([category, prompt, "Juggernaut XL v9", name, used_seed])

    del pipe
    torch.cuda.empty_cache()


def _generate_image_with_dreamlike_photoreal_20():
    """Bild mit Photoreal 2.0 erzeugen"""
    print("Generating synthetic images with Dreamlike PhotoReal 20")
    pipe = StableDiffusionPipeline.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for _ in range(SYNTHETIC_VARIANTEN_BEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_out, name = _image_output(prompt, category, "dreamlike_photoreal_20", used_seed, "known")
                image = pipe(
                    prompt,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen
                ).images[0]
                image.save(image_out)
                _write_csv([category, prompt, "Dreamlike PhotoReal 20", name, used_seed])

    del pipe
    torch.cuda.empty_cache()


def _generate_image_with_dreamshaper():
    """Bild mit Dreamshaper XL v2 Turbo erzeugen"""
    print("Generating synthetic images with Dreamshaper XL v2 Turbo")
    pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-xl-v2-turbo', torch_dtype=torch.float16,
                                                     variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for i in range(SYNTHETIC_VARIANTEN_UNBEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_out, name = _image_output(prompt, category, "dreamshaper", used_seed, "unknown")
                image = pipe(
                    prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=STEPS,
                    generator=gen
                ).images[0]
                image.save(image_out)
                _write_csv([category, prompt, "dreamshaper", name, used_seed])

    del pipe
    torch.cuda.empty_cache()


def create_images():
    """synthetische bekannte Fake-Bilder erzeugen"""
    _generate_image_with_stable_diffusion_15()
    _generate_image_with_dreamlike_photoreal_20()
    _generate_image_with_juggernaut_xl_v9()


def create_images_unbekannt():
    """synthetische unbekannte Fake-Bilder erzeugen"""
    _generate_image_with_dreamshaper()


if __name__ == "__main__":
    print("Starting ....")

    if not torch.cuda.is_available():
        print("WARN: CUDA nicht verfügbar – Ausführung auf CPU wird sehr langsam sein.", file=sys.stderr)

    login(token=HF_TOKEN, add_to_git_credential=True)

    create_images()
    create_images_unbekannt()

    print("DONE")
