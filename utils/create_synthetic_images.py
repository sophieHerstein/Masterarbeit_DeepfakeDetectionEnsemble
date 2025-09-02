import csv
import os
import random
import re
import sys

from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download, list_repo_files
import torch

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline, AutoPipelineForText2Image, \
    DPMSolverMultistepScheduler

from config import PROMPTS, CONFIG, CATEGORIES, VARIANTEN_BEKANNT, VARIANTEN_UNBEKANNT

# ------------------------------------------------------------------------------
# Pfade & Umgebungsvariablen
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------------------------------------------------------------------
# Globals (Pipelines werden einmalig geladen und wiederverwendet)
# ------------------------------------------------------------------------------
_HF_CACHE = {}

# Einheitliche Inferenzparameter
STEPS = 28
GUIDANCE_SCALE = 4.5
SIZE = 512
# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _slugify(text):
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9\-._]", "", text)
    return text[:60] if len(text) > 60 else text

def get_image_output(image_category, model, image_prompt, index, unknown = False):
    p = _slugify(image_prompt)
    name = f"{image_category}_synthetic_{model}_{p}_{index}.jpg"
    image_out = os.path.join(
        PROJECT_ROOT,
        CONFIG["images_path"],
        "unknown" if unknown else "known",
        image_category,
        "synthetic",
        model,
        name
    )
    os.makedirs(os.path.dirname(image_out), exist_ok=True)
    return image_out, name


def write_csv_row(image_category, image_prompt, model, image_path, seed):
    csv_file = os.path.join(PROJECT_ROOT, CONFIG["synthetic_images_log_path"])
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Kategorie", "Prompt", "Modell", "Path", "Seed"])
        writer.writerow([image_category, image_prompt, model, image_path, seed])

def make_generator():
    seed = random.randint(1, 100000)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g, seed

# ------------------------------------------------------------------------------
# Generatoren
# ------------------------------------------------------------------------------
def generate_image_with_stable_diffusion_xl_base_10():
    print("Generating synthetic images with Stable Diffusion XL Base 1.0")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for _ in range(VARIANTEN_BEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_output, name = get_image_output(category, "stable_diffusion_xl_base_10", prompt, used_seed)
                image = pipe(
                    prompt,
                    height=SIZE,
                    width=SIZE,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen
                ).images[0]
                image.save(image_output)
                write_csv_row(category, prompt, "Stable Diffusion XL Base 1.0", name, used_seed)

    del pipe
    torch.cuda.empty_cache()


def generate_image_with_juggernaut_xl_v9():
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
            for _ in range(VARIANTEN_BEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_output, name = get_image_output(category, "juggernaut_xl_v9", prompt, used_seed)
                image = pipe(
                    prompt=prompt,
                    height=SIZE,
                    width=SIZE,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen
                ).images[0]
                image.save(image_output)
                write_csv_row(category, prompt, "Juggernaut XL v9", name, used_seed)

    del pipe
    torch.cuda.empty_cache()


def generate_image_with_dreamlike_photoreal_20():
    print("Generating synthetic images with Dreamlike PhotoReal 20")
    pipe = StableDiffusionPipeline.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for _ in range(VARIANTEN_BEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_output, name = get_image_output(category, "dreamlike_photoreal_20", prompt, used_seed)
                image = pipe(
                    prompt,
                    height=SIZE,
                    width=SIZE,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen
                ).images[0]
                image.save(image_output)
                write_csv_row(category, prompt, "Dreamlike PhotoReal 20", name, used_seed)

    del pipe
    torch.cuda.empty_cache()


def generate_image_with_dreamshaper():
    print("Generating synthetic images with Dreamshaper XL v2 Turbo")
    pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-xl-v2-turbo', torch_dtype=torch.float16,
                                                     variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for i in range(VARIANTEN_UNBEKANNT):
                print(f"Create image for category '{category}' with prompt '{prompt}'")
                gen, used_seed = make_generator()
                image_output, name = get_image_output(category, "dreamshaper", prompt, used_seed, True)
                image = pipe(
                    prompt,
                    height=SIZE,
                    width=SIZE,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=STEPS,
                    generator=gen
                ).images[0]
                image.save(image_output)
                write_csv_row(category, prompt, "dreamshaper", name, used_seed)

    del pipe
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------
# High-level Wrapper
# ------------------------------------------------------------------------------
def create_images():
    generate_image_with_stable_diffusion_xl_base_10()
    generate_image_with_dreamlike_photoreal_20()
    generate_image_with_juggernaut_xl_v9()


def create_images_unbekannt():
    generate_image_with_dreamshaper()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting ....")

    if not torch.cuda.is_available():
        print("WARN: CUDA nicht verfügbar – Ausführung auf CPU wird sehr langsam sein.", file=sys.stderr)

    # Hugging Face Login
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN fehlt in .env")
    login(token=HF_TOKEN, add_to_git_credential=True)


    create_images()
    create_images_unbekannt()

    print("DONE")

