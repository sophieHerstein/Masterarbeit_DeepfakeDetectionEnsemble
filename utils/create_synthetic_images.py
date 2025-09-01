import csv
import os
import random
import re
import sys

from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download, list_repo_files
import torch

from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, StableDiffusionXLPipeline


from transformers import AutoTokenizer, AutoModel

from NextStep1Large.models.gen_pipeline import NextStepPipeline

from config import PROMPTS, CONFIG, CATEGORIES, VARIANTEN_BEKANNT, VARIANTEN_UNBEKANNT

# ------------------------------------------------------------------------------
# Pfade & Umgebungsvariablen
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEXTSTEP_DIR = os.path.join(PROJECT_ROOT, "NextStep1Large")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------------------------------------------------------------------
# Globals (Pipelines werden einmalig geladen und wiederverwendet)
# ------------------------------------------------------------------------------
_PIPELINES = {}
_HF_CACHE = {}

# Einheitliche Inferenzparameter
SD_STEPS = 28
SD_GUIDE_SD35 = 3.5
SD_GUIDE_OTHERS = 4.5

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _slugify(text):
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9\-._]", "", text)
    return text[:60] if len(text) > 60 else text

def get_image_output(image_category, model, image_prompt, index):
    p = _slugify(image_prompt)
    name = f"{image_category}_synthetic_{model}_{p}_{index}.jpg"
    image_out = os.path.join(
        PROJECT_ROOT,
        CONFIG["images_path"],
        image_category,
        "synthetic",
        model,
        name,
    )
    os.makedirs(os.path.dirname(image_out), exist_ok=True)
    return image_out


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
# Pipelines laden (einmalig)
# ------------------------------------------------------------------------------
def get_or_create_pipelines():
    if _PIPELINES:
        return _PIPELINES

    if not torch.cuda.is_available():
        print("WARN: CUDA nicht verfügbar – Ausführung auf CPU wird sehr langsam sein.", file=sys.stderr)

    # Hugging Face Login
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN fehlt in .env – benötigt für SD3.5/Juggernaut.")
    login(token=HF_TOKEN, add_to_git_credential=True)

    # SD 3.5 Large
    _PIPELINES["sd35"] = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Juggernaut XL v9 (SDXL Safetensors)
    files = list_repo_files("RunDiffusion/Juggernaut-XL-v9")
    safes = [f for f in files if f.lower().endswith(".safetensors")]
    if not safes:
        raise FileNotFoundError("Keine *.safetensors-Datei in RunDiffusion/Juggernaut-XL-v9 gefunden.")
    sdxl_candidates = [f for f in safes if re.search(r"(sdxl|xl)", f, re.IGNORECASE)]
    cand = sdxl_candidates[0] if sdxl_candidates else safes[0]
    if "jug_ckpt" not in _HF_CACHE:
        _HF_CACHE["jug_ckpt"] = hf_hub_download("RunDiffusion/Juggernaut-XL-v9", cand)
    _PIPELINES["jug"] = StableDiffusionXLPipeline.from_single_file(
        _HF_CACHE["jug_ckpt"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Dreamlike Photoreal 2.0
    _PIPELINES["dl20"] = StableDiffusionPipeline.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # NextStep 1 Large (lokal)
    tokenizer = AutoTokenizer.from_pretrained("stepfun-ai/NextStep-1-Large", trust_remote_code=True)
    model = AutoModel.from_pretrained(NEXTSTEP_DIR, trust_remote_code=True)
    _PIPELINES["ns1"] = NextStepPipeline(
        tokenizer=tokenizer,
        model=model,
        vae_name_or_path=os.path.join(NEXTSTEP_DIR, "vae"),
    ).to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    return _PIPELINES


# ------------------------------------------------------------------------------
# Generatoren
# ------------------------------------------------------------------------------
def generate_image_with_stable_diffusion_35(gen_pipes, image_prompt, image_category):
    print("Generating synthetic images with Stable Diffusion 3.5 Large")
    gen, used_seed = make_generator()
    image_output = get_image_output(image_category, "stable_diffusion_35", image_prompt, used_seed)
    image = gen_pipes["sd35"](
        image_prompt,
        num_inference_steps=SD_STEPS,
        guidance_scale=SD_GUIDE_SD35,
        generator=gen
    ).images[0]
    image.save(image_output)
    write_csv_row(image_category, image_prompt, "Stable Diffusion 3.5 Large", image_output, used_seed)


def generate_image_with_juggernaut_xl_v9(gen_pipes, image_prompt, image_category):
    print("Generating synthetic images with Juggernaut XL v9")
    gen, used_seed = make_generator()
    image_output = get_image_output(image_category, "juggernaut_xl_v9", image_prompt, used_seed)
    image = gen_pipes["jug"](
        prompt=image_prompt,
        num_inference_steps=SD_STEPS,
        guidance_scale=SD_GUIDE_OTHERS,
        generator=gen
    ).images[0]
    image.save(image_output)
    write_csv_row(image_category, image_prompt, "Juggernaut XL v9", image_output, used_seed)


def generate_image_with_dreamlike_photoreal_20(gen_pipes, image_prompt, image_category):
    print("Generating synthetic images with Dreamlike PhotoReal 20")
    gen, used_seed = make_generator()
    image_output = get_image_output(image_category, "dreamlike_photoreal_20", image_prompt, used_seed)
    image = gen_pipes["dl20"](
        image_prompt,
        num_inference_steps=SD_STEPS,
        guidance_scale=SD_GUIDE_OTHERS,
        generator=gen
    ).images[0]
    image.save(image_output)
    write_csv_row(image_category, image_prompt, "Dreamlike PhotoReal 20", image_output, used_seed)


def generate_image_with_nextstep_1_large(gen_pipes, image_prompt, image_category):
    print("Generating synthetic images with NextStep 1 Large")
    gen, used_seed = make_generator()
    image_output = get_image_output(image_category, "nextstep_1", image_prompt, used_seed)
    image = gen_pipes["ns1"].generate_image(
        image_prompt,
        hw=(512, 512),
        num_images_per_caption=1,
        cfg=7.5,
        cfg_img=1.0,
        cfg_schedule="constant",
        use_norm=False,
        num_sampling_steps=SD_STEPS,
        timesteps_shift=1.0,
        generator=gen
    )[0]
    image.save(image_output)
    write_csv_row(image_category, image_prompt, "NextStep-1-Large", image_output, used_seed)


# ------------------------------------------------------------------------------
# High-level Wrapper
# ------------------------------------------------------------------------------
def create_image(pipelines, image_prompt, image_category):
    print(f"Create image for category '{image_category}' with prompt '{image_prompt}'")
    generate_image_with_stable_diffusion_35(pipelines, image_prompt, image_category)
    generate_image_with_juggernaut_xl_v9(pipelines, image_prompt, image_category)
    generate_image_with_dreamlike_photoreal_20(pipelines, image_prompt, image_category)


def create_images_unbekannt(pipelines, image_prompt, image_category):
    print(f"Create images for unbekannten Datensatz for category '{image_category}' with prompt '{image_prompt}'")
    generate_image_with_nextstep_1_large(pipelines, image_prompt, image_category)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting ....")
    print("CUDA verfügbar:", torch.cuda.is_available())

    # Pipelines einmalig laden
    pipes = get_or_create_pipelines()

    # Hauptschleife
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            for i in range(VARIANTEN_BEKANNT):
                create_image(pipes, prompt, category)

            for j in range(VARIANTEN_UNBEKANNT):
                create_images_unbekannt(pipes, prompt, category)

    print("DONE")

