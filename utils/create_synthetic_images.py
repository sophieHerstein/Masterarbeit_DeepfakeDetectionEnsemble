# 3 Modelle, 3 Kategorien
# 1050 Bilder pro Kategorie, also 350 je Modell je Kategorie
# zusätzlich 150 Bilder je Kategorie von einem unbekannten Modell

import os
import csv
from config import PROMPTS, CONFIG, CATEGORIES, VARIANTEN_BEKANNT, VARIANTEN_UNBEKANNT
import torch
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline, StableDiffusionPipeline
from transformers import AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()  # sucht automatisch nach .env im Projektordner
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

def create_image(text_prompt, image_category, index):
    print(f"Create image for category '{image_category}' with prompt '{text_prompt}'")
    generate_image_with_stable_diffusion_35(text_prompt, image_category, index)
    generate_image_with_juggernaut_xl_v9(text_prompt, image_category, index)
    generate_image_with_dreamlike_photoreal_20(text_prompt, image_category, index)


def create_images_unbekannt(text_prompt, image_category, index):
    print(f"Create images for unbekannt")
    generate_image_with_nextstep_1_large(text_prompt, image_category, index)


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

# https://huggingface.co/stabilityai/stable-diffusion-3.5-large
def generate_image_with_stable_diffusion_35(text_prompt, image_category, index):
    print("Generating synthetic images with Stable Diffusion 3.5 Large")
    image_output = get_image_output(image_category, 'stable_diffusion_35', text_prompt, index)
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                    torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    image = pipe(
        text_prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]
    image.save(image_output)

# https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
def generate_image_with_juggernaut_xl_v9(text_prompt, image_category, index):
    print("Generating synthetic images with Juggernaut XL v9")
    image_output = get_image_output(image_category, 'juggernaut_xl_v9', text_prompt, index)
    pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9")
    image = pipe(text_prompt).images[0]
    image.save(image_output)

# https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0
def generate_image_with_dreamlike_photoreal_20(text_prompt, image_category, index):
    print("Generating synthetic images with Dreamlike PhotoReal 20")
    image_output = get_image_output(image_category, 'dreamlike_photoreal_20', text_prompt, index)
    pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(text_prompt).images[0]

    image.save(image_output)

# https://huggingface.co/stepfun-ai/NextStep-1-Large
def generate_image_with_nextstep_1_large(text_prompt, image_category, index):
    print("Generating synthetic images with nextstep 1 large")
    image_output = get_image_output(image_category, 'nextstep_1', text_prompt, index)
    model = AutoModelForCausalLM.from_pretrained("stepfun-ai/NextStep-1-Large", trust_remote_code=True,
                                                 torch_dtype="auto")
    model.to("cuda")

    image = model(text_prompt)
    image.save(image_output)

def get_image_output(image_category, model, text_prompt, index):
    image_out = os.path.join(CONFIG["images_path"], image_category, "synthetic", model, f"{image_category}_synthetic_{model}_{text_prompt.replace(" ", "-")}_{index}.jpg")
    os.makedirs(os.path.dirname(image_out), exist_ok=True)
    return image_out

if __name__ == "__main__":
    print("Starting ....")
    print("CUDA verfügbar:", torch.cuda.is_available())
    for category in CATEGORIES:
        for prompt in PROMPTS[category]:
            # for i in range(0, VARIANTEN_BEKANNT):
                create_image(prompt, category, 0)
                # create_image(prompt, category, i)
            # for j in range(0, VARIANTEN_UNBEKANNT):
                create_images_unbekannt(prompt, category, 0)
                # create_images_unbekannt(prompt, category, j)