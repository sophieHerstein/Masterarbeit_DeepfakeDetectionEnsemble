# 3 Modelle, 3 Kategorien
# 750 Bilder pro Kategorie, also 250 je Kategorie, je Modell
# gute diverse Prompts überlegen
import os

HUMAN_PROMPTS = ['Bli', 'blu'] #-> 25 Prompts á 10 Varianten
LANDSCAPE_PROMPTS = ['Bla', 'ble'] #-> 25 Prompts á 10 Varianten
BUILDING_PROMPTS = ['Blub', 'blob'] #-> 25 Prompts á 10 Varianten


# models: https://huggingface.co/stabilityai/stable-diffusion-3.5-large
# https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
# https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0
#


def create_image(text_prompt, category, index):
    print(f"Create image for prompt '{text_prompt}' with category '{category}'")
    model = "PLACEHOLDER"
    out = os.path.join('images', category, 'synthetic', model, f"{category}_sythetic_{model}_{prompt.replace(" ", "-")}_{index}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    #csv zum loggen


if __name__ == '__main__':
    for i in range(0, 10):
        for prompt in HUMAN_PROMPTS:
            create_image(prompt, 'human', i)

        for prompt in LANDSCAPE_PROMPTS:
            create_image(prompt, 'landscape', i)

        for prompt in BUILDING_PROMPTS:
            create_image(prompt, 'building', i)
