"""Manipulierten Anteil des Fake-Datensatzes erstellen"""
import os
import sys

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, \
    StableDiffusionInstructPix2PixPipeline

from huggingface_hub import login

from utils.config import EDIT_LIBRARY, CATEGORIES, CONFIG, MANIPULATED_VARIANTEN_BEKANNT, \
    MANIPULATED_HUMAN_VARIANTEN_BEKANNT, MANIPULATED_VARIANTEN_UNBEKANNT, RNG, GUIDANCE_SCALE, STEPS, HF_TOKEN
from utils.shared_methods import make_generator, write_csv_row, get_image_output

# Konstanten
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(PROJECT_ROOT, CONFIG["manipulated_images_log_path"])
CSV_HEADER = ["Kategorie", "OriginalPath", "Modell", "EditType", "InstructionOrPrompt", "Seed", "OutputPath"]

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


def _read_images_from(folder):
    """Bilder aus Ordner einlesen"""
    if not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files


def _choose_half(files):
    """Hälfte der Bilder für die Manipulation auswählen"""
    n = len(files) // 2
    return RNG.sample(files, n) if n > 0 else []


def write_csv(row):
    """zum Loggen der Informationen der manipulierten Bilder"""
    write_csv_row(CSV_PATH, CSV_HEADER, row)


def _image_output(prompt, category, model_name, used_seed, known_or_unknown):
    """Vorbereiten des Speicherns eines manipulierten Bildes"""
    return get_image_output(prompt, category, model_name, used_seed, PROJECT_ROOT, known_or_unknown, "manipulated")


def _dummy_checker(images):
    """Dummy Checker um Safety Check der Pipe zu disablen"""
    return images, [False] * len(images)


def _disable_safety_checker(pipe):
    """disablen des Safety Checks der Pipe"""
    pipe.safety_checker = _dummy_checker
    return pipe


def _resize(img, max_side=768):
    """Bildgröße für das manipulierende Modell anpassen"""
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = int((w * scale) // 8 * 8) or 8
    new_h = int((h * scale) // 8 * 8) or 8
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def _sorted_coords(ax, ay, bx, by, min_side, w, h):
    """Hilfsfunktion zum Sortieren der Koordinaten für das Einfügen der Maske"""
    x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
    y0, y1 = (ay, by) if ay <= by else (by, ay)

    if x1 - x0 < min_side:
        x1 = min(w - 1, x0 + min_side)
    if y1 - y0 < min_side:
        y1 = min(h - 1, y0 + min_side)
    return x0, y0, x1, y1


def _random_mask(img_size):
    """Zufällige Maske für die Inpainting Modelle"""
    w, h = img_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    min_side = max(8, min(w, h) // 12)

    count = RNG.randint(2, 4)
    for _ in range(count):
        try:
            ax = RNG.randint(0, max(0, w - 1))
            ay = RNG.randint(0, max(0, h - 1))
            bx = RNG.randint(0, max(0, w - 1))
            by = RNG.randint(0, max(0, h - 1))
            x0, y0, x1, y1 = _sorted_coords(ax, ay, bx, by, min_side, w, h)

            if RNG.random() < 0.5:
                draw.ellipse([x0, y0, x1, y1], fill=255)
            else:
                n = RNG.randint(3, 6)
                pts = []
                for _i in range(n):
                    px = RNG.randint(x0, x1)
                    py = RNG.randint(y0, y1)
                    pts.append((px, py))
                draw.polygon(pts, fill=255)
        except Exception:
            cx, cy = w // 2, h // 2
            x0 = max(0, cx - min_side)
            y0 = max(0, cy - min_side)
            x1 = min(w - 1, cx + min_side)
            y1 = min(h - 1, cy + min_side)
            draw.rectangle([x0, y0, x1, y1], fill=255)

    return mask

def _manipulate_image(img, pipe, prompt):
    """Bildmanipulation für Inpainting Modelle"""
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
    return out, seed


def _manipulate_image_with_stable_diffusion_15_img2img():
    """img2img Manipulation mit Stable Diffusion 1.5"""
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
                    out_path, image_name = _image_output(prompt, category, "stable_diffusion_15_imag2img", seed,
                                                         "known")
                    out.save(out_path)
                    write_csv([category, path, "stable_diffusion_15_imag2img", "img2img", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()


def _manipulate_image_with_stable_diffusion_inpainting():
    """Inpainting Manipulation mit Stable Diffusion"""
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
                    out, seed = _manipulate_image(img, pipe, prompt)
                    out_path, image_name = _image_output(prompt, category, "stable_diffusion_inpainting", seed, "known")
                    out.save(out_path)
                    write_csv([category, path, "stable_diffusion_inpainting", "inpaint", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()



def _manipulate_image_with_stable_diffusion_2_inpainting():
    """Inpainting Manipulation mit Stable Diffusion 2"""
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
                    out, seed = _manipulate_image(img, pipe, prompt)
                    out_path, image_name = _image_output(prompt, category, "stable_diffusion_2_inpainting", seed,
                                                         "known")
                    out.save(out_path)
                    write_csv(
                        [category, path, "stable_diffusion_2_inpainting", "inpaint", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()


def _manipulate_image_with_instruct_pix2pix():
    """Manipulation mit Instruct Pix2Pix"""
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
                    out_path, image_name = _image_output(prompt, category, "instruct_pix2pix", seed, "unknown")
                    out.save(out_path)
                    write_csv([category, path, "instruct_pix2pix", "instruction", prompt, seed, image_name])

    del pipe
    torch.cuda.empty_cache()


def _selected_known_for_category_and_manipulation(category, manipulation):
    """Aufteilung der bekannten Real-Bilder für die Manipulation vorbereiten"""
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
    else:
        if manipulation == 'sd_img2img':
            return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_15_IMG2IMG
        elif manipulation == 'sd_inpaint':
            return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_INPAINTING
        else:
            return FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_2_INPAINTING


def _selected_unknown_for_category(category):
    """Aufteilung der unbekannten Real-Bilder für die Manipulation vorbereiten"""
    if category == "human":
        return FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ
    elif category == "building":
        return FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET
    else:
        return FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE


def _dataset_path(*parts):
    """Pfad des Datensatzes unter dem die Real-Bilder liegen"""
    return os.path.join(PROJECT_ROOT, CONFIG["images_path"], *parts)


def _load_files(*parts):
    """Bilder laden"""
    return _read_images_from(_dataset_path(*parts))


def _split_into_three(files):
    """Bilder in drei gleich große Anteile unterteilen"""
    a, b, c = [], [], []
    for i, f in enumerate(files):
        if i % 3 == 0:
            a.append(f)
        elif i % 3 == 1:
            b.append(f)
        else:
            c.append(f)
    return a, b, c


def _prepare_known_dataset(
        all_files,
        label,
        target_all,
        target_img2img,
        target_inpaint,
        target_sd2_inpaint,
):
    """
    finale Bildaufteilung für Manipulation des bekannten Teils
    """
    chosen = _choose_half(all_files)

    target_all.clear()
    target_all.extend(chosen)

    img2img, inpaint, sd2 = _split_into_three(chosen)

    target_img2img.clear()
    target_inpaint.clear()
    target_sd2_inpaint.clear()
    target_img2img.extend(img2img)
    target_inpaint.extend(inpaint)
    target_sd2_inpaint.extend(sd2)

    print(f"[known] {label}: {len(chosen)} von {len(all_files)}")
    print(f"[img2img] {label}: {len(img2img)} von {len(chosen)}")
    print(f"[sd_inpaint] {label}: {len(inpaint)} von {len(chosen)}")
    print(f"[sd2_inpaint] {label}: {len(sd2)} von {len(chosen)}")
    print("")


def _prepare_unknown_dataset(
        all_files,
        label,
        target_all,
):
    """
    finale Bildaufteilung für Manipulation des unbekannten Teils
    """
    chosen = _choose_half(all_files)
    target_all.clear()
    target_all.extend(chosen)
    print(f"[unknown] {label}: {len(chosen)} von {len(all_files)}")
    print("")


def get_images_for_manipulation():
    """Bilder laden"""
    for category in CATEGORIES:
        if category == "human":
            k_ff_files = _load_files("known", "human", "realistic", "faceforensics")
            k_cb_files = _load_files("known", "human", "realistic", "celeba")
            u_ffhq_files = _load_files("unknown", "human", "realistic", "ffhq")

            _prepare_known_dataset(
                k_ff_files,
                label="human/FaceForensics",
                target_all=FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS,
                target_img2img=FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_15_IMG2IMG,
                target_inpaint=FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_INPAINTING,
                target_sd2_inpaint=FILES_FOR_MANIPULATION_HUMAN_KNOWN_FACEFORENSICS_SD_2_INPAINTING,
            )

            _prepare_known_dataset(
                k_cb_files,
                label="human/CelebA",
                target_all=FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA,
                target_img2img=FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_15_IMG2IMG,
                target_inpaint=FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_INPAINTING,
                target_sd2_inpaint=FILES_FOR_MANIPULATION_HUMAN_KNOWN_CELEBA_SD_2_INPAINTING,
            )

            _prepare_unknown_dataset(
                u_ffhq_files,
                label="human/FFHQ",
                target_all=FILES_FOR_MANIPULATION_HUMAN_UNKNOWN_FFHQ,
            )

        elif category == "building":
            k_files = _load_files("known", "building", "realistic", "architecture")
            u_files = _load_files("unknown", "building", "realistic", "imagenet")

            _prepare_known_dataset(
                k_files,
                label="building/architecture",
                target_all=FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE,
                target_img2img=FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_15_IMG2IMG,
                target_inpaint=FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_INPAINTING,
                target_sd2_inpaint=FILES_FOR_MANIPULATION_BUILDING_KNOWN_ARCHITECTURE_SD_2_INPAINTING,
            )

            _prepare_unknown_dataset(
                u_files,
                label="building/imagenet",
                target_all=FILES_FOR_MANIPULATION_BUILDING_UNKNOWN_IMAGENET,
            )

        elif category == "landscape":
            k_files = _load_files("known", "landscape", "realistic", "lhq")
            u_files = _load_files("unknown", "landscape", "realistic", "landscape")

            _prepare_known_dataset(
                k_files,
                label="landscape/LHQ",
                target_all=FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ,
                target_img2img=FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_15_IMG2IMG,
                target_inpaint=FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_INPAINTING,
                target_sd2_inpaint=FILES_FOR_MANIPULATION_LANDSCAPE_KNOWN_LHQ_SD_2_INPAINTING,
            )

            _prepare_unknown_dataset(
                u_files,
                label="landscape/LANDSCAPE",
                target_all=FILES_FOR_MANIPULATION_LANDSCAPE_UNKNOWN_LANDSCAPE,
            )


def manipulate_images():
    """Bekannte Bilder manipulieren"""
    _manipulate_image_with_stable_diffusion_15_img2img()
    _manipulate_image_with_stable_diffusion_inpainting()
    _manipulate_image_with_stable_diffusion_2_inpainting()


def manipulate_images_unbekannt():
    """unbekannte Bilder manipulieren"""
    _manipulate_image_with_instruct_pix2pix()


if __name__ == "__main__":
    print("Starting manipulated image generation ...")

    if not torch.cuda.is_available():
        print("WARN: CUDA nicht verfügbar – Ausführung auf CPU wird sehr langsam sein.", file=sys.stderr)

    login(token=HF_TOKEN)

    get_images_for_manipulation()

    manipulate_images()
    manipulate_images_unbekannt()

    print("DONE.")
