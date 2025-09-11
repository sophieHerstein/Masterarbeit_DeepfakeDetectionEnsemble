import os
import random
import shutil

RNG = random.Random(42)

def copy_random_images(src_folder, dst_folder, n, name_prefix="image"):
    # Alle Dateien im Quellordner auflisten
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # Nur Bilddateien (optional: nach Endung filtern)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if n > len(image_files):
        raise ValueError(f"Es gibt nur {len(image_files)} Bilder im Ordner, aber {n} wurden angefordert.")

    # Zufällige Auswahl treffen
    selected_files = RNG.sample(image_files, n)

    # Zielordner erstellen, falls er nicht existiert
    os.makedirs(dst_folder, exist_ok=True)

    # Kopieren mit neuem Namen
    for idx, filename in enumerate(selected_files, start=1):
        src_path = os.path.join(src_folder, filename)

        # Dateiendung behalten
        ext = os.path.splitext(filename)[1]

        # Neuen Namen zusammensetzen
        new_name = f"{name_prefix}_{idx}{ext}"

        dst_path = os.path.join(dst_folder, new_name)
        shutil.copy(src_path, dst_path)

    print(f"{n} Bilder erfolgreich nach '{dst_folder}' kopiert.")


if __name__ == '__main__':
    caleba_in = "../data_raw/img_align_celeba/img_align_celeba"
    caleba_out = "../images/known/human/realistic/celeba"
    ffhq_in = "../data_raw/ffhq/"
    ffhq_out = "../images/unknown/human/realistic/ffqh"
    landscape_in = "../data_raw/landscape/"
    landscape_out = "../images/unknown/landscape/realistic/landscape"
    lhq_in = "../data_raw/lhq/"
    lhq_out = "../images/known/landscape/realistic/lhq"
    architecture_in = "../data_raw/architecture/"
    architecture_out = "../images/known/building/realistic/architecture"
    imagenet_in = "../data_raw/imagenet/"
    imagenet_out = "../images/unknown/building/realistic/imagenet"
    copy_random_images(caleba_in, caleba_out, 1050, 'celeba')
    copy_random_images(ffhq_in, ffhq_out, 300, 'ffhq')
    copy_random_images(landscape_in, landscape_out, 150, 'landscape')
    copy_random_images(lhq_in, lhq_out, 2100, 'lhq')
    copy_random_images(architecture_in, architecture_out, 2100, 'architecture')
    copy_random_images(imagenet_in, imagenet_out, 150, 'imagenet')