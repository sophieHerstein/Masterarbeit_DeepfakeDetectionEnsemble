"""aufteilen der Bilder in Test-, Trainings- und Validierungsdaten"""
import os
import shutil

from utils.config import CONFIG, CATEGORIES, RNG

LABEL_DIR = {"real": "0_real", "fake": "1_fake"}
SUBSETS = ["train", "val", "test"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_AND_VAL_FOR_350 = 50
TEST_AND_VAL_FOR_2100 = 300
TEST_AND_VAL_FOR_1050 = 150
TEST_AND_VAL_FOR_175 = 25


def split_human():
    print("[INFO] Split images for human")
    human_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "human")
    human2_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "human_2")

    human_manipulated_root = os.path.join(human_root, "manipulated")
    human2_manipulated_root = os.path.join(human2_root, "manipulated")
    human_synthetic_root = os.path.join(human_root, "synthetic")
    human_realistic_root = os.path.join(human_root, "realistic")

    split_images(human_manipulated_root, "human", "manipulated")
    split_images(human2_manipulated_root, "human", "manipulated")
    split_images(human_synthetic_root, "human", "synthetic")
    split_images(human_realistic_root, "human", "realistic")


def split_building():
    print("[INFO] Split images for building")
    building_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "building")

    building_manipulated_root = os.path.join(building_root, "manipulated")
    building_synthetic_root = os.path.join(building_root, "synthetic")
    building_realistic_root = os.path.join(building_root, "realistic")

    split_images(building_manipulated_root, "building", "manipulated")
    split_images(building_synthetic_root, "building", "synthetic")
    split_images(building_realistic_root, "building", "realistic")


def split_landscape():
    print("[INFO] Split images for landscape")
    landscape_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "landscape")

    landscape_manipulated_root = os.path.join(landscape_root, "manipulated")
    landscape_synthetic_root = os.path.join(landscape_root, "synthetic")
    landscape_realistic_root = os.path.join(landscape_root, "realistic")

    split_images(landscape_manipulated_root, "landscape", "manipulated")
    split_images(landscape_synthetic_root, "landscape", "synthetic")
    split_images(landscape_realistic_root, "landscape", "realistic")


def split_images(root, category, image_type):
    if category == "human" and image_type == "manipulated":
        split = TEST_AND_VAL_FOR_175
    elif (category == "building" and image_type != "realistic") or (category == "landscape" and image_type != "realistic") or (category == "human" and image_type == "synthetic"):
        split = TEST_AND_VAL_FOR_350
    elif category == "human" and image_type == "realistic":
        split = TEST_AND_VAL_FOR_1050
    else:
        split = TEST_AND_VAL_FOR_2100
    for d in os.listdir(root):
        file_path = os.path.join(root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:split]
        val = files[split:split + split]
        train = files[split + split:]
        copy_files(file_path, test, "test", category, "1_fake")
        copy_files(file_path, train, "train", category, "1_fake")
        copy_files(file_path, val, "val", category, "1_fake")


def split_dataset():
    split_human()
    split_building()
    split_landscape()


def get_path(typ, category, fake_or_real):
    return os.path.join(PROJECT_ROOT, CONFIG["splited_images_path"], typ, category, fake_or_real)


def copy_files(source_path, arr, typ, category, fake_or_real):
    for f in arr:
        shutil.copy2(os.path.join(source_path, f), get_path(typ, category, fake_or_real))
    print(
        f"[INFO] Copied {len(os.listdir(source_path))} {typ} images for {fake_or_real} {category} images")


def shuffle_files(files):
    return RNG.shuffle(files)


if __name__ == "__main__":
    output_root = os.path.join(PROJECT_ROOT, CONFIG["splited_images_path"])

    for subset in SUBSETS:
        for cat in CATEGORIES:
            for label in ["real", "fake"]:
                os.makedirs(os.path.join(output_root, subset, cat, LABEL_DIR[label]), exist_ok=True)

    split_dataset()
