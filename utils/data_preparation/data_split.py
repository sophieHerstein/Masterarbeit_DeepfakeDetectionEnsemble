import os
import shutil
import random

from config import CONFIG, CATEGORIES

LABEL_DIR = {"real": "0_real", "fake": "1_fake"}
SUBSETS = ["train", "val", "test"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RNG = random.Random(42)

# building\
#     manipulated\
#         sd2\ -350- -> 250/50/50
#         sd15\ -350- -> 250/50/50
#         sd\ -350- -> 250/50/50
#     realistic\
#         architecture\ -2100- -> 1500/300/300
#     synthetic\
#         dreamlike\ -350- -> 250/50/50
#         juggernaut\ -350- -> 250/50/50
#         sd\ -350- -> 250/50/50
# human\
#     manipulated\
#         sd2\ -175-
#         sd15\ -175-
#         sd\ -175-
#     realistic\
#         celeba\ -1050- -> 750/150/150
#         faceforensics\ -1050- -> 750/150/150
#     synthetic\
#         dreamlike\ -350- -> 250/50/50
#         juggernaut\ -350- -> 250/50/50
#         sd\ -350- -> 250/50/50
# human_2\
#     manipulated\
#         sd2\ -175-
#         sd15\ -175-
#         sd\ -175-
# landscape\
#     manipulated\
#         sd2\ -350- -> 250/50/50
#         sd15\ -350- -> 250/50/50
#         sd\ -350- -> 250/50/50
#     realistic\
#         lhq\ -2100- -> 1500/300/300
#     synthetic\
#         dreamlike\ -350- -> 250/50/50
#         juggernaut\ -350- -> 250/50/50
#         sd\ -350- -> 250/50/50

TRAIN_FOR_350 = 250
TEST_AND_VAL_FOR_350 = 50
TRAIN_FOR_2100 = 1500
TEST_AND_VAL_FOR_2100 = 300
TRAIN_FOR_1050 = 750
TEST_AND_VAL_FOR_1050 = 150
TRAIN_FOR_175 = 125
TEST_AND_VAL_FOR_175 = 25


def split_human():
    print("[INFO] Split images for human")
    human_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "human")
    human2_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "human_2")

    human_manipulated_root = os.path.join(human_root, "manipulated")
    human2_manipulated_root = os.path.join(human2_root, "manipulated")
    human_synthetic_root = os.path.join(human_root, "synthetic")
    human_realistic_root = os.path.join(human_root, "realistic")

    for d in os.listdir(human_manipulated_root):
        file_path = os.path.join(human_manipulated_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_175]
        val = files[TEST_AND_VAL_FOR_175:TEST_AND_VAL_FOR_175 + TEST_AND_VAL_FOR_175]
        train = files[TEST_AND_VAL_FOR_175 + TEST_AND_VAL_FOR_175:]
        copy_files(file_path, test, "test", "human", "1_fake")
        copy_files(file_path, train, "train", "human", "1_fake")
        copy_files(file_path, val, "val", "human", "1_fake")

    for d in os.listdir(human2_manipulated_root):
        file_path = os.path.join(human2_manipulated_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_175]
        val = files[TEST_AND_VAL_FOR_175:TEST_AND_VAL_FOR_175 + TEST_AND_VAL_FOR_175]
        train = files[TEST_AND_VAL_FOR_175 + TEST_AND_VAL_FOR_175:]
        copy_files(file_path, test, "test", "human", "1_fake")
        copy_files(file_path, train, "train", "human", "1_fake")
        copy_files(file_path, val, "val", "human", "1_fake")

    for d in os.listdir(human_synthetic_root):
        file_path = os.path.join(human_synthetic_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_350]
        val = files[TEST_AND_VAL_FOR_350:TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350]
        train = files[TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350:]
        copy_files(file_path, test, "test", "human", "1_fake")
        copy_files(file_path, train, "train", "human", "1_fake")
        copy_files(file_path, val, "val", "human", "1_fake")

    for d in os.listdir(human_realistic_root):
        file_path = os.path.join(human_realistic_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_1050]
        val = files[TEST_AND_VAL_FOR_1050:TEST_AND_VAL_FOR_1050 + TEST_AND_VAL_FOR_1050]
        train = files[TEST_AND_VAL_FOR_1050 + TEST_AND_VAL_FOR_1050:]
        copy_files(file_path, test, "test", "human", "0_real")
        copy_files(file_path, train, "train", "human", "0_real")
        copy_files(file_path, val, "val", "human", "0_real")


def split_building():
    print("[INFO] Split images for building")
    building_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "building")

    building_manipulated_root = os.path.join(building_root, "manipulated")
    building_synthetic_root = os.path.join(building_root, "synthetic")
    building_realistic_root = os.path.join(building_root, "realistic")

    for d in os.listdir(building_manipulated_root):
        file_path = os.path.join(building_manipulated_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_350]
        val = files[TEST_AND_VAL_FOR_350:TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350]
        train = files[TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350:]
        copy_files(file_path, test, "test", "building", "1_fake")
        copy_files(file_path, train, "train", "building", "1_fake")
        copy_files(file_path, val, "val", "building", "1_fake")

    for d in os.listdir(building_synthetic_root):
        file_path = os.path.join(building_synthetic_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_350]
        val = files[TEST_AND_VAL_FOR_350:TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350]
        train = files[TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350:]
        copy_files(file_path, test, "test", "building", "1_fake")
        copy_files(file_path, train, "train", "building", "1_fake")
        copy_files(file_path, val, "val", "building", "1_fake")

    for d in os.listdir(building_realistic_root):
        file_path = os.path.join(building_realistic_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_2100]
        val = files[TEST_AND_VAL_FOR_2100:TEST_AND_VAL_FOR_2100 + TEST_AND_VAL_FOR_2100]
        train = files[TEST_AND_VAL_FOR_2100 + TEST_AND_VAL_FOR_2100:]
        copy_files(file_path, test, "test", "building", "0_real")
        copy_files(file_path, train, "train", "building", "0_real")
        copy_files(file_path, val, "val", "building", "0_real")


def split_landscape():
    print("[INFO] Split images for landscape")
    landscape_root = os.path.join(PROJECT_ROOT, CONFIG["images_path"], "known", "landscape")

    landscape_manipulated_root = os.path.join(landscape_root, "manipulated")
    landscape_synthetic_root = os.path.join(landscape_root, "synthetic")
    landscape_realistic_root = os.path.join(landscape_root, "realistic")

    for d in os.listdir(landscape_manipulated_root):
        file_path = os.path.join(landscape_manipulated_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_350]
        val = files[TEST_AND_VAL_FOR_350:TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350]
        train = files[TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350:]
        copy_files(file_path, test, "test", "landscape", "1_fake")
        copy_files(file_path, train, "train", "landscape", "1_fake")
        copy_files(file_path, val, "val", "landscape", "1_fake")

    for d in os.listdir(landscape_synthetic_root):
        file_path = os.path.join(landscape_synthetic_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_350]
        val = files[TEST_AND_VAL_FOR_350:TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350]
        train = files[TEST_AND_VAL_FOR_350 + TEST_AND_VAL_FOR_350:]
        copy_files(file_path, test, "test", "landscape", "1_fake")
        copy_files(file_path, train, "train", "landscape", "1_fake")
        copy_files(file_path, val, "val", "landscape", "1_fake")

    for d in os.listdir(landscape_realistic_root):
        file_path = os.path.join(landscape_realistic_root, d)
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        shuffle_files(files)
        test = files[:TEST_AND_VAL_FOR_2100]
        val = files[TEST_AND_VAL_FOR_2100:TEST_AND_VAL_FOR_2100 + TEST_AND_VAL_FOR_2100]
        train = files[TEST_AND_VAL_FOR_2100 + TEST_AND_VAL_FOR_2100:]
        copy_files(file_path, test, "test", "landscape", "0_real")
        copy_files(file_path, train, "train", "landscape", "0_real")
        copy_files(file_path, val, "val", "landscape", "0_real")


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

    # Ausgabeordner vorbereiten
    for subset in SUBSETS:
        for cat in CATEGORIES:
            for label in ["real", "fake"]:
                os.makedirs(os.path.join(output_root, subset, cat, LABEL_DIR[label]), exist_ok=True)

    split_dataset()
