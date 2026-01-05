import os
import shutil

from config import CONFIG, CATEGORIES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

unknown_test_base_path = os.path.join(PROJECT_ROOT, "images", "unknown")
known_test_base_path = os.path.join(PROJECT_ROOT, CONFIG["splited_images_path"], "test")


def prepare_unknown_test_dirs():
    os.makedirs(os.path.join(PROJECT_ROOT, CONFIG['unknown_test_dir'], "0_real"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, CONFIG['unknown_test_dir'], "1_fake"), exist_ok=True)
    for cat in CATEGORIES:
        path = os.path.join(unknown_test_base_path, cat)
        for d in os.listdir(path):
            print(f"[INFO] -UNKNOWN- Copying {d} images for {cat}")
            for d2 in os.listdir(os.path.join(path, d)):
                if d == "realistic":
                    for f in os.listdir(os.path.join(path, d, d2)):
                        shutil.copy2(os.path.join(path, d, d2, f),
                                     os.path.join(PROJECT_ROOT, CONFIG['unknown_test_dir'], "0_real"))
                else:
                    for f in os.listdir(os.path.join(path, d, d2)):
                        shutil.copy2(os.path.join(path, d, d2, f),
                                     os.path.join(PROJECT_ROOT, CONFIG['unknown_test_dir'], "1_fake"))


def prepare_known_test_dirs():
    os.makedirs(os.path.join(PROJECT_ROOT, CONFIG['known_test_dir'], "0_real"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, CONFIG['known_test_dir'], "1_fake"), exist_ok=True)
    for cat in CATEGORIES:
        path = os.path.join(known_test_base_path, cat)
        for d in os.listdir(path):
            print(f"[INFO] -KNOWN- Copying {d} images for {cat}")
            if d == "0_real":
                for f in os.listdir(os.path.join(path, d)):
                    shutil.copy2(os.path.join(path, d, f),
                                 os.path.join(PROJECT_ROOT, CONFIG['known_test_dir'], "0_real"))
            else:
                for f in os.listdir(os.path.join(path, d)):
                    shutil.copy2(os.path.join(path, d, f),
                                 os.path.join(PROJECT_ROOT, CONFIG['known_test_dir'], "1_fake"))


if __name__ == '__main__':
    prepare_unknown_test_dirs()
    prepare_known_test_dirs()
