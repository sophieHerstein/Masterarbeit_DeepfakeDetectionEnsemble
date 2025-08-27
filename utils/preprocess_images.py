import os
from PIL import Image, ImageFilter
from tqdm import tqdm

from utils.config import CONFIG


def get_grayscale(image_to_be_processed):
    img = Image.open(image_to_be_processed).convert('L')
    save_image(img)

def get_frequence(image_to_be_processed):
    #todo
    print(image_to_be_processed)

def get_edges(image_to_be_processed):
    img = Image.open(image_to_be_processed)
    img = img.convert("L")
    img = img.filter(ImageFilter.FIND_EDGES)

    save_image(img)


def save_image(processed_image):
    image_output = os.path.join(CONFIG["preprocessed_images_path"]) #todo: finalen pfad ergänzen
    processed_image.save(image_output)

#todo: prüfen obs funktioniert
if __name__ == "__main__":
    for root, _, files in os.walk(CONFIG["images_path"]): #todo richtigen pfad verwenden
        for file in tqdm(files, desc="Verarbeite Bilder"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:

                print("\nProcessing file:", file)
                get_grayscale(file)
                get_frequence(file)
                get_edges(file)

            except Exception as e:
                print(f"Fehler bei Datei {file}: {e}")
