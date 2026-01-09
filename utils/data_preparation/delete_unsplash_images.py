"""Unsplash Bilder aus Lizenzgründen vom lhq Datensatz entfernen"""

import json
import os

IMAGE_DIR = "../data_raw/lhq"
JSON_FILE = "../data_raw/metadata.json"

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))

deleted = 0
for entry in data:
    license_info = entry.get("license", "")
    filename = entry.get("filename", "")
    if "Unsplash License" in license_info:
        print(license_info)
        file_path = os.path.join(IMAGE_DIR, filename)
        print(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted += 1
            print(f"Gelöscht: {filename}")

print(f"\nFertig. {deleted} Bilder entfernt.")
