import os
import random

import cv2

RNG = random.Random(42)

frames_per_video = 1
video_format = ".mp4"
image_format = ".jpg"

source_path = "../data_raw/faceforensics/"

output_path = "../images/known/human/realistic/faceforensics"


def extract_frames(video, output_dir, i):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return

    frame_number = RNG.randint(0, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        filename = f"faceforensics_{i}{image_format}"
        cv2.imwrite(os.path.join(output_dir, filename), frame)

    cap.release()


video_files = [f for f in os.listdir(source_path) if f.endswith(video_format)]
video_files.sort()

os.makedirs(output_path, exist_ok=True)
index = 0
for file in video_files:
    index += 1
    video_id = os.path.splitext(file)[0]
    video_path = os.path.join(source_path, file)
    print(f"Extrahiere Frames aus: {video_path}")
    extract_frames(video_path, output_path, index)

while index < 1050:
    index += 1
    file = RNG.choice(video_files)
    video_path = os.path.join(source_path, file)
    print(f"Extrahiere Frames aus: {video_path}")
    extract_frames(video_path, output_path, index)

print("✅ Frame-Extraktion abgeschlossen.")
