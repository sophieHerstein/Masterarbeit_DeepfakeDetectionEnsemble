import os

import cv2

# Parameter
frames_per_video = 1
video_format = ".mp4"
image_format = ".jpg"

# Quelle und Zielverzeichnisse
source_path = "../data_raw/faceforensics/"

output_path = "../images/known/human/realistic/faceforensics"


# Funktion zur Frame-Extraktion
def extract_frames(video, output_dir, v_id):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // frames_per_video)

    count = 0
    saved = 0
    while cap.isOpened() and saved < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            filename = f"faceforensics_{v_id}{image_format}"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
        count += 1
    cap.release()



video_files = [f for f in os.listdir(source_path) if f.endswith(video_format)]
video_files.sort()

os.makedirs(output_path, exist_ok=True)
for file in video_files:
    video_id = os.path.splitext(file)[0]
    video_path = os.path.join(source_path, file)
    print(f"Extrahiere Frames aus: {video_path}")
    extract_frames(video_path, output_path, video_id)

print("✅ Frame-Extraktion abgeschlossen.")
