import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import shutil

# === Settings ===
dataset_path = "your path"  # path to the folder containing class folders
output_path = 'processed'
selected_classes = ['Archery', 'Basketball', 'BenchPress', 'Bowling', 'PushUps']
IMG_SIZE = 64
SEQUENCE_LENGTH = 64
SPLIT_RATIO = 0.8  # 80% for training, 20% for validation

# Create output directories
train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Function to extract frames from a video
def extract_frames(video_path, max_frames=SEQUENCE_LENGTH):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len(frames) % frame_interval == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    # Pad or crop frames to SEQUENCE_LENGTH
    if len(frames) < max_frames:
        frames += [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * (max_frames - len(frames))
    else:
        frames = frames[:max_frames]

    return np.array(frames)

# Process dataset
for class_name in os.listdir(dataset_path):
    if class_name not in selected_classes:
        continue  # Skip other classes

    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    videos = os.listdir(class_dir)
    random.shuffle(videos)
    split_index = int(len(videos) * SPLIT_RATIO)

    for idx, video_name in enumerate(tqdm(videos, desc=f'Processing {class_name}')):
        video_path = os.path.join(class_dir, video_name)
        try:
            frames = extract_frames(video_path)
            npy_filename = f"{class_name}_{video_name.split('.')[0]}.npy"
            save_dir = train_path if idx < split_index else val_path
            np.save(os.path.join(save_dir, npy_filename), frames)
        except Exception as e:
            print(f"⚠️ Error processing {video_name}: {e}")

print("\nPreprocessing finished and .npy files saved only for selected classes!")
