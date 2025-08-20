import os
import random
import pickle

# Set your dataset path
dataset_dir =    "your path"     

video_paths = []
labels = []
label_map = {}
label_id = 0

# Loop through activity folders
for activity in sorted(os.listdir(dataset_dir)):
    activity_path = os.path.join(dataset_dir, activity)
    if os.path.isdir(activity_path):
        # Assign a new label for each activity
        label_map[activity] = label_id
        for video in os.listdir(activity_path):
            if video.endswith(".avi"):
                video_paths.append(os.path.join(activity_path, video))
                labels.append(label_id)
        label_id += 1

# Shuffle dataset
combined = list(zip(video_paths, labels))
random.shuffle(combined)
video_paths, labels = zip(*combined)

# Split: 80% train, 20% val
split_idx = int(0.8 * len(video_paths))
train_paths = video_paths[:split_idx]
train_labels = labels[:split_idx]
val_paths = video_paths[split_idx:]
val_labels = labels[split_idx:]

# Save split
with open("video_paths_labels.pkl", "wb") as f:
    pickle.dump(((train_paths, train_labels), (val_paths, val_labels), label_map), f)

# Print summary
print(f"Dataset preparation complete!")
print(f"Total videos: {len(video_paths)}")
print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}")
print(f"Activities: {label_map}")
