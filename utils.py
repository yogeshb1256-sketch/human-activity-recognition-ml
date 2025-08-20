import cv2
import numpy as np

def extract_frames(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize to match model input
        frames.append(frame)

    cap.release()

    # Pad if fewer than 64 frames
    while len(frames) < num_frames:
        frames.append(np.zeros((64, 64, 3), dtype=np.uint8))

    return np.array(frames)

def preprocess_frames(frames):
    frames = frames.astype('float32') / 255.0
    return np.expand_dims(frames, axis=0)  # Shape: (1, 64, 64, 64, 3)

def predict_activity(preprocessed_frames, model):
    predictions = model.predict(preprocessed_frames)
    predicted_class = np.argmax(predictions)
    return predicted_class
