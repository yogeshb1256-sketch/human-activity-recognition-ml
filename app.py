from flask import Flask, render_template, request
import os
import tensorflow as tf
from utils import extract_frames, preprocess_frames, predict_activity
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'avi', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('har_model_5class.h5')

# Class labels
CLASS_LABELS = ['Archery', 'Basketball', 'BenchPress', 'Bowling', 'PushUps']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'video' not in request.files:
        return render_template('result.html', prediction="No file part")
    
    file = request.files['video']
    if file.filename == '':
        return render_template('result.html', prediction="No selected file")
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert to mp4 if it's an AVI file
        if filename.endswith('.avi'):
            mp4_path = filepath.rsplit('.', 1)[0] + '.mp4'
            cap = cv2.VideoCapture(filepath)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(mp4_path, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()
            out.release()
            os.remove(filepath)
            filepath = mp4_path
            filename = os.path.basename(filepath)

        # Process and predict
        frames = extract_frames(filepath)
        preprocessed = preprocess_frames(frames)
        prediction = predict_activity(preprocessed, model)
        predicted_label = CLASS_LABELS[prediction]

        return render_template('result.html', prediction=predicted_label)

    return render_template('result.html', prediction="Invalid file format")

if __name__ == '__main__':
    app.run(debug=True)
