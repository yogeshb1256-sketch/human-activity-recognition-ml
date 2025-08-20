import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Define paths
train_dir ="D:/NEW/processed/train"
val_dir = "D:/NEW/processed/val"

# Load .npy files and labels
def load_data(data_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            filepath = os.path.join(data_dir, file)
            data = np.load(filepath)
            label = file.split('_')[0]
            X.append(data)
            y.append(label)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = load_data(train_dir)
print("Loading validation data...")
X_val, y_val = load_data(val_dir)

# Encode class labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# One-hot encode labels
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded)
y_val_onehot = tf.keras.utils.to_categorical(y_val_encoded)

# Shuffle training data
X_train, y_train_onehot = shuffle(X_train, y_train_onehot, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0

print("Data loaded. Building model...")

# Build the model
model = Sequential([
    TimeDistributed(Conv2D(16, (3,3), activation='relu'), input_shape=(64, 64, 64, 3)),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
print("Training started...")
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=10,
    batch_size=4,
    verbose=1
)

# Save model
model.save("har_model_5class.h5")
print("Model training complete and saved as har_model_5class.h5")
