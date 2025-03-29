import os
import cv2
import numpy as np
import pandas as pd
import csv
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import logging

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'keypoint_classifier')
CSV_PATH = os.path.join(MODEL_DIR, 'keypoint.csv')
LABEL_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier_label.csv')

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load model if available
model_path = os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5')
label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.npy')

if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = load_model(model_path)
    le = LabelEncoder()
    le.classes_ = np.load(label_encoder_path, allow_pickle=True)
else:
    model = None
    le = LabelEncoder()


def extract_keypoints(image):
    """Extract hand keypoints from the image using Mediapipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append(landmark.x)
                keypoints.append(landmark.y)
            return keypoints
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the model using stored keypoints."""
    global model, le

    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'No training data found'}), 400

    try:
        # Load dataset
        data = pd.read_csv(CSV_PATH, header=0)
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(np.unique(y))
        y = to_categorical(y, num_classes)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Save model
        model.save(model_path)
        np.save(label_encoder_path, le.classes_)

        loss, accuracy = model.evaluate(X_test, y_test)
        return jsonify({'message': 'Model trained successfully', 'accuracy': accuracy})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def extract_landmarks(image):
    """Extracts 21 hand landmarks from an image using MediaPipe Hands."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions (needed for projection)
    image_height, image_width, _ = image.shape
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append(landmark.x * image_width)  # Scale X
                keypoints.append(landmark.y * image_height)  # Scale Y
                keypoints.append(landmark.z)  # Keep Z as is
            return keypoints  # Return the first detected hand keypoints

    return None  # No hand detected

logging.basicConfig(level=logging.INFO)
def calculate_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_points = []
    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_points.append([landmark_x, landmark_y])
    return landmark_points

def draw_landmarks(image, landmark_points):
    for landmark in landmark_points:
        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)

def recognize_gesture(landmark_points):
    try:
        landmark_flattened = np.array(landmark_points).flatten().reshape(1, -1)
        prediction = model.predict(landmark_flattened)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        logging.info(f'Recognized gesture: {predicted_label}')
        return predicted_label
    except Exception as e:
        logging.error(f'Error recognizing gesture: {e}')
        return None

@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture_api():
    if model is None or le is None:
        return jsonify({'error': 'No trained model found. Please train the model first.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    # Convert to OpenCV format
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Resize the image
    input_size = 256
    image_rgb = cv2.resize(image_rgb, (input_size, input_size))

    # Use a new instance of Hands() for each request to avoid timestamp issues
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    # Process the image
    results = hands.process(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Assuming you have these functions to process and recognize gestures
            landmark_list = calculate_landmark_list(image_rgb, hand_landmarks)
            draw_landmarks(image_rgb, landmark_list)
            predicted_label = recognize_gesture(landmark_list)
            if predicted_label:
                return jsonify({'gesture': predicted_label})

    return jsonify({'error': 'No hand detected'}), 400
@app.route('/reset_model', methods=['POST'])
def reset_model():
    """Reset model and delete training data."""
    try:
        # Remove model and label files
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(label_encoder_path):
            os.remove(label_encoder_path)
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        if os.path.exists(LABEL_PATH):
            os.remove(LABEL_PATH)

        return jsonify({'message': 'Model and data reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
