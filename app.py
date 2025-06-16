from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import threading
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import time
import uuid
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)
camera = cv2.VideoCapture(0)

recognizer = sr.Recognizer()
translator = Translator()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Constants
IMG_SIZE = 128
DATASET_PATH = "D:\\PROJECT\\gesture\\dataset\\asl_alphabet_train\\asl_alphabet_train"
MODEL_PATH = "gesture_model.pkl"

# Shared variables
deaf_to_normal_text = ""
normal_to_deaf_text = ""
deaf_audio_filename = ""
last_deaf_timestamp = 0
gesture_text = ""
model = None
label_map = {}

# Ensure static/audio folders exist
os.makedirs("static/audio", exist_ok=True)

# Load dataset
def load_dataset():
    X, y = [], []
    label_map_local = {}
    label_id = 0

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path {DATASET_PATH} does not exist.")
        return np.array([]), np.array([]), {}

    for label_name in sorted(os.listdir(DATASET_PATH)):
        label_dir = os.path.join(DATASET_PATH, label_name)
        if not os.path.isdir(label_dir):
            continue
        label_map_local[label_id] = label_name
        for i, img_file in enumerate(os.listdir(label_dir)):
            if i >= 300:
                break
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label_id)
        label_id += 1

    X = np.array(X).reshape(-1, IMG_SIZE * IMG_SIZE) / 255.0
    y = np.array(y)
    return X, y, label_map_local

# Train or load model
def train_or_load_model():
    global model, label_map
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        with open("label_map.txt", "r") as f:
            label_map = eval(f.read())
    else:
        X, y, label_map = load_dataset()
        if len(X) == 0:
            raise ValueError("No training data found.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        with open("label_map.txt", "w") as f:
            f.write(str(label_map))

        # Accuracy
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")

        # Line plot
        plt.figure(figsize=(6, 4))
        plt.plot(["Train", "Test"], [train_acc, test_acc], marker='o', linestyle='-', color='blue')
        plt.title("Training vs Testing Accuracy")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig("static/accuracy_plot.png")
        plt.close()

# Recognize gesture
def recognize_gesture(frame):
    global gesture_text
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).reshape(1, -1) / 255.0
            prediction = model.predict(resized)[0]
            gesture_text = label_map[prediction]
    else:
        gesture_text = ""

    return frame

# Video feed
def generate_frames():
    global deaf_to_normal_text, deaf_audio_filename, last_deaf_timestamp
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = recognize_gesture(frame)
        if gesture_text:
            translated = translator.translate(gesture_text, src='en', dest='ta')
            deaf_to_normal_text = translated.text
            last_deaf_timestamp = int(time.time())
            deaf_audio_filename = ""
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Background speech recognition
def recognize_speech_from_mic():
    global deaf_to_normal_text, deaf_audio_filename, last_deaf_timestamp
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for Deaf Person...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                english_text = recognizer.recognize_google(audio, language="en-IN")
                translated = translator.translate(english_text, src='en', dest='ta')
                deaf_to_normal_text = translated.text
                last_deaf_timestamp = int(time.time())
                tts = gTTS(text=translated.text, lang='ta')
                filename = f"{uuid.uuid4().hex}.mp3"
                path = os.path.join("static/audio", filename)
                tts.save(path)
                deaf_audio_filename = filename
        except Exception as e:
            print("Speech recognition error:", e)
        time.sleep(1)

threading.Thread(target=recognize_speech_from_mic, daemon=True).start()

# Routes
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/deaf')
def deaf():
    return render_template("deaf.html")

@app.route('/normal')
def normal():
    return render_template("normal.html")

@app.route('/accuracy')
def accuracy():
    return render_template("accuracy.html", time=int(time.time()))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_deaf_to_normal')
def get_deaf_to_normal():
    return jsonify({
        'translated_text': deaf_to_normal_text,
        'audio_url': url_for('static', filename=f'audio/{deaf_audio_filename}') if deaf_audio_filename else '',
        'timestamp': last_deaf_timestamp
    })

@app.route('/normal_reply', methods=['POST'])
def normal_reply():
    global normal_to_deaf_text
    try:
        data = request.get_json()
        tamil_text = data.get("text", "")
        translated = translator.translate(tamil_text, src='ta', dest='en')
        normal_to_deaf_text = translated.text
        return jsonify({"translated_text": translated.text})
    except Exception as e:
        print("Error translating Tamil to English:", e)
        return jsonify({"translated_text": "Translation Error"}), 500

@app.route('/get_normal_to_deaf')
def get_normal_to_deaf():
    return jsonify({'translated_text': normal_to_deaf_text})

# Run the app
if __name__ == '__main__':
    train_or_load_model()
    app.run(debug=True)
