from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import threading
import speech_recognition as sr
from googletrans import Translator
import os
import time
import uuid
import mediapipe as mp
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

recognizer = sr.Recognizer()
translator = Translator()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Shared variables
deaf_to_normal_text = ""
normal_to_deaf_text = ""
deaf_audio_filename = ""
last_deaf_timestamp = 0
gesture_text = ""

# --- Gesture Recognition ---
def recognize_gesture(frame):
    global gesture_text
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_tips_ids = [4, 8, 12, 16, 20]
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            fingers = []
            for tip in finger_tips_ids:
                if landmarks[tip][1] < landmarks[tip - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)
            if total_fingers == 5:
                gesture_text = "Hello"
            elif total_fingers == 0:
                gesture_text = "Bye"
            elif total_fingers == 1:
                gesture_text = "Yes"
            elif total_fingers == 2:
                gesture_text = "No"
            else:
                gesture_text = ""
    else:
        gesture_text = ""

    return frame

# --- Gesture Video Stream ---
def generate_frames():
    global deaf_to_normal_text, deaf_audio_filename, last_deaf_timestamp

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = recognize_gesture(frame)

        if gesture_text:
            print("🖐️ Gesture Detected:", gesture_text)
            translated = translator.translate(gesture_text, src='en', dest='ta')
            deaf_to_normal_text = translated.text
            last_deaf_timestamp = int(time.time())

            # No text-to-speech conversion
            deaf_audio_filename = ""

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Speech Thread for Deaf Person ---
def recognize_speech_from_mic():
    global deaf_to_normal_text, deaf_audio_filename, last_deaf_timestamp
    from gtts import gTTS  # Only used here for speech
    while True:
        try:
            with sr.Microphone() as source:
                print("🎧 Listening for Deaf Person (EN)...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("🧠 Recognizing...")

                english_text = recognizer.recognize_google(audio, language="en-IN")
                print("🗣️ English recognized:", english_text)

                translated = translator.translate(english_text, src='en', dest='ta')
                deaf_to_normal_text = translated.text
                last_deaf_timestamp = int(time.time())

                tts = gTTS(text=translated.text, lang='ta')
                filename = f"{uuid.uuid4().hex}.mp3"
                path = os.path.join("static", "audio", filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tts.save(path)
                deaf_audio_filename = filename

        except Exception as e:
            print("❌ Error in Deaf thread:", e)
        time.sleep(1)

# Start background speech thread
threading.Thread(target=recognize_speech_from_mic, daemon=True).start()

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/deaf')
def deaf():
    return render_template("deaf.html")

@app.route('/normal')
def normal():
    return render_template("normal.html")

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
        print("❌ Error translating Tamil to English:", e)
        return jsonify({"translated_text": "Translation Error"}), 500

@app.route('/get_normal_to_deaf')
def get_normal_to_deaf():
    return jsonify({'translated_text': normal_to_deaf_text})

if __name__ == '__main__':
    app.run(debug=True)
