import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image

# Load model
model = joblib.load("sign_model.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

st.title("🤟 Real-Time Sign Language Recognizer")

# Streamlit video capture
stframe = st.empty()
cap = cv2.VideoCapture(0)

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Cannot access camera.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            if len(landmarks) == 42:
                prediction = model.predict([landmarks])[0]
                cv2.putText(frame, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    # Display the frame
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB', use_column_width=True)

cap.release()
