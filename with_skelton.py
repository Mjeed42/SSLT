import cv2
import keras
import time
import numpy as np
import mediapipe as mp
from collections import deque

# Added for TTS from Code 1
from streamlit_TTS import text_to_speech
from gtts.lang import tts_langs

# Define available languages
langs = tts_langs().keys()

# Function to speak prediction if not repeated
def speak_prediction(prediction, lang='ar'):
    if prediction != getattr(speak_prediction, "last_spoken", None):
        text_to_speech(text=prediction, language=lang, key=f"tts_{prediction}_{lang}_{int(time.time())}")
        speak_prediction.last_spoken = prediction


# Load your trained model
model = keras.models.load_model('3_words_model_new.keras')
actions = ['assalam alaikum', 'sabah alkhair', 'kaif alhal']  # Replace with your real class names
threshold = 0.7  # Minimum confidence to accept prediction

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])  # shape (225,)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def main():
    sequence = []
    sentence = []

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detection
            image, results = mediapipe_detection(frame, holistic)

            # 2. Keypoint extraction
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]

            # 3. Prediction
            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                confidence = np.max(res)
                predicted_action = actions[np.argmax(res)]

                # 4. Update sentence
                if confidence > threshold:
                    if len(sentence) == 0 or predicted_action != sentence[-1]:
                        sentence.append(predicted_action)

                    # 🔊 Speak the prediction
                    speak_prediction(predicted_action, lang='ar')  # Change 'ar' to any desired language code

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # 5. Display probabilities (optional)
                print(f"Prediction: {predicted_action} ({confidence:.2f})")

            # 6. Visualization
            cv2.imshow('Sign Language Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
