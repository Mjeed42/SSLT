import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from collections import deque

#---------------------------Set page configuration------------------------------
st.set_page_config(
    page_title="SSLT - Real-Time Saudi Sign Language Translator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

#---------------------------Custom CSS for styling------------------------------
st.markdown("""
<style>
.Main-Title {
    font-size: 58px;
    font-weight: bold;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #004474, #ff5733, #28a745);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.Prediction-Text {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #004474, #ff5733, #28a745);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.Confidence-Text {
    font-size: 40px;
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

#---------------------------App Title & Intro-------------------------------
st.markdown("<div class='Main-Title'>🎙️ مترجم لغة الإشارة السعودية (SSLT)</div>", unsafe_allow_html=True)
st.markdown("---")

st.markdown('<p style="font-size: 24px; text-align: center;">مرحباً بك في تطبيق المترجم الإشاري الفوري!</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 18px; text-align: center;">قم بعرض إيماءة أمام الكاميرا، وسوف يُنطق الكلمة فوراً باللغة العربية.</p>', unsafe_allow_html=True)

#---------------------------Sidebar-------------------------------------
with st.sidebar:
    st.markdown("<h1 class='SideBar' style='color:orange;'>SSLT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🛠️ الإعدادات", expanded=True):
        predict_every_n_frames = st.slider("تكرار التنبؤ (كل N إطار)", 1, 120, 10)
        st.markdown("---")

    with st.expander("🪄 كيفية الاستخدام", expanded=True):
        st.markdown("""
        <ul style="font-size: 18px;">
            <li>اضغط على زر <strong>'تشغيل الكاميرا'</strong></li>
            <li>استخدم يديك لإظهار الإيماءة</li>
            <li>سيتم عرض ونطق الكلمة فورياً</li>
            <li>اضغط على زر <strong>'إيقاف الكاميرا'</strong> عند الانتهاء</li>
        </ul>
        """, unsafe_allow_html=True)

#---------------------------Load Model-------------------------------
@st.cache_resource
def load_model():
    return joblib.load("ssl_hand_gesture_model.pkl")

model = load_model()

labels = ["سنة", "أسبوع", "شهر", "الأربعاء", "الجمعة", "الثلاثاء", "الأثنين", "الأحد", "يوم", "الخميس", "السبت", "لغة الإشارة السعودية"]

#---------------------------Session State-------------------------------
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.last_prediction = None

#---------------------------Text-to-Speech Function---------------------------
def speak_prediction(text):
    if text == "..." or text is None or text == st.session_state.last_prediction:
        return
    st.session_state.last_prediction = text

    st.markdown(f"""
    <script>
        const msg = new SpeechSynthesisUtterance();
        msg.text = "{text}";
        msg.lang = "ar-SA";  // Arabic - Saudi Arabia
        window.speechSynthesis.speak(msg);
    </script>
    """, unsafe_allow_html=True)

#---------------------------MediaPipe Setup-----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#---------------------------Camera Logic-------------------------------
video_placeholder = st.empty()
prediction_placeholder = st.empty()

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("فشل الوصول إلى الكاميرا.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

        prediction = "..."
        confidence = 0.0

        if len(landmarks) == 126:
            df = pd.DataFrame([landmarks])
            prediction = model.predict(df)[0]
            confidence = 1.0  # Replace with predict_proba if available

            if prediction != st.session_state.last_prediction:
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                speak_prediction(prediction)

        # Show webcam feed
        with video_placeholder.container():
            st.image(frame, channels="BGR", use_container_width=True)

        # Show prediction + confidence
        with prediction_placeholder.container():
            if st.session_state.prediction:
                st.markdown(f"""
                <p class="Prediction-Text">
                التنبؤ: <strong>{st.session_state.prediction}</strong><br>
                <span class="Confidence-Text">الثقة: {int(st.session_state.confidence * 100)}%</span>
                </p>
                """, unsafe_allow_html=True)

        frame_count += 1

    cap.release()
else:
    prediction_placeholder.markdown('<p class="Prediction-Text" style="font-size: 40px; text-align:center;">🔴 الكاميرا معطلة. اضغط تشغيل لبدء التعرف.</p>', unsafe_allow_html=True)

#---------------------------Camera Controls-------------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥🟢 تشغيل الكاميرا", key="start_camera", use_container_width=True):
        st.session_state.camera_running = True
        st.session_state.prediction = None
        st.session_state.confidence = 0.0
        st.session_state.last_prediction = None

with col2:
    if st.button("🎥🛑 إيقاف الكاميرا", key="stop_camera", use_container_width=True):
        st.session_state.camera_running = False

st.markdown("---")

#---------------------------Footer------------------------------------
st.markdown(
    """
    <footer style="text-align: center; margin-top: 50px;">
        <p>© 2025 Saudi Sign Language Translator | تم بناؤه باستخدام Streamlit + MediaPipe + Scikit-learn</p>
    </footer>
    """,
    unsafe_allow_html=True
)
