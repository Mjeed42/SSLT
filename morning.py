import time
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from streamlit_TTS import text_to_speech

# ----------------------------- Set Page Configuration -----------------------------
st.set_page_config(
    page_title="🎙️ SSLT - Saudi Sign Language Translator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------- Custom CSS for Styling -----------------------------

st.markdown("""
<style>
.Main-Title {
    #font-size: 58px;
    font-weight: bold;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #004474, #ff5733, #28a745);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}



st.markdown("<div class='Main-Title'>🎙️ Arabic Sign Language Translator</div>", unsafe_allow_html=True)

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
.Cam-OFF-Text {
    font-size: 40px;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #999, #555);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.image-container {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- App Title & Intro -----------------------------
st.markdown("<div class='Main-Title'>🎙️ مترجم لغة الإشارة السعودية (SSLT)</div>", unsafe_allow_html=True)
st.markdown("---")

st.markdown('<p style="font-size: 24px; text-align: center;">مرحباً بك في تطبيق المترجم الإشاري الفوري!</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 18px; text-align: center;">قم بعرض إيماءة أمام الكاميرا، وسوف يُنطق الكلمة فوراً باللغة العربية.</p>', unsafe_allow_html=True)

# ----------------------------- Sidebar Settings -----------------------------
with st.sidebar:
    st.markdown("<h1 style='color:orange; font-size:58px;'>SSLT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🛠️ الإعدادات", expanded=True):
        predict_every_n_frames = st.slider("تكرار التنبؤ (كل N إطار)", 1, 120, 30)
        show_skeleton = st.checkbox("إظهار هيكل اليد", value=True)
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

# ----------------------------- Load Model and Labels -----------------------------
@st.cache_resource
def load_ssl_model():
    model = load_model('ssl_lstm_model3.h5')
    label_classes = np.load('label_classes3.npy', allow_pickle=True)
    return model, label_classes

model, label_classes = load_ssl_model()

SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

# ----------------------------- Session State -----------------------------
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.last_prediction = None

# ----------------------------- Text-to-Speech Function -----------------------------
def speak_prediction(text):
    if text == "..." or text is None or text == st.session_state.last_prediction:
        return
    st.session_state.last_prediction = text
    text_to_speech(text=text, language='ar', key=f"tts_{uuid.uuid4()}")

# ----------------------------- MediaPipe Setup -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ----------------------------- Camera Logic -----------------------------
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

        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if show_skeleton:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                for lm in hand_landmarks.landmark:
                    landmarks_list.extend([lm.x, lm.y, lm.z])

            if len(landmarks_list) == 63:
                landmarks_list += [0.0] * 63

            if len(landmarks_list) == 126:
                sequence.append(landmarks_list)

        else:
            sequence.clear()

        prediction = "..."
        confidence = 0.0

        frame_count += 1
        if len(sequence) == SEQUENCE_LENGTH and frame_count % predict_every_n_frames == 0:
            input_data = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
            prediction_probs = model.predict(input_data, verbose=0)[0]
            predicted_idx = np.argmax(prediction_probs)
            prediction = label_classes[predicted_idx]
            confidence = prediction_probs[predicted_idx]

            if prediction != st.session_state.last_prediction:
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                speak_prediction(prediction)

        # Show webcam feed in a small centered box
        with video_placeholder.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(frame, channels="BGR", width=320)

        # Show prediction + confidence
        with prediction_placeholder.container():
            if st.session_state.prediction:
                st.markdown(f"""
                <p class="Prediction-Text">
                التنبؤ: <strong>{st.session_state.prediction}</strong><br>
                <span class="Confidence-Text">الثقة: {int(st.session_state.confidence * 100)}%</span>
                </p>
                """, unsafe_allow_html=True)

        time.sleep(0.01)

    cap.release()
else:
    prediction_placeholder.markdown('<p class="Cam-OFF-Text" style="font-size: 40px; text-align:center;">🔴 الكاميرا معطلة. اضغط تشغيل لبدء التعرف.</p>', unsafe_allow_html=True)

# ----------------------------- Camera Controls -----------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥🟢 تشغيل الكاميرا", key="start_camera", use_container_width=True):
        st.session_state.camera_running = True
        st.session_state.prediction = None
        st.session_state.confidence = 0.0
        st.session_state.last_prediction = None
        sequence.clear()

with col2:
    if st.button("🎥🛑 إيقاف الكاميرا", key="stop_camera", use_container_width=True):
        st.session_state.camera_running = False

# ----------------------------- Footer ------------------------------------
st.markdown("---")
st.markdown(
    """
    <footer style="text-align: center; margin-top: 50px;">
        <p>© 2025 Saudi Sign Language Translator | تم بناؤه باستخدام Streamlit + MediaPipe + TensorFlow</p>
    </footer>
    """,
    unsafe_allow_html=True
)
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
.Cam-OFF-Text {
    font-size: 40px;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #999, #555);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.image-container {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- App Title & Intro -----------------------------
st.markdown("<div class='Main-Title'>🎙️ مترجم لغة الإشارة السعودية (SSLT)</div>", unsafe_allow_html=True)
st.markdown("---")

st.markdown('<p style="font-size: 24px; text-align: center;">مرحباً بك في تطبيق المترجم الإشاري الفوري!</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 18px; text-align: center;">قم بعرض إيماءة أمام الكاميرا، وسوف يُنطق الكلمة فوراً باللغة العربية.</p>', unsafe_allow_html=True)

# ----------------------------- Sidebar Settings -----------------------------
with st.sidebar:
    st.markdown("<h1 class='SideBar' style='color:orange;'>SSLT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🛠️ الإعدادات", expanded=True):
        predict_every_n_frames = st.slider("تكرار التنبؤ (كل N إطار)", 1, 120, 30)
        show_skeleton = st.checkbox("إظهار هيكل اليد", value=True)
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

# ----------------------------- Load Model and Labels -----------------------------
@st.cache_resource
def load_ssl_model():
    model = load_model('ssl_lstm_model3.h5')
    label_classes = np.load('label_classes3.npy', allow_pickle=True)
    return model, label_classes

model, label_classes = load_ssl_model()

SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

# ----------------------------- Session State -----------------------------
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.last_prediction = None

# ----------------------------- Text-to-Speech Function -----------------------------
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

# ----------------------------- MediaPipe Setup -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ----------------------------- Camera Logic -----------------------------
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

        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if show_skeleton:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                for lm in hand_landmarks.landmark:
                    landmarks_list.extend([lm.x, lm.y, lm.z])

            if len(landmarks_list) == 63:
                landmarks_list += [0.0] * 63

            if len(landmarks_list) == 126:
                sequence.append(landmarks_list)

        else:
            sequence.clear()

        prediction = "..."
        confidence = 0.0

        frame_count += 1
        if len(sequence) == SEQUENCE_LENGTH and frame_count % predict_every_n_frames == 0:
            input_data = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
            prediction_probs = model.predict(input_data, verbose=0)[0]
            predicted_idx = np.argmax(prediction_probs)
            prediction = label_classes[predicted_idx]
            confidence = prediction_probs[predicted_idx]

            if prediction != st.session_state.last_prediction:
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                speak_prediction(prediction)

        # Show webcam feed in a small centered box
        with video_placeholder.container():
            col1, col2, col3 = st.columns([1, 2, 1])  # Center alignment
            with col2:
                st.image(frame, channels="BGR", width=320)  # Fixed size

        # Show prediction + confidence
        with prediction_placeholder.container():
            if st.session_state.prediction:
                st.markdown(f"""
                <p class="Prediction-Text">
                التنبؤ: <strong>{st.session_state.prediction}</strong><br>
                <span class="Confidence-Text">الثقة: {int(st.session_state.confidence * 100)}%</span>
                </p>
                """, unsafe_allow_html=True)

    cap.release()
else:
    prediction_placeholder.markdown('<p class="Cam-OFF-Text" style="font-size: 40px; text-align:center;">🔴 الكاميرا معطلة. اضغط تشغيل لبدء التعرف.</p>', unsafe_allow_html=True)

# ----------------------------- Camera Controls -----------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥🟢 تشغيل الكاميرا", key="start_camera", use_container_width=True):
        st.session_state.camera_running = True
        st.session_state.prediction = None
        st.session_state.confidence = 0.0
        st.session_state.last_prediction = None
        sequence.clear()

with col2:
    if st.button("🎥🛑 إيقاف الكاميرا", key="stop_camera", use_container_width=True):
        st.session_state.camera_running = False

# ----------------------------- Footer ------------------------------------
st.markdown("---")
st.markdown(
    """
    <footer style="text-align: center; margin-top: 50px;">
        <p>© 2025 Saudi Sign Language Translator | تم بناؤه باستخدام Streamlit + MediaPipe + TensorFlow</p>
    </footer>
    """,
    unsafe_allow_html=True
)
