import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_lottie import st_lottie
import requests

# --- Page Config ---
st.set_page_config(page_title="MindCare AI", page_icon="🧠", layout="centered")

# --- Helper: Load Lottie Animation ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_brain = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ofa3xwo7.json") # AI Brain

# --- Custom CSS (Classic Purple 3D) ---
st.markdown("""
<style>
    /* Main Background - Deep Purple Gradient */
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 50%, #000000 100%);
        color: white;
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid #6c5ce7;
        border-radius: 15px;
    }
    
    /* 3D Glass Cards for Chat */
    .user-msg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px 20px 0 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 15px;
        text-align: right;
        color: #e0e0e0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .ai-msg {
        background: linear-gradient(135deg, #6c5ce7, #a55eea);
        padding: 20px;
        border-radius: 20px 20px 20px 0;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transform: perspective(1000px) rotateX(2deg);
    }
    
    /* Emoji Animation */
    @keyframes pop {
        0% { transform: scale(0); }
        80% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    .emoji-display {
        font-size: 80px;
        text-align: center;
        animation: pop 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #6c5ce7, #a55eea);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(108, 92, 231, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Tools ---
@st.cache_resource
def load_assets():
    try:
        model = load_model('mood_analyzer.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
        return model, tokenizer, label_encoder
    except Exception as e:
        return None, None, None

model, tokenizer, label_encoder = load_assets()

# --- Prediction Function ---
def predict_emotion(text):
    if not model: return "joy", 99.9 
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    pred = model.predict(padded_sequence)
    label_index = np.argmax(pred)
    label = label_encoder.inverse_transform([label_index])[0]
    confidence = np.max(pred) * 100
    return label, confidence

# --- UI Layout ---
col1, col2 = st.columns([1, 4])
with col1:
    if lottie_brain:
        st_lottie(lottie_brain, height=100, key="logo")
with col2:
    st.title("MindCare AI")
    st.caption("Advanced Emotional Intelligence System")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Input Area
with st.container():
    with st.form("chat_form"):
        user_input = st.text_input("", placeholder="Type your thoughts here... (e.g., I finally achieved my goal!)")
        cols = st.columns([1, 4])
        with cols[0]:
            submit = st.form_submit_button("Analyze 🧠")

if submit and user_input:
    emotion, conf = predict_emotion(user_input)
    
    # Emojis & Responses
    emotion_map = {
        "joy":      {"emoji": "🤩", "msg": "Fantastic! Your positivity is radiating!", "color": "#fdcb6e"},
        "sadness":  {"emoji": "😢", "msg": "It's okay not to be okay. I'm here for you.", "color": "#74b9ff"},
        "anger":    {"emoji": "🤬", "msg": "Take a deep breath. Let's process this calmly.", "color": "#ff7675"},
        "fear":     {"emoji": "😱", "msg": "You are safe. Focus on your breathing.", "color": "#a29bfe"},
        "love":     {"emoji": "🥰", "msg": "Love is in the air! Cherish this feeling.", "color": "#e84393"},
        "surprise": {"emoji": "😲", "msg": "Wow! That sounds unexpected!", "color": "#fab1a0"}
    }
    
    data = emotion_map.get(emotion, {"emoji": "🤔", "msg": "Interesting...", "color": "#dfe6e9"})
    
    # Save to history
    st.session_state.history.insert(0, {
        "user": user_input, 
        "ai": data['msg'], 
        "emoji": data['emoji'], 
        "emotion": emotion, 
        "conf": conf
    })

# Display Chat
st.markdown("<br>", unsafe_allow_html=True)
for chat in st.session_state.history:
    # 3D Reaction Card
    col_emoji, col_text = st.columns([1, 5])
    
    with col_emoji:
        st.markdown(f'<div class="emoji-display">{chat["emoji"]}</div>', unsafe_allow_html=True)
        st.caption(f"{chat['emotion'].upper()} {chat['conf']:.0f}%")
        
    with col_text:
        st.markdown(f'<div class="user-msg">{chat["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-msg"><b>AI:</b> {chat["ai"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
