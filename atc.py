import gdown
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle, os, re, requests, numpy as np
from time import sleep

# ----------------------- Streamlit Config -----------------------
st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖")

# ----------------------- Sidebar -----------------------
st.sidebar.markdown(
    "<h1 style='text-align: center; color: black;'>AI Support Chat Bot</h1>",
    unsafe_allow_html=True
)

st.sidebar.subheader("📊 Model Performance Summary")

TRAIN_ACCURACY = 0.9595
VAL_ACCURACY = 0.70
TEST_ACCURACY = 0.70

st.sidebar.subheader("Training Accuracy")
st.sidebar.write(f"✔️ {TRAIN_ACCURACY * 100:.2f}%")

st.sidebar.subheader("Validation Accuracy")
st.sidebar.write(f"📌 {VAL_ACCURACY * 100:.2f}%")

st.sidebar.subheader("Test Accuracy")
st.sidebar.write(f"🎯 {TEST_ACCURACY * 100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Notes")
st.sidebar.write("""
- Model: BiLSTM (Many-to-One)
- Trained for 30 epochs
- Early stopping restored epoch 29 weights
- Confusion Matrix: 52x52 (multi-class)
""")

# ----------------------- Styling -----------------------
st.markdown("""
<style>
:root {
    --primary-color: #5A5A5A;
    --background-color: #E5E5E5;
    --secondary-background-color: #F2F2F2;
    --text-color: #000000;
}
.main { background-color: #F2F2F2; }
.stTextInput>div>div>input {
    border-radius: 20px;
    border: 1px solid #5A5A5A;
    padding: 0.6em 1em;
}
.stButton>button {
    background-color: #5A5A5A;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------- Model & Assets -----------------------
MODEL_PATH = "newone_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1AVKznHoqchDEbTN06-LC0NWexrHRnBY-"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

max_len = 300



# ----------------------- Gemini API Setup (SECURE) -----------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

import os

GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please add it in Streamlit Environment Secrets.")
    st.stop()


# ----------------------- Helper Functions -----------------------
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s.,!?@#%&()-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def predict_queue(text):
    s = clean_text(text)
    seq = tokenizer.texts_to_sequences([s])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    idx = np.argmax(probs)
    queue = le.inverse_transform([idx])[0]
    return {"queue": queue, "confidence": float(probs[idx])}

import time

def generate_reply_with_gemini(ticket_body, predicted_queue):
    prompt = f"""
You are a professional and empathetic customer support representative.
The customer's issue belongs to: {predicted_queue}.
Write a short, helpful customer support reply.

Customer message: \"\"\"{ticket_body}\"\"\"
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    for attempt in range(3):  # retry 3 times
        try:
            resp = requests.post(
                GEMINI_API_URL,
                headers=headers,
                json=payload,
                timeout=20
            )

            if resp.status_code == 429:
                time.sleep(3)  # wait and retry
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            if attempt == 2:
                return "⚠️ Reply generation is temporarily unavailable. Please try again later."

# ----------------------- UI -----------------------
st.title("🤖 AI Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

examples = [
    "I ordered a novel last week but received the wrong edition.",
    "My software license key is not working.",
    "My payment failed but the amount was deducted.",
    "I cannot post new threads on the forum."
]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.write(f"**Predicted Queue:** {msg['queue']}")
        st.write(f"**Confidence:** {msg['confidence']:.4f}")
        st.write(msg["reply"])

st.markdown("---")

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Type your message:")
with col2:
    send_button = st.button("➤")

if send_button and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Bot is typing..."):
        pred = predict_queue(user_input)
        reply = generate_reply_with_gemini(user_input, pred["queue"])

    st.session_state.messages.append({
        "role": "bot",
        "queue": pred["queue"],
        "confidence": pred["confidence"],
        "reply": reply
    })
    st.rerun()

st.caption("Built with 🤖 and empathy | © 2025 AI Customer Support")


