import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
import pickle, os, requests, re, numpy as np
from time import sleep

import pickle
import os
import requests
import json
import re
import numpy as np




# 
# ----------------------- Streamlit Config -----------------------
st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖")

# 💙 Blue theme override
st.markdown("""
<style>
:root {
    --primary-color: #1E90FF;
    --background-color: #FFFFFF;
    --secondary-background-color: #F0F2F6;
    --text-color: #000000;
}

/* Main background */
.main {
    background-color: #F0F2F6;
}

/* Chat input box */
.stTextInput>div>div>input {
    border-radius: 20px;
    border: 1px solid #1E90FF;
    padding: 0.6em 1em;
}

/* Buttons */
.stButton>button {
    background-color: #1E90FF;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #5DADE2;
}

/* Animation for chat messages */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.chat-bubble {
    animation: fadeInUp 0.4s ease-in-out;
}
</style>
""", unsafe_allow_html=True)
model = load_model(r"C:\Users\welcome\OneDrive\Desktop\streamlit\env\Scripts\final_lstm_model.h5")

with open(r"C:\Users\welcome\OneDrive\Desktop\streamlit\env\Scripts\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(r"C:\Users\welcome\OneDrive\Desktop\streamlit\env\Scripts\label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

max_len = 200

# -----------------------
# ----------------------- Model & Assets -----------------------
# ----------------------- Gemini API Setup -----------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
os.environ["GEMINI_API_KEY"] ="AIzaSyCRFgZy91BkZ8xXxHUhzdH5EDldLVFFdMU"
# ----------------------- Helper Functions -----------------------
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\\S+", " ", s)
    s = re.sub(r"[^a-z0-9\\s\\.,!?@#%&()-]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def predict_queue(text):
    s = clean_text(text)
    seq = tokenizer.texts_to_sequences([s])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    idx = np.argmax(probs)
    queue = le.inverse_transform([idx])[0]
    return {"queue": queue, "confidence": float(probs[idx])}

def generate_reply_with_gemini(ticket_body, predicted_queue):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return "(Gemini API key not set.)"

    prompt = f"""
You are a professional and empathetic customer support representative.
The customer's issue belongs to: {predicted_queue}.
Write a detailed but concise customer support reply in 2 very short paragraphs and one closing line.
Leave a clear blank line between each paragraph to improve readability.

The customer's message may be in English or German.
If it is in German:
- First, write the full, natural-sounding response in German.
- Then, provide an **English translation** below, labeled as 'English Translation:'.
If the message is in English:
- Reply only in English.

Customer message: \"\"\"{ticket_body}\"\"\"
"""


    payload = {
        "model": "gemini-2.0-flash",
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        # 🧩 Replace double or single newlines with <br><br> for spacing between paragraphs
        formatted_text = re.sub(r'\n\s*\n', '<br><br>', text)
        formatted_text = re.sub(r'\n', ' ', formatted_text)
        return formatted_text
    except Exception as e:
        return f"(Failed to get Gemini reply: {e})"



# ----------------------- UI ----------------------
st.title("🤖 AI Customer Support Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Predefined queries
examples = [
    "Our company is encountering issues with the billing for our software subscriptions.",
    "Facing system performance challenges with data analytics tools, particularly during investment optimization tasks.",
    "A recent update to the platform caused conflicts with APIs, leading to errors in data synchronization. Restoring the previous version and clearing the cache resolved the problem",
   
]

# Display chat bubbles
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class='chat-bubble' style='background-color:#D6EAF8;
                        color:#000000;
                        padding:10px;
                        margin:5px 50px 5px 0;
                        border-radius:12px;
                        text-align:right;
                        box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                <b>You:</b><br>{msg['content']}
            </div>
            """,
            unsafe_allow_html=True)
    elif msg["role"] == "bot":
        queue = msg.get("queue", "N/A")
        confidence = msg.get("confidence", 0)
        reply = msg.get("reply", "")
        st.markdown(
            f"""
            <div class='chat-bubble' style='background-color:#E8F0FE;
                        color:#000000;
                        padding:10px;
                        margin:5px 0 5px 50px;
                        border-radius:12px;
                        text-align:left;
                        border-left:5px solid #1E90FF;
                        box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                <b>Bot:</b><br>
                <b>Predicted Queue:</b> {queue}<br>
                <b>Confidence:</b> {confidence:.4f}<br>
                <b>Reply:</b> {reply}
            </div>
            """,
            unsafe_allow_html=True)

st.markdown("---")

# Chat input bar
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Type your message here:", key="text_box",
                               value=st.session_state.input_text,
                               placeholder="Ask your question...")
with col2:
    send_button = st.button("➤")

# Suggested queries (shown only if input is empty)
if not st.session_state.input_text.strip():
    st.markdown("💡 **Suggested queries:**")
    for q in examples:
        if st.button(q):
            st.session_state.input_text = q
            st.rerun()

# Message processing
if send_button and user_input.strip():
    text = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": text})

    with st.spinner("Bot is typing..."):
        sleep(0.5)
        prediction = predict_queue(text)
        reply = generate_reply_with_gemini(text, prediction["queue"])

    st.session_state.messages.append({
        "role": "bot",
        "queue": prediction["queue"],
        "confidence": prediction["confidence"],
        "reply": reply
    })

    # 🧹 Clear input safely
    st.session_state.input_text = ""
    st.session_state.pop("text_box", None)  # safely remove the old key
    st.experimental_set_query_params(_clear="1")  # trigger UI rerun
    st.rerun()


