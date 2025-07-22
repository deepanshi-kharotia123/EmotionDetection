import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load resources
model = load_model("emotion_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Constants
max_len = 100
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emotion label map
label_map = {
    0: "😢 Sadness (e.g., disappointment, loneliness, grief)",
    1: "😊 Contentment (e.g., peace, calm, relaxation)",
    2: "❤️ Love (e.g., affection, care, compassion)",
    3: "😡 Anger (e.g., frustration, annoyance, resentment)",
    4: "😨 Fear (e.g., anxiety, worry, nervousness)",
    5: "😲 Surprise (e.g., amazement, disbelief, shock)"
}

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit App Layout
st.set_page_config(page_title="Emotion Detector", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Emotion Detection from Text 💬</h1>", unsafe_allow_html=True)
st.markdown("### Enter a sentence to detect the emotion behind it.")

user_input = st.text_area("📝 Type your sentence here:")

if st.button("🔍 Predict Emotion"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        # Preprocess and predict
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(padded)
        label_index = np.argmax(pred)
        emotion = label_map[label_index]
        confidence = np.max(pred) * 100

        # Display
        st.success(f"**Predicted Emotion:** {emotion}")
        st.info(f"🔎 Model Confidence: {confidence:.2f}%")
