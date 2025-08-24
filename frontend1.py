# frontend.py
import re
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# TensorFlow / Keras
from keras.models import load_model
from keras.utils import pad_sequences


# NLTK (auto-download needed corpora)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================
# App & Page Configuration
# =========================
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="💬",
    layout="centered",
)

# ---------------
# Constants/Paths
# ---------------
MODEL_PATH = Path("emotion_model.h5")
TOKENIZER_PATH = Path("tokenizer.pkl")
MAX_LEN = 100

EMOJI_LABELS = {
    0: "😢 Sadness (e.g., disappointment, loneliness, grief)",
    1: "😊 Contentment (e.g., peace, calm, relaxation)",
    2: "❤️ Love (e.g., affection, care, compassion)",
    3: "😡 Anger (e.g., frustration, annoyance, resentment)",
    4: "😨 Fear (e.g., anxiety, worry, nervousness)",
    5: "😲 Surprise (e.g., amazement, disbelief, shock)",
}


# =========================
# Utilities
# =========================
@st.cache_resource(show_spinner=False)
def ensure_nltk():
    """Ensure required NLTK corpora are available (download once)."""
    needed = ["stopwords", "wordnet", "omw-1.4"]
    for pkg in needed:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    return {
        "stop_words": set(stopwords.words("english")),
        "lemmatizer": WordNetLemmatizer(),
    }


def clean_text(text: str, stop_words, lemmatizer) -> str:
    """Lowercase, remove non-letters, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load Keras model and tokenizer; cache so they load once."""
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Model/tokenizer missing. Ensure 'emotion_model.h5' and 'tokenizer.pkl' "
            "are in the same folder as this script."
        )

    # Some older .h5 files carry optimizer state incompatible with current TF.
    # compile=False avoids loading the old optimizer config; recompile if needed.
    model = load_model(MODEL_PATH.as_posix(), compile=False)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def predict_emotion(model, tokenizer, text: str):
    """Preprocess -> sequence -> pad -> predict. Returns (label_idx, confidence, probs)."""
    nl = ensure_nltk()
    cleaned = clean_text(text, nl["stop_words"], nl["lemmatizer"])
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    probs = model.predict(padded, verbose=0)[0]
    label_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return label_idx, conf, probs, cleaned


# =========================
# UI
# =========================
# Header
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.2rem;">Emotion Detection from Text 💬</h1>
    <p style="text-align:center; color:#6b7280; margin-top:0;">
      Paste a sentence and I’ll infer the dominant emotion.
    </p>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This demo loads a trained Keras model (`emotion_model.h5`) and a tokenizer "
        "to classify the emotion expressed in text."
    )
    st.write(f"• Sequence max length: **{MAX_LEN}**")
    st.write("• NLTK: stopwords + lemmatization")
    st.divider()
    st.subheader("Quick Examples")
    examples = {
        "I miss my friends so much lately.": "Sadness",
        "What a beautiful day to relax.": "Contentment",
        "I love how supportive you are!": "Love",
        "Why did you do that?! I'm furious.": "Anger",
        "I'm worried about the exam tomorrow.": "Fear",
        "No way—did that really happen?!": "Surprise",
    }
    for txt in examples:
        if st.button(f"Try: {txt[:28]}…", use_container_width=True):
            st.session_state["__seed_text"] = txt

# Input
seed_text = st.session_state.get("__seed_text", "")
user_text = st.text_area(
    "📝 Type a sentence:",
    value=seed_text,
    height=140,
    placeholder="e.g., I can't believe how amazing this is!",
)

# Actions row
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    run_btn = st.button("🔍 Predict Emotion", use_container_width=True)
with colB:
    clear_btn = st.button("🧹 Clear", use_container_width=True)
with colC:
    show_probs = st.checkbox("Show class probabilities")

if clear_btn:
    st.session_state["__seed_text"] = ""
    st.rerun()

# Results
if run_btn:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        try:
            with st.status("Loading model & tokenizer…", expanded=False):
                model, tokenizer = load_artifacts()
                ensure_nltk()

            label_idx, conf, probs, cleaned = predict_emotion(model, tokenizer, user_text)

            # Pretty result
            st.success(f"**Predicted Emotion:** {EMOJI_LABELS[label_idx]}")
            st.info(f"🔎 Model Confidence: **{conf*100:.2f}%**")

            with st.expander("Preprocessed text (for transparency)"):
                st.code(cleaned or "(empty after preprocessing)", language="text")

            if show_probs:
                st.subheader("Class probabilities")
                # Build a small table with labels & probs
                rows = []
                for i in range(len(EMOJI_LABELS)):
                    rows.append(
                        {
                            "Class": EMOJI_LABELS[i].split(" ", 1)[0],
                            "Description": EMOJI_LABELS[i],
                            "Probability": float(probs[i]),
                        }
                    )
                # Display as a sorted table
                import pandas as pd

                df = pd.DataFrame(rows).sort_values("Probability", ascending=False, ignore_index=True)
                st.dataframe(df, use_container_width=True)

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)


# =========================
# Footer
# =========================
st.markdown(
    """
    <hr />
    <p style="text-align:center; color:#9ca3af">
      Built with Streamlit + TensorFlow. Ensure <code>emotion_model.h5</code> and
      <code>tokenizer.pkl</code> are present in this folder.
    </p>
    """,
    unsafe_allow_html=True,
)
