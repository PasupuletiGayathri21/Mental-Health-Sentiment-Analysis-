import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# ✅ Load model (use .h5 instead of .keras)
model = load_model("gru_model.h5")

# ✅ Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ Same as training
MAX_SEQUENCE_LENGTH = 100


# 🔥 Preprocessing (must match training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# 🔥 Prediction function
def predict_text(text):
    cleaned = preprocess_text(text)

    # Convert to sequence
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

    # Predict
    pred = model.predict(padded, verbose=0)[0]

    confidence = np.max(pred)
    label = label_encoder.classes_[np.argmax(pred)]

    # 🔥 Apply minimal rules ONLY if uncertain
    if confidence < 0.55:
        if "hopeless" in cleaned or "suicidal" in cleaned:
            return "very negative"
        if "love" in cleaned or "happy" in cleaned:
            return "positive"
        if "sad" in cleaned or "worried" in cleaned:
            return "negative"
        if "doctor" in cleaned or "treatment" in cleaned:
            return "neutral"

    return label


# 🎯 Streamlit UI
st.set_page_config(page_title="Mental Health Prediction", layout="centered")

st.title("🧠 Mental Health Prediction App")

st.write("Enter a sentence and the model will predict the mental health sentiment.")

text = st.text_area("✍️ Enter your text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter text")
    else:
        result = predict_text(text)
        st.success(f"✅ Prediction: **{result}**")