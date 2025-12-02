import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import warnings
warnings.filterwarnings("ignore")

# === Load trained components ===
@st.cache_resource
def load_model():
    model = joblib.load("genre_classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# === Feature extraction ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features = np.hstack([mfcc_mean, chroma_mean, spec_centroid, spec_bw, rolloff, zcr])
    return np.array(features).reshape(1, -1)

# === UI ===
st.set_page_config(page_title="🎵 Music Genre Classifier", page_icon="🎧", layout="centered")

st.title("🎶 Music Genre Classifier")
st.write("Upload a 30-second `.wav` file and I’ll predict its top 3 genres!")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")
    st.write("Analyzing audio... ⏳")

    try:
        features = extract_features(temp_path)
        features_scaled = scaler.transform(features)
        probs = model.predict_proba(features_scaled)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        genres = label_encoder.inverse_transform(top3_idx)
        confidences = probs[top3_idx] * 100

        st.success("✅ Prediction Complete!")
        for g, c in zip(genres, confidences):
            st.write(f"**{g.upper()}** — {c:.2f}%")
            st.progress(c / 100)

    except Exception as e:
        st.error(f"⚠️ Could not process the file: {e}")