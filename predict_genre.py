import librosa
import numpy as np
import joblib
import sys
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "genre_classifier.pkl"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "example.wav"

print("🎵 Loading model, scaler, and label encoder...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

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

print(f"🎶 Analyzing {AUDIO_PATH} ...")
features = extract_features(AUDIO_PATH)
features_scaled = scaler.transform(features)

# === Predict probabilities ===
probs = model.predict_proba(features_scaled)[0]
top3_idx = np.argsort(probs)[::-1][:3]  # top 3
genres = label_encoder.inverse_transform(top3_idx)
confidences = probs[top3_idx] * 100

print("\n🔊 Top 3 Predicted Genres:")
for g, c in zip(genres, confidences):
    print(f"   🎧 {g.upper():<12} — {c:.2f}% confidence")
