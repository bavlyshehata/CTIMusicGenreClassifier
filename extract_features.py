import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# === CONFIG ===
DATA_DIR = "data/Data/genres_original"
OUTPUT_FILE = "features.csv"

# === START TIMER ===
start_time = time.time()

genres = os.listdir(DATA_DIR)
data = []

for genre in tqdm(genres, desc="Genres"):
    genre_path = os.path.join(DATA_DIR, genre)
    for filename in tqdm(os.listdir(genre_path), desc=genre, leave=False):
        if not filename.endswith(".wav"):
            continue
        file_path = os.path.join(genre_path, filename)

        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None, duration=30)

            # --- Feature Extraction ---
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)

            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))

            # Combine all features
            features = np.hstack([
                mfcc_mean,
                chroma_mean,
                spec_centroid,
                spec_bw,
                rolloff,
                zcr,
                genre
            ])
            data.append(features)

        except Exception as e:
            print(f"⚠️  Skipping {file_path}: {e}")
            continue

# === SAVE TO CSV ===
columns = [f"mfcc_{i}" for i in range(13)] + \
          [f"chroma_{i}" for i in range(12)] + \
          ["spec_centroid", "spec_bw", "rolloff", "zcr", "label"]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

elapsed = time.time() - start_time
print(f"\n✅ Features saved to {OUTPUT_FILE}")
print(f"⏱️ Total time: {elapsed/60:.2f} minutes")
print(f"🎶 Processed {len(df)} tracks successfully.")
