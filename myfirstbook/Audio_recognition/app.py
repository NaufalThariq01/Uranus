import streamlit as st
import numpy as np
import librosa
import pickle
from streamlit_mic_recorder import mic_recorder
import os


# ============================================================
# Konfigurasi Global
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000   # Sample rate dataset + training


def extract_features(y, sr=SAMPLE_RATE):
    features = []

    # =====================================
    # 1. Basic Stats (9)
    # =====================================
    features += [
        np.mean(y),
        np.std(y),
        np.var(y),
        np.mean((y - np.mean(y))**3)/(np.std(y)**3 + 1e-6),   # skew
        np.mean((y - np.mean(y))**4)/(np.std(y)**4 + 1e-6),   # kurtosis
        np.sqrt(np.mean(y**2)),                               # rms_global
    ]

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    features.append(np.max(y) - np.min(y))                    # amplitude_range

    # =====================================
    # 2. Spectral (12)
    # =====================================
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features += [
        np.mean(spec_centroid), np.std(spec_centroid),
        np.mean(spec_bandwidth), np.std(spec_bandwidth),
        np.mean(spec_contrast), np.std(spec_contrast),
        np.mean(spec_rolloff), np.std(spec_rolloff),
        np.mean(spec_flatness), np.std(spec_flatness),
        np.mean(chroma), np.std(chroma)
    ]

    # =====================================
    # 3. MFCC: 40 coefficients (mean + std)
    # =====================================
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    for i in range(40):
        features.append(np.mean(mfcc[i]))

    for i in range(40):
        features.append(np.std(mfcc[i]))

    # =====================================
    # 4. Delta MFCC 40 (mean + std)
    # =====================================
    mfcc_delta = librosa.feature.delta(mfcc)

    for i in range(40):
        features.append(np.mean(mfcc_delta[i]))

    for i in range(40):
        features.append(np.std(mfcc_delta[i]))

    # =====================================
    # 5. RMS, Onset, Tempo, Autocorr Lag (5)
    # =====================================
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(np.mean(onset_env))
    features.append(np.std(onset_env))

    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    features.append(float(tempo[0]))

    autocorr = np.correlate(y, y, mode="full")
    mid = len(autocorr) // 2
    features.append(np.argmax(autocorr[mid+1:]) + 1)

    return np.array(features)

def load_all():
    model = pickle.load(open(os.path.join(BASE_DIR, "model_RandomForest.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    return model, scaler


model, scaler = load_all()


st.title("🎤 Audio Command Recognition")
st.write("Klasifikasi suara menggunakan fitur audio + Machine Learning")

tab1, tab2 = st.tabs(["🎙️ Rekam Suara", "📁 Upload File"])


with tab1:
    st.subheader("Rekam suara")

    audio_data = mic_recorder(
        start_prompt="Mulai Rekam",
        stop_prompt="Stop",
        format="wav"
    )

    if audio_data:
        st.audio(audio_data["bytes"], format="audio/wav")

        # ======================================
        # Decode audio bytes -> numpy int16
        # ======================================
        raw = audio_data["bytes"]
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # Normalisasi (-1 .. 1)
        y = y / 32768.0

        # Resample dari mic 48k → 16k
        y = librosa.resample(y, orig_sr=48000, target_sr=SAMPLE_RATE)

        # Minimal panjang 1 detik
        if len(y) < SAMPLE_RATE:
            st.error("Rekaman terlalu pendek! Coba rekam minimal 1 detik.")
            st.stop()

        # Ekstraksi fitur
        feats = extract_features(y).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        pred = model.predict(feats_scaled)[0]

        st.success(f"🎯 Prediksi: **{pred}**")


# ============================================================
# TAB 2: Upload File WAV
# ============================================================
with tab2:
    st.subheader("Upload File WAV")

    file = st.file_uploader("Pilih file WAV", type=['wav'])

    if file:
        st.audio(file, format="audio/wav")

        # Load menggunakan librosa (normal)
        y, sr = librosa.load(file, sr=SAMPLE_RATE)

        feats = extract_features(y).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        pred = model.predict(feats_scaled)[0]

        st.success(f"🎯 Prediksi: **{pred}**")
