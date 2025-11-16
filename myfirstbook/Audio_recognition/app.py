import streamlit as st
import pandas as pd
import numpy as np
import librosa
import pickle
import soundfile as sf
from streamlit_mic_recorder import mic_recorder
from scipy.signal import resample

# ============================================================
# Konfigurasi
# ============================================================
SAMPLE_RATE = 16000

# ============================================================
# Fungsi Ekstraksi Fitur (versi terbaru)
# ============================================================
def extract_features(y, sr=SAMPLE_RATE):
    features = []

    # 1. Basic stats
    features += [
        np.mean(y),
        np.std(y),
        np.var(y),
        np.mean((y - np.mean(y))**3)/(np.std(y)**3 + 1e-6), 
        np.mean((y - np.mean(y))**4)/(np.std(y)**4 + 1e-6),
        np.sqrt(np.mean(y**2)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.std(librosa.feature.zero_crossing_rate(y)),
        np.max(y) - np.min(y)
    ]

    # 2. Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features += [
        np.mean(spec_centroid), np.std(spec_centroid),
        np.mean(spec_bandwidth), np.std(spec_bandwidth),
        np.mean(spec_contrast), np.std(spec_contrast),
        np.mean(spec_rolloff), np.std(spec_rolloff),
        np.mean(spec_flatness), np.std(spec_flatness),
        np.mean(chroma), np.std(chroma)
    ]

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features.append(np.mean(mfcc[i]))
    for i in range(13):
        features.append(np.std(mfcc[i]))

    # Delta MFCC
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(13):
        features.append(np.mean(mfcc_delta[i]))
    for i in range(13):
        features.append(np.std(mfcc_delta[i]))

    # 4. Temporal / Energy
    rms = librosa.feature.rms(y=y)
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


# ============================================================
# Load model dan scaler + fitur terpilih
# ============================================================
@st.cache_resource
def load_all():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    selected_features = pickle.load(open("selected_features.pkl", "rb"))
    return model, scaler, selected_features


model, scaler, selected_features = load_all()

# ============================================================
# Streamlit UI
# ============================================================
st.title("🎤 Audio Command Recognition")
st.write("Sistem klasifikasi suara dengan fitur audio + Machine Learning")

tab1, tab2 = st.tabs(["🎙️ Rekam Suara", "📁 Upload File"])

# ============================================================
# TAB 1 — Microphone
# ============================================================
with tab1:
    st.subheader("Rekam suara")

    audio_data = mic_recorder(
        start_prompt="Mulai Rekam",
        stop_prompt="Berhenti",
        format="wav"
    )

    if audio_data:
        st.audio(audio_data["bytes"], format="audio/wav")

        # Load wav buffer
        y, sr = sf.read(audio_data["bytes"])

        # Pastikan mono
        if len(y.shape) == 2:
            y = np.mean(y, axis=1)

        # Resample ke SAMPLE_RATE
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Ekstraksi fitur
        feats = extract_features(y).reshape(1, -1)

        # Pilih fitur yang sama seperti training
        feats_df = pd.DataFrame(feats)
        feats_df = feats_df.iloc[:, selected_features]

        # Scaling
        feats_scaled = scaler.transform(feats_df)

        # Prediksi
        pred = model.predict(feats_scaled)[0]
        st.success(f"🎯 Prediksi: **{pred}**")

# ============================================================
# TAB 2 — Upload file WAV
# ============================================================
with tab2:
    st.subheader("Upload file WAV")

    file = st.file_uploader("Pilih file WAV", type=['wav'])

    if file:
        st.audio(file, format="audio/wav")

        y, sr = librosa.load(file, sr=SAMPLE_RATE)

        feats = extract_features(y).reshape(1, -1)
        feats_df = pd.DataFrame(feats)
        feats_df = feats_df.iloc[:, selected_features]

        feats_scaled = scaler.transform(feats_df)
        pred = model.predict(feats_scaled)[0]

        st.success(f"🎯 Prediksi: **{pred}**")
