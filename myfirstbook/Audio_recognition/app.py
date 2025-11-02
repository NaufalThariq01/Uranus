# =====================================
# app.py - Identifikasi suara "buka" / "tutup"
# =====================================

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
from scipy.stats import skew, kurtosis

# ======================
# Load Model dan Scaler
# ======================
@st.cache_resource
def load_model_scaler():
    model = pickle.load(open('model_RandomForest.pkl', 'rb'))  # ganti sesuai model terbaikmu
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model_scaler()

# ======================
# Fungsi Ekstraksi Fitur Statistik
# ======================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=48000, mono=True)
    y = y / np.max(np.abs(y))  # normalisasi
    y, _ = librosa.effects.trim(y, top_db=20)
    
    features = {
        'mean': np.mean(y),
        'std': np.std(y),
        'skew': skew(y),
        'kurtosis': kurtosis(y),
        'rms': np.mean(librosa.feature.rms(y=y)),
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
    }
    
    return pd.DataFrame([features])

# ======================
# Streamlit UI
# ======================
st.title("🎵 Identifikasi Suara: Buka / Tutup")
st.write("Upload file audio (.wav) untuk mengetahui apakah termasuk suara 'buka' atau 'tutup'.")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav"])

if uploaded_file is not None:
    # Simpan file sementara
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file, format="audio/wav")
    
    # Ekstraksi fitur
    X_new = extract_features("temp_audio.wav")
    
    # Normalisasi
    X_scaled = scaler.transform(X_new)
    
    # Prediksi
    pred_label = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"**Kategori:** {pred_label}")
    
    st.subheader("Probabilitas Kelas")
    for label, prob in zip(model.classes_, pred_proba):
        st.write(f"{label}: {prob:.2f}")
