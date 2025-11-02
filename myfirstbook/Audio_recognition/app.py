# =====================================
# app.py - Identifikasi suara "buka" / "tutup"
# =====================================
import os
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
    # Pastikan file diambil dari folder yang sama dengan app.py
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model_RandomForest.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    # Cek keberadaan file
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"❌ File scaler tidak ditemukan: {scaler_path}")
        st.stop()

    # Load model dan scaler
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    st.success("✅ Model dan Scaler berhasil dimuat.")
    return model, scaler


model, scaler = load_model_scaler()

# ======================
# Fungsi Ekstraksi Fitur Statistik
# ======================
def extract_features(file_path):
    """Ekstraksi fitur sederhana dari sinyal audio"""
    y, sr = librosa.load(file_path, sr=48000, mono=True)
    y = y / np.max(np.abs(y))  # normalisasi amplitude
    y, _ = librosa.effects.trim(y, top_db=20)  # potong bagian hening

    # Ekstraksi fitur dasar
    features = {
        "mean": np.mean(y),
        "std": np.std(y),
        "skew": skew(y),
        "kurtosis": kurtosis(y),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
    }

    # Pastikan urutan kolom konsisten
    feature_order = ["mean", "std", "skew", "kurtosis", "rms", "zcr"]
    df = pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)

    return df

# ======================
# Streamlit UI
# ======================
st.title("🎵 Identifikasi Suara: Buka / Tutup")
st.write("Upload file audio (.wav) untuk mengetahui apakah termasuk suara **'buka'** atau **'tutup'**.")

uploaded_file = st.file_uploader("📂 Pilih file audio", type=["wav"])

if uploaded_file is not None:
    # Simpan file sementara
    temp_audio = "temp_audio.wav"
    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Putar audio di Streamlit
    st.audio(uploaded_file, format="audio/wav")

    try:
        # Ekstraksi fitur
        X_new = extract_features(temp_audio)
        st.write("🧩 Fitur hasil ekstraksi:", X_new)

        # Normalisasi
        X_scaled = scaler.transform(X_new)

        # Prediksi
        pred_label = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]

        # Hasil prediksi
        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**Kategori:** {pred_label}")

        st.subheader("📊 Probabilitas Kelas")
        for label, prob in zip(model.classes_, pred_proba):
            st.write(f"- {label}: {prob:.2f}")

    except Exception as e:
        st.error(f"Terjadi error saat memproses audio: {e}")
