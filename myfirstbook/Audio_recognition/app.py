import streamlit as st
import numpy as np
import librosa
import pickle
import os
import soundfile as sf
from audiorecorder import audiorecorder

# ==========================
# Load model & scaler
# ==========================
def load_model_scaler():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model_KNN.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        st.success("✅ Model dan Scaler berhasil dimuat.")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_model_scaler()

# ==========================
# Fungsi ekstraksi fitur
# ==========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc1_std = np.std(mfcc[0])
    mfcc3_mean = np.mean(mfcc[2])
    return np.array([mfcc1_std, zcr, mfcc3_mean]).reshape(1, -1)

# ==========================
# Fungsi prediksi
# ==========================
def predict_audio(file_path):
    features = extract_features(file_path)
    X_scaled = scaler.transform(features)
    probs = model.predict_proba(X_scaled)[0]
    label = model.predict(X_scaled)[0]
    
    if label == 0:
        kategori = "Buka"
    elif label == 1:
        kategori = "Tutup"
    else:
        kategori = "Suara Tidak Dikenali"
    
    return kategori, probs

# ==========================
# UI Streamlit
# ==========================
st.title("🎙️ Voice Command Recognition")

audio = audiorecorder("Tekan untuk merekam", "Rekam selesai")

if len(audio) > 0:
    # Simpan hasil rekaman
    path = "temp_upload.wav"
    sf.write(path, audio.tobytes(), 44100, format="WAV")

    st.audio(path)

    st.info("⏳ Memproses audio...")
    kategori, probs = predict_audio(path)

    st.success(f"🎯 Hasil Prediksi: **{kategori}**")
    st.write("📊 Probabilitas:")
    for i, p in enumerate(probs):
        st.write(f"{i}: {p:.2f}")
