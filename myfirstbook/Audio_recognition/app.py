import streamlit as st
import numpy as np
import librosa
import pickle
import os
import soundfile as sf
from streamlit_mic_recorder import mic_recorder, speech_to_text

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

audio = mic_recorder(
    start_prompt="🎤 Mulai Rekam",
    stop_prompt="⏹️ Selesai Rekam",
    just_once=True,
    use_container_width=True
)

if len(audio) > 0:
    path = "temp_upload.wav"

    # --- Deteksi format audio yang dikembalikan ---
    if isinstance(audio, np.ndarray):
        data = audio
        sr = 44100

    elif isinstance(audio, dict):
        if "array" in audio:
            data = np.array(audio["array"], dtype=np.float32)
            sr = audio.get("sample_rate", 44100)
        elif "bytes" in audio:
            data = np.frombuffer(audio["bytes"], dtype=np.int16)
            sr = audio.get("sample_rate", 44100)
        elif "data" in audio:
            data = np.array(audio["data"], dtype=np.float32)
            sr = audio.get("sample_rate", 44100)
        else:
            st.error("❌ Format audio tidak dikenali. Struktur: " + str(audio))
            st.stop()

    else:
        try:
            from pydub import AudioSegment
            if isinstance(audio, AudioSegment):
                audio.export(path, format="wav")
                sr = 44100
                data = None
        except:
            st.error("❌ Tidak dapat menulis file audio.")
            st.stop()

    # --- Tulis file ---
    if data is not None:
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        sf.write(path, data, sr, format="WAV")

    st.audio(path)

    # --- Prediksi ---
    st.info("⏳ Memproses audio...")
    kategori, probs = predict_audio(path)

    st.success(f"🎯 Hasil Prediksi: **{kategori}**")
    st.write("📊 Probabilitas:")
    for i, p in enumerate(probs):
        label = "Buka" if i == 0 else "Tutup"
        st.write(f"- {label}: {p:.2f}")
