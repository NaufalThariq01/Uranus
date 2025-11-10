import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import warnings

# ======================
# Konfigurasi Global
# ======================
warnings.filterwarnings('ignore')
SAMPLE_RATE = 44100
THRESHOLD = 0.6
MIN_DURATION_SEC = 0.5

# ======================
# Load Model & Scaler
# ======================
@st.cache_resource
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

# ======================
# Fungsi Ekstraksi Fitur
# ======================
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfcc1_std = np.std(mfcc[0])
    mfcc3_mean = np.mean(mfcc[2])

    return [mfcc1_std, zcr, mfcc3_mean]

# ======================
# Fungsi Prediksi
# ======================
def predict_audio(path):
    X_new = np.array(extract_features(path)).reshape(1, -1)
    X_scaled = scaler.transform(X_new)
    probs = model.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(probs)
    label_num = model.classes_[pred_idx]

    label_map = {0: "Buka", 1: "Tutup"}
    label = label_map.get(label_num, "unknown")

    if max(probs) < THRESHOLD:
        label = "Penyusup"

    return label, probs

# ======================
# UI Streamlit
# ======================
st.title("🎵 Identifikasi Suara: Buka / Tutup / Penyusup")
st.write("Unggah file .wav atau rekam suara dari microphone.")

# --------------------
# Upload File
# --------------------
uploaded = st.file_uploader("📂 Pilih file audio (.wav)", type=["wav"])

# --------------------
# Rekam Mic
# --------------------
st.subheader("🎤 Rekam dari Microphone")
audio = mic_recorder(
    start_prompt="Mulai Rekam",
    stop_prompt="Selesai Rekam",
    just_once=True,
    use_container_width=True
)

# Status rekaman
if audio:
    st.info("✅ Rekaman selesai, tekan tombol Prediksi untuk memproses audio.")

# --------------------
# Tombol Prediksi
# --------------------
if st.button("🔍 Prediksi"):
    path = None
    data = None
    sr = SAMPLE_RATE

    # Tentukan sumber audio
    if uploaded:
        path = os.path.join(os.getcwd(), "temp_upload.wav")
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.audio(uploaded, format="audio/wav")

    elif audio:
        path = os.path.join(os.getcwd(), "temp_mic.wav")
        if isinstance(audio, dict):
            if "array" in audio:
                data = np.array(audio["array"], dtype=np.float32)
            elif "bytes" in audio and audio["bytes"]:
                data = np.frombuffer(audio["bytes"], dtype=np.int16)
        elif isinstance(audio, np.ndarray):
            data = audio

        if data is not None:
            # Normalisasi
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
            sf.write(path, data, sr)
            st.audio(path)

    else:
        st.warning("⚠️ Silakan upload file atau rekam dari microphone terlebih dahulu.")
        st.stop()

    # Cek durasi minimal
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    if len(y)/sr < MIN_DURATION_SEC:
        st.warning("⚠️ Rekaman terlalu pendek, coba ulangi.")
    else:
        # Prediksi
        label, probs = predict_audio(path)
        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**Kategori:** {label}")
        st.write("📊 Probabilitas:")
        for l, p in zip(model.classes_, probs):
            st.write(f"- {l}: {p:.2f}")
        if label == "Penyusup":
            st.warning("⚠️ Suara tidak dikenali!")
