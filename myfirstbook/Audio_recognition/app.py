import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import matplotlib.pyplot as plt
import warnings

# ======================
# Konfigurasi Global
# ======================
warnings.filterwarnings('ignore')
SAMPLE_RATE = 44100
THRESHOLD = 0.4  # turunkan sementara untuk deteksi penyusup
MIN_DURATION_SEC = 1.0 

# ======================
# Load Model & Scaler
# ======================
@st.cache_resource
def load_model_scaler():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model_RandomForest.pkl")
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

mic_path = os.path.join(os.getcwd(), "temp_mic.wav")

if audio:
    data = None
    sr = SAMPLE_RATE

    if isinstance(audio, dict):
        if "array" in audio:
            data = np.array(audio["array"], dtype=np.float32)
        elif "bytes" in audio and audio["bytes"]:
            data = np.frombuffer(audio["bytes"], dtype=np.int16).astype(np.float32)
    elif isinstance(audio, np.ndarray):
        data = audio.astype(np.float32)

    if data is not None:
        # Stereo -> mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Normalisasi aman
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        sf.write(mic_path, data, sr)
        st.audio(mic_path)
        st.success("✅ Rekaman tersimpan. Tekan tombol Prediksi untuk memproses.")

        # Tampilkan waveform
        st.subheader("📈 Waveform Rekaman")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(data, color='blue')
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

# --------------------
# Tombol Prediksi
# --------------------
if st.button("🔍 Prediksi"):
    path = None

    if uploaded:
        path = os.path.join(os.getcwd(), "temp_upload.wav")
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.audio(uploaded, format="audio/wav")
    elif os.path.exists(mic_path):
        path = mic_path
        st.audio(path)
    else:
        st.warning("⚠️ Silakan upload file atau rekam dari microphone terlebih dahulu.")
        st.stop()

    # Cek durasi minimal
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    duration = len(y)/sr
    if duration < MIN_DURATION_SEC:
        st.warning("⚠️ Audio terlalu pendek, coba ulangi.")
        st.stop()

    # Debug fitur
    features = extract_features(path)
    st.write("🔹 Fitur audio (mfcc1_std, zcr, mfcc3_mean):", features)

    # Prediksi
    label, probs = predict_audio(path)
    st.subheader("🎯 Hasil Prediksi")
    st.write(f"**Kategori:** {label}")
    st.write("📊 Probabilitas:")
    for l, p in zip(model.classes_, probs):
        st.write(f"- {l}: {p:.2f}")
    if label == "Penyusup":
        st.warning("⚠️ Suara tidak dikenali!")
