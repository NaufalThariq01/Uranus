import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
import warnings

# ======================
# Konfigurasi Global
# ======================
warnings.filterwarnings('ignore')
SAMPLE_RATE = 48000
THRESHOLD = 0.6

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

    # Mapping label angka ke teks
    label_map = {0: "buka", 1: "tutup"}
    label = label_map.get(label_num, "unknown")

    if max(probs) < THRESHOLD:
        label = "Penyusup"

    return label, probs, label_map

# ======================
# UI Streamlit
# ======================
st.title("🎵 Identifikasi Suara: Buka / Tutup / Penyusup")
st.write("Unggah file .wav atau rekam langsung dari microphone.")

# Upload File
uploaded = st.file_uploader("📂 Pilih file audio (.wav)", type=["wav"])
if uploaded:
    path = os.path.join(os.getcwd(), "temp_upload.wav")
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.audio(uploaded, format="audio/wav")

    label, probs, label_map = predict_audio(path)

    st.subheader("🎯 Hasil Prediksi")
    st.write(f"**Kategori:** {label}")

    st.write("📊 Probabilitas:")
    for i, p in zip(model.classes_, probs):
        st.write(f"- {label_map[i]}: {p:.2f}")

    if label == "Penyusup":
        st.warning("⚠️ Suara tidak dikenali!")

# Rekam Mic
st.header("🎤 Rekam dari Microphone")

class AudioProcessor(AudioProcessorBase):
    def recv_audio(self, frame):
        data = frame.to_ndarray().flatten()
        temp_path = os.path.join(os.getcwd(), "temp_mic.wav")
        sf.write(temp_path, data, SAMPLE_RATE)
        return frame

webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration=RTCConfiguration({})
)

if st.button("🔍 Deteksi dari Microphone"):
    try:
        temp_path = os.path.join(os.getcwd(), "temp_mic.wav")
        label, probs, label_map = predict_audio(temp_path)

        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**Kategori:** {label}")

        st.write("📊 Probabilitas:")
        for i, p in zip(model.classes_, probs):
            st.write(f"- {label_map[i]}: {p:.2f}")

        if label == "Penyusup":
            st.warning("⚠️ Suara tidak dikenali!")
    except Exception as e:
        st.error(f"Terjadi error: {e}")
