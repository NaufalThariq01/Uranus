import os
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
from scipy.stats import skew, kurtosis
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration

# ======================
# Load Model dan Scaler (tetap dari kode sebelumnya)
# ======================
@st.cache_resource
def load_model_scaler():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model_KNN.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"❌ File scaler tidak ditemukan: {scaler_path}")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    st.success("✅ Model dan Scaler berhasil dimuat.")
    return model, scaler

model, scaler = load_model_scaler()

# ======================
# Fungsi Ekstraksi Fitur Statistik (tetap)
# ======================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=48000, mono=True)
    y = y / np.max(np.abs(y))
    y, _ = librosa.effects.trim(y, top_db=20)
    features = {
        "mean": np.mean(y),
        "std": np.std(y),
        "skew": skew(y),
        "kurtosis": kurtosis(y),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
    }
    feature_order = ["mean", "std", "skew", "kurtosis", "rms", "zcr"]
    df = pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)
    return df

# ======================
# Threshold probabilitas minimal untuk dianggap valid
# ======================
THRESHOLD = 0.6

# ======================
# UI Streamlit
# ======================
st.title("🎵 Identifikasi Suara: Buka / Tutup / Penyusup")
st.write("Upload file audio (.wav) atau rekam langsung dari microphone.")

# ----------------------
# 1️⃣ Upload File Audio
# ----------------------
uploaded_file = st.file_uploader("📂 Pilih file audio", type=["wav"])

if uploaded_file is not None:
    temp_audio = "temp_audio.wav"
    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format="audio/wav")

    X_new = extract_features(temp_audio)
    st.write("🧩 Fitur hasil ekstraksi:", X_new)
    X_scaled = scaler.transform(X_new)
    pred_label = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    if max(pred_proba) < THRESHOLD:
        pred_label = "Penyusup"

    st.subheader("🎯 Hasil Prediksi")
    st.write(f"**Kategori:** {pred_label}")
    st.subheader("📊 Probabilitas Kelas")
    for label, prob in zip(model.classes_, pred_proba):
        st.write(f"- {label}: {prob:.2f}")
    if pred_label == "Penyusup":
        st.warning("⚠️ Suara tidak dikenali, dianggap penyusup!")

# ----------------------
# 2️⃣ Rekam dari Microphone
# ----------------------
class AudioProcessor(AudioProcessorBase):
    def recv_audio(self, frame):
        audio_bytes = frame.to_ndarray().flatten()
        librosa.output.write_wav("temp_mic.wav", audio_bytes, sr=48000)
        return frame

st.header("🎤 Rekam Suara dari Microphone")
webrtc_ctx = webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration=RTCConfiguration({})
)

if st.button("Deteksi dari Microphone"):
    try:
        X_new = extract_features("temp_mic.wav")
        st.write("🧩 Fitur hasil ekstraksi:", X_new)
        X_scaled = scaler.transform(X_new)
        pred_label = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]

        if max(pred_proba) < THRESHOLD:
            pred_label = "Penyusup"

        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**Kategori:** {pred_label}")
        st.subheader("📊 Probabilitas Kelas")
        for label, prob in zip(model.classes_, pred_proba):
            st.write(f"- {label}: {prob:.2f}")
        if pred_label == "Penyusup":
            st.warning("⚠️ Suara tidak dikenali, dianggap penyusup!")

    except Exception as e:
        st.error(f"Terjadi error: {e}")