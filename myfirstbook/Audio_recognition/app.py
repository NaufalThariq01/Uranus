import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration

# ======================
# Konfigurasi Global
# ======================
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
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        if y is None or len(y) == 0:
            raise ValueError("File audio kosong atau tidak valid.")
    except Exception as e:
        st.error(f"Gagal memuat file audio: {e}")
        return [0]*50

    y = y / (np.max(np.abs(y)) + 1e-6)
    f = []

    # ======================
    # Statistik
    # ======================
    f += [
        np.mean(y), np.std(y), np.var(y),
        np.mean((y - np.mean(y))**3) / (np.std(y)**3 + 1e-6),
        np.mean((y - np.mean(y))**4) / (np.std(y)**4 + 1e-6),
        np.sqrt(np.mean(y**2)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(y**2), np.std(y**2),
        np.max(y) - np.min(y)
    ]

    # ======================
    # Spektral (pakai STFT agar aman)
    # ======================
    S = np.abs(librosa.stft(y))
    feats = {
        "centroid": librosa.feature.spectral_centroid,
        "bandwidth": librosa.feature.spectral_bandwidth,
        "contrast": librosa.feature.spectral_contrast,
        "rolloff": librosa.feature.spectral_rolloff,
        "flatness": librosa.feature.spectral_flatness,
        "chroma": librosa.feature.chroma_stft
    }

    for name, func in feats.items():
        try:
            if name in ["contrast", "chroma"]:
                val = func(S=S, sr=sr)
            else:
                val = func(y=y, sr=sr)
            f += [np.mean(val), np.std(val)]
        except Exception as e:
            st.warning(f"⚠️ Gagal ekstrak fitur {name}: {e}")
            f += [0, 0]

    # ======================
    # MFCC
    # ======================
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        f += [np.mean(mfcc[i]) for i in range(5)]
        f += [np.std(mfcc[i]) for i in range(5)]
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak MFCC: {e}")
        f += [0]*10

    # ======================
    # Temporal
    # ======================
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        autocorr = np.correlate(y, y, mode='full')[len(y):]
        f += [
            librosa.get_duration(y=y, sr=sr),
            float(tempo[0]),
            np.mean(onset_env),
            np.argmax(autocorr) + 1,
            np.mean(np.abs(onset_env)),
            np.std(np.abs(onset_env))
        ]
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak fitur temporal: {e}")
        f += [0]*6

    return f

# ======================
# Fungsi Prediksi
# ======================
def predict_audio(path):
    X_new = np.array(extract_features(path)).reshape(1, -1)
    X_scaled = scaler.transform(X_new)
    probs = model.predict_proba(X_scaled)[0]
    label = model.classes_[np.argmax(probs)]
    if max(probs) < THRESHOLD:
        label = "Penyusup"
    return label, probs

# ======================
# UI Streamlit
# ======================
st.title("🎵 Identifikasi Suara: Buka / Tutup / Penyusup")
st.write("Unggah file .wav atau rekam langsung dari microphone.")

# Upload File
uploaded = st.file_uploader("📂 Pilih file audio (.wav)", type=["wav"])
if uploaded:
    path = "temp_upload.wav"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.audio(uploaded, format="audio/wav")

    label, probs = predict_audio(path)
    st.subheader("🎯 Hasil Prediksi")
    st.write(f"**Kategori:** {label}")
    st.write("📊 Probabilitas:")
    for l, p in zip(model.classes_, probs):
        st.write(f"- {l}: {p:.2f}")
    if label == "Penyusup":
        st.warning("⚠️ Suara tidak dikenali!")

# Rekam Mic
st.header("🎤 Rekam dari Microphone")

class AudioProcessor(AudioProcessorBase):
    def recv_audio(self, frame):
        data = frame.to_ndarray().flatten()
        sf.write("temp_mic.wav", data, 48000)
        return frame

webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration=RTCConfiguration({})
)

if st.button("🔍 Deteksi dari Microphone"):
    try:
        label, probs = predict_audio("temp_mic.wav")
        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**Kategori:** {label}")
        st.write("📊 Probabilitas:")
        for l, p in zip(model.classes_, probs):
            st.write(f"- {l}: {p:.2f}")
        if label == "Penyusup":
            st.warning("⚠️ Suara tidak dikenali!")
    except Exception as e:
        st.error(f"Terjadi error: {e}")
