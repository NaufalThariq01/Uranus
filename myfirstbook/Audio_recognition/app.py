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
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    features = []

    # ----- 1. Statistik (10 fitur) -----
    features.append(np.mean(y))                         # mean
    features.append(np.std(y))                          # std
    features.append(np.var(y))                          # var
    features.append(np.mean((y - np.mean(y))**3)/(np.std(y)**3 + 1e-6))  # skew
    features.append(np.mean((y - np.mean(y))**4)/(np.std(y)**4 + 1e-6))  # kurtosis
    features.append(np.sqrt(np.mean(y**2)))            # RMS
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))  # ZCR

    # Energy mean & std
    energy = y**2
    features.append(np.mean(energy))                   # energy_mean
    features.append(np.std(energy))                    # energy_std

    # Amplitude range (max-min)
    features.append(np.max(y) - np.min(y))            # amplitude_range

    # ----- 2. Spektral (20 fitur) -----
    # Spectral centroid & bandwidth
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bandwidth))
    features.append(np.std(spec_bandwidth))

    # Spectral contrast (7 bands → mean & std digabung)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.append(np.mean(spec_contrast))
    features.append(np.std(spec_contrast))

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(spec_rolloff))
    features.append(np.std(spec_rolloff))

    # Spectral flatness
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(spec_flatness))
    features.append(np.std(spec_flatness))

    # Chroma (12 bins → mean & std digabung)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma))
    features.append(np.std(chroma))

    # MFCC 5 koefisien → mean & std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i in range(5):
        features.append(np.mean(mfcc[i]))
    for i in range(5):
        features.append(np.std(mfcc[i]))

    # ----- 3. Temporal (6 fitur) -----
    # Duration
    features.append(librosa.get_duration(y=y, sr=sr))

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    features.append(float(tempo))

    # Onset rate
    features.append(np.mean(onset_env))

    # Autocorrelation lag (puncak pertama di autocorr)
    autocorr = np.correlate(y, y, mode='full')
    mid = len(autocorr)//2
    autocorr_lag = np.argmax(autocorr[mid+1:]) + 1
    features.append(autocorr_lag)

    # Envelope mean & std
    envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    features.append(np.mean(envelope))
    features.append(np.std(envelope))

    return features
 
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
