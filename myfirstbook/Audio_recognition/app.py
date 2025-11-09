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
    y_harm, y_perc = librosa.effects.hpss(y)

    # --- 1. Statistik dasar sinyal ---
    mean = np.mean(y)
    std = np.std(y)
    var = np.var(y)
    skew = np.mean((y - mean)**3) / (std**3 + 1e-6)
    kurtosis = np.mean((y - mean)**4) / (std**4 + 1e-6)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # --- 2. Energi ---
    energy = y ** 2
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    
    # --- 3. Rentang amplitudo ---
    amplitude_range = np.max(y) - np.min(y)
    
    # --- 4. Spektrum ---
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    
    # --- 5. Chroma & MFCC ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    
    # --- 6. Tempo & Onset ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    except Exception:
        tempo = 0.0
    onset_rate = librosa.onset.onset_detect(y=y, sr=sr).shape[0]
    
    # --- 7. Autocorrelation lag ---
    autocorr = np.correlate(y, y, mode='full')
    autocorr_lag = np.argmax(autocorr[len(autocorr)//2:])
    
    # --- 8. Envelope ---
    envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    
    # --- 9. Durasi ---
    duration = librosa.get_duration(y=y, sr=sr)
    
    # --- Gabungkan semua ke array fitur ---
    features = [
        mean, std, var, skew, kurtosis, rms, zcr,
        energy_mean, energy_std, amplitude_range,
        np.mean(spec_centroid), np.std(spec_centroid),
        np.mean(spec_bandwidth), np.std(spec_bandwidth),
        np.mean(spec_contrast), np.std(spec_contrast),
        np.mean(spec_rolloff), np.std(spec_rolloff),
        np.mean(spec_flatness), np.std(spec_flatness),
        np.mean(chroma), np.std(chroma),
        np.mean(mfcc[0]), np.mean(mfcc[1]), np.mean(mfcc[2]),
        np.mean(mfcc[3]), np.mean(mfcc[4]),
        np.std(mfcc[0]), np.std(mfcc[1]), np.std(mfcc[2]),
        np.std(mfcc[3]), np.std(mfcc[4]),
        duration, tempo, onset_rate, autocorr_lag,
        envelope_mean, envelope_std
    ]
    
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
    path = os.path.join(os.getcwd(), "temp_upload.wav")
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
        label, probs = predict_audio(temp_path)
        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**Kategori:** {label}")
        st.write("📊 Probabilitas:")
        for l, p in zip(model.classes_, probs):
            st.write(f"- {l}: {p:.2f}")
        if label == "Penyusup":
            st.warning("⚠️ Suara tidak dikenali!")
    except Exception as e:
        st.error(f"Terjadi error: {e}")
