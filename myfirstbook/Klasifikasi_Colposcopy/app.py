import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


model = load_model("colposcopy_lstm_model.h5")

CLASS_NAMES = [
    "Atrophy",
    "Inflammation",
    "Ectopy",
    "Normal",
    "LSIL",
    "HSIL"
]


def minmax_normalize(ts):
    ts = np.array(ts, dtype=float)
    if ts.max() - ts.min() == 0:
        return np.zeros_like(ts)
    return (ts - ts.min()) / (ts.max() - ts.min())


st.set_page_config(page_title="Colposcopy Diagnosis", layout="centered")

st.title("🩺 Colposcopy Diagnosis System")
st.write("Prediksi diagnosis colposcopy berdasarkan pola acetowhitening")

uploaded_file = st.file_uploader(
    "Upload time series acetowhite (180 nilai)",
    type=["csv", "txt"]
)

if uploaded_file is not None:


    try:
        data = pd.read_csv(uploaded_file, header=None).values.flatten()
    except:
        st.error("Format file tidak valid.")
        st.stop()

    if len(data) != 180:
        st.error("Data harus terdiri dari 180 nilai time series.")
        st.stop()

 
    data_norm = minmax_normalize(data)
    X_input = data_norm.reshape(1, 180, 1)

  
    pred_prob = model.predict(X_input)
    pred_class = np.argmax(pred_prob)
    confidence = np.max(pred_prob) * 100

    
    st.subheader("🧾 Hasil Prediksi")
    st.success(f"Diagnosis: **{CLASS_NAMES[pred_class]}**")
    st.write(f"Tingkat kepercayaan: **{confidence:.2f}%**")

    st.subheader("📊 Probabilitas Setiap Kelas")
    prob_df = pd.DataFrame({
        "Diagnosis": CLASS_NAMES,
        "Probabilitas (%)": pred_prob.flatten() * 100
    })

    st.bar_chart(prob_df.set_index("Diagnosis"))

