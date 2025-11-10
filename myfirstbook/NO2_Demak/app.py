import streamlit as st
import pickle
import numpy as np
import os

# ======================
# 🔧 Fungsi Load Model
# ======================
def load_model(day, forecast_type="1day"):
    """
    Memuat model dan scaler sesuai tipe forecast (1 hari atau 3 hari)
    """
    # Sesuaikan path folder model
    base_path = os.path.join("myfirstbook", "NO2_Demak")

    if forecast_type == "1day":
        folder = os.path.join(base_path, "no2_models")
        model_file = f"knn_day{day}.pkl"
        scaler_file = f"scaler_day{day}.pkl"
    elif forecast_type == "3day":
        folder = os.path.join(base_path, "no2_models_multiday")
        model_file = f"knn_day{day}_forecast3.pkl"
        scaler_file = f"scaler_day{day}_forecast3.pkl"
    else:
        raise ValueError("Tipe forecast tidak dikenal. Gunakan '1day' atau '3day'.")

    model_path = os.path.join(folder, model_file)
    scaler_path = os.path.join(folder, scaler_file)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"❌ Model atau scaler tidak ditemukan untuk day {day} ({forecast_type}).")
        st.write("Current working directory:", os.getcwd())
        st.write("Path dicari:", model_path)
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

# ======================
# 🌤️ UI Streamlit
# ======================
st.title("🌤️ Prediksi Kualitas Udara (NO₂) - Kabupaten Demak")
st.write("Gunakan data beberapa hari sebelumnya untuk memprediksi kadar NO₂ 1 hari atau 3 hari ke depan.")

# Pilihan model
forecast_type_label = st.radio("Pilih jenis prediksi:", ["1 Hari ke Depan", "3 Hari ke Depan"])
forecast_type_key = "1day" if forecast_type_label == "1 Hari ke Depan" else "3day"

day_option = st.selectbox("Gunakan berapa hari sebelumnya:", [2, 3, 4, 5])

# Muat model dan scaler
model, scaler = load_model(day_option, forecast_type_key)

# Tentukan jumlah fitur yang dibutuhkan
n_features = getattr(scaler, "n_features_in_", day_option)

st.subheader("Masukkan data kadar NO₂ dari hari-hari sebelumnya (dalam mol/m²)")
inputs = []
for i in range(n_features):
    value = st.number_input(
        f"NO₂ (t-{i+1})", 
        min_value=0.000001, 
        max_value=0.001000, 
        step=0.000001, 
        format="%.6f", 
        key=f"input_{i}"
    )
    inputs.append(value)

## ======================
# 🔮 Prediksi
# ======================
if st.button("Prediksi Sekarang"):
    X_input = np.array(inputs).reshape(1, -1)

    try:
        # Scaling input
        X_scaled = scaler.transform(X_input)

        # Prediksi
        prediction = model.predict(X_scaled)

        # ======================
        # 🔹 Konversi prediksi menjadi float tunggal
        # ======================
        if isinstance(prediction, np.ndarray):
            if prediction.ndim == 2:       # array 2D, ambil elemen pertama
                prediction_value = float(prediction[0, 0])
            else:                           # array 1D, ambil elemen pertama
                prediction_value = float(prediction[0])
        else:                               # sudah float tunggal
            prediction_value = float(prediction)

        # Tampilkan hasil prediksi
        st.success(f"💡 Hasil prediksi kadar NO₂: **{prediction_value:.8f} mol/m²**")

        # ======================
        # 🔹 Interpretasi kategori kualitas udara
        # ======================
        if prediction_value <= 0.000019:
            kategori = "🟢 RENDAH"
        elif 0.000019 < prediction_value <= 0.000035:
            kategori = "🟡 SEDANG"
        else:
            kategori = "🔴 TINGGI"

        st.info(f"Kategori Kualitas Udara: {kategori}")

    except ValueError as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Pastikan jumlah input sesuai dengan model yang dipilih.")

# ======================
# 📋 Informasi tambahan
# ======================
with st.expander("ℹ️ Informasi Threshold"):
    st.markdown("""
    **Kategori Kadar NO₂ (mol/m²):**
    - 🟢 RENDAH : ≤ 0.000019  
    - 🟡 SEDANG : 0.000019 - 0.000035  
    - 🔴 TINGGI : > 0.000035  

    **Statistik Data:**
    - Minimum: 0.000010 mol/m²  
    - Maksimum: 0.000090 mol/m²
    """)


