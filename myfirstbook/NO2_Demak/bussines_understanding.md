# 🏙️ Analisis dan Prediksi Kualitas Udara Berdasarkan Kadar NO₂ di Daerah Demak

## 📘 1. Latar Belakang

Kualitas udara merupakan salah satu faktor penting yang memengaruhi kesehatan manusia dan keseimbangan lingkungan. Salah satu parameter utama dalam penilaian kualitas udara adalah konsentrasi gas Nitrogen Dioksida (NO₂). Gas ini umumnya dihasilkan dari aktivitas manusia seperti pembakaran bahan bakar kendaraan bermotor, industri, serta proses pembangkitan energi.

Kadar NO₂ yang tinggi dapat menyebabkan gangguan kesehatan, terutama pada sistem pernapasan, serta menjadi indikator utama terjadinya pencemaran udara. Oleh karena itu, memantau dan memprediksi kadar NO₂ secara berkala sangat penting untuk mendukung kebijakan pengendalian polusi dan perencanaan lingkungan yang berkelanjutan.

Melalui kemajuan teknologi penginderaan jauh, kini kita dapat memperoleh data kadar NO₂ dari satelit seperti Sentinel-5P, yang dapat diakses melalui Copernicus Data Space menggunakan antarmuka OpenEO API. Data ini memberikan pengamatan harian terhadap distribusi NO₂ di berbagai wilayah, termasuk daerah Demak dan sekitarnya.

Dalam penelitian ini, dilakukan pemodelan menggunakan pendekatan machine learning berbasis deret waktu (time series) untuk memprediksi kadar NO₂ di hari berikutnya berdasarkan data 3, 4, dan 5 hari sebelumnya. Model yang digunakan adalah K-Nearest Neighbors (KNN) Regressor, yang bertujuan untuk mengenali pola historis perubahan kadar NO₂ dan memberikan hasil prediksi yang mendekati nilai aktual.

## ❓ 2. Rumusan Masalah

Berdasarkan latar belakang tersebut, dapat dirumuskan beberapa permasalahan utama sebagai berikut:

Bagaimana memperoleh dan mengolah data kadar NO₂ dari satelit Sentinel-5P untuk daerah Demak secara harian?

Bagaimana proses preprocessing data dilakukan agar data siap digunakan untuk analisis dan pemodelan?

Sejauh mana model K-Nearest Neighbors (KNN) mampu memprediksi kadar NO₂ berdasarkan data 3, 4, dan 5 hari sebelumnya?

Bagaimana tingkat akurasi model dalam memprediksi kualitas udara berdasarkan hasil evaluasi seperti MSE, R², dan MAPE?

## 🎯 3. Tujuan Penelitian

Tujuan dari penelitian ini adalah:

Mengambil dan memproses data kadar NO₂ dari satelit Sentinel-5P menggunakan OpenEO API untuk wilayah Demak.

Melakukan preprocessing seperti interpolasi data hilang dan pembuatan fitur lag (3, 4, dan 5 hari sebelumnya).

Membangun dan mengevaluasi model KNN Regressor untuk memprediksi kadar NO₂ hari berikutnya.

Menganalisis hasil evaluasi model menggunakan metrik Mean Squared Error (MSE), R² Score, dan Mean Absolute Percentage Error (MAPE) untuk menilai tingkat keakuratan prediksi.

## 🧠 4. Manfaat Penelitian

Penelitian ini diharapkan dapat memberikan manfaat sebagai berikut:

Menyediakan pendekatan berbasis data untuk memprediksi kualitas udara secara harian.

Membantu masyarakat dan instansi pemerintah dalam pemantauan polusi udara serta pengambilan keputusan preventif terhadap peningkatan kadar NO₂.

Menjadi dasar bagi pengembangan sistem peramalan kualitas udara otomatis berbasis pembelajaran mesin di masa depan.

## ⚙️ 5. Metode Singkat

Proses penelitian ini melibatkan beberapa tahap utama:

Pengumpulan Data: Mengambil data NO₂ harian menggunakan OpenEO dari satelit Sentinel-5P.

Preprocessing: Meliputi interpolasi nilai yang hilang, pembuatan fitur lag (3, 4, 5 hari), serta normalisasi data.

Modeling: Menggunakan algoritma KNN Regressor untuk memprediksi kadar NO₂ berdasarkan data historis.

Evaluasi: Mengukur performa model menggunakan MSE, R², dan MAPE, serta menafsirkan tingkat keakuratan hasil prediksi.