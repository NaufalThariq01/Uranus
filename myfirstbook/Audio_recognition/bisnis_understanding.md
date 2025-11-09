# 🧩 1. Business Understanding
## 1.1 Latar Belakang

Dalam era digital dan otomasi, sistem pengenalan suara (voice recognition) memiliki peran penting dalam berbagai bidang, mulai dari *smart home*, perintah suara (*voice command*), hingga sistem keamanan cerdas. Salah satu penerapan sederhana namun relevan adalah identifikasi suara **“buka”** dan **“tutup”**, yang dapat digunakan untuk mendeteksi aktivitas fisik seperti membuka atau menutup pintu, laci, atau wadah tertentu.

Pengenalan suara biasanya dilakukan menggunakan pendekatan berbasis analisis frekuensi seperti **Spectrogram** atau **Mel-Frequency Cepstral Coefficients (MFCC)**. Namun, metode tersebut membutuhkan komputasi yang cukup tinggi dan tidak selalu efisien untuk aplikasi sederhana atau perangkat dengan sumber daya terbatas.

Penelitian ini mencoba pendekatan **ringan namun efektif**, yaitu dengan mengekstraksi **36 fitur utama** dari sinyal audio yang dikelompokkan menjadi tiga jenis utama: **fitur statistik, spektral, dan temporal**.

---

### 🔹 Fitur Statistik
Merepresentasikan karakteristik bentuk gelombang secara numerik berdasarkan distribusi amplitudo sinyal.

Beberapa fitur yang digunakan:
- Mean amplitude – rata-rata nilai amplitudo sinyal
- Standard deviation – ukuran sebaran amplitudo terhadap rata-rata
- Variance
- Skewness – tingkat kemiringan distribusi sinyal
- Kurtosis – tingkat keruncingan distribusi
- Root Mean Square (RMS) – kekuatan energi rata-rata sinyal
- Zero Crossing Rate (ZCR) – frekuensi perubahan tanda positif/negatif pada sinyal
- Energy mean & std
- Amplitude max & min

➡️ **Total: 10 fitur statistik**

---

### 🔹 Fitur Spektral
Menggambarkan karakteristik frekuensi dan energi dalam domain spektrum sinyal audio.

Beberapa fitur yang digunakan:
- Spectral centroid (mean, std) – posisi pusat energi spektrum
- Spectral bandwidth (mean, std) – lebar pita frekuensi utama
- Spectral contrast (mean, std) – perbedaan antara puncak dan lembah energi spektrum
- Spectral roll-off (mean, std) – frekuensi di mana energi kumulatif mencapai 85–95%
- Spectral flatness (mean, std) – tingkat keseragaman spektrum
- Chroma features (mean, std) – distribusi energi per nada musik (12 bin)
- MFCC (mean, std untuk 5 koefisien utama) – representasi mel-scale dari spektrum

➡️ **Total: 20 fitur spektral**

---

### 🔹 Fitur Temporal
Berhubungan dengan aspek waktu dan dinamika perubahan sinyal audio.

Beberapa fitur yang digunakan:
- Tempo (BPM) – kecepatan ritme sinyal
- Duration (detik) – panjang sinyal audio
- Onset rate – laju kemunculan serangan bunyi
- Autocorrelation lag – pola pengulangan periodik sinyal
- Envelope mean & std – perubahan amplop energi terhadap waktu
- Attack time & decay time – waktu naik dan turun energi sinyal

➡️ **Total: 6 fitur temporal**

---

### 🔹 Total Keseluruhan

| Jenis Fitur | Jumlah Fitur | Contoh |
|--------------|---------------|---------|
| Statistik | 10 | mean, std, skewness |
| Spektral | 20 | spectral centroid, MFCC |
| Temporal | 6 | tempo, duration |

**Total: 36 fitur audio**

## 1.2 Tujuan Penelitian

Tujuan utama penelitian ini adalah:

Mengidentifikasi dan mengklasifikasikan suara “buka” dan “tutup” berdasarkan kombinasi fitur statistik, spektral, dan temporal.

Menentukan jenis fitur yang paling berpengaruh dalam membedakan kedua jenis suara tersebut.

Membangun model klasifikasi berbasis machine learning (seperti K-Nearest Neighbors, Random Forest, atau Naive Bayes) untuk mengenali pola suara.

Mengembangkan aplikasi sederhana berbasis Streamlit untuk menguji hasil model dengan input suara langsung dari pengguna.

## 1.3 Rumusan Masalah

1. Bagaimana mengekstraksi fitur statistik, spektral, dan temporal dari sinyal audio “buka” dan “tutup”?

2. Fitur mana yang memberikan performa terbaik dalam membedakan dua jenis suara tersebut?

3. Algoritma klasifikasi apa yang paling akurat untuk mengenali suara “buka” dan “tutup”?

4. Bagaimana menerapkan hasil model ke dalam aplikasi interaktif untuk mendeteksi suara secara real-time?

## 1.4 Batasan Masalah

Data suara direkam langsung menggunakan perekam ponsel oleh dua orang berbeda.

Masing-masing orang menghasilkan 50 sampel suara “buka” dan 50 sampel suara “tutup”, dengan total 200 data audio (100 dari setiap orang).

Dataset dibagi dalam dua folder berdasarkan sumber suara (dua orang berbeda).

Tujuan tambahan: sistem dapat mengenali suara hanya jika berasal dari dua orang tersebut; jika suara berasal dari orang lain, akan muncul peringatan “Anda bukan pengguna terdaftar / penyusup”.

Penelitian hanya fokus pada dua kelas (buka dan tutup), tanpa melibatkan jenis suara lain.

Transformasi fitur dilakukan dengan tiga pendekatan (statistik, spektral, temporal), namun hanya fitur dengan performa terbaik yang digunakan dalam tahap akhir modelling.

## 1.5 Manfaat Penelitian

Memberikan alternatif metode pengenalan suara yang ringan, cepat, dan efisien, tanpa membutuhkan GPU atau transformasi kompleks.

Menjadi dasar pengembangan sistem keamanan atau otomatisasi berbasis audio pattern recognition.

Menunjukkan perbandingan efektivitas antara fitur statistik, spektral, dan temporal dalam konteks klasifikasi dua suara sederhana.

Memberikan contoh implementasi praktis pengenalan suara ke dalam aplikasi interaktif berbasis Streamlit.

## 1.6 Sumber Dataset

Dataset direkam langsung menggunakan perekam ponsel, terdiri dari:

* Folder Dataset_Voice_pertama: 50 suara “buka” dan 50 suara “tutup” dari orang pertama,

* Folder Dataset_Voice_kedua: 50 suara “buka” dan 50 suara “tutup” dari orang kedua.

Dataset ini digunakan agar sistem mampu:

* Membedakan pola suara “buka” dan “tutup”,

* Mengenali suara hanya dari dua orang terdaftar, dan

* Menolak atau memberi peringatan bila suara berasal dari orang lain (penyusup).