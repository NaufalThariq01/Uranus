# 🧠 2. Data Understanding
## 2.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini berasal dari dua sumber data hasil rekaman langsung menggunakan perekam ponsel. Masing-masing dataset direkam oleh dua orang berbeda untuk mensimulasikan sistem pengenalan suara berbasis identitas.

Setiap dataset memiliki dua kelas utama:

* Buka → berisi rekaman suara kata “buka”

* Tutup → berisi rekaman suara kata “tutup”

Tujuan dari pembagian ini adalah agar sistem dapat:

* Mengenali perbedaan antara suara “buka” dan “tutup”.

* Membedakan apakah suara tersebut berasal dari pengguna yang dikenal (dua orang perekam dataset).

* Memberikan peringatan (“Anda bukan pengguna terdaftar / penyusup”) jika input berasal dari orang lain.

Struktur folder dataset adalah sebagai berikut:
Uranus/
 └── myfirstbook/
     └── Audio_recognition/
         ├── Dataset_Voice_pertama/
         │   ├── Buka_wav/
         │   └── Tutup_wav/
         └── Dataset_Voice_kedua/
             ├── Buka/
             └── Tutup/

📸 Contoh struktur folder (dari dataset kedua):
Uranus\myfirstbook\Audio_recognition\Dataset_Voice_kedua\Buka
Uranus\myfirstbook\Audio_recognition\Dataset_Voice_kedua\Tutup

## 2.2 Spesifikasi Dataset

| Atribut                | Nilai / Keterangan                            |
| ---------------------- | --------------------------------------------- |
| Format File            | `.wav`                                        |
| Sampling Rate          | 16.000 Hz                                     |
| Durasi per audio       | ± 1 detik                                     |
| Jumlah kelas           | 2 (buka, tutup)                               |
| Jumlah dataset         | 2 sumber suara (2 orang berbeda)              |
| Jumlah data per orang  | 50 suara “buka” + 50 suara “tutup” = 100 file |
| Total data keseluruhan | 200 file audio                                |

## 2.3 Eksplorasi Awal

Dari eksplorasi awal terhadap data audio:

* Setiap file direkam dengan durasi relatif seragam (~2 detik).

* Sampling rate konsisten di 16 kHz untuk semua file.

* Beberapa file memiliki noise latar belakang ringan (napas, bunyi kipas, atau gema ruangan).

* Pola suara antara dua individu berbeda secara signifikan, terutama dari segi pitch dan intensitas.

* Distribusi jumlah data antar kelas seimbang (masing-masing 50 per kelas per orang).

* Visualisasi awal (waveform dan spektrogram) menunjukkan perbedaan yang konsisten:

* Suara “buka” memiliki amplitudo meningkat perlahan dan durasi sedikit lebih panjang.

* Suara “tutup” memiliki ledakan energi cepat di awal dan durasi lebih pendek.

## 2.4 Rencana Ekstraksi Fitur

Tahapan analisis fitur dibagi menjadi tiga kategori utama yang mencakup total **36 fitur audio**: fitur statistik, fitur spektral, dan fitur temporal.  
Setiap kategori mewakili karakteristik berbeda dari sinyal audio dan akan dievaluasi secara terpisah untuk menentukan kelompok fitur yang paling efektif dalam membedakan suara **“buka”** dan **“tutup”**.

---

### 🧮 1. Fitur Statistik

Berbasis sinyal **time-series mentah (y)** tanpa transformasi frekuensi.  
Fitur ini merepresentasikan bentuk dan distribusi amplitudo sinyal.

Fitur yang diekstraksi meliputi:

- Mean amplitude  
- Standard deviation  
- Variance  
- Skewness  
- Kurtosis  
- Root Mean Square (RMS)  
- Zero Crossing Rate (ZCR)  
- Energy mean  
- Energy std  
- Amplitude max dan min  

➡️ **Total: 10 fitur statistik**

---

### 🎵 2. Fitur Spektral

Dihitung dari **transformasi frekuensi** menggunakan fungsi `librosa.feature`.  
Fitur ini menggambarkan persebaran energi dalam domain spektrum suara.

Fitur yang diekstraksi meliputi:

- Spectral centroid (mean, std)  
- Spectral bandwidth (mean, std)  
- Spectral contrast (mean, std)  
- Spectral roll-off (mean, std)  
- Spectral flatness (mean, std)  
- Chroma features (mean, std)  
- MFCC (mean, std untuk 5 koefisien utama)  

➡️ **Total: 20 fitur spektral**

---

### ⏱️ 3. Fitur Temporal

Berhubungan dengan **durasi, ritme, dan dinamika energi sinyal** dalam domain waktu.

Fitur yang diekstraksi meliputi:

- Tempo (BPM)  
- Duration (detik)  
- Onset rate  
- Autocorrelation lag  
- Envelope mean dan std  
- Attack time dan decay time  

➡️ **Total: 6 fitur temporal**

---

### 📊 Total Keseluruhan

| Jenis Fitur | Jumlah Fitur | Contoh |
|--------------|---------------|---------|
| Statistik | 10 | mean, std, RMS |
| Spektral | 20 | MFCC, spectral centroid |
| Temporal | 6 | tempo, duration |

**Total keseluruhan: 36 fitur audio**

## 2.5 Rencana Analisis

* Eksplorasi visual – waveform, spektrogram, dan distribusi fitur tiap kelas.

* Statistical summary – analisis perbedaan mean dan variansi antar kelas.

* Korelasi fitur – untuk menghindari fitur yang redundan.

* Evaluasi model – menguji performa tiga kelompok fitur (statistik, spektral, temporal) menggunakan model KNN, Random Forest, dan Naive Bayes.

* Pemilihan fitur terbaik – menentukan fitur yang paling kontributif terhadap akurasi model.

## 2.6 Insight Awal

Suara “buka” cenderung memiliki pola energi meningkat perlahan dan amplitudo puncak di akhir.

Suara “tutup” menampilkan energi awal tinggi lalu cepat menurun.

Fitur RMS dan Spectral Centroid menunjukkan pola pemisahan yang cukup jelas antar kelas.

Variasi antar individu bisa menjadi dasar pengembangan sistem verifikasi suara (voice identity) di tahap lanjutan.