# 🧩 4. Data Preparation
## 4.1 Tujuan

Tahapan Data Preparation bertujuan untuk menyiapkan data audio mentah dari dataset Kaggle agar siap digunakan dalam proses modeling. Proses ini mencakup pengorganisasian ulang dataset, pembersihan awal, penyeragaman format, serta ekstraksi fitur statistik yang akan digunakan sebagai input model klasifikasi.

## 4.2 Proses Penyiapan Data
### a. Struktur dan Integrasi Data

Dataset Kaggle terdiri dari dua folder utama:

* train → berisi 150 file audio kelas buka dan 150 file audio kelas tutup

* val (validation/test) → berisi 50 file audio kelas buka dan 50 file audio kelas tutup

Semua file audio dikumpulkan ke dalam satu struktur baru agar mudah diproses oleh program:

dataset/
├── train/
│   ├── buka/
│   └── tutup/
└── test/
    ├── buka/
    └── tutup/

Struktur ini digunakan untuk memudahkan pipeline pembacaan data saat proses ekstraksi fitur menggunakan Python.

### b. Penyeragaman Format Audio

Semua file audio diseragamkan agar memiliki spesifikasi yang sama:

* Format: .wav

* Sample rate: 48.000 Hz

* Kanal: Mono

* Durasi: ±1–2 detik

Proses ini dilakukan menggunakan library librosa dengan fungsi librosa.load(path, sr=48000, mono=True) agar semua sinyal memiliki bentuk time series yang konsisten.

### c. Preprocessing Awal

Sebelum melakukan ekstraksi fitur, dilakukan beberapa tahap preprocessing:

* Normalisasi amplitudo
Menyamakan skala amplitudo agar sinyal berada di rentang -1 hingga 1.

* Penghapusan noise
Menggunakan pendekatan spectral gating untuk mengurangi gangguan latar seperti suara angin atau klik.

* Trimming (pemotongan diam)
Menghapus bagian diam pada awal dan akhir rekaman agar fokus hanya pada bagian utama suara.

* Padding (penyeragaman durasi)
Jika durasi kurang dari panjang target, bagian kosong diisi dengan nol (zero padding) agar panjang sinyal seragam.

### d. Ekstraksi Fitur Statistik

Setiap file audio kemudian dikonversi menjadi vektor fitur numerik yang mewakili karakteristik statistik sinyal waktu.
Beberapa fitur yang digunakan adalah:

| Jenis Fitur                  | Keterangan                                   |
| ---------------------------- | -------------------------------------------- |
| **Mean**                     | Nilai rata-rata amplitudo sinyal             |
| **Standard Deviation**       | Variasi amplitudo sinyal terhadap mean       |
| **Skewness**                 | Kemiringan distribusi amplitudo sinyal       |
| **Kurtosis**                 | Kepuncakan distribusi sinyal                 |
| **Root Mean Square (RMS)**   | Ukuran energi rata-rata sinyal               |
| **Zero Crossing Rate (ZCR)** | Jumlah perpotongan sinyal terhadap sumbu nol |

Nilai-nilai ini diambil langsung dari sinyal time domain tanpa transformasi spektral (tanpa FFT atau MFCC), sesuai dengan batasan masalah penelitian.

### e. Penyusunan Dataset Fitur

Setiap file .wav menghasilkan satu baris data berisi nilai-nilai fitur dan labelnya (“buka” atau “tutup”).
Contoh tabel hasil akhir:

| file          | mean  | std  | skew  | kurtosis | rms  | zcr   | label |
| ------------- | ----- | ---- | ----- | -------- | ---- | ----- | ----- |
| buka_001.wav  | 0.014 | 0.29 | -0.45 | 2.13     | 0.22 | 0.041 | buka  |
| tutup_087.wav | 0.008 | 0.25 | 0.33  | 1.97     | 0.19 | 0.038 | tutup |

Dataset ini kemudian disimpan ke file fitur_statistik.csv yang akan digunakan pada tahap modeling.

## 4.3 Hasil Akhir

Tahap Data Preparation menghasilkan dataset terstruktur yang telah melalui:

Penyeragaman format audio

Pembersihan dan normalisasi sinyal

Ekstraksi fitur statistik

Pelabelan dan penyimpanan hasil ke format CSV

Dataset inilah yang siap digunakan pada proses pelatihan model klasifikasi untuk membedakan suara “buka” dan “tutup”.