# 🧩 4. Data Preparation

## 4.1 Tujuan

Tahapan **Data Preparation** bertujuan untuk menyiapkan data audio mentah dari dua sumber rekaman berbeda agar siap digunakan pada tahap **ekstraksi fitur dan pemodelan klasifikasi suara**.  
Proses ini mencakup pengorganisasian dataset, penyeragaman format file, pembersihan sinyal, serta penyusunan dataset fitur yang siap dipakai untuk pelatihan model.

---

## 4.2 Proses Penyiapan Data

### 🎛️ a. Struktur dan Integrasi Data

Dataset berasal dari dua sumber rekaman dengan struktur awal sebagai berikut:

Uranus/
└── myfirstbook/
└── Audio_recognition/
├── Dataset_Voice_pertama/
│ ├── Buka_wav/
│ └── Tutup_wav/
└── Dataset_Voice_kedua/
├── Buka/
└── Tutup/


Kedua dataset ini kemudian digabung dan disusun ulang ke dalam format baru agar mudah diproses secara terstruktur:

dataset_merged/
├── train/
│ ├── buka/
│ └── tutup/
└── test/
├── buka/
└── tutup/


Struktur ini digunakan untuk memudahkan proses pembacaan dan pembagian data (train-test split) pada tahap selanjutnya.

---

### 🎧 b. Penyeragaman Format Audio

Dataset pertama memiliki sebagian file dengan format **.aac**, sementara **Dataset_Voice_kedua** sudah sepenuhnya dalam format **.wav**.  
Agar konsisten, file dari dataset pertama dikonversi ke `.wav` menggunakan *audio converter* (FFmpeg atau Audacity).

Spesifikasi akhir audio:

| Parameter | Nilai Standar |
|------------|----------------|
| Format | `.wav` |
| Sample rate | 16.000 Hz |
| Kanal | Mono |
| Durasi rata-rata | ±1–2 detik |

Setelah konversi, seluruh data audio dari kedua dataset memiliki spesifikasi yang sama dan siap diproses.

---

### 🧹 c. Preprocessing Awal

Sebelum dilakukan ekstraksi fitur, sinyal audio melalui beberapa tahap *preprocessing* agar data lebih bersih dan seragam:

| Tahap | Deskripsi |
|--------|------------|
| **Normalisasi amplitudo** | Menyelaraskan skala amplitudo ke rentang -1 hingga 1 |
| **Noise reduction** | Mengurangi gangguan latar (napas, kipas, gema) menggunakan teknik *spectral gating* |
| **Trimming silence** | Menghapus bagian diam di awal dan akhir rekaman |
| **Padding durasi** | Menyeragamkan panjang sinyal menggunakan *zero padding* bila durasi kurang dari target |

Tahapan ini memastikan setiap file memiliki bentuk sinyal dan energi yang setara sebelum diekstraksi.

---

### 🧮 d. Ekstraksi Fitur Audio

Setiap file audio `.wav` dikonversi menjadi vektor numerik berisi **36 fitur audio** yang merepresentasikan karakteristik utama sinyal dari tiga kategori:

| Jenis Fitur | Contoh | Jumlah |
|--------------|---------|---------|
| **Statistik** | mean, std, RMS, skewness | 10 |
| **Spektral** | spectral centroid, MFCC, chroma | 20 |
| **Temporal** | duration, tempo, onset rate | 6 |

Ekstraksi dilakukan menggunakan kombinasi fungsi dari **librosa** dan **NumPy**, mencakup karakteristik domain waktu dan frekuensi.

Contoh potongan kode ekstraksi fitur:

```python
import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    features = []

    # Statistik
    features += [np.mean(y), np.std(y), np.var(y)]
    features += [np.mean((y - np.mean(y))**3)/(np.std(y)**3)]
    features += [np.mean((y - np.mean(y))**4)/(np.std(y)**4)]
    features += [np.sqrt(np.mean(y**2))]
    features += [np.mean(librosa.feature.zero_crossing_rate(y))]

    # Spektral
    features += [
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    ]

    # Tambahan MFCC (5 koefisien utama)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5), axis=1)
    features += list(mfcc)
    return features
```

### 📋 e. Penyusunan Dataset Fitur

Setiap file menghasilkan satu baris data berisi nilai-nilai fitur dan label kelasnya (“buka” atau “tutup”).
Contoh tabel hasil akhir:

| file          | mean  | std  | skew  | kurtosis | rms  | zcr   | mfcc1 | mfcc2 | ... | label |
| ------------- | ----- | ---- | ----- | -------- | ---- | ----- | ----- | ----- | --- | ----- |
| buka_001.wav  | 0.014 | 0.29 | -0.45 | 2.13     | 0.22 | 0.041 | -12.3 | 4.7   | ... | buka  |
| tutup_087.wav | 0.008 | 0.25 | 0.33  | 1.97     | 0.19 | 0.038 | -15.2 | 3.9   | ... | tutup |

Hasil ekstraksi seluruh file disimpan dalam satu dataset dengan format .csv:
fitur_audio_36.csv

## 4.3 Hasil Akhir

Tahap Data Preparation menghasilkan dataset audio yang telah melalui proses:

✅ Penyeragaman format audio antar dataset (konversi AAC → WAV bila perlu)
✅ Normalisasi dan pembersihan sinyal (noise, trimming, padding)
✅ Ekstraksi 36 fitur audio (statistik, spektral, temporal)
✅ Pelabelan dan penyimpanan hasil ke file .csv
