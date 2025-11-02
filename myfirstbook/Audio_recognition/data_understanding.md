# 🧠 2. Data Understanding

## 2.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini terdiri dari dua sumber data audio utama, yaitu:

Dataset 48k
path:
Uranus\myfirstbook\Audio_recognition\dataset48k

Dataset ini merupakan kumpulan file audio dengan sampling rate 48 kHz yang digunakan untuk melatih dan menguji model pengenalan suara. Struktur foldernya dibagi menjadi dua subset utama:

Train → digunakan untuk proses pelatihan model

Val → digunakan untuk proses validasi

Masing-masing subset memiliki dua kelas, yaitu:

* buka → berisi file suara “buka”

Contoh path:

...\dataset48k\train\buka\buka48k-buka_0.wav

Terdiri dari 150 file (buka_0.wav – buka_149.wav)

* tutup → berisi file suara “tutup”

Contoh path:

...\dataset48k\train\tutup\tutup48k-tutup_0.wav

Terdiri dari 150 file (tutup_0.wav – tutup_149.wav)

Sementara folder validation (val) masing-masing berisi 50 file (nomor 150–199) untuk kedua kelas tersebut.

| Subset | Kelas | Jumlah Sampel | Rentang Nama File                               |
| ------ | ----- | ------------- | ----------------------------------------------- |
| train  | buka  | 150           | buka48k-buka_0.wav – buka48k-buka_149.wav       |
| train  | tutup | 150           | tutup48k-tutup_0.wav – tutup48k-tutup_149.wav   |
| val    | buka  | 50            | buka48k-buka_150.wav – buka48k-buka_199.wav     |
| val    | tutup | 50            | tutup48k-tutup_150.wav – tutup48k-tutup_199.wav |

## 2.2. Deskripsi Dataset

Format file: .wav

Sampling rate: 16.000 Hz

Durasi per audio: ±1 detik

Jumlah kelas: 2 kelas (buka, tutup)

Ukuran data: ±50–100 sampel per kelas (dapat ditambah untuk keseimbangan)

## 2.3. Eksplorasi Awal 


Setiap file audio direkam dengan sampling rate tetap.

Data bersih, namun beberapa file memiliki noise (suara latar).

Variasi suara antar individu cukup tinggi → perlu normalisasi.

Distribusi jumlah data per kelas relatif seimbang (jika tidak, akan dilakukan balancing).

## 2.4. Rencana Analisis Fitur

Ciri-ciri (fitur) yang akan diekstrak dari data audio berbasis time series statistik, meliputi:

Mean amplitude (rata-rata energi)

Standard deviation (keragaman sinyal)

Zero Crossing Rate (ZCR)

Root Mean Square Energy (RMS)

Spectral Centroid

Spectral Bandwidth

Spectral Roll-off

Fitur-fitur ini nantinya akan digunakan sebagai input ke model klasifikasi (misalnya KNN, SVM, atau Random Forest).

## 2.5. Insight Awal

Berdasarkan visualisasi (akan dibuat di tahap berikut):

Suara “buka” cenderung memiliki pola energi yang lebih panjang dan naik perlahan.

Suara “tutup” cenderung memiliki puncak energi yang tajam di awal lalu turun cepat.

Fitur RMS dan Spectral Centroid menunjukkan perbedaan pola yang bisa dimanfaatkan model untuk klasifikasi.