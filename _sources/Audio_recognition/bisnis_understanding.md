# 🧩 1. Business Understanding
## 1.1 Latar Belakang

Dalam era digital dan otomasi, sistem yang mampu mengenali suara secara otomatis memiliki peran penting dalam berbagai bidang seperti smart home, voice command system, hingga safety monitoring. Salah satu penerapan sederhana namun bermanfaat adalah identifikasi suara “buka” dan “tutup”, yang dapat digunakan untuk mendeteksi aktivitas fisik seperti membuka atau menutup pintu, kotak, atau wadah tertentu.

Pengenalan suara umumnya dilakukan menggunakan teknik analisis frekuensi seperti spectrogram atau MFCC. Namun, pendekatan tersebut memerlukan komputasi yang relatif tinggi. Oleh karena itu, penelitian ini mencoba menggunakan pendekatan yang lebih ringan, yaitu analisis fitur statistik dari sinyal time series audio, untuk membedakan antara dua jenis suara: “buka” dan “tutup”.

Dengan pendekatan ini, sistem diharapkan tetap dapat mengenali pola suara secara efisien tanpa memerlukan transformasi domain frekuensi.

## 1.2 Tujuan Penelitian

Tujuan utama penelitian ini adalah:

Mengidentifikasi dan mengklasifikasikan suara “buka” dan “tutup” berdasarkan karakteristik statistik dari sinyal audio.

Menguji efektivitas fitur statistik seperti mean, standard deviation, skewness, kurtosis, RMS, dan zero crossing rate dalam membedakan dua kategori suara.

Membangun model klasifikasi berbasis machine learning (seperti K-Nearest Neighbors, Random Forest, atau Naive Bayes) untuk mengenali pola dari fitur tersebut.

## 1.3 Rumusan Masalah

Berdasarkan latar belakang di atas, rumusan masalah yang dapat diambil adalah:

Bagaimana cara mengekstraksi fitur statistik dari sinyal audio “buka” dan “tutup” secara efektif?

Fitur statistik apa yang paling berpengaruh dalam membedakan suara “buka” dan “tutup”?

Algoritma klasifikasi apa yang paling akurat untuk mengenali pola tersebut?

## 1.4 Batasan Masalah

Data suara yang digunakan berasal dari dataset online di Kaggle yang berisi berbagai jenis suara “buka” dan “tutup”.

Ciri yang digunakan hanya fitur statistik dari sinyal time series audio, tanpa transformasi ke domain frekuensi (seperti FFT atau MFCC).

Model yang diuji terbatas pada algoritma machine learning dasar (KNN, Random Forest, Naive Bayes).

Fokus penelitian adalah pada proses identifikasi dua kelas: “buka” dan “tutup”, bukan pengenalan suara secara umum.

## 1.5 Manfaat Penelitian

Memberikan alternatif metode pengenalan suara yang ringan dan efisien, tanpa perlu transformasi spektral yang kompleks.

Menjadi dasar pengembangan sistem monitoring suara otomatis, misalnya deteksi aktivitas pintu atau wadah.

Memberikan kontribusi dalam pengembangan aplikasi berbasis audio recognition menggunakan fitur statistik sederhana.

## 1.6 Sumber Dataset

Dataset audio yang digunakan diperoleh secara online dari Kaggle, yang judulnya "Audio Recognition: "Buka" and "Tutup" ",link: https://www.kaggle.com/datasets/muhammadridhoisdi/audio-recognition-buka-and-tutup?resource=download.  
Dengan dua folder yaitu train dan test(val) ,train berisi 150 sample suara buka dan 150 sample suara tutup sedangkan test(val) berisi 50 sampe suara buka dan 50 sample suara tutup