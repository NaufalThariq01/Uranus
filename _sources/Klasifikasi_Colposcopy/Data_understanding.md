# Data Understanding

Data yang digunakan dalam penelitian ini berasal dari proses pemeriksaan serviks yang diberikan asam asetat. Setelah diberikan asam asetat, jaringan serviks yang abnormal akan mengalami perubahan warna menjadi putih secara bertahap, fenomena ini disebut acetowhitening. Dataset ini merepresentasikan perubahan intensitas pemutihan tersebut dalam bentuk deret waktu (time series), sehingga setiap data mencerminkan bagaimana intensitas acetowhite berubah dari waktu ke waktu.

Data ini bersumber dari : (https://github.com/KarinaGF/ColposcopyData)
paper asli: Karina Gutiérrez-Fragoso et al. Optimization of
Classification Strategies of Acetowhite Temporal Patterns towards
Improving Diagnostic Performance of Colposcopy. Computational and
Mathematical Methods in Medicine Volume 2017, Article ID 5989105,
10 pages https://doi.org/10.1155/2017/5989105

Data train dan test telah terbagi saat mengintall dataset,Data train dan test berformat ts(time series)

## Jumlah Data


Total instance: 200
Panjang deret waktu: 182
Kelas:
- Nilai 0 menunjukkan "atrofi" dan mencakup 15 kasus
- Nilai 1 menunjukkan "inflamasi" dan mencakup 24 kasus
- Nilai 2 menunjukkan "ektopi" dan mencakup 20 kasus
- Nilai 3 menunjukkan "normal" dan mencakup 48 kasus
- Nilai 4 menunjukkan "lesi intraepitel skuamosa derajat rendah (LSIL)" dan mencakup 37 kasus
- Nilai 5 mewakili "lesi intraepitel skuamosa tingkat tinggi (LSIL)" dan mencakup 56 kasus

pemisahan uji kereta default yang dibuat melalui partisi acak.

## Karakteristik Data

- Data berbentuk deret waktu (time series) yang merupakan hasil rekaman perubahan nilai intensitas acetowhite dari waktu ke waktu.
- Data hanya terdiri dari satu fitur utama, yaitu nilai intensitas acetowhite, serta label diagnosis.
- Setiap kelas diagnosis memiliki pola temporal yang berbeda
