<img width="585" height="312" alt="{1D2E9530-4196-4750-BC09-5AB60F7B4C93}" src="https://github.com/user-attachments/assets/608cfe9d-f304-4e83-b478-11a373fac6fa" /># Eksplorasi Data

Setelah memahami struktur dataset, langkah berikutnya adalah melakukan eksplorasi data untuk mendapatkan wawasan lebih dalam mengenai pola dan karakteristik data.

---

## Statistik Deskriptif

 ![Tabel Iris di PostgreSQL yang menampilkan deteksi outlier](../_build/html/_static/images/statistik_deskriptif_postgre.png)


## Deteksi Outlier

 ![Tabel Iris di PostgreSQL yang menampilkan deteksi outlier](../_build/html/_static/images/deteksi_outlier_postgre.png)

Untuk mendeteksi outlier, dibuat scatter chart dengan kombinasi variabel numerikal:

1. Sepal length vs sepal width  
   - Titik data dari ketiga spesies menyebar rapat.  
   - Tidak ditemukan nilai yang menyendiri jauh dari kelompok → tidak ada outlier ekstrim.  

2. Petal length vs petal width  
   - Terlihat bahwa Iris-setosa membentuk kluster terpisah dengan ukuran kelopak lebih kecil.  
   - Hal ini bukan outlier kesalahan data, melainkan perbedaan alami antar spesies.  

Dengan demikian, tidak ada data yang perlu dihapus karena outlier pada dataset ini merupakan bagian dari variasi kelas.

## Visualisasi

Eksplorasi dilakukan menggunakan **Power BI** dengan fokus pada:

- Nilai minimum dan maksimum setiap kolom  
- Nilai rata-rata setiap kolom  
- Distribusi jumlah tiap kelas (ditampilkan dalam grafik batang)  

### Hasil Visualisasi
- MySQL  
  ![Eksplorasi dengan MySQL](../_build/html/_static/images/eksplorasi_data_mysql.png)

- PostgreSQL  
  ![Eksplorasi dengan PostgreSQL](../_build/html/_static/images/eksplorasi_data_postgre.png)

---

## Insight Awal

(Tuliskan poin-poin hasil pengamatan dari eksplorasi data, misalnya perbedaan karakteristik antar spesies)

---

## Ringkasan

(Tuliskan kesimpulan dari eksplorasi data)
