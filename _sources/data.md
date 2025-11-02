# Data Understanding

Pada bagian ini, saya melakukan pemahaman awal terhadap dataset **Iris** yang digunakan dalam proyek ini. Dataset diambil dari Kaggle, lalu disimpan pada database **MySQL** dan **PostgreSQL**.

---

## Struktur Data

Dataset Iris yang digunakan memiliki:
- Jumlah baris: 150 (setiap baris mewakili 1 bunga iris)
- Jumlah kolom: 5 (4 atribut numerikal dan 1 atribut kategorikal).

Daftar kolom dan tipe data:

### MySQL
| Kolom        | Tipe Data SQL | Tipe Analisis |
|--------------|---------------|---------------|
| sepal_length | FLOAT         | Numerikal     |
| sepal_width  | FLOAT         | Numerikal     |
| petal_length | FLOAT         | Numerikal     |
| petal_width  | FLOAT         | Numerikal     |
| species      | VARCHAR(50)   | Kategorikal   |

![Tabel Iris di MySQL](_build/html/_static/images/data_iris_mysql.png)

---

### PostgreSQL
| Kolom        | Tipe Data SQL    | Tipe Analisis |
|--------------|-----------------|---------------|
| sepal_length | DOUBLE PRECISION| Numerikal     |
| sepal_width  | DOUBLE PRECISION| Numerikal     |
| petal_length | DOUBLE PRECISION| Numerikal     |
| petal_width  | DOUBLE PRECISION| Numerikal     |
| species      | TEXT            | Kategorikal   |

![Tabel Iris di PostgreSQL](_build/html/_static/images/data_iris_postgre.png)

---

## Kualitas Data

- **Missing Values:**
![Tabel Iris di PostgreSQL yang menampilkan missing value](_build/html/_static/images/missing_value_postgre.png)  
  - **Missing Values:** Tidak ditemukan missing values pada dataset Iris (semua kolom terisi penuh).

- **Duplicate Rows:**
  ![Tabel Iris di PostgreSQL yang menampilkan duplicates](_build/html/_static/images/duplicates_postgre.png)
 - **Duplicate Rows:** Tidak ditemukan baris duplikat pada dataset.

---

## Distribusi Awal

![Tabel Iris di PostgreSQL yang menampilkan count of species](_build/html/_static/images/eksplorasi_data_postgre.png)  

- Distribusi kategori:
  Hasil: masing-masing 50 data untuk Setosa, Versicolor, Virginica.
  Setosa (50), Versicolor (50), Virginica (50).


- Rentang nilai numerikal:  
  - sepal_length: min 4.3, max 7.9  
  - sepal_width: min 2.0, max 4.4  
  - petal_length: min 1.0, max 6.9  
  - petal_width: min 0.1, max 2.5  


---

# Deteksi Outlier pada Dataset Iris dengan PyCaret

Dokumen ini menjelaskan langkah-langkah deteksi outlier pada dataset **Iris** menggunakan **PyCaret**.  
Tiga model yang digunakan:
- ABOD (Angle-Based Outlier Detection)
- SOD (Subspace Outlier Detection)
- IForest (Isolation Forest)

---

## 1. Import Library

```python
import pandas as pd
from pycaret.anomaly import *
import matplotlib.pyplot as plt
```

## 2. Load Dataset

```python
# Baca dataset iris
data = pd.read_csv("data_iris.csv")

# Hapus kolom target (species) karena bukan fitur numerik
data = data.drop(columns=["species"])
```
## 3. Set up Pycaret

```python
# Setup anomaly detection
s = setup(data, session_id=123)
```
## 4. Model ABOD

```python
# Buat model ABOD dengan fraction 0.1 (10% data diasumsikan outlier)
abod = create_model('abod', fraction=0.1)

# Assign hasil prediksi ke dataset
results_abod = assign_model(abod)

# Tampilkan ringkasan hasil (10 baris pertama)
print("Ringkasan Hasil Outlier Detection (ABOD):")
print(results_abod[['sepal_length','sepal_width','petal_length','petal_width','Anomaly','Anomaly_Score']].head(10))

# Visualisasi scatter plot
plt.figure(figsize=(8,6))
plt.scatter(
    results_abod['sepal_length'], results_abod['sepal_width'],
    c=results_abod['Anomaly'], cmap='coolwarm', edgecolor='k', marker='o'
)
plt.title("Visualisasi Outlier dengan ABOD")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
```
## Informasi Setup

| Description             | Value                  |
|--------------------------|------------------------|
| Session id              | 123                    |
| Original data shape     | (150, 4)               |
| Transformed data shape  | (150, 4)               |
| Numeric features        | 4                      |
| Preprocess              | True                   |
| Imputation type         | simple                 |
| Numeric imputation      | mean                   |
| Categorical imputation  | mode                   |
| CPU Jobs                | -1                     |
| Use GPU                 | False                  |
| Log Experiment          | False                  |
| Experiment Name         | anomaly-default-name   |
| USI                     | e496                   |

---

## Ringkasan Hasil Outlier Detection (10 baris pertama)

| sepal_length | sepal_width | petal_length | petal_width | Anomaly | Anomaly_Score |
|--------------|-------------|--------------|-------------|---------|---------------|
| 5.8          | 2.8         | 5.1          | 2.4         | 1       | -0.385082     |
| 5.7          | 4.4         | 1.5          | 0.4         | 1       | -0.592311     |
| 7.2          | 3.6         | 6.1          | 2.5         | 1       | -0.612383     |
| 6.7          | 2.5         | 5.8          | 1.8         | 1       | -0.084224     |
| 4.9          | 2.5         | 4.5          | 1.7         | 1       | -0.050388     |
| 7.7          | 3.8         | 6.7          | 2.2         | 1       | -0.129286     |
| 7.7          | 2.6         | 6.9          | 2.3         | 1       | -0.692992     |
| 6.3          | 3.3         | 6.0          | 2.5         | 1       | -0.364475     |
| 5.2          | 2.7         | 3.9          | 1.4         | 1       | -0.885601     |
| 4.6          | 3.6         | 1.0          | 0.2         | 1       | -0.771745     |

---

## Statistik Outlier

- **Total Data**     : 150  
- **Normal Data**    : 135  
- **Outlier Data**   : 15  
- **Persentase Outlier** : 10.00%

---

## Ringkasan Jumlah Data

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Normal   | 135    | 90.0       |
| Outlier  | 15     | 10.0       |
### Visualisasi Deteksi Outlier ABOD
![Menampilkan visual deteksi Outlier](_build/html/_static/images/visual_deteksi_outlier_abod.png)  

## 5. Model SOD

```python
# Buat model SOD dengan fraction 0.1
sod = create_model('sod', fraction=0.1)

# Assign hasil prediksi ke dataset
results_sod = assign_model(sod)

# Tampilkan ringkasan hasil (10 baris pertama)
print("Ringkasan Hasil Outlier Detection (SOD):")
print(results_sod[['sepal_length','sepal_width','petal_length','petal_width','Anomaly','Anomaly_Score']].head(10))

# Visualisasi scatter plot
plt.figure(figsize=(8,6))
plt.scatter(
    results_sod['sepal_length'], results_sod['sepal_width'],
    c=results_sod['Anomaly'], cmap='coolwarm', edgecolor='k', marker='o'
)
plt.title("Visualisasi Outlier dengan SOD")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
```
## Informasi Setup

| Description             | Value                  |
|--------------------------|------------------------|
| Session id              | 123                    |
| Original data shape     | (150, 4)               |
| Transformed data shape  | (150, 4)               |
| Numeric features        | 4                      |
| Preprocess              | True                   |
| Imputation type         | simple                 |
| Numeric imputation      | mean                   |
| Categorical imputation  | mode                   |
| CPU Jobs                | -1                     |
| Use GPU                 | False                  |
| Log Experiment          | False                  |
| Experiment Name         | anomaly-default-name   |
| USI                     | bfe8                   |

---

## Ringkasan Hasil Outlier Detection (10 baris pertama)

| sepal_length | sepal_width | petal_length | petal_width | Anomaly | Anomaly_Score |
|--------------|-------------|--------------|-------------|---------|---------------|
| 7.2          | 3.0         | 5.8          | 1.6         | 1       | 0.307002      |
| 6.1          | 2.6         | 5.6          | 1.4         | 1       | 0.450000      |
| 4.5          | 2.3         | 1.3          | 0.3         | 1       | 0.444297      |
| 6.7          | 2.5         | 5.8          | 1.8         | 1       | 0.610000      |
| 4.9          | 2.5         | 4.5          | 1.7         | 1       | 0.450000      |
| 5.8          | 2.8         | 5.1          | 2.4         | 1       | 0.429011      |
| 7.1          | 3.0         | 5.9          | 2.1         | 1       | 0.285045      |
| 6.0          | 2.2         | 5.0          | 1.5         | 1       | 0.450000      |
| 7.2          | 3.2         | 6.0          | 1.8         | 1       | 0.284429      |
| 4.6          | 3.6         | 1.0          | 0.2         | 1       | 0.332340      |

---

## Statistik Outlier

- **Total Data**     : 150  
- **Normal Data**    : 135  
- **Outlier Data**   : 15  
- **Persentase Outlier** : 10.00%

---

## Ringkasan Jumlah Data

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Normal   | 135    | 90.0       |
| Outlier  | 15     | 10.0       |
### Visualisasi Deteksi Outlier SOD
![Menampilkan visual deteksi Outlier](_build/html/_static/images/visual_deteksi_outlier_sod.png)  

## 6. Model IFOREST

```python
## Buat model Isolation Forest dengan fraction 0.1
iforest = create_model('iforest', fraction=0.1)

# Assign hasil prediksi ke dataset
results_iforest = assign_model(iforest)

# Tampilkan ringkasan hasil (10 baris pertama)
print("Ringkasan Hasil Outlier Detection (IFOREST):")
print(results_iforest[['sepal_length','sepal_width','petal_length','petal_width','Anomaly','Anomaly_Score']].head(10))

# Visualisasi scatter plot
plt.figure(figsize=(8,6))
plt.scatter(
    results_iforest['sepal_length'], results_iforest['sepal_width'],
    c=results_iforest['Anomaly'], cmap='coolwarm', edgecolor='k', marker='o'
)
plt.title("Visualisasi Outlier dengan IFOREST")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
```
## Informasi Setup

| Description             | Value                  |
|--------------------------|------------------------|
| Session id              | 123                    |
| Original data shape     | (150, 4)               |
| Transformed data shape  | (150, 4)               |
| Numeric features        | 4                      |
| Preprocess              | True                   |
| Imputation type         | simple                 |
| Numeric imputation      | mean                   |
| Categorical imputation  | mode                   |
| CPU Jobs                | -1                     |
| Use GPU                 | False                  |
| Log Experiment          | False                  |
| Experiment Name         | anomaly-default-name   |
| USI                     | e496                   |

---

## Ringkasan Hasil Outlier Detection (10 baris pertama)

| sepal_length | sepal_width | petal_length | petal_width | Anomaly | Anomaly_Score |
|--------------|-------------|--------------|-------------|---------|---------------|
| 5.8          | 2.8         | 5.1          | 2.4         | 1       | -0.385082     |
| 5.7          | 4.4         | 1.5          | 0.4         | 1       | -0.592311     |
| 7.2          | 3.6         | 6.1          | 2.5         | 1       | -0.612383     |
| 6.7          | 2.5         | 5.8          | 1.8         | 1       | -0.084224     |
| 4.9          | 2.5         | 4.5          | 1.7         | 1       | -0.050388     |
| 7.7          | 3.8         | 6.7          | 2.2         | 1       | -0.129286     |
| 7.7          | 2.6         | 6.9          | 2.3         | 1       | -0.692992     |
| 6.3          | 3.3         | 6.0          | 2.5         | 1       | -0.364475     |
| 5.2          | 2.7         | 3.9          | 1.4         | 1       | -0.885601     |
| 4.6          | 3.6         | 1.0          | 0.2         | 1       | -0.771745     |

---

## Statistik Outlier

- **Total Data**     : 150  
- **Normal Data**    : 135  
- **Outlier Data**   : 15  
- **Persentase Outlier** : 10.00%

---

## Ringkasan Jumlah Data

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Normal   | 135    | 90.0       |
| Outlier  | 15     | 10.0       |
### Visualisasi Deteksi Outlier IFOREST
![Menampilkan visual deteksi Outlier](_build/html/_static/images/visual_deteksi_outlier_iforest.png)  
