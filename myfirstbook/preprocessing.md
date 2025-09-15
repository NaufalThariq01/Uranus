## Data Preprocessing

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
#Menghapus Outlier

menggunakan 3 model:
## 1. MODEL ABOD

```python
# 3. Buat model Isolation Forest
abod = create_model('abod', fraction=0.1) 
results = assign_model(abod)

# menghapus data outlier
cleaned_data = results[results["Anomaly"] == 0].copy()

print("Data asli: ", results.shape)
print("Data setelah hapus outlier: ", cleaned_data.shape)

# buat file csv baru
cleaned_data.to_csv("iris_ABOD_cleaned.csv", index=False)
print("Dataset berhasil disimpan")
# Visualisasi perbandingan sebelum & sesudah preprocessing
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Sebelum pembersihan (ada outlier)
axes[0].scatter(
    results['sepal_length'], results['sepal_width'],
    c=results['Anomaly'], cmap='coolwarm', edgecolor='k'
)
axes[0].set_title("Sebelum Hapus Outlier (ABOD)")
axes[0].set_xlabel("Sepal Length")
axes[0].set_ylabel("Sepal Width")

# Sesudah pembersihan (hanya normal data)
axes[1].scatter(
    cleaned_data['sepal_length'], cleaned_data['sepal_width'],
    c='blue', edgecolor='k'
)
axes[1].set_title("Sesudah Hapus Outlier (ABOD)")
axes[1].set_xlabel("Sepal Length")
axes[1].set_ylabel("Sepal Width")

plt.tight_layout()
plt.show()
```
## Visualisasi 

## 2. MODEL IFOREST

```python
# 3. Buat model Isolation Forest
iforest = create_model('iforest', fraction=0.1) 
results = assign_model(iforest)

# menghapus data outlier
cleaned_data = results[results["Anomaly"] == 0].copy()

print("Data asli: ", results.shape)
print("Data setelah hapus outlier: ", cleaned_data.shape)

# buat file csv baru
cleaned_data.to_csv("iris_IFOREST_cleaned.csv", index=False)
print("Dataset berhasil disimpan")
# Visualisasi perbandingan sebelum & sesudah preprocessing
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Sebelum pembersihan (ada outlier)
axes[0].scatter(
    results['sepal_length'], results['sepal_width'],
    c=results['Anomaly'], cmap='coolwarm', edgecolor='k'
)
axes[0].set_title("Sebelum Hapus Outlier (IFOREST)")
axes[0].set_xlabel("Sepal Length")
axes[0].set_ylabel("Sepal Width")

# Sesudah pembersihan (hanya normal data)
axes[1].scatter(
    cleaned_data['sepal_length'], cleaned_data['sepal_width'],
    c='blue', edgecolor='k'
)
axes[1].set_title("Sesudah Hapus Outlier (IFOREST)")
axes[1].set_xlabel("Sepal Length")
axes[1].set_ylabel("Sepal Width")

plt.tight_layout()
plt.show()
```
## 3. MODEL SOD

```python
# 3. Buat model Isolation Forest
sod = create_model('sod', fraction=0.1) 
results = assign_model(sod)

# menghapus data outlier
cleaned_data = results[results["Anomaly"] == 0].copy()

print("Data asli: ", results.shape)
print("Data setelah hapus outlier: ", cleaned_data.shape)

# buat file csv baru
cleaned_data.to_csv("iris_SOD_cleaned.csv", index=False)
print("Dataset berhasil disimpan")
# Visualisasi perbandingan sebelum & sesudah preprocessing
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Sebelum pembersihan (ada outlier)
axes[0].scatter(
    results['sepal_length'], results['sepal_width'],
    c=results['Anomaly'], cmap='coolwarm', edgecolor='k'
)
axes[0].set_title("Sebelum Hapus Outlier (SOD)")
axes[0].set_xlabel("Sepal Length")
axes[0].set_ylabel("Sepal Width")

# Sesudah pembersihan (hanya normal data)
axes[1].scatter(
    cleaned_data['sepal_length'], cleaned_data['sepal_width'],
    c='blue', edgecolor='k'
)
axes[1].set_title("Sesudah Hapus Outlier (SOD)")
axes[1].set_xlabel("Sepal Length")
axes[1].set_ylabel("Sepal Width")

plt.tight_layout()
plt.show()
```
