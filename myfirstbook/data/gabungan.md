# Hasil Penggabungan Data dari Dua Database

Pada bagian ini ditunjukkan bagaimana kita dapat menggabungkan data dari **PostgreSQL** dan **MySQL** menggunakan Python.  
Tujuannya adalah membuat satu dataset gabungan untuk dianalisis lebih lanjut atau digunakan di Power BI.

---

## 📌 Kode Python

Di bawah ini adalah script Python untuk menggabungkan tabel dari dua database berbeda:

```python
import pandas as pd
from sqlalchemy import create_engine

# --- Koneksi ke PostgreSQL ---
pg_user = "postgres"
pg_pass = "123"
pg_host = "localhost"
pg_port = "5432"
pg_db   = "data_iris"

engine_pg = create_engine(f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}")

# Ambil kolom dari PostgreSQL
query_pg = "SELECT sepal_length, sepal_width FROM public.iris"
df_pg = pd.read_sql(query_pg, engine_pg)

# --- Koneksi ke MySQL ---
my_user = "root"
my_pass = ""   # password kosong
my_host = "localhost"
my_port = "3306"
my_db   = "data_iris"

engine_my = create_engine(f"mysql+mysqlconnector://{my_user}:{my_pass}@{my_host}:{my_port}/{my_db}")

# Ambil kolom dari MySQL
query_my = "SELECT petal_length, petal_width, species FROM iris"
df_my = pd.read_sql(query_my, engine_my)

# --- Gabungkan data secara horizontal (berdasarkan urutan baris) ---
df_merge = pd.concat([df_pg, df_my], axis=1)

# Output ke Power BI
dataset = df_merge
```
# 📊 Penjelasan

  -Dari PostgreSQL diambil 2 kolom: sepal_length dan sepal_width.

  -Dari MySQL diambil 3 kolom: petal_length, petal_width, dan species.

  -Data digabungkan secara horizontal dengan pd.concat(..., axis=1) sehingga baris tetap berurutan dan kolom saling melengkapi.

  -Hasil akhirnya berupa dataset lengkap yang terdiri dari 5 kolom.

# 📷 Tabel Setelah Digabung

ini adalah tabel data iris yang saya gabung dari dua database berbeda

![ini adalah tabel data iris yang saya gabung dari dua database berbeda](../_static/images/gabungan.png)
