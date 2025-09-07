# Eksplorasi Data

Saya menggunakan data iris yang saya ambil dari kaggle untuk saya upload ke database lokal di postgree dan mysql

struktur dari table nya adalah sebagai berikut:

MYSQL:

CREATE TABLE iris (
    sepal_length FLOAT,
    sepal_width FLOAT,
    petal_length FLOAT,
    petal_width FLOAT,
    species VARCHAR(50)
);

POSTGRE:

CREATE TABLE iris (
    sepal_length double precision,
    sepal_width double precision,
    petal_length double precision,
    petal_width double precision,
    species varchar(50)
);

![ini adalah tabel data iris yang saya simpan di myqsl lokal](/_static/images/data_iris_mysql.png)

![ini adalah tabel data iris yang saya simpan di postgre lokal](/_static/images/data_iris_postgre.png)

Lalu saya melakukan eksplorasi data menggunakan POWER BI dimana saya mencari 
- min max dari setiap kolom
- rata rata dari setiap kolom
- jumlah setiap kelas ditampilkan dalam grafik batang

![ini adalah hasil eksplorasi data di power BI menggunakan database mysql](/_static/images/eksplorasi_data_mysql.png)

![ini adalah hasil eksplorasi data di power BI menggunakan database postgre](/_static/images/eksplorasi_data_postgre.png)
