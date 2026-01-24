#  Supervised Learning - Sistem Prediksi Jaringan

Di project ini gue make beberapa teknologi utama untuk membangun sistem **prediksi jaringan berbasis Machine Learning**. Pemilihan stack ini bukan asal pakai, tapi karena sesuai kebutuhan training model, analisis data, dan proses eksperimen.

<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/flask/flask.png" width="90" />
</p>

<h1 align="center">Machine Learning</h1>

<p align="center">
  Web Flask untuk prediksi status jaringan <b>(normal / gangguan)</b> menggunakan model Machine Learning <b>Random Forest</b> dan <b>Naive Bayes</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-purple?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-Numerical%20Computing-blue?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-green" />
  <img src="https://img.shields.io/badge/Seaborn-Visualization-0aa6c2" />
</p>

---

## 🚀 Fitur Utama
- Input parameter jaringan:
  - bandwidth
  - latency
  - packet loss
  - uptime
- Pilih algoritma:
  - Random Forest
  - Naive Bayes
- Output:
  - status prediksi: **normal / gangguan**
  - probabilitas prediksi
- Menampilkan ringkasan metrik evaluasi model pada dashboard

---

## 🔥 Tech Stack
| Komponen | Teknologi |
|---------|-----------|
| Backend | Flask |
| Machine Learning | scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS (Tailwind optional) |

---

### <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
Python jadi bahasa utama karena paling fleksibel buat dunia data dan Machine Learning. Hampir semua library ML dan pengolahan data support Python, jadi lebih gampang untuk eksperimen dan scaling project.

### <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
Gue pakai Jupyter Notebook supaya proses development lebih enak:
- bisa jalanin kode bertahap
- gampang analisis output
- cocok buat dokumentasi eksperimen ML
- visualisasi hasil prediksi lebih jelas

Notebook juga membantu untuk tracking progress, apalagi project ini masih **beta** dan banyak trial-error.

###  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
Scikit-learn dipakai sebagai core library untuk Machine Learning karena:
- punya banyak algoritma yang stabil (baseline model)
- preprocessing lengkap
- evaluasi model (accuracy, precision, recall, dll) gampang dipakai
- cocok buat bikin model prediksi dengan workflow yang rapi

### <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
Pandas dipakai untuk:
- baca dataset (CSV/Excel)
- bersihin data (missing value, duplikat)
- transform data jadi siap training
- analisis sebelum masuk ke model

Karena project ini banyak bermain di data, jadi gua gunain pandas dan itu wajib banget.

### <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
NumPy dipakai sebagai pondasi komputasi numerik:
- operasi array/matrix cepat
- dukungan input untuk training model
- bantu hitungan matematis/statistik saat preprocessing atau evaluasi

---

# Masih Belajar 
 
Saat ini project masih dalam tahap **BETA** dan masih terus dikembangkan, jadi beberapa fitur / hasil model kemungkinan masih belum final.

> ⚠️ Status: **On Progress (Beta)**  
> 🔥 Project ini terbuka untuk diskusi dan pengembangan bareng.

---

## 🚀 Cara Jalanin Datanya
1. Clone repository:
   ```bash
   git clone https://github.com/Raffa-shi/Prediksi-Machine-Learning.git
   cd Prediksi-Machine-Learning
Install Manual di Terminal
- py pip install -r requirements.txt
- py pip install numpy pandas scikit-learn matplotlib jupyter
- jupyter notebook

---
##  🌐 buat Enviroment baru

- python -m venv venv
- Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
- .\venv\Scripts\activate
- python -m pip install --upgrade pip
- pip install -r requirements.txt
- Python: Select Interpreter (ctrl + shift + p) select interpreter local venv 

jalankan train models

- python train_models.py

Jalankan Flask
- python app.py

> Note: karena project ini masih tahap **progress**, tech stack masih bisa bertambah (misalnya Seaborn untuk visualisasi lebih instant atau library khusus untuk network/graph).
