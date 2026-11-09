# 💳 Credit Risk Assessment Engine (TA-03)

Aplikasi Web end-to-end berbasis **Regresi Logistik** untuk memprediksi kelayakan kredit dan risiko gagal bayar (*default*) nasabah kartu kredit berdasarkan data demografis serta riwayat transaksi keuangan.

Proyek ini menggunakan dataset **`UCI_Credit_Card.csv`** (30.000 data nasabah) dan dibangun menggunakan kerangka kerja **Flask**, **Scikit-Learn**, dan **Tailwind CSS**.

---

## 👨‍💻 Identitas Mahasiswa

* **Nama:** Ari Mauludin
* **NIM:** 301240044
* **Kelas:** 5C (Teknik Informatika)
* **Mata Praktikum:** Pemodelan dan Simulasi
* **Program Studi:** S1 Teknik Informatika
* **Instansi:** Universitas Bale Bandung (UNIBBA)

---

## 📊 Ringkasan Evaluasi Model

Model melatih 30.000 data nasabah dengan pembagian rasio 80:20 (Stratified Train-Test Split) serta menerapkan strategi penyeimbangan kelas (`class_weight='balanced'`).

| Metrik Evaluasi | Baseline Model | Balanced Model (`class_weight='balanced'`) |
| :--- | :---: | :---: |
| **True Positive (TP)** | 318 | **823** |
| **False Negative (FN)** | 1.009 | **504** |
| **Recall (Gagal Bayar)** | 23.96% | **62.02%** |
| **F1-Score** | 0.3553 | **0.4613** |
| **ROC-AUC Score** | 0.7076 | **0.7081** |

> **Metrik Vital (Recall):** Dalam manajemen risiko kredit, meminimalkan *False Negative* (memprediksi nasabah berisiko sebagai nasabah baik) adalah prioritas utama untuk mencegah kerugian langsung modal bank. Model *Balanced* berhasil meningkatkan *Recall* hingga **62.02%**.

---

## 🛠️ Teknologi & Library

* **Bahasa Pemrograman:** Python 3.11
* **Machine Learning & Preprocessing:** Scikit-Learn, Pandas, NumPy, Joblib
* **Web Framework:** Flask
* **Front-End Design:** HTML5, Tailwind CSS, JavaScript (Fetch API)
* **Production Server:** Gunicorn

---

## 📁 Struktur Repositori

```text
.
├── app.py                  # Script aplikasi Flask (Web Server & API API endpoint)
├── train_model.py          # Script eksekusi training model & pembuat file .joblib
├── model_logistic.joblib   # Artefak model Regresi Logistik terlatih
├── scaler.joblib           # Object StandardScaler untuk normalisasi input
├── feature_names.joblib    # Daftar urutan kolom fitur dataset
├── UCI_Credit_Card.csv     # Dataset riwayat kredit (Kaggle / UCI)
├── requirements.txt        # Daftar dependensi library Python
├── Procfile                # Konfigurasi deployment server (Render/Heroku)
├── README.md               # Dokumentasi proyek
└── templates/
    └── index.html          # Antarmuka dashboard web frontend