# Credit Risk Scoring — Data Science Competition Logika UI 2026

Model prediksi probabilitas gagal bayar (*default probability*) nasabah pinjaman, dibangun dari 9 tabel data mentah (profil, transaksi, histori pembayaran, data biro kredit, dsb) menjadi 70 fitur siap-model, dengan blend dua gradient boosting model.

**Kompetisi:** Data Science Competition — Logika UI 2026
**Role:** Data Scientist / ML Engineer

---

## 📌 Problem Statement

Memprediksi kemungkinan nasabah gagal membayar pinjaman (*credit default*) berdasarkan data historis multi-sumber: profil nasabah, perilaku rekening, transaksi e-commerce, histori pembayaran pinjaman sebelumnya, dan data biro kredit (bureau). Tantangan utama: menggabungkan 9 tabel relasional secara **leakage-safe** (hanya menggunakan informasi yang tersedia *sebelum* waktu keputusan pinjaman) dan menangani ketidakseimbangan kelas (±13.6% kelas positif/default).

## 🧠 Pendekatan

### 1. Ingestion & Feature Engineering (Analytical Base Table)
Membangun ABT dari 9 tabel mentah menggunakan **Polars** (lazy evaluation untuk efisiensi memori pada data besar):

- `customer_profiles` — join 1-to-1 dengan validasi anti-duplikasi.
- `account_behavior` — agregasi rata-rata income credit, hari saldo rendah, tren saldo (hanya histori sebelum `decision_datetime`, leakage-safe).
- `ecommerce_transactions` — fitur belanja (total spend, spend 30 hari terakhir, rasio COD, diversitas kategori, frekuensi order, resensi order).
- `repayment_history` — rasio pembayaran, frekuensi telat bayar, worst DPD (*days past due*), termasuk versi berjendela waktu (**3/6/12 bulan**).
- `bureau_observations` — skor biro rata-rata/terbaru, volatilitas skor, rasio utilisasi kredit, jumlah inquiry 6 bulan terakhir.
- `previous_applications` — rasio penolakan/persetujuan pengajuan sebelumnya, jarak hari dari pengajuan terakhir.

Hasil akhir: **70 fitur** (`train_abt4.parquet` / `test_abt4.parquet`), tervalidasi agar kolom train dan test sejajar (kecuali target).

### 2. Modeling — Blend LightGBM + CatBoost
- **Cross-validation:** 5-fold Stratified K-Fold, diulang pada **3 random seed** berbeda, rata-rata prediksi test antar seed untuk stabilitas.
- **Model 1 — LightGBM:** custom evaluation metric **KS Statistic** (Kolmogorov-Smirnov) via `feval`, regularisasi manual (`lambda_l1`, `lambda_l2`, `min_data_in_leaf`).
- **Model 2 — CatBoost:** native categorical feature handling, early stopping berbasis AUC.
- **Blending:** weighted average — 30% LightGBM + 70% CatBoost.

### 3. Evaluasi
- Metrik utama: **KS Statistic** — mengukur separasi maksimum antara distribusi kumulatif probabilitas kelas "lancar" vs "macet".
- Visualisasi: histogram distribusi probabilitas per kelas + kurva CDF dengan titik potong KS.
- Feature importance (Information Gain) dari LightGBM untuk interpretasi model.

## 📊 Hasil

- Out-of-fold KS Score ≈ **0.48–0.49** (konsisten antar fold).
- 70 fitur final tervalidasi tanpa data leakage (semua fitur time-aware terhadap `decision_datetime`).

## 🛠️ Tech Stack

`Python` `Polars` `LightGBM` `CatBoost` `Scikit-learn` `Matplotlib` `Kaggle`

## 📁 Struktur Repo

```
logika-ui-credit-risk-scoring/
├── notebooks/
│   └── DSC0135_LearningByDoing_Notebook.ipynb   # ingestion, feature engineering, modeling, submission
└── README.md
```

> **Catatan:** dataset mentah adalah milik penyelenggara kompetisi (Logika UI) dan tidak diunggah ulang di repo ini. Lihat halaman kompetisi untuk mengunduh dataset asli.

## 📸 Visualisasi

> Tambahkan screenshot grafik distribusi KS Statistic dan feature importance dari notebook.

---

*Dikerjakan sebagai bagian dari tim "LearningByDoing" pada Data Science Competition Logika UI 2026.*
