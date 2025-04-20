# Sistem Pengenalan Wajah dan Deteksi Suku

Aplikasi ini mengimplementasikan dua fitur utama:
1. **Face Similarity (Kemiripan Wajah)**: Mengidentifikasi dan membandingkan wajah untuk menentukan apakah dua gambar wajah yang berbeda berasal dari orang yang sama.
2. **Deteksi Suku/Etnis**: Mengklasifikasikan wajah seseorang ke dalam kategori suku/etnis berdasarkan fitur wajah.

## Persyaratan Sistem

- Python 3.8+
- Webcam (untuk pengambilan gambar)
- GPU (opsional, untuk mempercepat proses training)

## Instalasi

1. Clone repository ini:
```
git clone <repository-url>
cd <repository-directory>
```

2. Buat virtual environment (opsional tapi direkomendasikan):
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Penggunaan

1. Jalankan aplikasi Streamlit:
```
streamlit run app.py
```

2. Buka browser dan akses URL yang ditampilkan (biasanya http://localhost:8501)

3. Ikuti langkah-langkah berikut:
   - Tambahkan wajah ke dataset menggunakan fitur "Tambah Wajah"
   - Gunakan fitur "Face Similarity" untuk membandingkan wajah
   - Gunakan fitur "Deteksi Suku" untuk mengklasifikasikan suku/etnis

## Struktur Dataset

Dataset disimpan dalam struktur folder berikut:
```
dataset/
├── person_name/
│   ├── ethnicity/
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   └── ...
│   └── ...
└── metadata.csv
```

File `metadata.csv` berisi informasi tentang setiap gambar:
```
path_gambar,nama,suku,ekspresi,sudut,pencahayaan
dataset/person_name/ethnicity/img_0.jpg,person_name,ethnicity,normal,frontal,normal
```

## Ketentuan Dataset

- Minimal 15 orang berbeda
- Minimal 4 gambar per orang
- Variasi gambar: ekspresi, sudut, pencahayaan, jarak
- Minimal 3 suku/etnis berbeda
- Resolusi minimal 224×224 piksel
- Format file JPEG atau PNG 