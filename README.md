# Tugas Besar 2 IF3270 Pembelajaran Mesin
## Convolutional Neural Network dan Recurrent Neural Network

**Kelompok 60:**
- Eduardus Alvito Kristiadi (13522004)
- Rici Trisna Putra (13522026)  
- Dimas Bagoes Hendrianto (13522112)

---

## Deskripsi Singkat Repository

Implementasi tugas besar ini berisi ini berisi implementasi dari scratch untuk Convolutional Neural Network (CNN), Simple Recurrent Neural Network (RNN), dan Long Short-Term Memory (LSTM) menggunakan hanya library NumPy. Proyek ini mencakup pelatihan model menggunakan Keras pada dataset CIFAR-10 untuk CNN dan dataset NusaX-Sentiment untuk RNN dan LSTM, serta implementasi forward propagation yang dapat memuat bobot dari model Keras yang telah dilatih.

### Fitur Utama
- **CNN dari Scratch**: Implementasi layer Conv2D, Pooling, Dense, dan aktivasi untuk klasifikasi gambar
- **RNN dari Scratch**: Implementasi SimpleRNN unidirectional dan bidirectional untuk klasifikasi teks
- **LSTM dari Scratch**: Implementasi sel LSTM lengkap dengan gate mechanism untuk klasifikasi teks
- **Weight Loading**: Kemampuan memuat bobot dari model Keras yang telah dilatih
- **Batch Processing**: Dukungan inference dalam batch untuk efisiensi
- **Validation System**: Sistem perbandingan hasil antara implementasi custom dan Keras

---

## Setup dan Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip package manager

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd tugas-besar-2-if3270
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Siapkan data**
   - Dataset CIFAR-10 akan diunduh otomatis oleh Keras
   - Pastikan dataset NusaX-Sentiment sudah tersedia di folder `src/data/`

---

## Cara Menjalankan Program

### 1. Training Model dengan Keras

Untuk melatih semua model (CNN, RNN, dan LSTM):
```bash
python src/pretrained\ model/main_pretrained.py --model all
```

Untuk melatih model specific:
```bash
# CNN saja
python src/pretrained\ model/main_pretrained.py --model cnn

# RNN saja  
python src/pretrained\ model/main_pretrained.py --model rnn

# LSTM saja
python src/pretrained\ model/main_pretrained.py --model lstm
```

### 2. Menjalankan Custom Forward Propagation

Untuk demo dan validasi semua implementasi custom:
```bash
python src/forward\ propagation\ custom/main_custom.py --demo all --validate
```

Untuk demo model specific:
```bash
# CNN custom implementation
python src/forward\ propagation\ custom/main_custom.py --demo cnn

# RNN custom implementation
python src/forward\ propagation\ custom/main_custom.py --demo rnn

# LSTM custom implementation  
python src/forward\ propagation\ custom/main_custom.py --demo lstm
```

### 3. Testing dan Validation

Untuk menjalankan testing lengkap:
```bash
python src/forward\ propagation\ custom/test_forward_propagation.py
```

### 4. Bug Fixes (jika diperlukan)

Jika mengalami error dalam implementasi CNN:
```bash
python fixes.py
```

---

## Experiment yang Dilakukan

### CNN (CIFAR-10)
1. **Pengaruh jumlah layer konvolusi**: Variasi 2, 3, 4 layer
2. **Pengaruh banyak filter per layer**: Variasi [16,32,64], [32,64,128], [64,128,256]
3. **Pengaruh ukuran filter**: Variasi [3,3,3], [5,5,5], [3,5,7]
4. **Pengaruh jenis pooling**: Max pooling vs Average pooling

### RNN (NusaX-Sentiment)
1. **Pengaruh jumlah layer RNN**: Variasi 1, 2, 3 layer
2. **Pengaruh banyak cell per layer**: Variasi [32,16], [64,32], [128,64]
3. **Pengaruh arah RNN**: Unidirectional vs Bidirectional

### LSTM (NusaX-Sentiment)
1. **Pengaruh jumlah layer LSTM**: Variasi 1, 2, 3 layer
2. **Pengaruh banyak cell per layer**: Variasi [32,16], [64,32], [128,64]
3. **Pengaruh arah LSTM**: Unidirectional vs Bidirectional

---

## Output dan Hasil

### File Output yang Dihasilkan
- **models/**: Model yang telah dilatih dalam format .h5
- **results/**: Grafik training curves dan hasil experiment
- **custom_results/**: Hasil demo dan validasi implementasi custom
- **test_results/**: Laporan testing dan perbandingan performa

### Metrics yang Digunakan
- **Macro F1-Score**: Metrik utama untuk evaluasi semua model
- **Prediction Agreement**: Tingkat kesamaan prediksi antara Keras dan custom implementation
- **Inference Time**: Perbandingan waktu inference

---

## Implementasi Teknis

### CNN Custom Implementation
- Layer yang didukung: Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten
- Aktivasi: ReLU, Softmax
- Padding: Same dan Valid
- Batch processing untuk efisiensi

### RNN Custom Implementation  
- Layer yang didukung: Embedding, SimpleRNN, Bidirectional RNN, Dense
- Aktivasi: Tanh (RNN), Softmax (output)
- Dukungan sequence processing dengan return_sequences

### LSTM Custom Implementation
- Implementasi lengkap sel LSTM dengan 4 gate (input, forget, cell, output)
- Aktivasi: Sigmoid untuk gate, Tanh untuk cell state dan output
- Dukungan bidirectional processing
- Memory management untuk cell state dan hidden state

---

## Pembagian Tugas Anggota Kelompok

### Eduardus Alvito Kristiadi (13522004)
- Implementasi CNN forward propagation dari scratch
- Development utility functions dan helper classes
- Testing dan validation system untuk CNN
- Dokumentasi dan laporan bagian CNN

### Rici Trisna Putra (13522026)
- Implementasi LSTM forward propagation dari scratch
- Training script untuk model LSTM menggunakan Keras
- Analisis hasil experiment LSTM
- Dokumentasi dan laporan bagian LSTM

### Dimas Bagoes Hendrianto (13522112)
- Implementasi RNN forward propagation dari scratch  
- Training script untuk model RNN menggunakan Keras
- Integration testing dan main demonstration scripts
- Dokumentasi dan laporan bagian RNN

### Pembagian Tugas Kolektif
- Design arsitektur sistem dan structure repository
- Integration testing dan debugging
- Analisis perbandingan hasil antar model
- Finalisasi laporan dan dokumentasi
