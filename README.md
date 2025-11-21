Vision Transformer Comparison

## 1. Deskripsi Proyek

##### Repositori ini berisi implementasi dan perbandingan beberapa arsitektur Vision Transformer
- vit-base
- swin-tiny
- deit-small

##### Dataset ODIR(Ocular Disease Intelligent Recognition) penyakit mata sumber dari Kaggle 
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Proyek ini mencakup:
- Training model (ViT, Swin, dsb.)
- Perhitungan metrik evaluasi lengkap:
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Confusion matrix (CSV)
- Parameter model (trainable / non-trainable)
- Inference time
- Mode training sederhana (tanpa K-Fold)
- Visualisasi metrik 

## 2. Struktur Folder

```text
VisionTransformer-Comparison/
├─ src/
│  ├─ dataloader.py
│  ├─ datareader.py
│  ├─ model.py
│  └─ train.py
├─ dataset/           # opsional, jika training pakai folder lokal
│  └─ ODIR/           # berisi CATARACT, DR, GLAUCOMA, NORMAL
├─ results/           # hasil training (csv, checkpoint, dll.)
├─ quick_mini_test/   # contoh hasil k-Fold lama (opsional)
├─ requirements.txt
└─ LOGBOOK.md

```

### 4. Cara Menjalankan di Google Colab

#### **Langkah 1 — Clone Repository**
```bash
!git clone https://github.com/Intan207/VisionTransformer-Comparison.git
%cd VisionTransformer-Comparison
```

#### **Langkah 2 — Download Dataset ODIR dari Google Drive**
link Dataset:

https://drive.google.com/file/d/1fY9y2fXl-HHKZhZ2EBEANA0zdXnd8W0p/view
Gunakan **gdown** agar tidak perlu login Google:

```bash
!pip install gdown
!gdown 1fY9y2fXl-HHKZhZ2EBEANA0zdXnd8W0p
```

Setelah file berhasil diunduh, lakukan unzip agar folder dataset tersedia:

```bash
!unzip ODIR.zip -d dataset/
```

Struktur akhirnya akan menjadi:
```
dataset/
   ODIR/
      CATARACT/
      DR/
      GLAUCOMA/
      NORMAL/
```

#### **Langkah 3 — Jalankan Training**
- Vit-base
```bash
!python -m src.train \
  --dataset-root /content/dataset/ODIR \
  --model vit-base \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 2 \
  --img-size 224 \
  --val-split 0.2 \
  --outdir results_vit
```

- Swin-tiny
```bash
!python -m src.train \
  --dataset-root /content/dataset/ODIR \
  --model swin-tiny \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 2 \
  --img-size 224 \
  --val-split 0.2 \
  --outdir results_swin
```

- Deit-small
```bash
!python -m src.train \
  --dataset-root /content/dataset/ODIR \
  --model vit-base \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 2 \
  --img-size 224 \
  --val-split 0.2 \
  --outdir results_deit
```


