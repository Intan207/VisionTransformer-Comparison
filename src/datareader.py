"""
datareader.py — scan folder dataset ODIR dan bikin file results/kfold.json
Dipakai SEKALI SAJA untuk membagi dataset jadi K-Fold.
Setelah itu training cukup pakai kfold.json yang sudah jadi.
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# ====== KONFIGURASI DASAR (SESUAIKAN DENGAN STRUKTUR FOLDER KAMU) ======

# Lokasi dataset ODIR di lokal (relative dari root repo)
# Struktur yang diharapkan:
#   dataset/
#       ODIR/
#           CATARACT/
#           DR/
#           GLAUCOMA/
#           NORMAL/
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset" / "ODIR"

# Nama-nama kelas (harus sama dengan nama folder)
CLASSES: List[str] = ["CATARACT", "DR", "GLAUCOMA", "NORMAL"]

# Jumlah fold
K_FOLDS: int = 5

# Ekstensi gambar yang diterima
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ====== FUNGSI PEMBANTU ======

def scan_items(root: Path, classes: List[str]) -> List[Dict[str, Any]]:
    """
    Men-scan folder dataset dan mengembalikan list:
    [
      {"path": "C:/.../dataset/ODIR/CATARACT/xxx.jpg", "label": "CATARACT"},
      ...
    ]
    """
    items: List[Dict[str, Any]] = []
    root = Path(root)

    for cls in classes:
        p = root / cls
        if not p.exists():
            raise FileNotFoundError(f"Folder kelas '{cls}' tidak ditemukan di {root}")

        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                items.append({"path": str(f.resolve()), "label": cls})

    if not items:
        raise RuntimeError(f"Tidak ada gambar ditemukan di {root}")

    return items


def make_stratified_kfold(root: Path, classes: List[str], k: int = 5):
    """
    Membagi dataset jadi k fold secara stratified:
    - Tiap fold punya distribusi kelas yang mirip
    - Untuk tiap fold:
        - ada list 'train'
        - ada list 'val'
    """
    random.seed(42)

    # Kumpulkan item per kelas
    by_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in classes}
    for it in scan_items(root, classes):
        by_class[it["label"]].append(it)

    # Acak urutan dalam tiap kelas
    for c in classes:
        random.shuffle(by_class[c])

    # Siapkan struktur fold
    folds = [{"train": [], "val": []} for _ in range(k)]

    # Bagi per kelas
    for c in classes:
        arr = by_class[c]
        n = len(arr)

        # Hitung ukuran tiap bagian untuk kelas ini
        sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]

        idx = 0
        parts: List[List[Dict[str, Any]]] = []
        for s in sizes:
            parts.append(arr[idx:idx + s])
            idx += s

        # Untuk setiap fold i:
        #   - val: part ke-i
        #   - train: gabungan part lain
        for i in range(k):
            val = parts[i]
            train = [x for j, part in enumerate(parts) if j != i for x in part]
            folds[i]["val"].extend(val)
            folds[i]["train"].extend(train)

    return folds


# ====== MAIN SCRIPT: BIKIN K-FOLD DAN SIMPAN KE JSON ======

def main():
    print("=== Generate K-Fold JSON ===")
    print(f"Repo root     : {REPO_ROOT}")
    print(f"DATASET_ROOT  : {DATASET_ROOT}")
    print(f"Kelas         : {CLASSES}")
    print(f"Jumlah fold   : {K_FOLDS}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {DATASET_ROOT}")

    # Buat folder results/ kalau belum ada
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Bangun folds
    folds = make_stratified_kfold(DATASET_ROOT, CLASSES, k=K_FOLDS)

    # Simpan ke kfold.json di results/
    kjson_path = results_dir / "kfold.json"
    payload = {
        "classes": CLASSES,
        "dataset_root": str(DATASET_ROOT.resolve()),
        "folds": folds,
    }

    with kjson_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved kfold.json ke: {kjson_path.resolve()}")

    # Ringkasan distribusi per fold
    from collections import Counter
    print("\nRingkasan per fold:")
    for i, fd in enumerate(folds, 1):
        c_tr = Counter([e["label"] for e in fd["train"]])
        c_va = Counter([e["label"] for e in fd["val"]])
        print(f"Fold {i}:")
        print(f"  train = {dict(c_tr)}")
        print(f"  val   = {dict(c_va)}")


if __name__ == "__main__":
    main()
