# export_kfold_json.py — membuat kfold.json lokal (tanpa Colab)
import os, json, random
from pathlib import Path
from collections import Counter

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def scan_items(root, classes):
    items = []
    root = Path(root)
    for cls in classes:
        d = root / cls
        if not d.exists():
            raise FileNotFoundError(f"Folder kelas '{cls}' tidak ditemukan di {d}")
        for f in d.rglob("*"):
            if f.suffix.lower() in IMG_EXT:
                items.append({"path": str(f.resolve()), "label": cls})
    return items

def make_kfold(root, classes, k=5):
    all_items = scan_items(root, classes)
    by_class = {c: [] for c in classes}
    for item in all_items:
        by_class[item["label"]].append(item)

    for c in classes:
        random.shuffle(by_class[c])

    folds = [ {"train": [], "val": []} for _ in range(k) ]
    for c in classes:
        arr = by_class[c]
        n = len(arr)
        sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        idx = 0
        parts = []
        for s in sizes:
            parts.append(arr[idx:idx+s])
            idx += s

        for i in range(k):
            val = parts[i]
            train = [x for j, part in enumerate(parts) if j != i for x in part]
            folds[i]["val"].extend(val)
            folds[i]["train"].extend(train)

    return folds

if __name__ == "__main__":
    DATASET_ROOT = "dataset/ODIR"
    CLASSES = ["CATARACT", "DR", "GLAUCOMA", "NORMAL"]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    folds = make_kfold(DATASET_ROOT, CLASSES, k=5)

    out_json = results_dir / "kfold.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "classes": CLASSES,
            "dataset_root": str(Path(DATASET_ROOT).resolve()),
            "folds": folds
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ kfold.json berhasil dibuat di: {out_json}")
    print(f"Total fold: {len(folds)}")
