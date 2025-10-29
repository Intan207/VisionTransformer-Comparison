# export_kfold_json.py
# Ekspor indeks K-Fold ke JSON (dengan opsi train_balanced/undersample)
import argparse, json, os
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold

# ===== Whitelist kelas (sesuai proyek saat ini) =====
ALLOWED_CLASSES = {"NORMAL", "DR", "GLAUCOMA", "CATARACT"}

def pick_dataset_root() -> str:
    candidates = [
        r"C:\Users\Intan\Documents\Tugas Akhir\DATASET",
        r"C:\Users\perma\Documents\Tugas Akhir\DATASET",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise SystemExit("Tidak menemukan DATASET root. Edit list di pick_dataset_root().")

def undersample_to_min(idxs: List[int], labels_all: List[int], rng: np.random.RandomState) -> List[int]:
    byc: Dict[int, List[int]] = {}
    for i in idxs:
        byc.setdefault(labels_all[i], []).append(i)
    min_count = min(len(v) for v in byc.values())
    out = []
    for _, v in byc.items():
        if len(v) <= min_count:
            out.extend(v)
        else:
            sel = rng.choice(v, size=min_count, replace=False).tolist()
            out.extend(sel)
    rng.shuffle(out)
    return out

def pretty_counts(idxs: List[int], labels_all: List[int], class_names: List[str]) -> str:
    c = Counter([labels_all[i] for i in idxs])
    total = sum(c.values())
    parts = [f"{class_names[k]}={c.get(k,0)}" for k in range(len(class_names))]
    return f"{total} -> " + ", ".join(parts)

def load_flat_dataset(root: Path, dataset_used: Tuple[str, ...]):
    """
    Return:
      samples: List[Tuple[path, class_index]]
      class_names: List[str] (sorted, subset of ALLOWED_CLASSES present)
    """
    present = set()
    folders = []
    for name in dataset_used:
        sub = (root / name).resolve()
        if not sub.exists():
            print(f"[WARN] {sub} tidak ada; lewati.")
            continue
        ds = ImageFolder(root=str(sub))
        valid = ALLOWED_CLASSES.intersection(set(ds.classes))
        if not valid:
            print(f"[WARN] {sub} tidak punya kelas {ALLOWED_CLASSES}; lewati.")
            continue
        folders.append((ds, valid))
        present.update(valid)

    if not folders:
        raise SystemExit("Tidak ada subdataset valid. Cek struktur ODIR/<kelas>/...")

    class_names = sorted(list(present))
    map_cls = {c: i for i, c in enumerate(class_names)}

    samples, targets = [], []
    for ds, valid in folders:
        vset = set(valid)
        for p, lbl in ds.samples:
            cname = ds.classes[lbl]
            if cname in vset:
                idx = map_cls[cname]
                samples.append((p, idx))
                targets.append(idx)

    if not samples:
        raise SystemExit("Tidak ada sampel setelah filter 4 kelas.")
    return samples, targets, class_names

def main():
    ap = argparse.ArgumentParser(description="Export K-Fold splits ke JSON.")
    ap.add_argument("--dataset-root", default=None, help="Root DATASET (berisi folder ODIR).")
    ap.add_argument("--dataset-used", nargs="+", default=["ODIR"], help="Subfolder yang digunakan (default: ODIR).")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--balance-train", action="store_true", help="Sertakan versi train_balanced (undersample ke kelas minimal).")
    ap.add_argument("--out", default="kfold_splits.json")
    args = ap.parse_args()

    root = Path(args.dataset_root or pick_dataset_root())
    print(f"[INFO] DATASET_ROOT: {root}")

    # Seed
    rng = np.random.RandomState(args.seed)

    samples, targets, class_names = load_flat_dataset(root, tuple(args.dataset_used))
    all_idx = np.arange(len(samples))
    targets_arr = np.array(targets)

    # Info global
    glob = Counter(targets)
    print("[GLOBAL]", {class_names[k]: glob.get(k, 0) for k in range(len(class_names))})

    # Safety k-folds
    min_per_class = min(glob.values())
    if args.k_folds > min_per_class:
        print(f"[WARN] k_folds={args.k_folds} > jumlah minimal per kelas={min_per_class} -> pakai {min_per_class}")
        args.k_folds = max(min_per_class, 2)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True)

    folds = []
    for fold_id, (tr, va) in enumerate(skf.split(all_idx, targets_arr)):
        tr = tr.tolist(); va = va.tolist()
        print(f"\n== FOLD {fold_id} ==")
        print("Train BEFORE :", pretty_counts(tr, targets, class_names))
        print("Val          :", pretty_counts(va, targets, class_names))

        fold_entry = {
            "train": [{"path": samples[i][0], "label_idx": int(samples[i][1]), "label": class_names[samples[i][1]]} for i in tr],
            "val":   [{"path": samples[i][0], "label_idx": int(samples[i][1]), "label": class_names[samples[i][1]]} for i in va],
            "summary": {
                "train_before": Counter([targets[i] for i in tr]),
                "val":          Counter([targets[i] for i in va]),
            },
        }

        # Optional undersampled train
        if args.balance_train:
            tr_bal = undersample_to_min(tr, targets, rng)
            print("Train AFTER  :", pretty_counts(tr_bal, targets, class_names))
            fold_entry["train_balanced"] = [
                {"path": samples[i][0], "label_idx": int(samples[i][1]), "label": class_names[samples[i][1]]}
                for i in tr_bal
            ]
            fold_entry["summary"]["train_after"] = Counter([targets[i] for i in tr_bal])

        # Convert Counter to plain dict with label names for readability
        def _pretty(counter_obj):
            return {class_names[k]: int(v) for k, v in dict(counter_obj).items()}
        fold_entry["summary"] = {
            k: _pretty(v) for k, v in fold_entry["summary"].items()
        }

        folds.append(fold_entry)

    out = {
        "classes": class_names,
        "k_folds": args.k_folds,
        "seed": args.seed,
        "dataset_root": str(root),
        "dataset_used": args.dataset_used,
        "balance_train": bool(args.balance_train),
        "folds": folds,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_path.resolve()}")

if __name__ == "__main__":
    main()
