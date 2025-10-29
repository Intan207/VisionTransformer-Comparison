# dataloader.py
# K-Fold + penyeimbangan TRAIN (random undersample ke kelas minimal per fold)
# Whitelist 4 kelas: NORMAL, DR, GLAUCOMA, CATARACT
# Auto-pick dataset root (Intan / perma - Windows)

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, cast

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# ===== Kelas yang diizinkan (sesuai kebutuhan TA) =====
ALLOWED_CLASSES = {"NORMAL", "DR", "GLAUCOMA", "CATARACT"}


def pick_dataset_root() -> str:
    """
    Menentukan dataset root secara otomatis untuk lingkungan Windows lokal.
    Mengembalikan path yang pertama kali ditemukan.
    """
    candidates = [
        r"C:\Users\Intan\Documents\Tugas Akhir\DATASET",
        r"C:\Users\perma\Documents\Tugas Akhir\DATASET",
    ]
    for c in candidates:
        p = Path(c)
        print(f"[CHECK] dataset_root kandidat: {p} -> exists? {p.exists()}")
        if p.exists():
            print(f"[USE] dataset_root: {p}")
            return str(p)
    raise SystemExit(
        "[FATAL] Tidak menemukan folder DATASET. "
        "Edit list di pick_dataset_root()."
    )


class CombinedDataset(Dataset):
    """
    Dataset sederhana berbasis daftar (path gambar, label_idx).
    Aman terhadap file corrupt: jika 1 file gagal dibuka, akan maju ke file berikutnya.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        samples: list of (path, label_idx) setelah digabung & remap label
        transform: augmentasi/normalisasi torchvision
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Hindari loop tak berujung jika banyak file rusak:
        start = idx
        while True:
            path, target = self.samples[idx]
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")        # pastikan 3 channel
                break
            except Exception:
                # Jika gagal buka, lompat ke item berikutnya
                idx = (idx + 1) % len(self.samples)
                if idx == start:
                    raise RuntimeError("Semua gambar gagal dibuka.")
        return (self.transform(img) if self.transform else img), int(target)


def _undersample_to_min(
    indices: Iterable[int],
    labels_all: List[int],
    rng: np.random.RandomState,
) -> List[int]:
    """
    Menyeimbangkan indeks TRAIN menjadi jumlah minimal per kelas.
    - Kumpulkan indeks tiap kelas
    - Ambil acak tanpa replacement sebanyak min_count
    - Shuffle hasil agar acak
    """
    by_class: dict[int, List[int]] = {}
    for i in indices:
        y = labels_all[i]
        by_class.setdefault(y, []).append(i)

    min_count = min(len(v) for v in by_class.values())       # target undersample = kelas paling kecil
    balanced: List[int] = []
    for _, idxs in by_class.items():
        if len(idxs) <= min_count:
            balanced.extend(idxs)                            # sudah <= min_count -> ambil semua
        else:
            sel = rng.choice(idxs, size=min_count, replace=False)  # turunkan ke min_count
            balanced.extend(sel.tolist())

    rng.shuffle(balanced)                                    # acak urutan akhir
    return balanced


def _pretty_counts(count_dict: Counter, class_names: List[str]) -> str:
    """
    Util untuk menampilkan distribusi kelas dengan rapi.
    """
    total = sum(count_dict.values())
    parts = [f"{class_names[i]}={count_dict.get(i, 0)}" for i in range(len(class_names))]
    return f"{total} sampel -> " + ", ".join(parts)


def dataloader_eye(
    resize_size: int = 224,
    random_rotation: int = 10,
    normalize_with_imagenet: bool = True,
    dataset_path_global: str | None = None,
    random_state: int = 2025,
    batch_size: int = 16,
    k_folds: int = 5,
    dataset_used: Tuple[str, ...] = ("ODIR",),   # subfolder di DATASET_ROOT
    num_workers: int = 0,
    balance_train: bool = True,                  # ✅ undersample TRAIN ke kelas minimal per fold
) -> Iterator[Tuple[DataLoader, DataLoader, List[str]]]:
    """
    Generator yang menghasilkan (train_loader, val_loader, class_names) untuk tiap fold.

    Struktur dataset yang diharapkan:
      <DATASET_ROOT>/
        ODIR/
          NORMAL/
          DR/
          GLAUCOMA/
          CATARACT/
    """
    # --- Tentukan root dataset ---
    if dataset_path_global is None:
        dataset_path_global = pick_dataset_root()

    # --- Seed untuk reproducibility ---
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    # --- Transform (train & eval) ---
    tf_tr = [
        T.Resize((resize_size, resize_size)),    # samakan ukuran
        T.RandomHorizontalFlip(),                # augmentasi ringan
        T.RandomRotation(random_rotation),       # augmentasi ringan
        T.ToTensor(),                            # ke tensor [0..1]
    ]
    tf_ev = [
        T.Resize((resize_size, resize_size)),
        T.ToTensor(),
    ]
    if normalize_with_imagenet:
        # Normalisasi mean/std ImageNet (umum untuk pretrained)
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tf_tr.append(norm)
        tf_ev.append(norm)
    tf_tr, tf_ev = T.Compose(tf_tr), T.Compose(tf_ev)

    # --- Load subdataset yang tersedia & filter whitelist kelas ---
    root = Path(dataset_path_global).resolve()
    print(f"[INFO] DATASET_ROOT : {root} | exists? {root.exists()}")

    loaded: List[Tuple[ImageFolder, set[str]]] = []
    present_classes: set[str] = set()

    for name in dataset_used:
        sub = (root / name).resolve()
        print(f"[INFO] cek subroot : {sub} | exists? {sub.exists()}")
        if not sub.exists():
            print(f"[WARN] {sub} tidak ditemukan; dilewati.")
            continue

        ds = ImageFolder(root=str(sub))                         # baca folder per kelas
        valid_here = ALLOWED_CLASSES.intersection(set(ds.classes))  # batasi ke 4 kelas whitelist
        if not valid_here:
            print(f"[WARN] {sub} tidak berisi kelas {ALLOWED_CLASSES}; dilewati.")
            continue

        loaded.append((ds, valid_here))
        present_classes.update(valid_here)

    if not loaded:
        tried = [str((root / n).resolve()) for n in dataset_used]
        raise ValueError(
            "No valid datasets found. Pastikan struktur:\n"
            "<DATASET_ROOT>\\ODIR\\{NORMAL,DR,GLAUCOMA,CATARACT}\\*.jpg\n"
            "Folder dicek: " + ", ".join(tried)
        )

    # --- Buat mapping kelas terpadu (stabil) ---
    class_names = sorted(list(present_classes))                   # contoh: ['CATARACT','DR','GLAUCOMA','NORMAL']
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"[INFO] Classes (used): {class_names}")

    # --- Gabungkan semua sampel & remap label lokal -> label terpadu ---
    combined_samples: List[Tuple[str, int]] = []                  # [(path, new_lbl), ...]
    combined_targets: List[int] = []                              # [new_lbl, new_lbl, ...]

    for ds, valid in loaded:
        vset = set(valid)
        for p, lbl in ds.samples:
            cname = ds.classes[lbl]
            if cname in vset:                                     # hanya kelas whitelist
                new_lbl = class_to_idx[cname]                     # remap nama -> index terpadu
                combined_samples.append((p, new_lbl))
                combined_targets.append(new_lbl)

    if not combined_samples:
        raise ValueError("[FATAL] Tidak ada sampel setelah filtering kelas.")

    # --- Info distribusi global (sebelum split) ---
    global_counts = Counter(combined_targets)
    print("[GLOBAL] Distribusi sebelum split:", _pretty_counts(global_counts, class_names))

    # --- Safety: k_folds tidak boleh melampaui jumlah di kelas terkecil ---
    min_count = min(global_counts.values())
    if k_folds > min_count:
        print(f"[WARN] k_folds={k_folds} > jumlah minimal per kelas={min_count} -> k_folds diturunkan ke {min_count}")
        k_folds = max(min_count, 2)

    # --- Siapkan pembagi Stratified K-Fold ---
    rng = np.random.RandomState(random_state)
    targets_arr = np.array(combined_targets)
    all_idx = np.arange(len(combined_samples))
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # === Generator per fold ===
    for tr_idx, va_idx in skf.split(all_idx, targets_arr):
        tr_idx = tr_idx.tolist()
        va_idx = va_idx.tolist()

        # Logging distribusi BEFORE
        train_counts_before = Counter([combined_targets[i] for i in tr_idx])
        val_counts = Counter([combined_targets[i] for i in va_idx])
        print("[FOLD] Train BEFORE :", _pretty_counts(train_counts_before, class_names))
        print("[FOLD] Val          :", _pretty_counts(val_counts, class_names))

        # ✅ Undersample TRAIN ke kelas minimal per fold
        if balance_train:
            tr_idx = _undersample_to_min(tr_idx, combined_targets, rng)
            train_counts_after = Counter([combined_targets[i] for i in tr_idx])
            print("[BALANCE] Train AFTER  :", _pretty_counts(train_counts_after, class_names))

        # --- Buat dataset & dataloader ---
        ds_tr = CombinedDataset([combined_samples[i] for i in tr_idx], transform=tf_tr)
        ds_va = CombinedDataset([combined_samples[i] for i in va_idx], transform=tf_ev)

        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False if num_workers > 0 else False,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False if num_workers > 0 else False,
        )

        yield dl_tr, dl_va, class_names


# ==== Demo opsional (tidak dipakai saat diimport oleh train.py) ====
if __name__ == "__main__":
    gen = dataloader_eye(
        dataset_used=("ODIR",),
        k_folds=5,
        batch_size=16,
        num_workers=0,
        balance_train=True,   # ✅ konsisten: undersample per fold
    )
    dl_tr, dl_va, class_names = next(gen)

    # Beri tahu Pylance tipe sebenarnya agar len() tidak warning
    ds_tr = cast(CombinedDataset, dl_tr.dataset)
    ds_va = cast(CombinedDataset, dl_va.dataset)

    print(f"[OK] Train: {len(ds_tr)} | Val: {len(ds_va)} | Classes: {class_names}")
