# src/dataloader.py — baca kfold.json → PyTorch DataLoader (robust + cross-platform)
import os, json, warnings
from typing import Dict, List, Tuple, Generator, Any
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, UnidentifiedImageError

# Root repo (…/tugas-akhir), supaya path tidak tergantung OS / user
THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../tugas-akhir/src
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))        # .../tugas-akhir
DATASET_ROOT = os.path.join(ROOT_DIR, "dataset")                # .../tugas-akhir/dataset

# Mean & std bawaan ImageNet—umum dipakai buat model pretrained (ResNet/ConvNeXt)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    """Samakan semua 'acak' supaya eksperimen bisa diulang."""
    random.seed(seed)                    # random di Python
    torch.manual_seed(seed)              # random di PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)     # random di GPU

    # supaya beberapa operasi lebih deterministik
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fix_path(path: str) -> str:
    """
    Normalisasi path gambar supaya:
    - Path absolute yang mengandung 'dataset/...' (misal hasil kfold dari Windows / Linux)
      diarahkan ke ROOT_DIR/dataset/...
    - Path relatif 'dataset/ODIR/...' juga diarahkan ke ROOT_DIR/dataset/ODIR/...
    """
    if not path:
        raise ValueError("Empty path in kfold.json")

    # ganti backslash → slash dulu
    p = path.replace("\\", "/")
    low = p.lower()
    key = "dataset/"

    # kalau di path ada 'dataset/...'
    if key in low:
        # ambil bagian SETELAH 'dataset/'
        idx = low.index(key) + len(key)
        rel_after = p[idx:]  # contoh: 'ODIR/NORMAL/123.jpg'
        full = os.path.join(DATASET_ROOT, rel_after)
        return os.path.normpath(full)

    # kalau path diawali 'dataset/...'
    if low.startswith(key):
        rel_after = p[len(key):]  # buang 'dataset/'
        full = os.path.join(DATASET_ROOT, rel_after)
        return os.path.normpath(full)

    # kalau path relatif biasa → anggap relatif ke ROOT_DIR
    if not os.path.isabs(path):
        full = os.path.join(ROOT_DIR, p)
        return os.path.normpath(full)

    # kalau sudah absolut dan tidak mengandung 'dataset/' → biarkan saja
    return os.path.normpath(path)


class SimpleFundus(Dataset):
    """
    Dataset sederhana untuk gambar fundus.
    entries: list dict { "path": "...", "label": "NORMAL"/"DR"/... }
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        class_to_idx: Dict[str, int],
        img_size: int = 224,
        augment: bool = False,
    ):
        self.entries = entries               # daftar semua gambar + label
        self.class_to_idx = class_to_idx     # mapping label teks -> index angka

        # augmentasi untuk training
        aug = []
        if augment:
            aug = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)],
                    p=0.3,
                ),
                T.RandomRotation(degrees=10),
            ]

        base = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        # pipeline transform lengkap
        self.tf = T.Compose(aug + base)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        raw_path = e["path"]
        label = e["label"]

        # perbaiki path agar cross-platform
        path = fix_path(raw_path)

        # cek file fisik ada
        if not os.path.exists(path):
            warnings.warn(f"[dataloader] file not found after fix_path: {path}")
            raise FileNotFoundError(path)

        # buka gambar, pastikan RGB
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as err:
            warnings.warn(f"[dataloader] cannot read image: {path} ({err})")
            raise

        # label teks -> index angka
        y = self.class_to_idx[label]
        return self.tf(img), torch.tensor(y, dtype=torch.long)


def load_folds(
    json_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = 224,
    seed: int = 42,
    use_aug_on_train: bool = True,
) -> Generator[Tuple[DataLoader, DataLoader, List[str]], None, None]:
    """
    Membaca kfold.json dan mengembalikan generator:
    yield dl_tr, dl_va, classes  untuk tiap fold.
    """

    set_seed(seed)

    # Pastikan path kfold.json relatif ke ROOT_DIR kalau dikasih relatif
    if not os.path.isabs(json_path):
        json_path = os.path.join(ROOT_DIR, json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # optimisasi untuk GPU
    pin = torch.cuda.is_available()
    persistent = (num_workers > 0)

    for fd in data["folds"]:

        ds_tr = SimpleFundus(
            fd["train"],
            class_to_idx,
            img_size=img_size,
            augment=use_aug_on_train,
        )
        ds_va = SimpleFundus(
            fd["val"],
            class_to_idx,
            img_size=img_size,
            augment=False,
        )

        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=(2 if persistent else None),
        )

        dl_va = DataLoader(
            ds_va,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=(2 if persistent else None),
        )

        yield dl_tr, dl_va, classes


if __name__ == "__main__":
    # Demo kecil: test ambil 1 batch (optional, tidak dipakai train.py)
    demo_json = os.path.join("results", "kfold.json")
    dl_tr, dl_va, classes = next(
        load_folds(demo_json, batch_size=8, num_workers=0, img_size=224)
    )
    xb, yb = next(iter(dl_tr))
    print("ROOT_DIR:", ROOT_DIR)
    print("Classes:", classes, "| Batch:", xb.shape, yb.shape)
