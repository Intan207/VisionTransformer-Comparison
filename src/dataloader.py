# src/dataloader.py — baca kfold.json → PyTorch DataLoader (cross-platform, simple)
import os, json, warnings
from typing import Dict, List, Tuple, Generator, Any
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, UnidentifiedImageError

# Root repo (…/tugas-akhir)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../tugas-akhir/src
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..")) # .../tugas-akhir
DATASET_ROOT = os.path.join(ROOT_DIR, "dataset")         # .../tugas-akhir/dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fix_path(path: str) -> str:
    """
    Normalisasi path gambar supaya sederhana:
    - Kalau path ABSOLUTE (mulai dari '/' atau '/content/...' atau sejenisnya) → pakai apa adanya
    - Kalau path RELATIF (misal: 'dataset/ODIR/NORMAL/xxx.jpg') → gabung dengan ROOT_DIR
    """
    if not path:
        raise ValueError("Empty path in kfold.json")

    # Samakan jadi '/' supaya tidak pusing backslash
    p = path.replace("\\", "/")

    # 1) Kalau sudah absolute (contoh: /content/DATASET/ODIR/...), jangan diubah-ubah lagi
    if os.path.isabs(p):
        return os.path.normpath(p)

    # 2) Kalau relatif (misal: 'dataset/ODIR/NORMAL/xxx.jpg'), anggap relatif ke ROOT_DIR
    full = os.path.join(ROOT_DIR, p)
    return os.path.normpath(full)


class SimpleFundus(Dataset):
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        class_to_idx: Dict[str, int],
        img_size: int = 224,
        augment: bool = False,
    ):
        self.entries = entries
        self.class_to_idx = class_to_idx

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
        self.tf = T.Compose(aug + base)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        raw_path = e["path"]
        label = e["label"]

        path = fix_path(raw_path)

        if not os.path.exists(path):
            warnings.warn(f"[dataloader] file not found after fix_path: {path}")
            raise FileNotFoundError(path)

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as err:
            warnings.warn(f"[dataloader] cannot read image: {path} ({err})")
            raise

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

    set_seed(seed)

    # kfold.json boleh relatif terhadap ROOT_DIR
    if not os.path.isabs(json_path):
        json_path = os.path.join(ROOT_DIR, json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    pin = torch.cuda.is_available()
    persistent = (num_workers > 0)

    for fd in data["folds"]]:
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
    demo_json = os.path.join("results", "kfold.json")
    dl_tr, dl_va, classes = next(
        load_folds(demo_json, batch_size=8, num_workers=0, img_size=224)
    )
    xb, yb = next(iter(dl_tr))
    print("ROOT_DIR:", ROOT_DIR)
    print("Classes:", classes, "| Batch:", xb.shape, yb.shape)
