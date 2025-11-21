# train.py — Training dan evaluasi model (tanpa K-Fold)

import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.dataloader import SimpleFundus, set_seed
from src.model import build_model


def count_params(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    model_size_mb = total * 4 / (1024**2)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "model_size_mb": model_size_mb,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    losses: List[float] = []
    preds: List[int] = []
    gts: List[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        losses.append(float(loss.item()))
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(yb.cpu().tolist())

    val_loss = sum(losses) / max(1, len(losses))
    if len(gts) == 0:
        f1 = prec = rec = acc = 0.0
    else:
        f1 = float(f1_score(gts, preds, average="macro"))
        prec = float(precision_score(gts, preds, average="macro", zero_division=0))
        rec = float(recall_score(gts, preds, average="macro", zero_division=0))
        acc = float(sum(1 for a, b in zip(gts, preds) if a == b) / len(gts))

    return val_loss, acc, f1, prec, rec, gts, preds


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 2,
):
    model.eval()

    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xb = xb.to(device)
        _ = model(xb)

    total_time = 0.0
    n_images = 0

    for xb, yb in loader:
        xb = xb.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model(xb)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time += t1 - t0
        n_images += xb.size(0)

    if n_images == 0 or total_time <= 0.0:
        return 0.0, 0.0, 0.0, 0

    avg_ms = (total_time / n_images) * 1000.0
    throughput = n_images / total_time

    return total_time, avg_ms, throughput, n_images


def save_confusion_matrix_csv(cm, classes, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["true/pred"] + list(classes)
        writer.writerow(header)
        for i, row in enumerate(cm):
            writer.writerow([classes[i]] + list(row))


def save_per_class_metrics_csv(
    prec_cls,
    rec_cls,
    f1_cls,
    support,
    classes,
    path: Path,
):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for c, p, r, f1, s in zip(classes, prec_cls, rec_cls, f1_cls, support):
            writer.writerow([c, float(p), float(r), float(f1), int(s)])


def train_model(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    wd: float = 1e-4,
    ckpt_path: str | None = None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_f1 = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        train_correct = 0
        train_total = 0

        for xb, yb in tqdm(dl_tr, desc=f"[epoch {ep}/{epochs}]", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            nb += 1

            preds = logits.argmax(1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss = total_loss / max(1, nb)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(model, dl_va, device)

        print(
            f"[ep {ep:02d}] "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, ckpt_path)

    return best_state, best_f1, history


def scan_dataset(root_dir: Path) -> Tuple[List[Dict], List[str]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root_dir}")

    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"Tidak ada subfolder kelas di {root_dir}")

    entries: List[Dict] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls in classes:
        cls_dir = root_dir / cls
        for fname in os.listdir(cls_dir):
            p = cls_dir / fname
            if p.suffix.lower() in exts and p.is_file():
                entries.append(
                    {
                        "path": str(p.resolve()),
                        "label": cls,
                    }
                )

    if not entries:
        raise ValueError(f"Tidak ada file gambar valid di {root_dir}")

    return entries, classes


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path ke root dataset (berisi subfolder kelas)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit-base",
        choices=[
            "resnet-152",
            "convnext-base",
            "vit-base",
            "swin-tiny",
            "deit-small",
            "timm",
        ],
        help="Pilih arsitektur model",
    )
    parser.add_argument(
        "--timm-name",
        type=str,
        default="",
        help="Nama model timm asli, misal: vit_small_patch16_224 (dipakai jika --model timm)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: {device} | GPU: {gpu_name}")
    else:
        print(f"Device: {device} (CPU)")

    set_seed(42)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    entries, classes = scan_dataset(dataset_root)
    num_classes = len(classes)
    print(f"Ditemukan {len(entries)} gambar, {num_classes} kelas: {classes}")

    labels = [e["label"] for e in entries]
    train_idx, val_idx = train_test_split(
        range(len(entries)),
        test_size=args.val_split,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    train_entries = [entries[i] for i in train_idx]
    val_entries = [entries[i] for i in val_idx]

    print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")

    class_to_idx = {c: i for i, c in enumerate(classes)}

    ds_tr = SimpleFundus(
        train_entries,
        class_to_idx,
        img_size=args.img_size,
        augment=True,
    )
    ds_va = SimpleFundus(
        val_entries,
        class_to_idx,
        img_size=args.img_size,
        augment=False,
    )

    pin = torch.cuda.is_available()
    persistent = args.num_workers > 0

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=(2 if persistent else None),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=(2 if persistent else None),
    )

    model = build_model(
        name=args.model,
        num_classes=num_classes,
        pretrained=True,
        timm_name=(args.timm_name or None),
    )

    param_info = count_params(model)
    print(
        f"Params — total: {param_info['total_params']:,} | "
        f"trainable: {param_info['trainable_params']:,} | "
        f"non-trainable: {param_info['non_trainable_params']:,} | "
        f"~size: {param_info['model_size_mb']:.2f} MB"
    )

    ckpt_path = outdir / f"{args.model}_best.pt"
    best_state, best_f1, history = train_model(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        ckpt_path=str(ckpt_path),
    )

    if best_state is None:
        raise RuntimeError("best_state kosong. Training mungkin gagal atau tidak ada batch yang diproses.")

    # simpan kurva training/validation ke CSV
    curves_path = outdir / f"{args.model}_training_curves.csv"
    if history:
        with curves_path.open("w", newline="") as f:
            fieldnames = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        print("Kurva training/validation disimpan di:", curves_path)

    model.load_state_dict(best_state)
    model.to(device)
    val_loss, val_acc, val_f1, val_prec, val_rec, gts, preds = evaluate(model, dl_va, device)

    print(
        f"[FINAL] val_loss={val_loss:.4f} | acc={val_acc:.4f} | "
        f"f1={val_f1:.4f} | prec={val_prec:.4f} | rec={val_rec:.4f}"
    )

    results_row: Dict[str, float | int | str] = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "total_params": param_info["total_params"],
        "trainable_params": param_info["trainable_params"],
        "non_trainable_params": param_info["non_trainable_params"],
        "model_size_mb": param_info["model_size_mb"],
        "ckpt": str(ckpt_path),
    }

    if len(gts) > 0:
        labels_int = list(range(num_classes))
        prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
            gts,
            preds,
            labels=labels_int,
            zero_division=0,
        )
        cm = confusion_matrix(gts, preds, labels=labels_int)

        cm_path = outdir / f"{args.model}_confusion_matrix.csv"
        perclass_path = outdir / f"{args.model}_perclass_metrics.csv"

        save_confusion_matrix_csv(cm, classes, cm_path)
        save_per_class_metrics_csv(prec_cls, rec_cls, f1_cls, support, classes, perclass_path)

        print(f"Confusion matrix disimpan di: {cm_path}")
        print(f"Per-class metrics disimpan di: {perclass_path}")

        results_row["confusion_matrix_csv"] = str(cm_path)
        results_row["perclass_metrics_csv"] = str(perclass_path)
    else:
        print("[WARN] gts/preds kosong, tidak bisa hitung confusion matrix & per-class metrics.")
        results_row["confusion_matrix_csv"] = ""
        results_row["perclass_metrics_csv"] = ""

    total_time_s, avg_ms, throughput, n_images = measure_inference_time(
        model,
        dl_va,
        device=device,
        warmup_batches=2,
    )
    print(
        f"[INFERENCE] images={n_images} | "
        f"total_time={total_time_s:.4f} s | avg={avg_ms:.4f} ms/img | "
        f"throughput={throughput:.2f} img/s"
    )

    results_row["inference_total_time_s"] = total_time_s
    results_row["inference_avg_ms_per_image"] = avg_ms
    results_row["inference_throughput_img_per_s"] = throughput
    results_row["inference_n_images"] = n_images

    csv_path = outdir / f"{args.model}_results_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_row.keys()))
        writer.writeheader()
        writer.writerow(results_row)

    print("Hasil disimpan di:", csv_path)
    print("Selesai.")


if __name__ == "__main__":
    main()
