# train.py — K-Fold training untuk Vision Transformer (tanpa W&B)
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm.auto import tqdm

# ==== IMPORT DARI PACKAGE src ====
from src.dataloader import load_folds      # pakai dataloader.py kamu
from src.model import build_model          # pakai model.py kamu


# =======================
#  Utility: hitung parameter
# =======================
def count_params(model: nn.Module):
    """
    Hitung jumlah parameter model:
    - total
    - trainable
    - non-trainable
    - estimasi ukuran model (MB) dengan asumsi float32 (4 byte)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    model_size_mb = total * 4 / (1024 ** 2)  # float32 = 4 byte

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "model_size_mb": model_size_mb,
    }


# =======================
#  Evaluasi (macro + simpan gts/preds)
# =======================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluasi di validation set:
    - kembalikan loss rata-rata
    - macro accuracy, f1, precision, recall
    - serta daftar ground-truth (gts) dan prediksi (preds)
    """
    model.eval()
    losses = []
    preds = []
    gts = []

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
        f1   = float(f1_score(gts, preds, average="macro"))
        prec = float(precision_score(gts, preds, average="macro", zero_division=0))
        rec  = float(recall_score(gts, preds, average="macro", zero_division=0))
        acc  = float(sum(1 for a, b in zip(gts, preds) if a == b) / len(gts))

    return val_loss, acc, f1, prec, rec, gts, preds


# =======================
#  Inference time measurement
# =======================
@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 2,
):
    """
    Ukur waktu inferensi di seluruh loader:
    - Lakukan warm-up beberapa batch (tidak dihitung)
    - Ukur waktu total untuk 1 pass penuh di loader
    - Kembalikan total_time (detik), avg_ms_per_image, throughput (img/s), n_images
    """
    model.eval()

    # Warm-up
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xb = xb.to(device)
        _ = model(xb)

    # Timed pass
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

        total_time += (t1 - t0)
        n_images += xb.size(0)

    if n_images == 0 or total_time <= 0.0:
        return 0.0, 0.0, 0.0, 0

    avg_ms = (total_time / n_images) * 1000.0
    throughput = n_images / total_time

    return total_time, avg_ms, throughput, n_images


# =======================
#  Save confusion matrix & per-class metrics
# =======================
def save_confusion_matrix_csv(cm, classes, path: Path):
    """
    Simpan confusion matrix ke CSV dengan header kelas.
    """
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
    """
    Simpan precision/recall/F1 per kelas ke CSV.
    """
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for c, p, r, f1, s in zip(classes, prec_cls, rec_cls, f1_cls, support):
            writer.writerow([c, float(p), float(r), float(f1), int(s)])


# =======================
#  Train satu fold
# =======================
def train_one_fold(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    wd: float = 1e-4,
    ckpt_path: str | None = None,
    fold_id: int | None = None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0

        for xb, yb in tqdm(dl_tr, desc=f"[fold {fold_id}] epoch {ep}/{epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            nb += 1

        train_loss = total_loss / max(1, nb)
        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(model, dl_va, device)

        print(
            f"[fold {fold_id} | ep {ep:02d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={val_f1:.4f}"
        )

        # simpan model terbaik berdasarkan F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, ckpt_path)

    return best_state, best_f1


# =======================
#  MAIN (CLI)
# =======================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kfold-json",
        type=str,
        required=True,
        help="Path ke kfold.json (hasil pembagian fold)",
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
            "timm",          # generic timm model (butuh --timm-name)
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
    parser.add_argument("--outdir", type=str, default="checkpoints")

    # subset fold (kalau mau jalankan sebagian saja)
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=0)  # 0 = sampai terakhir

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: {device} | GPU: {gpu_name}")
    else:
        print(f"Device: {device} (CPU)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # muat semua fold dari kfold.json
    all_folds = list(
        load_folds(
            args.kfold_json,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
        )
    )
    if not all_folds:
        raise ValueError("Tidak ada fold di JSON.")

    classes = all_folds[0][2]
    num_classes = len(classes)
    total_folds = len(all_folds)

    start_fold = max(1, args.start_fold)
    end_fold = args.end_fold if args.end_fold > 0 else total_folds
    if start_fold < 1 or end_fold > total_folds or start_fold > end_fold:
        raise ValueError(f"Rentang fold tidak valid: start={start_fold}, end={end_fold}, total={total_folds}")

    selected = all_folds[start_fold - 1 : end_fold]
    print(f"Jalankan fold {start_fold}..{end_fold} dari total {total_folds}")
    print("Kelas:", classes)

    # tempat simpan hasil tiap fold
    fold_results = []

    for idx, (dl_tr, dl_va, _) in enumerate(selected, start=start_fold):
        print(f"\n========== FOLD {idx} ({args.model}) ==========")

        # bangun model
        model = build_model(
            name=args.model,
            num_classes=num_classes,
            pretrained=True,
            timm_name=(args.timm_name or None),
        )

        # hitung parameter (sekali per fold)
        param_info = count_params(model)
        print(
            f"Params — total: {param_info['total_params']:,} | "
            f"trainable: {param_info['trainable_params']:,} | "
            f"non-trainable: {param_info['non_trainable_params']:,} | "
            f"~size: {param_info['model_size_mb']:.2f} MB"
        )

        ckpt_path = outdir / f"{args.model}_fold{idx}_best.pt"
        best_state, best_f1 = train_one_fold(
            model=model,
            dl_tr=dl_tr,
            dl_va=dl_va,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            ckpt_path=str(ckpt_path),
            fold_id=idx,
        )

        if best_state is None:
            raise RuntimeError("best_state kosong. Training mungkin gagal atau tidak ada batch yang diproses.")

        # evaluasi ulang dengan best_state (untuk final skor per fold + confusion matrix + per-class metrics)
        model.load_state_dict(best_state)
        model.to(device)
        val_loss, val_acc, val_f1, val_prec, val_rec, gts, preds = evaluate(model, dl_va, device)

        print(
            f"[FOLD {idx} FINAL] val_loss={val_loss:.4f} | acc={val_acc:.4f} | "
            f"f1={val_f1:.4f} | prec={val_prec:.4f} | rec={val_rec:.4f}"
        )

        # =============================
        #  Per-class metrics & confusion matrix
        # =============================
        if len(gts) > 0:
            labels = list(range(num_classes))
            prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
                gts,
                preds,
                labels=labels,
                zero_division=0,
            )
            cm = confusion_matrix(gts, preds, labels=labels)

            # simpan CSV confusion matrix & per-class metrics per fold
            cm_path = outdir / f"{args.model}_fold{idx}_confusion_matrix.csv"
            perclass_path = outdir / f"{args.model}_fold{idx}_perclass_metrics.csv"

            save_confusion_matrix_csv(cm, classes, cm_path)
            save_per_class_metrics_csv(prec_cls, rec_cls, f1_cls, support, classes, perclass_path)

            print(f"Confusion matrix disimpan di: {cm_path}")
            print(f"Per-class metrics disimpan di: {perclass_path}")
        else:
            cm_path = None
            perclass_path = None
            print("[WARN] gts/preds kosong, tidak bisa hitung confusion matrix & per-class metrics.")

        # =============================
        #  Inference time (validation set)
        # =============================
        total_time_s, avg_ms, throughput, n_images = measure_inference_time(
            model,
            dl_va,
            device=device,
            warmup_batches=2,
        )
        print(
            f"[FOLD {idx} INFERENCE] images={n_images} | "
            f"total_time={total_time_s:.4f} s | avg={avg_ms:.4f} ms/img | "
            f"throughput={throughput:.2f} img/s"
        )

        fold_results.append(
            {
                "fold": idx,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "total_params": param_info["total_params"],
                "trainable_params": param_info["trainable_params"],
                "non_trainable_params": param_info["non_trainable_params"],
                "model_size_mb": param_info["model_size_mb"],
                "inference_total_time_s": total_time_s,
                "inference_avg_ms_per_image": avg_ms,
                "inference_throughput_img_per_s": throughput,
                "inference_n_images": n_images,
                "ckpt": str(ckpt_path),
                "confusion_matrix_csv": str(cm_path) if cm_path is not None else "",
                "perclass_metrics_csv": str(perclass_path) if perclass_path is not None else "",
            }
        )

    # simpan hasil summary ke CSV
    csv_path = outdir / f"{args.model}_kfold_results.csv"
    if fold_results:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fold_results[0].keys()))
            writer.writeheader()
            writer.writerows(fold_results)
        print("Hasil K-Fold disimpan di:", csv_path)

    # rata-rata skor utama
    avg_loss = sum(fr["val_loss"] for fr in fold_results) / len(fold_results)
    avg_acc  = sum(fr["val_acc"] for fr in fold_results) / len(fold_results)
    avg_f1   = sum(fr["val_f1"] for fr in fold_results) / len(fold_results)

    print(
        f"\nRATA-RATA dari {len(fold_results)} fold: "
        f"val_loss={avg_loss:.4f} | acc={avg_acc:.4f} | f1={avg_f1:.4f}"
    )

    print("Selesai.")


if __name__ == "__main__":
    main()
