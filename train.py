# train.py — K-Fold training (TRAIN di-undersample ke kelas minimal),
# simpan best per fold, tulis results CSV, dan log lengkap ke Weights & Biases.
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ====== PROYEK KAMU ======
from dataloader import dataloader_eye                      # generator fold -> (train_loader, val_loader, class_names)
from model import build_resnet152, build_convnext_base     # builder model pretrained

# ====== W&B ======
try:
    import importlib
    wandb = importlib.import_module("wandb")
except Exception as e:
    # If wandb is not installed (e.g., in static analysis or minimal environments),
    # provide a small no-op stub so the rest of the code can run without errors.
    print(f"[WARN] wandb tidak tersedia: {e} — menggunakan stub lokal (no-op).")

    class _WandbPlot:
        @staticmethod
        def confusion_matrix(preds, y_true, class_names):
            # Return a simple structure so wandb.log can accept it without crashing.
            return {"preds": list(preds), "y_true": list(y_true), "class_names": list(class_names)}

    class _WandbStub:
        plot = _WandbPlot

        def finish(self):
            return None

        def login(self):
            return None

        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    wandb = _WandbStub()


# ----------------------------------------------------------
# MODEL FACTORY
# ----------------------------------------------------------
def build_model(name: str, num_classes: int) -> nn.Module:
    """
    Membangun model pretrained sesuai argumen --model, lalu
    mengganti head klasifikasi terakhir -> num_classes.
    """
    name = name.lower()
    if name in ("resnet-152", "resnet152"):
        return build_resnet152(num_classes, pretrained=True)      # ResNet-152 + head baru
    elif name in ("convnext-base", "convnext"):
        return build_convnext_base(num_classes, pretrained=True)  # ConvNeXt-Base + head baru
    else:
        raise ValueError(f"Unknown model: {name}")


# ----------------------------------------------------------
# UTIL METRIK / LOGGING
# ----------------------------------------------------------
CLASS_NAMES_DEFAULT = ["NORMAL", "DR", "GLAUCOMA", "CATARACT"]

def _acc_from_lists(preds: List[int], gts: List[int]) -> float:
    t_preds = torch.tensor(preds)
    t_gts   = torch.tensor(gts)
    return (t_preds == t_gts).float().mean().item()

def _log_confusion(gts: List[int], preds: List[int], class_names: List[str] | None):
    """Log confusion matrix ke W&B."""
    cnames = class_names or CLASS_NAMES_DEFAULT
    try:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=preds, y_true=gts, class_names=cnames
            )
        })
    except Exception as e:
        print(f"[WARN] gagal log confusion matrix ke W&B: {e}")


# ----------------------------------------------------------
# TRAIN 1 FOLD
# ----------------------------------------------------------
def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    use_es: bool = True,       # early stopping opsional
    es_patience: int = 5,      # sabar 5 epoch
) -> Tuple[Dict[str, torch.Tensor], float, List[int], List[int], List[Dict[str, float]]]:
    """
    Melatih 1 fold:
      - Optimizer: AdamW
      - Loss: CrossEntropy
      - Seleksi model terbaik berdasarkan F1 (macro) pada VALID
      - Return:
          best_state: state_dict model terbaik (disalin ke CPU, aman untuk disimpan)
          best_f1: nilai F1 terbaik (macro)
          last_gts, last_preds: GT & pred dari model terbaik (untuk report ringkas)
          hist: list dict metrik per-epoch untuk di-log ke W&B
    """
    model = model.to(device)                                 # pindahkan model ke device (GPU/CPU)
    crit = nn.CrossEntropyLoss()                             # loss klasifikasi multi-kelas
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # optimizer

    best_f1: float = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None     # state dict terbaik (di CPU)
    last_preds: Optional[List[int]] = None                   # prediksi terakhir saat best
    last_gts: Optional[List[int]] = None                     # label GT terakhir saat best

    hist: List[Dict[str, float]] = []

    # Early stopping berdasarkan val_loss (hemat waktu)
    best_val_loss = math.inf
    bad = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()

        # -------------------- TRAIN --------------------
        model.train()
        run_loss = 0.0
        nbatches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nbatches += 1

        epoch_train_loss = run_loss / max(1, nbatches)  # train_loss

        # -------------------- VALID --------------------
        model.eval()
        preds: List[int] = []
        gts: List[int] = []
        val_losses: List[float] = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss = crit(logits, y).item()          # hitung val loss
                val_losses.append(vloss)
                p = logits.argmax(1)
                preds.extend(p.cpu().tolist())
                gts.extend(y.cpu().tolist())

        if len(gts) == 0:
            raise RuntimeError("Validation set kosong pada fold ini.")

        val_loss_mean = float(sum(val_losses) / max(1, len(val_losses)))
        acc = _acc_from_lists(preds, gts)
        f1  = f1_score(gts, preds, average="macro")
        prec = precision_score(gts, preds, average="macro", zero_division=0)
        rec  = recall_score(gts, preds, average="macro", zero_division=0)

        print(
            f"[ep {ep:02d}] "
            f"loss={epoch_train_loss:.3f} val_loss={val_loss_mean:.3f} "
            f"acc={acc:.3f} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f}"
        )

        # simpan history untuk W&B
        hist.append({
            "epoch": ep,
            "train_loss": float(epoch_train_loss),
            "val_loss": float(val_loss_mean),
            "val_acc": float(acc),
            "val_f1": float(f1),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "epoch_time_s": time.time() - t0,
        })

        # ---- simpan model terbaik berdasar F1 macro ----
        if f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            last_preds, last_gts = preds[:], gts[:]

        # ---- early stopping berdasarkan val_loss ----
        if use_es:
            if val_loss_mean < best_val_loss - 1e-6:
                best_val_loss = val_loss_mean
                bad = 0
            else:
                bad += 1
                if bad >= es_patience:
                    print(f"Early stopping (val_loss) at epoch {ep}")
                    break

    assert best_state is not None and last_preds is not None and last_gts is not None, \
        "best_state tidak terisi — periksa loop training/validasi."

    return best_state, best_f1, last_gts, last_preds, hist


# ----------------------------------------------------------
# CSV APPEND
# ----------------------------------------------------------
def append_results_csv(csv_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    """
    Menambahkan baris hasil ke file CSV.
      - Membuat parent folder jika belum ada.
      - Menulis header jika file belum ada.
      - Menulis baris: fold, acc, f1, precision, recall
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["fold", "acc", "f1", "precision", "recall"]
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)                               # tulis header sekali di awal
        for r in rows:
            w.writerow([r[h] for h in header])               # tulis baris sesuai urutan header


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main() -> None:
    """
    Entry point:
      - Parse argumen CLI
      - Siapkan generator K-Fold (dari dataloader_eye)
      - Loop tiap fold -> latih, evaluasi, simpan best checkpoint, tulis CSV, log ke W&B
      - Cetak ringkasan Mean F1 di akhir
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=None, help="Jika None pakai auto-pick di dataloader.py")
    ap.add_argument("--model", default="resnet-152", choices=["resnet-152", "convnext-base"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    # --- W&B init (di sini supaya config terisi dari args) ---
    wandb.finish()
    try:
        wandb.login()
    except Exception:
        pass
    wandb.init(
        project="eye-disease-ta",
        name=f"{args.model}-k{args.k_folds}",
        config={
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "k_folds": args.k_folds,
            "num_workers": args.num_workers,
            "dataset_root": args.dataset_root,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # ✅ Generator fold: undersample TRAIN ke kelas minimal per fold
    fold_gen = dataloader_eye(
        dataset_path_global=args.dataset_root,               # root yang berisi subfolder ODIR/
        k_folds=args.k_folds,                                # jumlah fold K-Fold
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_train=True,                                  # aktifkan undersampling TRAIN per fold
    )

    # Siapkan folder keluaran
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_dir) / f"{args.model}.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    mean_f1 = 0.0

    # --------- Loop tiap fold ---------
    for fold_id, pack in enumerate(fold_gen):
        train_loader, val_loader, class_names = pack
        print(f"\n========== FOLD {fold_id} ==========")
        print(f"Classes: {class_names}")

        # Bangun model pretrained + head baru
        model = build_model(args.model, num_classes=len(class_names))

        # Latih 1 fold dan ambil model terbaik (berdasar F1) + history metrik per-epoch
        best_state, best_f1, gts, preds, hist = train_one_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr
        )

        # ---- LOG per-epoch ke W&B (lengkap + tag fold) ----
        for it in hist:
            wandb.log({
                "fold": fold_id,
                "epoch": it["epoch"],
                "train_loss": it["train_loss"],
                "val_loss": it["val_loss"],
                "val_acc": it["val_acc"],
                "val_f1": it["val_f1"],
                "val_precision": it["val_precision"],
                "val_recall": it["val_recall"],
                "epoch_time_s": it.get("epoch_time_s", None),
                # "lr": current_lr,  # tambahkan jika kamu menambahkan scheduler & menyimpan LR di hist
            })

        # Simpan checkpoint terbaik untuk fold ini
        out_path = save_dir / f"{args.model}_fold{fold_id}_best.pt"
        torch.save(best_state, out_path)
        print(f"[SAVE] {out_path}")

        # Hitung metrik ringkas di fold (berdasar prediksi terbaik)
        acc = _acc_from_lists(preds, gts)
        prec = precision_score(gts, preds, average="macro", zero_division=0)
        rec  = recall_score(gts, preds, average="macro", zero_division=0)

        # Siapkan baris CSV
        row = {
            "fold": fold_id,
            "acc": f"{acc:.4f}",
            "f1": f"{best_f1:.4f}",
            "precision": f"{prec:.4f}",
            "recall": f"{rec:.4f}",
        }
        all_rows.append(row)
        append_results_csv(results_csv, [row])

        # (Opsional) Laporan per-kelas untuk dokumentasi
        try:
            print(classification_report(gts, preds, target_names=class_names, digits=3))
        except Exception:
            pass

        # Log ringkasan per-fold & confusion matrix
        wandb.log({
            "fold": fold_id,
            "fold_acc": float(acc),
            "fold_f1": float(best_f1),
            "fold_precision": float(prec),
            "fold_recall": float(rec),
        })
        _log_confusion(gts, preds, class_names)

        mean_f1 += best_f1

    # --------- Ringkasan akhir ---------
    if len(all_rows) > 0:
        mean_f1 /= len(all_rows)
        print(f"\n[RESULT] Mean F1 over {len(all_rows)} folds: {mean_f1:.4f}")
        print(f"[RESULT] CSV tersimpan di: {results_csv.resolve()}")
        wandb.log({"mean_f1_over_folds": float(mean_f1)})

    wandb.finish()


if __name__ == "__main__":
    main()
