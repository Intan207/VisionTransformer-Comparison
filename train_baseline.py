# train_baseline.py — baseline ringan tanpa pretrained (TinyCNN)
# - Konsumsi generator dari dataloader.py (K-Fold + undersample train)
# - Simpan model terbaik per fold (berdasarkan F1 macro)
# - Tulis results CSV: results/baseline_cnn.csv

import argparse
from pathlib import Path
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from dataloader import dataloader_eye  # yield: (train_loader, val_loader, class_names)


# ====== Util reproducibility ======
def set_seed(seed: int = 2025):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====== Model kecil (ringan CPU) ======
class TinyCNN(nn.Module):
    """
    CNN kecil untuk klasifikasi 224x224 RGB -> num_classes.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


# ====== Loop 1 fold ======
def train_one_fold(model, train_loader, val_loader, device, epochs: int, lr: float, weight_decay: float = 1e-4):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Matikan torch.compile agar tidak butuh C++ compiler
    # (biarkan training berjalan mode eager)


    best_f1 = -1.0
    best_state = None
    last_preds, last_gts = None, None

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()

        # valid
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x).argmax(1)
                preds += p.cpu().tolist()
                gts += y.cpu().tolist()

        acc = (torch.tensor(preds) == torch.tensor(gts)).float().mean().item()
        f1 = f1_score(gts, preds, average="macro")
        prec = precision_score(gts, preds, average="macro", zero_division=0)
        rec = recall_score(gts, preds, average="macro", zero_division=0)
        print(f"[ep {ep:02d}] loss={run_loss:.3f} acc={acc:.3f} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            # simpan state dict di CPU biar aman diserialisasi
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            last_preds, last_gts = preds[:], gts[:]

    return best_state, best_f1, last_gts, last_preds


# ====== Tulis CSV hasil ======
def append_results_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["fold", "acc", "f1", "precision", "recall"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])


# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=None, help="Jika None pakai auto-pick di dataloader.py")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="baseline_cnn", help="prefix nama file hasil & checkpoint")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # generator fold dari dataloader.py (sudah K-Fold + undersample train)
    fold_gen = dataloader_eye(
        dataset_path_global=args.dataset_root,
        k_folds=args.k_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_train=True,
    )

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_dir) / f"{args.tag}.csv"

    mean_f1 = 0.0
    n_folds = 0

    for fold_id, pack in enumerate(fold_gen):
        train_loader, val_loader, class_names = pack
        num_classes = len(class_names)
        print(f"\n========== FOLD {fold_id} ==========")
        print(f"Classes: {class_names}")

        model = TinyCNN(num_classes=num_classes)
        best_state, best_f1, gts, preds = train_one_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr
        )

        # simpan model terbaik fold ini
        out_path = save_dir / f"{args.tag}_fold{fold_id}_best.pt"
        torch.save(best_state, out_path)
        print(f"[SAVE] {out_path}")

        # pastikan kita punya preds/gts valid; jika None, muat state terbaik dan hitung ulang pada val_loader
        if gts is None or preds is None:
            # muat model ke device dan apply best state jika tersedia
            if best_state is not None:
                try:
                    model.load_state_dict(best_state)
                except Exception:
                    # jika best_state tidak cocok, skip loading
                    pass
            model = model.to(device)
            preds, gts = [], []
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    p = model(x).argmax(1)
                    preds += p.cpu().tolist()
                    gts += y.cpu().tolist()

        # metrik ringkas utk CSV (aman karena preds/gts kini list)
        if len(preds) == 0 or len(gts) == 0:
            acc = 0.0
            prec = 0.0
            rec = 0.0
        else:
            acc = (torch.tensor(preds) == torch.tensor(gts)).float().mean().item()
            prec = precision_score(gts, preds, average="macro", zero_division=0)
            rec = recall_score(gts, preds, average="macro", zero_division=0)

        row = {
            "fold": fold_id,
            "acc": f"{acc:.4f}",
            "f1": f"{best_f1:.4f}",
            "precision": f"{prec:.4f}",
            "recall": f"{rec:.4f}",
        }
        append_results_csv(results_csv, [row])

        # laporan per-kelas (opsional untuk dokumentasi)
        try:
            print(classification_report(gts, preds, target_names=class_names, digits=3))
        except Exception:
            pass

        mean_f1 += best_f1
        n_folds += 1

    if n_folds > 0:
        mean_f1 /= n_folds
        print(f"\n[RESULT] Mean F1 over {n_folds} folds: {mean_f1:.4f}")
        print(f"[RESULT] CSV tersimpan di: {results_csv.resolve()}")


if __name__ == "__main__":
    main()
