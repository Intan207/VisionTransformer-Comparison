# train.py — K-Fold training sederhana + W&B (opsional)
import os, csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm.auto import tqdm

# ==== IMPORT DARI PACKAGE src (penting untuk RunPod/Colab/Windows) ====
from src.dataloader import load_folds      # pakai dataloader.py kamu
from src.model import build_model          # pakai model.py kamu


# =======================
#  Evaluasi sederhana
# =======================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
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
    # guard against empty ground-truth / prediction lists
    if len(gts) == 0:
        f1 = 0.0
        prec = 0.0
        rec = 0.0
        acc = 0.0
    else:
        f1   = float(f1_score(gts, preds, average="macro"))
        prec = float(precision_score(gts, preds, average="macro", zero_division=0))
        rec  = float(recall_score(gts, preds, average="macro", zero_division=0))
        acc  = float(sum(1 for a, b in zip(gts, preds) if a == b) / len(gts))

    return val_loss, acc, f1, prec, rec


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
    use_wandb: bool = False,
    fold_id: int | None = None,
    wandb_run=None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_f1 = -1.0
    best_state = None
    wandb_disabled = False  # kalau W&B error di tengah, kita matikan

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
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, dl_va, device)

        print(
            f"[fold {fold_id} | ep {ep:02d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={val_f1:.4f}"
        )

        # logging ke W&B (jika aktif & tidak error)
        if use_wandb and (wandb_run is not None) and (not wandb_disabled):
            try:
                import wandb
                wandb.log(
                    {
                        "epoch": ep,
                        "fold": fold_id,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_precision": val_prec,
                        "val_recall": val_rec,
                    }
                )
            except Exception as e:
                print(f"[WARN] wandb.log gagal, W&B dimatikan untuk sisa training. Error: {e}")
                wandb_disabled = True

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
        default="resnet-152",
        choices=["resnet-152", "convnext-base"],
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

    # W&B opsional
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project", type=str, default="eye-disease-ta")
    parser.add_argument("--group", type=str, default="")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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

    # setup W&B kalau diminta
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            # kalau API key belum ada, wandb.init biasanya akan error → kita tangkap
            wandb_run = wandb.init(
                project=args.project,
                group=(args.group or None),
                name=f"{args.model}_kfold_local",
                config={
                    "model": args.model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "wd": args.wd,
                    "img_size": args.img_size,
                    "num_classes": num_classes,
                    "classes": classes,
                    "k_folds": total_folds,
                    "run_folds": f"{start_fold}-{end_fold}",
                },
            )
        except Exception as e:
            print(f"[WARN] wandb.init gagal, logging W&B dimatikan. Error: {e}")
            args.use_wandb = False
            wandb_run = None

    # tempat simpan hasil tiap fold
    fold_results = []

    for idx, (dl_tr, dl_va, _) in enumerate(selected, start=start_fold):
        print(f"\n========== FOLD {idx} ({args.model}) ==========")
        model = build_model(args.model, num_classes, pretrained=True)

        ckpt_path = str(outdir / f"{args.model}_fold{idx}_best.pt")
        best_state, best_f1 = train_one_fold(
            model=model,
            dl_tr=dl_tr,
            dl_va=dl_va,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            ckpt_path=ckpt_path,
            use_wandb=args.use_wandb,
            fold_id=idx,
            wandb_run=wandb_run,
        )

        # pastikan best_state dihasilkan
        if best_state is None:
            raise RuntimeError("best_state kosong. Training mungkin gagal atau tidak ada batch yang diproses.")

        # evaluasi ulang dengan best_state (opsional)
        model.load_state_dict(best_state)
        model.to(device)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, dl_va, device)

        print(
            f"[FOLD {idx} FINAL] val_loss={val_loss:.4f} | acc={val_acc:.4f} | "
            f"f1={val_f1:.4f} | prec={val_prec:.4f} | rec={val_rec:.4f}"
        )

        fold_results.append(
            {
                "fold": idx,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "ckpt": ckpt_path,
            }
        )

    # simpan hasil ke CSV
    csv_path = outdir / f"{args.model}_kfold_results.csv"
    if fold_results:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fold_results[0].keys()))
            writer.writeheader()
            writer.writerows(fold_results)
        print("Hasil K-Fold disimpan di:", csv_path)

    # rata-rata skor
    avg_loss = sum(fr["val_loss"] for fr in fold_results) / len(fold_results)
    avg_acc  = sum(fr["val_acc"] for fr in fold_results) / len(fold_results)
    avg_f1   = sum(fr["val_f1"] for fr in fold_results) / len(fold_results)

    print(
        f"\nRATA-RATA dari {len(fold_results)} fold: "
        f"val_loss={avg_loss:.4f} | acc={avg_acc:.4f} | f1={avg_f1:.4f}"
    )

    if args.use_wandb and wandb_run is not None:
        try:
            import wandb
            wandb.log(
                {
                    "avg/val_loss": avg_loss,
                    "avg/val_acc": avg_acc,
                    "avg/val_f1": avg_f1,
                }
            )
            wandb.finish()
        except Exception as e:
            print(f"[WARN] wandb.log/finish gagal di akhir. Error: {e}")

    print("Selesai.")


if __name__ == "__main__":
    main()
