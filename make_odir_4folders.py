# sample_from_4folders.py
# Ambil tepat N=per_class file acak per kelas dari dataset yang sudah 4 folder
# dan salin/hardlink ke folder tujuan kedua.

import argparse, os, random, shutil
from pathlib import Path

DEF_CLASSES = ["CATARACT", "DR", "GLAUCOMA", "NORMAL"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path):
    ensure_dir(dst.parent)
    try:
        if not dst.exists():
            os.link(src, dst)  # hemat disk (hardlink, satu drive)
    except Exception:
        shutil.copy2(src, dst) # fallback: copy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root dataset sumber yang berisi 4 subfolder kelas")
    ap.add_argument("--dst", required=True, help="Root dataset tujuan hasil sampling")
    ap.add_argument("--per-class", type=int, default=276)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--classes", nargs="*", default=DEF_CLASSES)
    ap.add_argument("--oversample", action="store_true", help="Jika kelas < target, gandakan acak")
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    print(f"[SRC] {src}")
    print(f"[DST] {dst}")
    ensure_dir(dst)

    for cname in args.classes:
        sdir = src / cname
        if not sdir.exists():
            raise SystemExit(f"[ERROR] Folder kelas tidak ditemukan: {sdir}")
        files = [p for p in sdir.iterdir() if p.is_file()]
        n = len(files)
        target = args.per_class
        print(f"[INFO] {cname}: {n} file tersedia; target={target}")

        if n == 0:
            raise SystemExit(f"[ERROR] Kelas '{cname}' kosong.")

        if n >= target:
            chosen = random.sample(files, target)
        else:
            if not args.oversample:
                raise SystemExit(f"[ERROR] {cname} hanya {n} < {target}. Tambahkan --oversample jika tetap ingin {target}.")
            extra = random.choices(files, k=target - n)
            chosen = files + extra

        ddir = dst / cname
        ensure_dir(ddir)
        for i, src_path in enumerate(chosen, 1):
            out = ddir / src_path.name
            if out.exists():  # jika oversample bisa duplikat nama
                out = ddir / f"{src_path.stem}_{i}{src_path.suffix}"
            copy_or_link(src_path, out)

        print(f"[DONE] {cname}: {len([p for p in ddir.iterdir() if p.is_file()])} file")

    print("\n[OK] Dataset sampling selesai:", dst)

if __name__ == "__main__":
    main()
