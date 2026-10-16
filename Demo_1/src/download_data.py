"""Adım 1 - Veri setini Kaggle'dan indir.

Prototipte görseller hazır veri setinden gelir. Gerçek sistemde bu adımın yerini
IP kameralardan gelen görüntü akışı alacak (bkz. predict.py --source rtsp://...).

Kullanım:
    python -m src.download_data
    python -m src.download_data --path /zaten/indirilmis/klasor   # indirmeyi atla
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import data_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--path", default=None, help="Veri seti zaten indirildiyse kök klasörü")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ddir = data_dir(cfg, args.data_dir)
    ddir.mkdir(parents=True, exist_ok=True)

    if args.path:
        raw = Path(args.path).resolve()
    else:
        import kagglehub

        handle = cfg["dataset"]["kaggle_handle"]
        print(f"[indir] {handle} indiriliyor (~3 GB, Colab'da 1-3 dk sürer)...")
        # Herkese açık veri setleri için Kaggle hesabı/token gerekmez.
        raw = Path(kagglehub.dataset_download(handle))

    if not raw.exists():
        raise SystemExit(f"Klasör bulunamadı: {raw}")

    # prepare_data.py ham verinin nerede olduğunu buradan okur
    (ddir / "raw_path.txt").write_text(str(raw), encoding="utf-8")
    print(f"[indir] Ham veri: {raw}")
    print(f"[indir] Yol kaydedildi: {ddir / 'raw_path.txt'}")


if __name__ == "__main__":
    main()
