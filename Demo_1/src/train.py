"""Adım 3 - Model eğitimi.

Üç model aynı veri, aynı görsel boyutu ve aynı augmentation ayarlarıyla
eğitilir ki karşılaştırma adil olsun. Hepsi COCO üzerinde ön-eğitimli
ağırlıklardan başlar (transfer learning).

Kullanım:
    python -m src.train --model yolo11
    python -m src.train --model yolov8
    python -m src.train --model rtdetr
    python -m src.train --model rtdetr --resume        # Colab koptuysa kaldığı yerden
    python -m src.train --model yolo11 --epochs 3      # hızlı deneme
"""

from __future__ import annotations

import argparse
import shutil

from src.utils import ARCHS, load_config, load_model, output_dir, processed_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=ARCHS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="config'teki değeri ezer")
    ap.add_argument("--batch", type=int, default=None, help="config'teki değeri ezer")
    ap.add_argument("--device", default=None, help="0, cpu ... (varsayılan: otomatik)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg, mcfg = cfg["train"], cfg["models"][args.model]
    data = processed_yaml(cfg, args.data_dir)
    if not data.exists():
        raise SystemExit(f"{data} yok. Önce: python -m src.prepare_data")

    runs = output_dir(cfg, args.output_dir) / "runs"
    run_dir = runs / args.model
    last = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last.exists():
            raise SystemExit(f"Devam edilecek checkpoint yok: {last}")
        print(f"[eğitim] {last} üzerinden devam ediliyor...")
        load_model(args.model, last).train(resume=True)
    else:
        if run_dir.exists():
            print(f"[eğitim] UYARI: {run_dir} zaten var, üzerine yazılacak (devam için --resume).")
        model = load_model(args.model, mcfg["weights"])
        params = {
            "data": str(data),
            "imgsz": tcfg["imgsz"],
            "epochs": args.epochs or tcfg["epochs"],
            "batch": args.batch or mcfg["batch"],
            "patience": tcfg["patience"],
            "workers": tcfg["workers"],
            "cache": tcfg["cache"],
            "amp": tcfg["amp"],
            "optimizer": mcfg.get("optimizer", "auto"),
            "seed": cfg["seed"],
            "deterministic": True,
            "project": str(runs),
            "name": args.model,
            "exist_ok": True,
            "plots": True,
            **tcfg["augment"],
        }
        if "lr0" in mcfg:
            params["lr0"] = mcfg["lr0"]
        if args.device is not None:
            params["device"] = args.device
        print(f"[eğitim] {args.model} ({mcfg['weights']}) eğitiliyor:", {k: v for k, v in params.items() if k not in tcfg["augment"]})
        model.train(**params)

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dst = output_dir(cfg, args.output_dir) / "weights" / f"{args.model}_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"[eğitim] En iyi ağırlık: {best}\n[eğitim] Kopyası: {dst}")


if __name__ == "__main__":
    main()
