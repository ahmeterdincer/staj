"""Adım 4 - Modelleri test setinde değerlendir ve karşılaştır.

Eğitimi biten tüm modelleri (outputs/runs/*/weights/best.pt) eğitimde HİÇ
görmedikleri test seti üzerinde ölçer:

  mAP50        : IoU >= 0.5 ile ortalama hassasiyet (tespit "yaklaşık doğru yerde" mi)
  mAP50-95     : 0.5-0.95 arası IoU'ların ortalaması (kutu ne kadar sıkı oturuyor)
  Precision    : verilen alarmların ne kadarı doğru (yanlış alarm az mı)
  Recall       : gerçek yangınların ne kadarı yakalandı (kaçırılan az mı)
  Gecikme (ms) : tek görsel için ön işleme + çıkarım + son işleme süresi

Yangın tespitinde Recall genelde daha kritiktir: kaçırılan bir yangın,
yanlış bir alarmdan çok daha pahalıdır.

Kullanım:
    python -m src.evaluate
    python -m src.evaluate --models yolo11 rtdetr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.utils import ARCHS, IMG_EXTS, available_models, load_config, load_model, output_dir, processed_yaml

SERIES_COLORS = {"yolo11": "#2a78d6", "yolov8": "#eb6834", "rtdetr": "#1baf7a"}
LABELS = {"yolo11": "YOLO11", "yolov8": "YOLOv8", "rtdetr": "RT-DETR"}


def measure_latency(model, images: list[Path], imgsz: int, warmup: int = 5) -> float:
    for p in images[:warmup]:
        model.predict(str(p), imgsz=imgsz, verbose=False)
    times = []
    for p in images[warmup:]:
        r = model.predict(str(p), imgsz=imgsz, verbose=False)[0]
        times.append(sum(r.speed.values()))
    return round(float(np.mean(times)), 2) if times else float("nan")


def evaluate_model(arch, weights, data_yaml: Path, split: str, cfg: dict, eval_dir: Path) -> dict:
    model = load_model(arch, weights)
    m = model.val(data=str(data_yaml), split=split, imgsz=cfg["train"]["imgsz"], batch=16,
                  project=str(eval_dir), name=arch, exist_ok=True, plots=True, verbose=False)
    box = m.box
    row = {
        "model": arch,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        name = str(m.names[int(c)]).lower()
        row[f"{name}_AP50"] = round(float(ap50), 4)
        row[f"{name}_recall"] = round(float(r), 4)

    cfgd = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    img_dir = Path(cfgd["path"]) / cfgd[split]
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)[:105]
    row["latency_ms"] = measure_latency(model, imgs, cfg["train"]["imgsz"])
    row["params_M"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    row["size_MB"] = round(Path(weights).stat().st_size / 1e6, 1)
    return row


def save_markdown(rows: list[dict], path: Path) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_charts(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("mAP50", "mAP50"), ("mAP50-95", "mAP50-95"), ("fire_AP50", "Ateş AP50"),
               ("smoke_AP50", "Duman AP50"), ("precision", "Precision"), ("recall", "Recall")]
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e8e7e2"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(metrics))
    width = 0.8 / len(rows)
    for i, r in enumerate(rows):
        vals = [r.get(k, 0) for k, _ in metrics]
        bars = ax1.bar(x + (i - (len(rows) - 1) / 2) * width, vals, width * 0.92,
                       color=SERIES_COLORS.get(r["model"], "#888"), label=LABELS.get(r["model"], r["model"]))
        ax1.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, color=ink2, padding=2)
    ax1.set_xticks(x, [lbl for _, lbl in metrics])
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Test seti doğruluk metrikleri (yüksek = iyi)", fontsize=11, color=ink)
    ax1.legend(frameon=False, fontsize=9, ncol=len(rows), loc="upper left")

    for r in rows:
        ax2.scatter(r["latency_ms"], r["mAP50-95"], s=90, color=SERIES_COLORS.get(r["model"], "#888"),
                    edgecolor="white", linewidth=2, zorder=3)
        ax2.annotate(LABELS.get(r["model"], r["model"]), (r["latency_ms"], r["mAP50-95"]),
                     xytext=(8, 4), textcoords="offset points", fontsize=9, color=ink)
    ax2.set_xlabel("Gecikme (ms / görsel, düşük = iyi)", fontsize=9, color=ink2)
    ax2.set_ylabel("mAP50-95", fontsize=9, color=ink2)
    ax2.set_title("Hız - doğruluk dengesi", fontsize=11, color=ink)
    ax2.margins(x=0.35, y=0.25)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=ink2, labelsize=8)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", choices=ARCHS, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_yaml = processed_yaml(cfg, args.data_dir)
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} yok. Önce: python -m src.prepare_data")
    split = "test" if "test" in yaml.safe_load(data_yaml.read_text(encoding="utf-8")) else "val"

    models = available_models(cfg, args.output_dir)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    if not models:
        raise SystemExit("Değerlendirilecek eğitilmiş model yok. Önce: python -m src.train --model ...")

    out = output_dir(cfg, args.output_dir)
    report_dir = out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arch, w in models.items():
        print(f"[değerlendir] {arch} ({split} seti)...")
        rows.append(evaluate_model(arch, w, data_yaml, split, cfg, out / "eval"))
        print(f"[değerlendir] {rows[-1]}")

    (report_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_markdown(rows, report_dir / "model_comparison.md")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(report_dir / "model_comparison.csv", index=False)
    except ImportError:
        pass
    save_charts(rows, report_dir / "model_comparison.png")

    best = max(rows, key=lambda r: r["mAP50-95"])
    print("\n" + (report_dir / "model_comparison.md").read_text(encoding="utf-8"))
    print(f"En yüksek mAP50-95: {LABELS[best['model']]} ({best['mAP50-95']})")
    print(f"Raporlar: {report_dir}")


if __name__ == "__main__":
    main()
