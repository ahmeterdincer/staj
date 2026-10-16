"""Adım 5 - Tahmin (inference): görsel, klasör, video veya canlı kamera akışı.

Çıktı: işaretlenmiş görseller/video + her biri için % ateş / % duman raporu
(results.json).

Kullanım:
    python -m src.predict --model yolo11 --source ornek.jpg
    python -m src.predict --model rtdetr --source klasor/
    python -m src.predict --model yolov8 --source video.mp4
    python -m src.predict --weights yol/best.pt --source ornek.jpg

    # İleride gerçek sistem: IP kamera (RTSP) veya webcam
    python -m src.predict --model yolo11 --source "rtsp://kullanici:sifre@192.168.1.10:554/stream1"
    python -m src.predict --model yolo11 --source 0 --max-frames 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from src.fire_analysis import analyze, draw_overlay, summarize_frames
from src.utils import (ARCHS, DISPLAY_NAMES, VIDEO_EXTS, infer_arch, load_config, load_model,
                       output_dir, trained_weights)


def is_stream_or_video(source: str) -> bool:
    s = source.lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://")) or Path(s).suffix in VIDEO_EXTS


def describe(report: dict) -> str:
    parts = [f"{DISPLAY_NAMES[c]}: güven %{report[c]['confidence']:.1f}, alan %{report[c]['area_percent']:.1f}"
             for c in ("fire", "smoke")]
    return f"{report['status']:14s} | " + " | ".join(parts)


def run_images(model, source, kwargs, alarm, save_dir: Path) -> list[dict]:
    reports = []
    for r in model.predict(source=source, stream=True, verbose=False, **kwargs):
        rep = analyze(r, alarm)
        name = Path(r.path).name
        cv2.imwrite(str(save_dir / f"{Path(name).stem}_pred.jpg"), draw_overlay(r, rep))
        reports.append({"image": name, **rep})
        print(f"{name:40s} {describe(rep)}")
    return reports


def run_video(model, source, kwargs, alarm, save_dir: Path, max_frames: int | None) -> dict:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    fps = fps if 1 <= fps <= 120 else 25

    out_name = (Path(source).stem if Path(source).suffix else "stream") + "_pred.mp4"
    writer, frame_reports, t0 = None, [], time.time()
    try:
        for i, r in enumerate(model.predict(source=src, stream=True, verbose=False, **kwargs)):
            rep = analyze(r, alarm)
            frame = draw_overlay(r, rep)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(save_dir / out_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            frame_reports.append({"frame": i, **rep})
            if i % 50 == 0:
                print(f"kare {i:6d}  {describe(rep)}")
            if max_frames and i + 1 >= max_frames:
                break
    except KeyboardInterrupt:
        print("Durduruldu.")
    finally:
        if writer is not None:
            writer.release()

    summary = summarize_frames(frame_reports)
    summary["processing_fps"] = round(len(frame_reports) / max(time.time() - t0, 1e-6), 1)
    summary["output_video"] = str(save_dir / out_name)
    return {"summary": summary, "frames": frame_reports}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=ARCHS, help="Eğitilmiş modeli kullan (outputs/runs/<model>/weights/best.pt)")
    g.add_argument("--weights", help="Herhangi bir .pt dosyası")
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["inference"]
    if args.model:
        arch, weights = args.model, trained_weights(cfg, args.model, args.output_dir)
    else:
        arch, weights = infer_arch(args.weights), Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Ağırlık dosyası bulunamadı: {weights}")

    save_dir = Path(args.save_dir) if args.save_dir else output_dir(cfg, args.output_dir) / "predictions" / (args.model or weights.stem)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(arch, weights)
    kwargs = {"conf": args.conf or icfg["conf"], "iou": icfg["iou"], "imgsz": cfg["train"]["imgsz"]}

    if is_stream_or_video(args.source):
        result = run_video(model, args.source, kwargs, icfg["alarm"], save_dir, args.max_frames)
        s = result["summary"]
        print(f"\nÖzet: {s['status']} | {s['frames']} kare, {s['processing_fps']} FPS")
        for c in ("fire", "smoke"):
            print(f"  {DISPLAY_NAMES[c]}: karelerin %{s[c]['frames_detected_percent']}'inde, "
                  f"en yüksek güven %{s[c]['max_confidence']}, en büyük alan %{s[c]['max_area_percent']}")
    else:
        result = run_images(model, args.source, kwargs, icfg["alarm"], save_dir)

    (save_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nÇıktılar: {save_dir}")


if __name__ == "__main__":
    main()
