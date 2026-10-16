"""Adım 6 - Gradio web demosu.

Görsel veya video yükle -> model seç -> kutular + % ateş / % duman.

Kullanım:
    python -m src.app            # http://127.0.0.1:7860
    python -m src.app --share    # Colab'da: herkese açık geçici link üretir
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from src.fire_analysis import analyze, draw_overlay, summarize_frames
from src.utils import DISPLAY_NAMES, available_models, load_config, load_model, output_dir

LABELS = {"yolo11": "YOLO11", "yolov8": "YOLOv8", "rtdetr": "RT-DETR"}


def build_app(cfg: dict, out_override: str | None) -> gr.Blocks:
    models_found = available_models(cfg, out_override)
    if not models_found:
        raise SystemExit("Eğitilmiş model bulunamadı (outputs/runs/*/weights/best.pt). Önce modelleri eğit.")
    icfg = cfg["inference"]
    imgsz = cfg["train"]["imgsz"]
    cache: dict = {}

    def get_model(arch):
        if arch not in cache:
            cache[arch] = load_model(arch, models_found[arch])
        return cache[arch]

    def report_md(rep: dict) -> str:
        icon = {"YANGIN ALARMI": "🔴", "DUMAN UYARISI": "🟠", "SUPHELI": "🟡", "NORMAL": "🟢"}[rep["status"]]
        rows = "\n".join(
            f"| {DISPLAY_NAMES[c]} | %{rep[c]['confidence']:.1f} | %{rep[c]['area_percent']:.2f} | {rep[c]['count']} |"
            for c in ("fire", "smoke")
        )
        return (f"### {icon} {rep['status']}\n\n| Sınıf | Güven | Kapladığı alan | Tespit sayısı |\n"
                f"|---|---|---|---|\n{rows}\n\n"
                "*Güven: modelin en emin tespiti. Alan: tespit kutularının görseldeki toplam payı.*")

    def predict_image(image_rgb, arch, conf):
        if image_rgb is None:
            return None, {}, "Bir görsel yükleyin."
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Ultralytics numpy girdiyi BGR bekler
        r = get_model(arch).predict(bgr, conf=conf, iou=icfg["iou"], imgsz=imgsz, verbose=False)[0]
        rep = analyze(r, icfg["alarm"])
        label = {DISPLAY_NAMES[c]: rep[c]["confidence"] / 100 for c in ("fire", "smoke")}
        return cv2.cvtColor(draw_overlay(r, rep), cv2.COLOR_BGR2RGB), label, report_md(rep)

    def predict_video(video_path, arch, conf, progress=gr.Progress()):
        if not video_path:
            return None, "Bir video yükleyin."
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
        out_path = Path(tempfile.mkdtemp()) / "sonuc.mp4"
        writer, reps = None, []
        stream = get_model(arch).predict(video_path, stream=True, conf=conf, iou=icfg["iou"], imgsz=imgsz, verbose=False)
        for i, r in enumerate(stream):
            rep = analyze(r, icfg["alarm"])
            frame = draw_overlay(r, rep)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            reps.append(rep)
            if total:
                progress(min(1.0, (i + 1) / total), desc=f"Kare {i + 1}/{total}")
        if writer is not None:
            writer.release()
        s = summarize_frames(reps)
        md = f"### Video özeti: {s['status']}  ({s['frames']} kare)\n\n| Sınıf | Görüldüğü kareler | En yüksek güven | En büyük alan |\n|---|---|---|---|\n"
        md += "\n".join(
            f"| {DISPLAY_NAMES[c]} | %{s[c]['frames_detected_percent']} | %{s[c]['max_confidence']} | %{s[c]['max_area_percent']} |"
            for c in ("fire", "smoke")
        )
        return str(out_path), md

    comp_md = output_dir(cfg, out_override) / "reports" / "model_comparison.md"
    choices = [(LABELS[a], a) for a in models_found]

    with gr.Blocks(title="Yangın & Duman Tespiti") as demo:
        gr.Markdown("# 🔥 Yangın & Duman Tespiti\nGörsel veya video yükleyin; seçilen model ateş ve dumanı işaretleyip yüzdeleri raporlasın.")
        with gr.Row():
            arch = gr.Dropdown(choices=choices, value=choices[0][1], label="Model")
            conf = gr.Slider(0.05, 0.95, value=icfg["conf"], step=0.05, label="Güven eşiği")
        with gr.Tab("Görsel"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Girdi")
                img_out = gr.Image(type="numpy", label="Sonuç")
            with gr.Row():
                lbl = gr.Label(label="Güven skoru", num_top_classes=2)
                md = gr.Markdown()
            gr.Button("Tespit et", variant="primary").click(predict_image, [img_in, arch, conf], [img_out, lbl, md])
        with gr.Tab("Video"):
            with gr.Row():
                vid_in = gr.Video(label="Girdi")
                vid_out = gr.Video(label="Sonuç")
            vmd = gr.Markdown()
            gr.Button("Videoyu işle", variant="primary").click(predict_video, [vid_in, arch, conf], [vid_out, vmd])
        if comp_md.exists():
            with gr.Tab("Model karşılaştırması"):
                gr.Markdown(comp_md.read_text(encoding="utf-8"))
                png = comp_md.with_suffix(".png")
                if png.exists():
                    gr.Image(str(png), show_label=False)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_app(cfg, args.output_dir).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
