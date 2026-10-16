"""Model çıktısını "% ateş / % duman" raporuna çevirir.

Her sınıf için iki ayrı yüzde üretilir:

  confidence   : Modelin o sınıftaki en emin tespitinin güven skoru.
                 "Görüntüde ateş VAR mı?" sorusunun cevabı. (örn. %91)
  area_percent : O sınıfa ait tüm kutuların birleşiminin görselin toplam
                 alanına oranı. "Yangın/duman ne kadar BÜYÜK?" sorusunun
                 cevabı. (örn. %12.4) Kutular üst üste binerse alan iki kez
                 sayılmaz. Not: kutu tabanlı bir yaklaşımdır, kutu içindeki
                 her piksel ateş değildir; gerçek piksel oranı için
                 segmentasyon modeli gerekir.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils import ASCII_NAMES

CLASSES = ("fire", "smoke")
MASK_MAX_SIDE = 640  # alan hesabı için maske çözünürlüğü (hız için küçültülür)


def analyze(result, alarm_cfg: dict) -> dict:
    """result: tek bir ultralytics Results nesnesi."""
    h, w = result.orig_shape
    names = {int(k): str(v).lower() for k, v in result.names.items()}
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
    confs = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
    clss = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)

    s = min(1.0, MASK_MAX_SIDE / max(h, w))
    mh, mw = max(1, round(h * s)), max(1, round(w * s))

    report = {}
    for cname in CLASSES:
        sel = np.array([names.get(c) == cname for c in clss], dtype=bool)
        if not sel.any():
            report[cname] = {"detected": False, "confidence": 0.0, "area_percent": 0.0, "count": 0}
            continue
        mask = np.zeros((mh, mw), dtype=bool)
        for x1, y1, x2, y2 in xyxy[sel] * s:
            mask[int(max(0, y1)):int(np.ceil(y2)), int(max(0, x1)):int(np.ceil(x2))] = True
        report[cname] = {
            "detected": True,
            "confidence": round(float(confs[sel].max()) * 100, 1),
            "area_percent": round(float(mask.mean()) * 100, 2),
            "count": int(sel.sum()),
        }

    if report["fire"]["confidence"] >= alarm_cfg["fire_conf"] * 100:
        status = "YANGIN ALARMI"
    elif report["smoke"]["confidence"] >= alarm_cfg["smoke_conf"] * 100:
        status = "DUMAN UYARISI"
    elif report["fire"]["detected"] or report["smoke"]["detected"]:
        status = "SUPHELI"
    else:
        status = "NORMAL"
    report["status"] = status
    report["detections"] = [
        {"class": names.get(int(c), str(c)), "confidence": round(float(p) * 100, 1),
         "box_xyxy": [round(float(v), 1) for v in b]}
        for b, p, c in zip(xyxy, confs, clss)
    ]
    return report


STATUS_COLORS = {  # BGR
    "YANGIN ALARMI": (40, 40, 220),
    "DUMAN UYARISI": (0, 165, 255),
    "SUPHELI": (0, 200, 255),
    "NORMAL": (60, 160, 60),
}


def draw_overlay(result, report: dict) -> np.ndarray:
    """Kutuları çizer + üste durum ve yüzdeleri içeren bir bilgi şeridi ekler."""
    img = result.plot()  # BGR, kutular + etiketler
    h, w = img.shape[:2]
    scale = max(0.5, w / 1280)
    bar_h = int(70 * scale)
    bar = np.full((bar_h, w, 3), 25, np.uint8)
    color = STATUS_COLORS[report["status"]]
    cv2.rectangle(bar, (0, 0), (int(12 * scale), bar_h), color, -1)
    cv2.putText(bar, report["status"], (int(22 * scale), int(30 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color, max(1, int(2 * scale)))
    line = "   ".join(
        f"{ASCII_NAMES[c]}: guven %{report[c]['confidence']:.0f} | alan %{report[c]['area_percent']:.1f}"
        for c in CLASSES
    )
    cv2.putText(bar, line, (int(22 * scale), int(58 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (235, 235, 235), max(1, int(1.5 * scale)))
    return np.vstack([bar, img])


def summarize_frames(frame_reports: list[dict]) -> dict:
    """Video için kare bazlı raporları tek bir özete indirger."""
    n = len(frame_reports) or 1
    out = {"frames": len(frame_reports)}
    for c in CLASSES:
        det = [r[c] for r in frame_reports if r[c]["detected"]]
        out[c] = {
            "frames_detected": len(det),
            "frames_detected_percent": round(100 * len(det) / n, 1),
            "max_confidence": max((d["confidence"] for d in det), default=0.0),
            "max_area_percent": max((d["area_percent"] for d in det), default=0.0),
            "mean_area_percent": round(sum(d["area_percent"] for d in det) / len(det), 2) if det else 0.0,
        }
    statuses = [r["status"] for r in frame_reports]
    for s in ("YANGIN ALARMI", "DUMAN UYARISI", "SUPHELI"):
        if s in statuses:
            out["status"] = s
            break
    else:
        out["status"] = "NORMAL"
    return out
