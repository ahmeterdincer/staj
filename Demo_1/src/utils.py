"""Ortak yardımcılar: config okuma, yol çözme, model yükleme."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

ARCHS = ("yolo11", "yolov8", "rtdetr")

# Arayüzde gösterilecek Türkçe sınıf adları
DISPLAY_NAMES = {"fire": "Ateş", "smoke": "Duman"}
# cv2.putText Türkçe karakter çizemez, görsel üstü yazılar için ASCII karşılıklar
ASCII_NAMES = {"fire": "ATES", "smoke": "DUMAN"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def data_dir(cfg: dict, override: str | None = None) -> Path:
    """Öncelik: CLI argümanı > FSD_DATA_DIR ortam değişkeni > config."""
    return _resolve(override or os.environ.get("FSD_DATA_DIR") or cfg["paths"]["data_dir"])


def output_dir(cfg: dict, override: str | None = None) -> Path:
    return _resolve(override or os.environ.get("FSD_OUTPUT_DIR") or cfg["paths"]["output_dir"])


def processed_yaml(cfg: dict, data_override: str | None = None) -> Path:
    return data_dir(cfg, data_override) / "processed" / "data.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_model(arch: str, weights: str | Path):
    """RT-DETR ağırlıkları YOLO sınıfıyla değil RTDETR sınıfıyla yüklenmeli."""
    from ultralytics import RTDETR, YOLO

    return RTDETR(str(weights)) if arch == "rtdetr" else YOLO(str(weights))


def trained_weights(cfg: dict, arch: str, out_override: str | None = None) -> Path:
    return output_dir(cfg, out_override) / "runs" / arch / "weights" / "best.pt"


def available_models(cfg: dict, out_override: str | None = None) -> dict[str, Path]:
    """Eğitimi tamamlanmış (best.pt'si olan) modeller."""
    found = {}
    for arch in ARCHS:
        w = trained_weights(cfg, arch, out_override)
        if w.exists():
            found[arch] = w
    return found


def infer_arch(weights: str | Path) -> str:
    """Dışarıdan verilen bir .pt dosyasının mimarisini adından tahmin et."""
    s = str(weights).lower()
    return "rtdetr" if "rtdetr" in s or "rt-detr" in s else "yolo"
