"""Çevrimdışı (offline) augmentation pipeline'ı.

Ultralytics eğitim sırasında zaten online augmentation yapıyor (mosaic, HSV,
flip, ölçek, kaydırma - bkz. config.yaml > train.augment). Burada onun
yapmadığı, güvenlik kamerasına özgü bozulmaları simüle ediyoruz:

    Senaryo                         Dönüşüm
    -----------------------------   ---------------------------------------
    Odak kayması / hareket          GaussianBlur, MotionBlur
    Gece, düşük ışıkta sensör       GaussNoise, ISONoise
    IP kamera akışının sıkıştırması ImageCompression (JPEG artefaktları)
    Düşük çözünürlüklü kamera       Downscale (küçült -> büyüt)
    Gece / ters ışık / parlama      RandomBrightnessContrast, RandomGamma
    Kızılötesi (IR) gece modu       ToGray
    Sisli / puslu hava              RandomFog (az: duman ile karışmasın)
    Kameranın açılı takılması       Perspective, Affine (eğme, döndürme)

Albumentations kutuları (bbox) görselle birlikte dönüştürür, yani etiketler
geometrik dönüşümlerden sonra da doğru kalır.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np


def build_offline_pipeline() -> A.Compose:
    return A.Compose(
        [
            # --- Geometrik (kamera açısı) ---
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.Affine(rotate=(-8, 8), shear=(-6, 6), scale=(0.9, 1.1), p=0.4),
            # --- Optik / sensör bozulmaları ---
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 9)),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.08)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                ],
                p=0.4,
            ),
            A.Downscale(scale_range=(0.4, 0.8), p=0.25),
            A.ImageCompression(quality_range=(25, 70), p=0.5),
            # --- Işık koşulları ---
            A.OneOf(
                [
                    # Negatif ağırlıklı parlaklık: gece/karanlık sahneler daha olası
                    A.RandomBrightnessContrast(brightness_limit=(-0.45, 0.15), contrast_limit=0.2),
                    A.RandomGamma(gamma_limit=(60, 140)),
                    A.CLAHE(clip_limit=3.0),
                ],
                p=0.6,
            ),
            A.ToGray(p=0.1),
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.08),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,  # dönüşümde %70'inden fazlası kadraj dışına çıkan kutuyu at
            clip=True,
        ),
    )


def apply(pipeline: A.Compose, image_bgr: np.ndarray, boxes: list[list[float]], classes: list[int]):
    """boxes: YOLO formatında [[cx, cy, w, h], ...] (normalize)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = pipeline(image=rgb, bboxes=boxes, class_labels=classes)
    img = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
    return img, [list(map(float, b)) for b in out["bboxes"]], [int(c) for c in out["class_labels"]]


# Önizleme çizimleri için sınıf renkleri (BGR): fire = turuncu, smoke = mavi
BOX_COLORS = {0: (52, 104, 235), 1: (214, 120, 42)}


def draw_yolo_boxes(image_bgr: np.ndarray, boxes, classes, names: list[str]) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    for (cx, cy, bw, bh), c in zip(boxes, classes):
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = BOX_COLORS.get(c, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, names[c], (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
