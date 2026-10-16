"""Adım 2 - Veri ön işleme (data preprocessing).

Yapılanlar sırasıyla:
  1. Ham veri setindeki train/val/test klasörlerini ve sınıf adlarını bul
  2. Etiketleri doğrula ve temizle
       - bozuk satırlar, geçersiz sayılar, bilinmeyen sınıflar atılır
       - görsel dışına taşan kutular kırpılır (clip), çok küçük kutular atılır
       - aynı kutunun tekrarları silinir
  3. Tabakalı (stratified) alt küme seç: ateş / duman / ikisi / arka plan
     oranları korunur, arka plan oranı config'ten ayarlanır
  4. Görselleri doğrula (açılamayan, çok küçük görseller atılır) ve split'ler
     arası kopyaları ele (test görseli train'de olmasın -> veri sızıntısı yok)
  5. Görselleri uzun kenar 640 olacak şekilde küçült (en-boy oranı korunur;
     YOLO etiketleri normalize olduğu için değişmez). Modelin istediği
     640x640 letterbox + 0-1 normalizasyonu Ultralytics eğitimde kendisi yapar.
  6. Train setine çevrimdışı augmentation uygula (bkz. augmentations.py)
  7. data.yaml, istatistikler ve önizleme görsellerini yaz

Kullanım:
    python -m src.prepare_data
    python -m src.prepare_data --raw-dir /veri/seti/koku --data-dir /content/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.utils import IMG_EXTS, data_dir, load_config, output_dir, set_seed

SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
CLASS_ALIASES = {"flame": "fire", "flames": "fire", "fires": "fire", "smokes": "smoke"}
CATEGORIES = ("both", "fire", "smoke", "background")
PROCESS_ORDER = ("test", "val", "train")  # önce test/val: train'deki kopyalar bunlara göre elenir


# --------------------------------------------------------------------------- #
# 1. Ham veri setini keşfet
# --------------------------------------------------------------------------- #
def find_splits(root: Path) -> dict[str, tuple[Path, Path]]:
    """{split: (images_dir, labels_dir)} - klasör yapısından bağımsız arar."""
    found: dict[str, tuple[Path, Path]] = {}
    candidates = sorted(root.rglob("images"), key=lambda p: len(p.parts))
    for img_dir in candidates:
        if not img_dir.is_dir():
            continue
        split = SPLIT_ALIASES.get(img_dir.parent.name.lower())
        if split and split not in found:
            found[split] = (img_dir, img_dir.parent / "labels")
    # Alternatif yapı: images/train, labels/train
    for img_dir in candidates:
        for sub in (img_dir.iterdir() if img_dir.is_dir() else []):
            split = SPLIT_ALIASES.get(sub.name.lower())
            if sub.is_dir() and split and split not in found:
                found[split] = (sub, img_dir.parent / "labels" / sub.name)
    return found


def read_source_names(root: Path) -> list[str]:
    for y in sorted(root.rglob("*.yaml"), key=lambda p: len(p.parts)):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            names = d["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=int)]
            print(f"[hazırla] Sınıf adları {y.name} dosyasından okundu: {names}")
            return [str(n).strip().lower() for n in names]
    print("[hazırla] UYARI: data.yaml bulunamadı, D-Fire varsayılanı kullanılıyor: ['smoke', 'fire']")
    return ["smoke", "fire"]


# --------------------------------------------------------------------------- #
# 2. Etiket doğrulama
# --------------------------------------------------------------------------- #
def parse_label_file(path: Path, class_map: dict[int, int], min_box: float, issues: Counter):
    """YOLO etiket dosyasını oku, temizle. Dönüş: (boxes [[cx,cy,w,h]], classes)."""
    boxes, classes, seen = [], [], set()
    if not path.exists():  # YOLO kuralı: etiket dosyası yoksa görsel arka plandır
        return boxes, classes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues["hatalı_satır"] += 1
            continue
        try:
            c = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            issues["sayı_değil"] += 1
            continue
        if not all(map(math.isfinite, (cx, cy, w, h))):
            issues["sayı_değil"] += 1
            continue
        if c not in class_map:
            issues["bilinmeyen_sınıf"] += 1
            continue
        # xyxy'ye çevir, [0,1]'e kırp, geri çevir
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if (x1, y1, x2, y2) != (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2):
            issues["kırpılan_kutu"] += 1
        w, h = x2 - x1, y2 - y1
        if w < min_box or h < min_box:
            issues["çok_küçük_kutu"] += 1
            continue
        box = (round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), round(w, 6), round(h, 6))
        key = (class_map[c], box)
        if key in seen:
            issues["tekrar_kutu"] += 1
            continue
        seen.add(key)
        boxes.append(list(box))
        classes.append(class_map[c])
    return boxes, classes


def category_of(classes: list[int], fire_id: int, smoke_id: int) -> str:
    s = set(classes)
    if fire_id in s and smoke_id in s:
        return "both"
    if fire_id in s:
        return "fire"
    if smoke_id in s:
        return "smoke"
    return "background"


# --------------------------------------------------------------------------- #
# 3. Tabakalı örnekleme
# --------------------------------------------------------------------------- #
def stratified_order(records: list[dict], n: int | None, bg_ratio: float, rng: random.Random) -> list[dict]:
    """Kotalara göre sıralanmış aday listesi döndürür. Kota dolmazsa (bozuk
    görsel vs.) işleme adımı listenin devamından yedek alır."""
    by_cat = {c: [r for r in records if r["category"] == c] for c in CATEGORIES}
    for lst in by_cat.values():
        rng.shuffle(lst)
    if n is None or n >= len(records):
        out = [r for c in CATEGORIES for r in by_cat[c]]
        rng.shuffle(out)
        return out

    n_bg = min(round(n * bg_ratio), len(by_cat["background"]))
    n_pos = n - n_bg
    pos_total = sum(len(by_cat[c]) for c in ("both", "fire", "smoke")) or 1
    quotas = {c: round(n_pos * len(by_cat[c]) / pos_total) for c in ("both", "fire", "smoke")}
    quotas["background"] = n_bg

    primary, reserve = [], []
    for c in CATEGORIES:
        primary += by_cat[c][: quotas[c]]
        reserve += by_cat[c][quotas[c]:]
    rng.shuffle(primary)
    rng.shuffle(reserve)
    return primary + reserve


# --------------------------------------------------------------------------- #
# 4-5. Görsel doğrulama, kopya eleme, yeniden boyutlandırma
# --------------------------------------------------------------------------- #
def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def write_label(path: Path, boxes, classes) -> None:
    path.write_text(
        "".join(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b, c in zip(boxes, classes)),
        encoding="utf-8",
    )


def process_split(split, candidates, n, out_dir, ds_cfg, seen_hashes, issues):
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    target = len(candidates) if n is None else min(n, len(candidates))
    kept = []
    for rec in candidates:
        if len(kept) >= target:
            break
        raw = rec["image"].read_bytes()
        digest = hashlib.md5(raw).hexdigest()
        if digest in seen_hashes:
            issues["kopya_görsel"] += 1
            continue
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            issues["açılamayan_görsel"] += 1
            continue
        if min(img.shape[:2]) < 32:
            issues["çok_küçük_görsel"] += 1
            continue
        seen_hashes.add(digest)

        img = resize_max_side(img, ds_cfg["resize_max_side"])
        name = f"{split}_{len(kept):05d}"
        cv2.imwrite(str(img_out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, ds_cfg["jpeg_quality"]])
        write_label(lbl_out / f"{name}.txt", rec["boxes"], rec["classes"])
        kept.append({**rec, "out_name": name})

    if len(kept) < target:
        print(f"[hazırla] UYARI: {split} için {target} istendi, {len(kept)} geçerli görsel bulundu.")
    return kept


# --------------------------------------------------------------------------- #
# 6. Çevrimdışı augmentation
# --------------------------------------------------------------------------- #
def offline_augment(out_dir, kept, aug_cfg, ds_cfg, names, rng, report_dir):
    from src.augmentations import apply, build_offline_pipeline, draw_yolo_boxes

    pipe = build_offline_pipeline()
    img_dir, lbl_dir = out_dir / "images" / "train", out_dir / "labels" / "train"
    chosen = rng.sample(kept, round(len(kept) * aug_cfg["fraction"]))
    made, failed, previews = 0, 0, []
    for rec in chosen:
        img = cv2.imread(str(img_dir / f"{rec['out_name']}.jpg"))
        for k in range(aug_cfg["copies_per_image"]):
            try:
                aug, boxes, classes = apply(pipe, img, rec["boxes"], rec["classes"])
            except Exception as e:  # tek bir görselin hatası pipeline'ı durdurmasın
                failed += 1
                if failed <= 3:
                    print(f"[hazırla] augmentation atlandı ({rec['out_name']}): {e}")
                continue
            keep = [(b, c) for b, c in zip(boxes, classes) if b[2] >= ds_cfg["min_box_size"] and b[3] >= ds_cfg["min_box_size"]]
            # Pozitif bir görsel augmentation sonrası tüm kutularını kaybettiyse yanlış
            # "arka plan" örneği olmasın diye kaydetme
            if rec["boxes"] and not keep:
                continue
            boxes, classes = [b for b, _ in keep], [c for _, c in keep]
            name = f"{rec['out_name']}_aug{k}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), aug, [cv2.IMWRITE_JPEG_QUALITY, ds_cfg["jpeg_quality"]])
            write_label(lbl_dir / f"{name}.txt", boxes, classes)
            made += 1
            if len(previews) < 6 and boxes:
                previews.append((draw_yolo_boxes(img, rec["boxes"], rec["classes"], names),
                                 draw_yolo_boxes(aug, boxes, classes, names)))
    print(f"[hazırla] Çevrimdışı augmentation: {made} yeni görsel üretildi ({failed} hata).")
    if previews:
        save_aug_preview(previews, report_dir / "augmentation_preview.jpg")
    return made


def save_aug_preview(pairs, path: Path, cell=320) -> None:
    def fit(im):
        h, w = im.shape[:2]
        s = cell / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.full((cell, cell, 3), 255, np.uint8)
        y, x = (cell - im.shape[0]) // 2, (cell - im.shape[1]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        return canvas

    rows = [np.hstack([fit(a), np.full((cell, 8, 3), 255, np.uint8), fit(b)]) for a, b in pairs]
    header = np.full((36, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(header, "Orijinal", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(header, "Augmented", (cell + 18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.imwrite(str(path), np.vstack([header, *rows]))


# --------------------------------------------------------------------------- #
# 7. Raporlar
# --------------------------------------------------------------------------- #
def save_stats_plot(stats: dict, names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in ("train", "val", "test") if s in stats]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    labels = ["ateş+duman", "sadece ateş", "sadece duman", "arka plan"]
    for ax, split in zip(axes, splits):
        vals = [stats[split]["categories"].get(c, 0) for c in CATEGORIES]
        bars = ax.bar(labels, vals, color="#2a78d6", width=0.6)
        ax.bar_label(bars, fontsize=8, color="#52514e", padding=2)
        ax.set_title(f"{split}  ({stats[split]['images']} görsel)", fontsize=10, color="#0b0b0b")
        ax.tick_params(axis="x", labelsize=8, rotation=20, colors="#52514e")
        ax.tick_params(axis="y", labelsize=8, colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("İşlenmiş veri seti - görsel kategorileri", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def split_stats(records, names):
    box_counts = Counter(c for r in records for c in r["classes"])
    return {
        "images": len(records),
        "categories": dict(Counter(r["category"] for r in records)),
        "boxes": {names[i]: box_counts.get(i, 0) for i in range(len(names))},
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--raw-dir", default=None, help="Ham veri kökü (varsayılan: download_data çıktısı)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    ddir = data_dir(cfg, args.data_dir)
    if args.raw_dir:
        raw_root = Path(args.raw_dir)
    else:
        rp = ddir / "raw_path.txt"
        if not rp.exists():
            raise SystemExit("Ham veri bulunamadı. Önce: python -m src.download_data")
        raw_root = Path(rp.read_text(encoding="utf-8").strip())

    out_dir = ddir / "processed"
    report_dir = output_dir(cfg, args.output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        print(f"[hazırla] Eski işlenmiş veri siliniyor: {out_dir}")
        shutil.rmtree(out_dir)

    # 1. keşif
    splits = find_splits(raw_root)
    if "train" not in splits:
        raise SystemExit(f"{raw_root} altında train/images klasörü bulunamadı.")
    print("[hazırla] Bulunan split'ler:", {k: str(v[0]) for k, v in splits.items()})

    names = list(ds["classes"])
    src_names = read_source_names(raw_root)
    class_map = {}
    for i, n in enumerate(src_names):
        n = CLASS_ALIASES.get(n, n)
        if n in names:
            class_map[i] = names.index(n)
        else:
            print(f"[hazırla] UYARI: kaynak sınıf '{n}' hedef sınıflarda yok, atlanacak.")
    print(f"[hazırla] Sınıf eşlemesi (kaynak id -> hedef id): {class_map}  hedef: {names}")
    fire_id, smoke_id = names.index("fire"), names.index("smoke")

    # 2. etiketleri tara
    issues: Counter = Counter()
    records: dict[str, list[dict]] = {}
    source_stats = {}
    for split, (img_dir, lbl_dir) in splits.items():
        recs = []
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            boxes, classes = parse_label_file(lbl_dir / f"{img.stem}.txt", class_map, ds["min_box_size"], issues)
            recs.append({"image": img, "boxes": boxes, "classes": classes,
                         "category": category_of(classes, fire_id, smoke_id)})
        records[split] = recs
        source_stats[split] = split_stats(recs, names)
        print(f"[hazırla] Kaynak {split}: {source_stats[split]}")

    # Kaynakta val yoksa train'den ayır
    if "val" not in records:
        rng.shuffle(records["train"])
        k = max(1, len(records["train"]) // 10)
        records["val"], records["train"] = records["train"][:k], records["train"][k:]

    # 3-5. örnekle + işle
    seen_hashes: set[str] = set()
    kept_all = {}
    for split in PROCESS_ORDER:
        if split not in records:
            continue
        n = ds["subset"].get(split)
        order = stratified_order(records[split], n, ds["background_ratio"], rng)
        kept_all[split] = process_split(split, order, n, out_dir, ds, seen_hashes, issues)

    # 6. augmentation
    n_aug = 0
    if ds["offline_aug"]["enabled"]:
        n_aug = offline_augment(out_dir, kept_all["train"], ds["offline_aug"], ds, names, rng, report_dir)

    # 7. data.yaml + raporlar
    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        **({"test": "images/test"} if "test" in kept_all else {}),
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    stats = {s: split_stats(kept_all[s], names) for s in kept_all}
    if "train" in stats:
        stats["train"]["offline_augmented_extra"] = n_aug
    summary = {"source": source_stats, "processed": stats, "cleaning": dict(issues)}
    (report_dir / "dataset_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_stats_plot(stats, names, report_dir / "dataset_stats.png")

    print("\n[hazırla] Temizlik özeti:", dict(issues) or "sorun yok")
    for s, st in stats.items():
        print(f"[hazırla] {s:5s}: {st['images']} görsel, kutular {st['boxes']}, kategoriler {st['categories']}")
    print(f"[hazırla] data.yaml -> {out_dir / 'data.yaml'}")
    print(f"[hazırla] Raporlar -> {report_dir}")


if __name__ == "__main__":
    main()
