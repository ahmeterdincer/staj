# 🔥 Yangın & Duman Tespiti (Prototip)

Görüntülerde **ateş** ve **dumanı** tespit eden bir nesne tespiti (object detection) sistemi.
Üç modern detektör aynı veriyle eğitilip karşılaştırılır: **YOLO11**, **YOLOv8**, **RT-DETR**.

Her görsel için çıktı:

```
YANGIN ALARMI  | Ateş: güven %91.2, alan %12.4 | Duman: güven %76.0, alan %31.8
```

- **Güven:** modelin o sınıftaki en emin tespiti. "Ateş var mı?" sorusunun cevabı.
- **Alan:** o sınıfa ait tespit kutularının görselde kapladığı toplam pay. "Ne kadar büyük?" sorusunun cevabı.
- **Durum:** `YANGIN ALARMI` / `DUMAN UYARISI` / `SUPHELI` / `NORMAL` (eşikler `config.yaml > inference.alarm` altında).

---

## Akış

```
[Kaggle veri seti]  ──>  download_data  ──>  prepare_data  ──>  train (x3)  ──>  evaluate  ──>  predict / app
 (ileride: IP kamera)                       temizle, örnekle,    YOLO11          test setinde     % ateş
                                            küçült, augment      YOLOv8          karşılaştır      % duman
                                                                 RT-DETR
```

| Adım | Dosya | Ne yapar |
|---|---|---|
| 1 | `src/download_data.py` | D-Fire veri setini Kaggle'dan indirir (hesap gerekmez) |
| 2 | `src/prepare_data.py` | Etiket temizliği, tabakalı alt küme, kopya eleme, 640px'e küçültme, data.yaml |
| 2b | `src/augmentations.py` | Kamera koşullarını simüle eden çevrimdışı augmentation |
| 3 | `src/train.py` | Seçilen modeli fine-tune eder |
| 4 | `src/evaluate.py` | Test setinde mAP / precision / recall / hız karşılaştırması |
| 5 | `src/predict.py` | Görsel, klasör, video veya RTSP akışında tahmin + JSON rapor |
| 6 | `src/app.py` | Gradio web demosu |
| – | `src/fire_analysis.py` | Model çıktısını yüzde raporuna çeviren mantık |
| – | `configs/config.yaml` | **Tüm ayarlar burada** |

## Veri seti

**D-Fire** veri setinin YOLO formatına çevrilmiş Kaggle sürümü kullanılıyor:
[sayedgamal99/smoke-fire-detection-yolo](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)
(~21 bin görsel, train/val/test ayrılmış, lisansı CC0). Sınıflar: `fire`, `smoke`. Ayrıca hiç ateş/duman
içermeyen görseller de var. Bunlar yanlış alarmı azaltmak için önemli.

Varsayılan olarak bunun **alt kümesi** kullanılıyor: train için 4000, val için 800, test için 800 görsel.
Train'e ayrıca ~1200 augmented görsel ekleniyor. Tam seti kullanmak için `config.yaml` içinde
`dataset.subset` altındaki değerleri `null` yap.

> İlk araştırmada önerilen setler neden kullanılmadı: `amergr/fire-and-smoke-dataset` Kaggle'da yok.
> `dataclusterlabs/fire-and-smoke-dataset` ise ücretli bir setin 100 görsellik tanıtım örneği, üç detektörü
> eğitmeye yetmez.

## Ön işleme ve augmentation

**Ön işleme (`prepare_data.py`):**
1. Bozuk etiket satırları, geçersiz sayılar ve bilinmeyen sınıflar atılır. Görsel dışına taşan kutular kırpılır,
   çok küçük ve tekrarlanan kutular silinir.
2. Tabakalı örnekleme yapılır: "ateş+duman / sadece ateş / sadece duman / arka plan" oranları korunur.
3. Açılamayan görseller atılır. Split'ler arası kopyalar elenir, böylece test görseli train'e sızmaz.
4. Görseller uzun kenarı 640 piksel olacak şekilde, en-boy oranı korunarak küçültülür. 640×640 letterbox
   ve 0-1 normalizasyonu eğitim sırasında Ultralytics tarafından yapılır.

**Augmentation iki katmanlı:**

| Katman | Nerede | Neler |
|---|---|---|
| Online (her epoch farklı) | `config.yaml > train.augment` | Mosaic, mixup, HSV, yatay çevirme, ölçek, kaydırma, döndürme, eğme, perspektif |
| Offline (kamera senaryoları) | `augmentations.py` | Bulanıklık, sensör gürültüsü, JPEG sıkıştırma, düşük çözünürlük, gece/karanlık, IR (gri), sis, açılı kamera |

Bilinçli olarak yapılmayanlar:
- **Dikey çevirme:** alev ve duman yukarı doğru yükselir, ters görüntü gerçekçi değil.
- **Büyük renk tonu kayması:** ateşin turuncu-kırmızı rengi ayırt edici bir özellik.

Sonuçlar `outputs/reports/augmentation_preview.jpg` dosyasında görülebilir.

## Colab'da çalıştırma

1. Bu klasörü (`configs/`, `src/`, `notebooks/`, `requirements.txt`) Google Drive'ına
   `MyDrive/fire_smoke_detection/` olarak yükle. Klasörün boyutu birkaç yüz KB.
2. Drive'da `notebooks/colab_train.ipynb` dosyasını Colab ile aç.
3. `Runtime > Change runtime type > GPU` seç, sonra hücreleri sırayla çalıştır.

Veri seti doğrudan Colab sunucusuna indiği için senin internetinden ~3 GB harcanmaz. Eğitim çıktıları
Drive'a yazılır, bu yüzden oturum koparsa kaybolmaz. Kaldığın yerden devam etmek için:
`python -m src.train --model rtdetr --resume`.

## Komutlar

```bash
pip install -r requirements.txt
python -m src.download_data
python -m src.prepare_data
python -m src.train --model yolo11        # yolov8 | rtdetr
python -m src.evaluate
python -m src.predict --model yolo11 --source ornek.jpg
python -m src.app --share
```

## Çıktılar

```
outputs/
├── reports/       dataset_stats.png, augmentation_preview.jpg, model_comparison.{png,md,csv,json}
├── runs/<model>/  Ultralytics eğitim çıktıları (weights/best.pt, results.png, confusion_matrix.png ...)
├── eval/<model>/  test seti grafikleri (PR eğrisi, confusion matrix)
├── weights/       <model>_best.pt kopyaları
└── predictions/   işaretlenmiş görseller/videolar + results.json
```

## Bilinen sınırlar / sonraki adımlar

- **Alan yüzdesi kutu tabanlıdır.** Kutunun içindeki her piksel ateş değildir. Gerçek piksel oranı için
  segmentasyon modeli (örn. YOLO11-seg) gerekir.
- **Video kararları kare bazlıdır.** Gerçek sistemde tek karelik yanlış alarmı bastırmak için zamansal
  filtre eklenmeli, örn. "son 2 saniyenin %60'ında ateş varsa alarm ver".
- **IP kamera:** `predict.py --source rtsp://...` şimdiden çalışıyor. Sonraki adımlar: çoklu kamera, alarm
  bildirimi (e-posta/Telegram) ve ONNX/TensorRT export ile hızlandırma.
