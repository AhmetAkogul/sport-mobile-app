# analysis — hareket analizi motoru

sport-mobile-app'in görüntü analizi tarafı (Python). Kameradan kişiyi ve
iskeletini bulur, hareketi tanır, tekrarları sayar, her tekrarın formuna karar
verir ve bildirim üretir. İki kullanım:

- **Mobil:** kullanıcı ön/arka kamerayla kendini çeker; tek kişi, yalnız MediaPipe.
- **Salon:** salon kameraları; çok kişi, kişi takibi, antrenör paneline bildirim.

Araştırma denemesidir; klinik değerlendirme değildir.

## Kurulum

Python 3.13. Komutlar bu klasörde (`analysis/`) çalıştırılır.

```bash
python3 -m venv .venv && source .venv/bin/activate
make kurulum          # bağımlılıklar (opencv-contrib-python 5.0.0.93)
make kontrol          # ortam doğrulama
make test             # testler
make lint             # ruff (önce: make kurulum-dev)
```

⚠️ `opencv-python` kurmayın: `opencv-contrib-python` ile aynı `cv2` modülünü
sağlar, ikisi yan yana duramaz.

## Canlı modlar

Poz modelleri (MediaPipe, RTMW) ayrı bir model ortamında çalışır ve depoya
girmez; yolları `make` değişkenleriyle verin:

| Değişken | Ne |
|---|---|
| `MP_PYTHON` | MediaPipe + rtmlib + onnxruntime kurulu Python |
| `MP_MODEL` | MediaPipe `pose_landmarker_full` `.task` dosyası |
| `RTMW_MODEL`, `DEDEKTOR` | RTMW-x ve YOLOX `.onnx` dosyaları (salon) |
| `HAREKET_MODELI`, `FORM_MODELLERI` | eğitilmiş hareket ve form modelleri (`.joblib`) |

```bash
make antrenor-mobil MP_PYTHON=... MP_MODEL=...     # mobil: tek kişi, MediaPipe
make antrenor MP_PYTHON=... MP_MODEL=... RTMW_MODEL=... DEDEKTOR=...   # salon
make panel                                           # antrenör paneli: http://localhost:8765
```

Video dosyasından denemek için `ARGS="--video yol.mp4"`, IP kamera için
`ARGS="--akis rtsp://..."`. Olaylar `out/olaylar.jsonl`'e yazılır; panel bu
dosyayı izler.

## Klasörler

| Klasör | İçerik |
|---|---|
| `mono/` | tek kamera hattı, canlı mod, antrenör, olay şeması, panel |
| `pose3d/` | çoklu kamera üçgenleme, iskelet, çok kişi 3B takip |
| `eval/` | ölçüler, hareket tanıma, tekrar kararları |
| `calib/`, `uncertainty/` | kalibrasyon ve belirsizlik |
| `capture/` | kayıt ve senkron |
| `veri/` | veri seti okuyucular |
| `scripts/` | deney betikleri (`make reproduce`) |
| `tests/` | testler |

Veri setleri ve model dosyaları depoya girmez (`.gitignore`).
