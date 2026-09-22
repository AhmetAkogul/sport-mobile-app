# Telefon / hazır poz modeli hattı

`phone.py` mevcut telefon hattı iskeletidir. `mediapipe_model.py` gerçek Tasks
modelini mevcut `Poz2B` arayüzüne bağlar; `run_video.py` yerel videodan JSONL üretir.
Karar ve model kimliği: [0030](../docs/kararlar/0030-mediapipe-gercek-model.md).

İsteğe bağlı ortam (ana kalibrasyon ortamını değiştirmeden):

```bash
python3 -m venv data/mediapipe-env
data/mediapipe-env/bin/python -m pip install -r mono/requirements-mediapipe.txt
```

Modeli karar kaydındaki resmî adresten `data/models/` altına alın. Video çalıştırma:

```bash
data/mediapipe-env/bin/python -m mono.run_video \
  --video data/ornek.avi \
  --model data/models/pose_landmarker_full-float16-v1.task \
  --output data/poz-sonucu --max-frames 300
```

Çıktı dizini yeni olmalıdır. `poses.jsonl` her kare için eklem isimleri, özgün
piksel koordinatları, güven/görünürlük, kare indeksi ve model kimliğini taşır.
`summary.json` işlenmiş/tespit edilmiş kare sayısı, giriş içerik kimliği ve
bitiş nedenini içerir. `frame_limit` tüm videonun işlendiği anlamına gelmez.
OpenCV kaynak sonu ile bazı okuma hatalarını ayıramaz. Eksik tespitler silinmez.
Süreç yerel kitaplık çökmesiyle kapanırsa durum `running` kalabilir; otomatik
başarı kabul edilmemelidir. Normal Python hataları `failed` olarak kaydedilir.

`MediaPipeEstimator` bir bağlam yöneticisidir ve `phone.hatti_kostur`'un beklediği
çağrılabilir arayüzü sağlar. Bu turda telefon 3B baseline'ının gerçek doğruluğu
sınanmadı. Çoklu kişi takibi, VIDEO modu ve fiziksel değerlendirme henüz yoktur.
