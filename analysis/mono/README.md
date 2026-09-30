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

## Poz önizlemesi

Supervision kurulumu ayrı ortamda `pip install -r mono/requirements-visualization.txt`
ile yapılır. Kaydedilmiş bir sonuçtan türetilmiş video üretmek için:

```bash
data/mediapipe-env/bin/python -m mono.visualize \
  --video data/mediapipe-demo/repeated-official-sample.avi \
  --poses-dir data/mediapipe-demo/run-3 \
  --output data/mediapipe-demo/overlay-1
```

`overlay.avi` ve `preview.png` türetilmiş çıktıdır; kaynak video ve `poses.jsonl`
değiştirilmez. Çıktı sabit FPS önizlemesidir, ses içermez ve doğruluk/etik veya
fiziksel doğrulama iddiası taşımaz. Ayrıntılı karar: [0032](../docs/kararlar/0032-supervision-gorsellestirme.md).

## RTMPose alternatifi

`RTMPoseEstimator(detector_path, pose_path)` aynı BGR -> Poz2B arayüzünü sağlar.
Ayrı ortamda kurulum ve model kimlikleri: [0031](../docs/kararlar/0031-rtmpose-on-karsilastirma.md).
Kişi tespiti + RTMPose-M birlikte çalışır; birden fazla kişi açıkça reddedilir.

`mono.benchmark` aynı videoyu iki ayrı ortamda karşılaştırmak içindir. Örnek:

```bash
data/rtmpose-env/bin/python -m mono.benchmark --backend rtmpose \
  --video data/mediapipe-demo/repeated-official-sample.avi \
  --model data/models/rtmpose/pose.onnx --detector data/models/rtmpose/detector.onnx \
  --output data/yeni-rtmpose-olcumu.json
```

Model başlangıcı ve video çözme süresi kare gecikmesinin dışında tutulur;
tepe RSS tüm süreci içerir. Çıktı doğruluk alanını null bırakır. Elimizdeki
tekrarlanan resim videosu doğruluk, gerçek hareket veya çok kişi testi değildir.
Telefon hattı CLI'sında `--intrinsics` 3x3 JSON dizi, `--lengths` ise şu biçimde
bir JSON listesidir: `[{"a":"sag_kalca","b":"sag_diz","metre":0.42}]`.
