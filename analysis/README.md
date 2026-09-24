# Çoklu Kamera 3B Hareket Analizi

> Telefon uygulamaları egzersiz formunu, ölçemeyecekleri bir hassasiyetle değerlendiriyor.
> Biz kalibre edilmiş çoklu kamera düzeneğiyle bu hatayı ölçüyor, sınırını belirliyor ve
> telefon kestirimini düzeltmeyi öğretiyoruz.

**Bitirme projesi · 2026–2027**

## Hızlı başlangıç

```bash
make kurulum     # bağımlılıkları kur
make kontrol     # ortam doğru mu?
make test        # testler (atlananlar nedeniyle listelenir)
make reproduce   # 9 deneyin sayı ve şekilleri + sayı kilidi (~90 s)
```

`make kontrol` "Her şey yerinde" demeden çalışmaya başlamayın.

**Birebir aynı ortam** için tam sürümler `requirements-lock.txt`'te:
`pip install -r requirements-lock.txt`. 23 Eylül 2026'da boş bir sanal ortamda
bu dosyayla kurulum, 510 test ve `make reproduce` (2758 sayının hepsi kilitle
aynı) doğrulandı. Kilit `make kilit-surum` ile yenilenir.

Geliştirme: `make kurulum-dev` (ruff, pytest-cov), `make lint`, `make kapsam`.

**Demo** (isteğe bağlı model ortamında, bkz. `docs/kararlar/0030`, `0031`):
`python -m mono.demo --video v.mp4 --model m.task --intrinsics K.json --lengths L.json --output out/demo`
→ işaretli video + kare başına iki form kararı (yalın eşik / belirsizliği bilen).
Ayrıntı ve örnek sonuç: `docs/deney/2026-09-23-telefon-hatti-uctan-uca.md`.

## Katmanlar

| Katman | Rol | Hedef hata |
|---|---|---|
| Profesyonel (çoklu kamera) | Yer gerçeği olarak kullanılacak | hedef < 10 mm (Kapı 2), fiziksel olarak doğrulanmadı |
| Tüketici (tek telefon) | Ölçülen nesne | ~150–250 mm (literatür) |
| Düzeltme katmanı | Katkımız | Ölçülecek |

## Dizin yapısı

| Dizin | İçerik |
|---|---|
| `calib/` | Kalibrasyon: ChArUco tespiti, `calibrateMultiview` sarmalayıcı |
| `uncertainty/` | Monte Carlo, kovaryans, kalibrasyon kararlılığı |
| `capture/` | Senkron kayıt, oturum yönetimi |
| `pose3d/` | Çoklu görüşten poz + triangulation |
| `mono/` | Telefon (tek kamera) hattı |
| `correction/` | Düzeltme modeli |
| `eval/` | **Tüm metrikler** — tek giriş noktası |
| `scripts/` | Yardımcı betikler |
| `docs/kararlar/` | Teknik karar kayıtları (neden böyle yaptık) |
| `docs/toplanti/` | Haftalık toplantı tutanakları |
| `data/` | Ham veri — **git'e girmez** |

## Önce okuyun

| Belge | Ne için |
|---|---|
| [PROJE-PLANI.md](PROJE-PLANI.md) | Takvim, kapılar, görev dağılımı, riskler |
| [KATKI.md](KATKI.md) | Nasıl çalışıyoruz: dal, commit, inceleme, geri alma |
| [KOD-PLANI.md](KOD-PLANI.md) | Ne yazacağız, hangi sırayla, hangi kabul testiyle |
| [docs/kararlar/](docs/kararlar/) | Neden bu teknolojiyi seçtik |

## Durum

**Faz 0 — Karar ve kurulum.** Ayrıntı için `PROJE-PLANI.md`.

Depo iskeleti kurulu, ortam doğrulandı (Python 3.13 + OpenCV 5.0.0, `make kontrol` yeşil).
İlk ölçüm modülü eklendi: `eval.metrics.reprojection_error`; 15 test geçti.
`make reproduce` bilinen yapay nokta hatalarının JSON özetini üretir; gerçek
kalibrasyon raporu değildir. [Geliştirme kaydı](docs/deney/2026-09-22-metric-baseline.md).
Sıradaki iş: ChArUco üretimi ve tespiti, ardından sentetik kalibrasyon doğrulaması.

## Hazır teknoloji entegrasyonu

Roboflow, Supervision, hazır poz modelleri ve diğer araçların görevleri ile
ortak veri sözleşmesi [HAZIR-TEKNOLOJILER.md](HAZIR-TEKNOLOJILER.md) içinde.
Yeni modül veya bağımlılık eklemeden önce bu belgeyi okuyun. Hazır araçları
uygun yerde kullanın; model çıktısını kamera/kare kimliği ve koordinat sistemiyle
birlikte aktarın. Aday araçların entegrasyonu henüz tamamlanmış değildir.
