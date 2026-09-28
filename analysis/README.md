# Çoklu Kamera 3B Hareket Analizi

> Telefon uygulamaları egzersiz formunu, ölçemeyecekleri bir hassasiyetle değerlendiriyor.
> Biz bu hatayı kalibre edilmiş çoklu kamera düzeneğiyle ölçüyor, sınırını belirliyor ve
> telefon kestirimini düzeltmeyi öğretiyoruz. Ürün olarak: mobil uygulama kullanıcıya
> yalnız ölçebildiği koşulda geri bildirim veriyor, salon kameraları problemli hareketi
> antrenöre bildiriyor (`docs/kararlar/0073`).

**Bitirme projesi · 2026–2027**

## Ürün hedefi

Proje iki bağlı ürün yüzeyinden oluşur:

- **Mobil uygulama:** Kullanıcı ön veya arka telefon kamerasıyla egzersiz yaparken
  kadraj, bakış açısı ve ölçülebilirlik rehberi alır; sistem yalnızca desteklenen
  koşullarda form geri bildirimi verir, aksi durumda açıkça "ölçülemez" der.
- **Spor salonu sistemi:** Sabit çoklu kamera düzeneği istasyon/anonim kişi
  düzeyinde hareketi analiz eder ve problemli ölçüm veya form olayını antrenöre
  bildirir. Üyelik/kişi tanıma ilk sürümün dışındadır; sonraki fazda ayrıca
  etik, güvenlik ve açık rıza incelemesi gerektirir.

Mobil ve salon arayüzleri aynı ölçüm, belirsizlik ve olay sözleşmesini kullanır.
Salon bildirimi klinik teşhis değildir; "inceleme önerilir", "ölçülemez" veya
"belirli form koşulu gözlendi" sınırlarında kalır.

## Hızlı başlangıç

```bash
make kurulum     # bağımlılıkları kur
make kontrol     # ortam doğru mu?
make test        # testler (atlananlar nedeniyle listelenir)
make reproduce   # 9 deneyin sayı ve şekilleri + sayı kilidi (~90 s)
```

`make kontrol` "Her şey yerinde" demeden çalışmaya başlamayın. Tam test ve sayı
kilidi yalnız belgelenen `.venv` ortamında geçerli kabul edilir; sistem Python'ı
ile yapılan kısmi çalıştırmalar kanıt sayılmaz.

**Birebir aynı ortam** için tam sürümler `requirements-lock.txt`'te:
`pip install -r requirements-lock.txt`. Kurulumdan sonra `make test` ve
`make reproduce` çalıştırılmalı; sonuçlar tarihli deney kaydıyla birlikte
raporlanır. Kilit `make kilit-surum` ile yenilenir.

Geliştirme: `make kurulum-dev` (ruff, pytest-cov), `make lint`, `make kapsam`.

**Demo** (isteğe bağlı model ortamında, bkz. `docs/kararlar/0030`, `0031`):
`python -m mono.demo --video v.mp4 --model m.task --intrinsics K.json --lengths L.json --output out/demo`
→ işaretli video + kare başına iki form kararı (yalın eşik / belirsizliği bilen).
Ayrıntı ve örnek sonuç: `docs/deney/2026-09-23-telefon-hatti-uctan-uca.md`.

**Canlı kamera:** `make canli` (MediaPipe ortamı ve modeli `MP_PYTHON`, `MP_MODEL`
ile değiştirilebilir; kayıt için `ARGS="--kaydet out/deneme.mp4"`). Bilgisayar
kamerasından squat: gövde açısı, açıya göre düzeltilmiş valgus, tekrar kararı;
60°'nin üstünde "bu açıdan ölçülemez". Kameraya önden durun. Araştırma denemesi:
gerçek salon videolarında "dizler içe"yi ayıramadı (`docs/deney/2026-09-28-profil-ve-etiketsiz-yon.md`).

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

**Durum — gerçek veride telefon hattı ve ürün kapsamı.** Araştırma çekirdeği
REHAB24-6, EC3D ve Fitness-AQA üzerinde sınanıyor. Mobil kullanıcı akışı ile
salon antrenör olay akışı ürün yüzeyi olarak tanımlandı; fiziksel çoklu kamera
referansı ve Kapı 4 henüz tamamlanmış kabul edilmiyor. Ayrıntı için
`PROJE-PLANI.md` ve `docs/kararlar/0073-mobil-ve-salon-urun-kapsami.md`.

Ortam: Python 3.13 + `opencv-contrib-python` 5.0.0.93, `.venv` içinde (`make kontrol`).
Test sayısı ve güncel bulgular için tek kaynak `CLAUDE.md` "Şu anki durum" bölümüdür
(sayı burada tekrarlanmaz ki eskimesin). Çalışan uçtan uca parçalar: `make canli`
(tek kişi, tam vücut), `make canli-coklu` (çok kişi, kimlik, hareket etiketi),
`make antrenor` (hareket tanıma, tekrar sayımı, tekrar kararı, ipucu; 0035–0040).

## Hazır teknoloji entegrasyonu

Roboflow, Supervision, hazır poz modelleri ve diğer araçların görevleri ile
ortak veri sözleşmesi [HAZIR-TEKNOLOJILER.md](HAZIR-TEKNOLOJILER.md) içinde.
Yeni modül veya bağımlılık eklemeden önce bu belgeyi okuyun. Hazır araçları
uygun yerde kullanın; model çıktısını kamera/kare kimliği ve koordinat sistemiyle
birlikte aktarın. Aday araçların entegrasyonu henüz tamamlanmış değildir.
