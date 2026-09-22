# Kayıt altyapısı

`session.py` oturum meta verisini doğrular; `record.py` yapılandırmadaki N kamera
veya video dosyasından eksiksiz kare grupları kaydeder. Donanım gerektirmeyen
testler sentetik görüntü ve video kullanır; insan veya canlı kamera kaydı yapılmaz.

## Kullanım

Proje kökünden, hazırlanmış yapılandırma ile:

```bash
.venv/bin/python -m capture.record --config data/session-config.json --output data/sessions/deneme-001 --max-frames 100
```

Örnek yapılandırma (dosya yolları çalıştırılan dizine göredir):

```json
{
  "participant_code": "synthetic",
  "lighting": "Sentetik sabit görüntü",
  "notes": "İnsansız kayıt testi",
  "cameras": [
    {"camera_id": "left", "source": "data/left.avi", "settings": "640x480, 30 fps, sentetik"},
    {"camera_id": "right", "source": "data/right.avi", "settings": "640x480, 30 fps, sentetik"}
  ]
}
```

`source` tamsayı olduğunda kamera indeksi, metin olduğunda OpenCV kaynağıdır.
`settings` deney sırasında kullanılan ayarların açıklamasıdır. İsteğe bağlı
`width`, `height`, `fps` alanları sürücü ayarlarını uygular/doğrular; ayrıntı aşağıdadır. Katılımcı kodu, ışık, kamera ayarı, kaynak ve benzersiz
kamera kimliği olmadan kayıt başlamaz. Gerçek isim yerine kod kullanılır.

## Çıktı sözleşmesi

- `session.json`: sürüm, yapılandırma, UTC başlangıç/bitiş, durum ve grup sayısı.
- `batches/000000/<camera_id>.png`: her kameranın aynı grup indeksindeki karesi.
- `batches/000000/frames.json`: grubun yolları, boyutları ve zaman damgaları.
- `frames.jsonl`: tamamlanan grupların satır bazlı indeksi.

Çıktı dizini yeni olmalıdır; mevcut oturum üzerine yazılmaz. Her kaynak için önce
`grab`, ardından tüm kaynaklar için `retrieve` çağrılır. Bir grubun bütün PNG'leri
ve meta verisi yazıldıktan sonra grup dizini atomik taşımayla görünür olur.
Yazma/çözme hatasında yarım grup silinir; önceki tam gruplar korunur ve kaynaklar
serbest bırakılır. Beklenmedik süreç/güç kesilmesinde `.pending` kalabilir;
tamamlanmış kayıt sayılmaz. Disk dayanıklılığı için fsync garantisi verilmez.
İndeks yazılırken süreç kesilirse tamamlanmış grup indeks dışında kalabilir;
`batches/*/frames.json` esas alınır. Kurtarma için aşağıdaki yeni dizine aktarım aracı kullanılır.

`completed`, istenen sayıda grubun kaydedildiğini söyler. Kaynak sonu/kopması
OpenCV `grab` sonucundan ayırt edilemediğinden `source_stopped` olarak kaydedilir;
komut çıkış kodu 2 olur. Çözme/yazma hatası `failed`, Ctrl+C `interrupted` olur.

## Zaman ve sınırlar

`grab_started_ns` / `grab_finished_ns` aynı süreçteki monoton bilgisayar saatidir.
Sensör pozlama zamanı değildir; video dosyalarında video PTS'si değildir.
Aynı grup indeksi ve eşit kare sayısı fiziksel eşzamanlılık kanıtlamaz. Donanım
zaman damgası, tetikleme, video PTS eşleştirmesi ve gerçek senkron kaymasının
ölçülmesi sonraki aşamadır. Canlı kaynakta sürücü çağrısı kilitlenirse ayrı kaynak süreci süre sınırıyla
sonlandırılır; ayrıntı aşağıdadır. PNG hattı yüksek hızlı/uzun süreli kayıt için doğrulanmadı.

## 22 Eylül 2026 geliştirme kaydı

Kod planının Adım 3 kapsamındaki ilk kayıt altyapısı geliştirildi.
`tests/test_capture.py`: **25 test geçti**. Kontroller: zorunlu meta veri,
benzersiz kaynaklar, eşit kare sayıları, zaman damgaları, kısa kaynakta durma,
açma/çözme/yazma hataları, kesinti sonrası tam grupların korunması, üzerine
yazmayı engelleme, grab/retrieve sırası ve iki gerçek OpenCV video okuyucusuyla
sentetik MJPG videoların okunup PNG olarak geri doğrulanması.

Fiziksel kamera, insan katılımcı veya gerçek senkronizasyon testi yapılmadı.
`calib/` ve ortak plan dosyaları bu çalışma kapsamında değiştirilmedi.

## Model entegrasyonuna hazırlık

`capture.reader.iter_batches(session_dir)` her adımda eksiksiz bir kamera grubu
verir. Her `RecordedFrame`, özgün BGR uint8 görüntüyü, kamera kimliğini, grup
indeksini, oturum yolunu ve zaman damgasının anlamını taşır. RGB isteyen model
adaptörü dönüşümü kendisi yapar; boyut değiştiriyorsa ters dönüşümü saklar.

Okuyucu grup sayısı/sırası, kamera kimlikleri, görüntü boyutu ve zamanların
sırasını doğrular. Bozuk bir gruptan kısmi sonuç vermez. Kapatılmış oturum
gerektirir; eksik JSONL indeksi yerine grup meta verisini okur. Bu işlem
çökmüş oturumu otomatik onarma değildir; bunun için `capture.recover` kullanılır.

22 Eylül ek doğrulama: 14 okuyucu testi geçti; 25 kayıt ve 15 metrik testiyle
birlikte **54 test başarılı**. Hazır model/Roboflow bağlantısı henüz uygulanmadı;
ortak aktarım sınırı hazırlandı. Entegrasyon planı: [HAZIR-TEKNOLOJILER.md](../HAZIR-TEKNOLOJILER.md).

## Kesilen oturumu kurtarma

Önce kayıt işlemini durdurun. Yeni bir çıktı dizini seçin:

```bash
python -m capture.recover --source data/sessions/kesilen --output data/sessions/kurtarilan
```

Araç tam grup dizinlerini kopyalayıp görüntüleri ve meta veriyi okuyucu üzerinden
kontrol eder, JSONL indeksini yeniden kurar. Özgün kayda dokunmaz; `.pending`
içeriğini değiştirmeden dışarıda bırakır. Bozuk bir tam grubu sessizce atlamaz;
çıktı yayımlamadan hata verir. Mevcut çıktı dizinine yazmaz. İşlem sırasında
kaynak meta verisi veya grup listesi değişirse durur. Bu kontrol canlı kayıt
kilidi değildir; çalışmakta olan kaydı kurtarmak desteklenmez.

Kurtarılan oturum `recovered` durumundadır. Özgün meta veri
`recovery-original-session.json` içinde korunur. Kurtarma zamanı ayrı yazılır;
bilinmeyen özgün bitiş zamanı uydurulmaz. Kurtarma kaydı sürdürmez ve kayıp
kareleri geri getirmez. Tüm kayıt kopyalandığı için ek disk alanı gerekir.
Yeni oturumun `session.json` dosyası en son yayımlanır; yayın sırasında güç
kesilirse eksik çıktı otomatik temizlenmez. Özgün kaynak yine korunur.

## Tamamlanma durumu — 22 Eylül ek çalışması

- Kayıt ve model için okuma: sentetik kaynaklarla doğrulandı.
- Kesilen kayıttan ayrı dizine kurtarma: kısmi indeks, eski grup sayısı,
  bozuk görüntü, eksik grup, bağlantılı dosya, mevcut hedef, kaynak değişikliği
  ve komut satırı üzerinden sınandı.
- Kamera adlarının büyük/küçük harfle dosya çakıştırması engellendi.
- Okuyucuyla uyumsuz görüntü türleri kayıt aşamasında reddediliyor.
- Toplam **68 test geçti**: kayıt 29, okuyucu 14, kurtarma 10, metrik 15.

Capture bütünüyle tamamlanmış değildir. Açık kabul işleri:

- [x] Kaynak çağrılarında süreç tabanlı zaman aşımı ve sonlandırma; sentetik kilitlenmelerle doğrulandı. Gerçek sürücü testi açık.
- [x] Çözünürlük/FPS uygulama ve sürücüden geri okuma altyapısı. Gerçek kamera testi açık.
- [x] Yerel videoları OpenCV POS_MSEC ve açık ofsetlerle eşleştiren ayrı plan üretimi. Eşleşmiş görüntü okuyucusu eklendi; doğrudan PTS okuyucu ve VFR doğrulaması henüz yok.
- [ ] Donanım/sensör zaman damgaları ve gerçek kameralar arası kayma ölçümü.
- [x] Dört yerel kaynakla 300 grup/1200 görüntü kaydı ve sentetik disk dolması hata testleri.
- [ ] Uzun süreli gerçek kamera yükü, gerçek disk dolması ve fiziksel kesinti deneyi.

Bu turda gerçek kamera açılmadı; fiziksel kabul kapıları kapanmadı.

## Kaynak zaman aşımı — 22 Eylül güncellemesi

Varsayılan kayıt yolu her kamera/video kaynağını `spawn` ile ayrı süreçte açar.
`capture/source.py` OpenCV'nin hazır `VideoCapture` API'sini kullanır. Ana süreç
kayıt/indeks yazımını yönetir; kaynak süreci yalnızca görüntü alma işini yapar.
Sürücü bir çağrıda kilitlendiğinde Python iş parçacığı iptaline güvenmek yerine
kaynak süreci sonlandırılır. Büyük görüntünün IPC üzerinden alınması da ana
süreçte bloklayan bir `recv` ile yapılmaz; yanıt süre sınırıyla beklenir.

Varsayılanlar: açılma **15 saniye**, her grab/retrieve/release **5 saniye**.
Komut satırında `--open-timeout` ve `--source-timeout`; Python API'sinde
`open_timeout_s` ve `source_timeout_s` ile ayarlanır. Başlangıç süresi Python alt
sürecinin yüklenmesini de içerir. Süre dolduğunda sonlandırma/join adımları kısa,
sınırlı ek beklemeler yapabilir; bu gerçek zamanlı işletim sistemi garantisi değildir.

Python betiğinde kayıt çağrısını `if __name__ == "__main__":` altında yapın.
Enjekte edilen `capture_factory` doğrudan çağrılır; test/özel adaptör kullananın
zaman aşımı sorumluluğu kendisindedir. Meta veride bu yol `custom_factory`,
varsayılan yol `process` olarak belirtilir. Canlı cihazlar için varsayılanı kullanın.

Zaman aşımı oturumu `failed` yapar; daha önce tamamlanan gruplar korunur.
Kapanma hataları da `cleanup_errors` alanında saklanır; başarılı bitmiş gibi
raporlanmaz. `release` tekrar çağrılabilir. Grab zaman aralığı ana süreçteki
istek/yanıtı kapsar, IPC gecikmesini de içerir; pozlama zamanı değildir.

Doğrulama: **84 test geçti**. Önceki 68 teste eklenen 16 test; açma/grab/retrieve/
release kilitlenmeleri, alt süreç hatası ve ani çıkışı, süreç ve alıcı iş parçacığı
kapanması, hatalı süreler, korunmuş tam gruplar, kapanma hatası ve gerçek MJPG video
ile komut satırından süreç yalıtımlı kaydı kapsıyor. Gerçek kamera çalıştırılmadı.
Disk yazımındaki kilitlenmelere süre sınırı ve sürücü/işletim sistemi düzeyinde
fiziksel cihazın yeniden açılabilirliği bu doğrulamanın kapsamında değildir.

## Çözünürlük ve FPS — 22 Eylül güncellemesi

Her kamera yapılandırmasına isteğe bağlı `width`, `height`, `fps` eklenebilir:

```json
{"camera_id": "left", "source": 0, "settings": "Hedef 640x480 / 30 fps", "width": 640, "height": 480, "fps": 30}
```

Eski yapılandırmalar geçerlidir. Belirtilmeyen değerler değiştirilmez; sürücüden
okunabiliyorsa raporlanır. `source` tamsayı kamera indeksiyse ayarlar uygulanır.
Metin dosya/URL kaynağında çözünürlük ve FPS değiştirilmez; girilen değerler
beklenti olarak doğrulanır. Böylece video zaman çizelgesi değiştirilmiş gibi
raporlanmaz. Metin cihaz yollarında ayar uygulama bu sürümde desteklenmez.

Tüm ayarlar uygulandıktan sonra değerler tekrar okunur. Sürücü `set` isteğini
reddederse veya istenen değer doğrulanamazsa oturum kare kaydetmeden `failed`
olur. FPS için tolerans %1 veya 0.1 fps (büyük olan), boyutlar için tam eşitliktir.
Gelen her görüntünün boyutu da istenen boyutlarla karşılaştırılır. Yanlış boyutta
bir kare grubun tamamını durdurur; önceki tam gruplar korunur.

`session.json` içindeki `camera_settings`, kamera başına `requested`, `reported`,
`set_accepted`, `mode` ve `issues` alanlarını içerir. Sıfır/NaN/sonsuz sürücü
okumaları bilinmeyen (`null`) olarak saklanır. FPS hâlâ sürücü beyanıdır;
ölçülmüş kare üretim hızı veya fiziksel senkronizasyon sonucu değildir.
Pozlama, kazanç ve otomatik odak ayarları henüz uygulanmaz.

Özellik okuma/yazma çağrıları da kaynak sürecinde zaman aşımına tabidir.
**106 test geçti:** ayar kabulü/reddi, ayarların birbirini değiştirmesi, yanlış
kare boyutu, FPS toleransı, geçersiz sayılar, alt süreçte get/set kilitlenmesi
ve gerçek MJPG dosyasından çözünürlük/FPS okuma doğrulandı. Gerçek kamera açılmadı.

## Farklı FPS videoları eşleştirme — 22 Eylül güncellemesi

`capture.alignment` yerel video dosyalarını tarar ve kaynak kare indekslerini
zamanlarına göre eşleyen JSON planı üretir. Mevcut `record_session` aynı turda
kare alma davranışını korur; bu ayrı çevrimdışı araçtır. Kaydı otomatik yeniden
örneklemez ve video/PNG üretmez. Plan `open_aligned_batches` ile görüntü gruplarına çevrilir.

```json
{
  "videos": [
    {"camera_id": "left", "path": "data/left.avi"},
    {"camera_id": "right", "path": "data/right.avi"}
  ],
  "reference_camera": "left",
  "offsets_ms": {"left": 0, "right": 125},
  "tolerance_ms": 20
}
```

```bash
python -m capture.alignment --config data/alignment-config.json --output data/alignment.json
```

Ortak zaman = dosya zamanı + ofset. Örnekte right dosyasının sıfır zamanı ortak
çizelgede 125 ms olur. Ofsetler kullanıcı/deney tarafından belirlenir; sıfır da
bilinçli olarak yazılmalıdır. Bu araç ofseti görüntüden bulmaz veya fiziksel
senkronizasyonu doğrulamaz. Tolerans hareket hızına göre deneyde seçilmelidir.

Zaman, çözülen kareden sonra OpenCV `CAP_PROP_POS_MSEC` ile alınır. Bu,
kapsayıcıdan doğrudan PTS okuma garantisi değildir. Tekrarlanan, geriye giden,
negatif veya sonlu olmayan değerlerde araç hata verir; FPS üzerinden zaman
uydurmaz. Zaman desteğini sınamak için en az iki kare gerekir. `--max-frames`
varsayılanı 100000; sınır aşılırsa kısmi başarı raporu yerine hata üretilir.
OpenCV grab=false kaynak sonu ve bazı okuma hatalarını ayırt edemeyebilir;
bozuk/kesik video dosyalarının tamlığını bu araç kanıtlamaz.

Referans kareler sırayla ele alınır; diğer kaynaklarda en yakın kullanılmamış
kare seçilir. Eşit uzaklıkta erken kare tercih edilir. Tam grubun en erken/en
geç zaman farkı tolerans içindeyse kabul edilir. Kare tekrarı ve interpolasyon
yoktur. Açgözlü yöntemdir; küresel en fazla eşleşme veya en küçük hata garantisi
yoktur. Daha düşük FPS'li kaynağı referans seçmek genellikle daha az boş aday
üretir; referans otomatik seçilmez.

JSON planı kaynak yollarını, kare sayılarını, ofsetleri, eşleşen grup/kareleri,
eşleşmeyen referans karelerini ve tüm kullanılmayan kare indekslerini içerir.
Sıfır grup çıkarsa rapor yine yazılır, komut çıkış kodu 2 olur. Var olan raporun
üzerine yazılmaz. Plan sürüm 2 kaynak videoların SHA-256 içerik kimliklerini taşır; videoları
kopyalamaz. Okuyucu kimlikleri doğrular. Planı kullanırken özgün videoları değiştirmeyin.

24 yeni test geçti: 10/20 FPS gerçek MJPG dosyaları ve komut satırı, ofset,
eşleşme toleransı, tüm grup zaman farkı, tekrar kullanmama, geçersiz zamanlar,
tarama sınırı ve üzerine yazmayı engelleme. VFR/telefon kodekleri ve gerçek
kamera zaman kayması henüz test edilmedi.

## Eşleşmiş görüntüleri modele aktarma — 22 Eylül güncellemesi

```python
from capture.aligned_reader import open_aligned_batches

# Bağımsız betikte bu çağrıyı if __name__ == "__main__": altında çalıştırın.
with open_aligned_batches("data/alignment.json") as batches:
    for frames in batches:
        for frame in frames:
            # Model adaptörüne frame.image_bgr ve kimlik/zaman alanlarını aktarın.
            print(frame.camera_id, frame.source_frame_index, frame.aligned_time_ms)
```

Her adım tüm kameraların `AlignedFrame` demetidir. Görüntü özgün boyutta BGR
uint8 olarak gelir. Kamera kimliği, kaynak dosya yolu, grup numarası, kaynak
kare indeksi, dosya zamanı ve ofsetli ortak zaman korunur. Hazır model adaptörü
RGB dönüşümü/yeniden boyutlandırmayı kendi sözleşmesine göre yapmalıdır.
Bu değişiklik henüz MediaPipe, Roboflow veya başka bir modeli çalıştırmaz.

Yeni taramalar video içeriğinin SHA-256 özetini tarama öncesi/sonrası hesaplar;
plan sürümü **2** oldu. Eski veya elle oluşturulmuş içerik kimliksiz planlar
okuyucu tarafından reddedilir; `capture.alignment` ile yeniden üretilmelidir.
Okuyucu tüm planı ve kaynak içerik kimliklerini ilk görüntüden önce doğrular.
Kareler rastgele seek ile değil baştan sıralı çözülür. Seçili karelerin POS_MSEC
zamanı plana karşı **0.001 ms** toleransla kontrol edilir; bu sayısal tutarlılık
kontrolüdür, sensör hassasiyeti iddiası değildir.

Yanlış kimlik/sıra, tekrar kullanılmış indeks, eksik kamera, tutarsız ofset,
tolerans aşımı, erken dosya sonu veya yanlış kare zamanı hata verir. Bir grubun
kısmi görüntüleri sunulmaz; önceki tam gruplar daha önce tüketilmiş olabilir.
`with` bloğu erken bitse veya hata olsa da kaynaklar kapanır. Çözme/get çağrıları
mevcut süreç zaman aşımı altyapısını kullanır. Dosyalar kullanım sırasında sabit
kalmalıdır: başlangıçta tam içerik özeti, gruplar arasında boyut/mtime/inode
kontrolü yapılır; her karede dosyanın tamamı yeniden hash edilmez.

Gerçek 10/20 FPS MJPG dosyalarında yalnız indeksleri değil seçilen görüntülerin
piksel değerlerini de doğruladık. Tarama → plan → görüntü okuma uçtan uca testi,
plan bozulması, kaynak değişmesi, zaman uyuşmazlığı, erken dosya sonu, boş plan
ve erken çıkışta kaynakların kapanması sınandı. VFR, telefon kodekleri ve fiziksel
kamera senkronizasyonu hâlâ açık kabul işleridir.

Bu güncelleme sonunda capture ve metrik testlerinin toplamı: **147 geçti**.

## Dört kaynaklı doğrulama ve disk hataları — 22 Eylül güncellemesi

```bash
python -m capture.validate --output data/yeni-dogrulama --cameras 4 --frames 300
```

Araç yeni çıktı dizininde numaralı MJPG videolar oluşturur, süreç yalıtımlı kayıt
hattından geçirir ve kaydedilen PNG'leri kaynak videolardan yeniden çözülen
karelerle piksel düzeyinde karşılaştırır. Kaynaklar ve kayıtlar data altında
kalır. `validation.json` yalnızca tüm kontroller geçince yazılır. Mevcut dizine
tekrar yazılmaz. Kodlayıcı kayıpları nedeniyle karşılaştırma ham sentetik resme
değil, kaynak videonun çözülen resmine karşı yapılır.

Bu makinedeki koşu: 4 kaynak, 320×240, kaynak başına 300 kare; **1200 görüntü
birebir doğrulandı**. Kayıt aşaması 1.138 saniye, kayıt dosyaları 6,225,847 bayt.
Bu süre kaynak üretimini/son doğrulamayı içermez; önceden kaydedilmiş dosyalar
çevrimdışı okundu. Sonuç gerçek kamera FPS'si, senkron kayması veya uzun süreli
kararlılık ölçümü değildir.

Disk hata testlerinde ikinci grupta ENOSPC, ardından son meta veri yazımında
ENOSPC; yarım grup temizleme izni reddi; başarılı kare yazımından sonra son durum
kaydı hatası enjekte edildi. Asıl hata artık kapanış hatasıyla gizlenmez; ek hata
exception notu olarak korunur. Kaynaklar serbest bırakılır, tam gruplar korunur.
Disk gerçekten doluysa son durum dosyası yazılamayabilir ve oturum `recording`
görünebilir. Bu durumda yer açıldıktan sonra `capture.recover` ile yeni dizine
kurtarma yapılır. Güç kesilmesi ve gerçek disk doldurma denenmedi.

Bu tur sonunda capture/metrik testlerinin toplamı **150 geçti**. Ayrıca dört
kaynaklı doğrulama koşusu başarılı. Var olan `.venv` bağlantısı değiştirilmedi.
