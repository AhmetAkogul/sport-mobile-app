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
`settings` deney sırasında kullanılan ayarların açıklamasıdır; sürücü ayarlarını
uygulamaz veya doğrulamaz. Katılımcı kodu, ışık, kamera ayarı, kaynak ve benzersiz
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
`batches/*/frames.json` esas alınır. Otomatik kurtarma henüz yoktur.

`completed`, istenen sayıda grubun kaydedildiğini söyler. Kaynak sonu/kopması
OpenCV `grab` sonucundan ayırt edilemediğinden `source_stopped` olarak kaydedilir;
komut çıkış kodu 2 olur. Çözme/yazma hatası `failed`, Ctrl+C `interrupted` olur.

## Zaman ve sınırlar

`grab_started_ns` / `grab_finished_ns` aynı süreçteki monoton bilgisayar saatidir.
Sensör pozlama zamanı değildir; video dosyalarında video PTS'si değildir.
Aynı grup indeksi ve eşit kare sayısı fiziksel eşzamanlılık kanıtlamaz. Donanım
zaman damgası, tetikleme, video PTS eşleştirmesi ve gerçek senkron kaymasının
ölçülmesi sonraki aşamadır. Canlı kaynakta sürücü çağrısı kilitlenirse zaman aşımı
henüz uygulanmaz. PNG hattı yüksek hızlı/uzun süreli kayıt için doğrulanmadı.

## 22 Eylül 2026 geliştirme kaydı

Kod planının Adım 3 kapsamındaki ilk kayıt altyapısı geliştirildi.
`tests/test_capture.py`: **25 test geçti**. Kontroller: zorunlu meta veri,
benzersiz kaynaklar, eşit kare sayıları, zaman damgaları, kısa kaynakta durma,
açma/çözme/yazma hataları, kesinti sonrası tam grupların korunması, üzerine
yazmayı engelleme, grab/retrieve sırası ve iki gerçek OpenCV video okuyucusuyla
sentetik MJPG videoların okunup PNG olarak geri doğrulanması.

Fiziksel kamera, insan katılımcı veya gerçek senkronizasyon testi yapılmadı.
`calib/` ve ortak plan dosyaları bu çalışma kapsamında değiştirilmedi.
