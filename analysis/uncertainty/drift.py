"""Kalibrasyon suruklenmesi: ayni duzenek haftalar sonra hala guvenilir mi?

Tezin ikinci ozgun kaniti (docs/kararlar/0006): kalici kurulumda kalibrasyonun
**zaman icindeki suruklenmesi**. Bir toolbox tek oturumun belirsizligini verir
(`uncertainty.montecarlo`); "bu kalibrasyona kac hafta guvenilir" sorusunu hicbir
arac olcmez. Cevap, ayni duzenekte tekrarlanan kalibrasyon kayitlarinin
(`calib.io.Kabin`) zaman serisinden cikar.

Veri akisi (haftalik ritim, PROJE-PLANI.md):

1. Ayni board, ayni ayarlarla kisa kalibrasyon oturumu (~15 dk).
2. `kalibre_et` -> `calib.io.kalibrasyondan(...)` -> `calib.io.yaz(...)`
   -- tarih/surum damgali JSON diske duser (`data/kalibrasyon/` altina).
3. Bu modul, kayitlari zaman sirasina dizer ve parametre bazinda sapma yuzdesi
   uretir. Kapi 2 olcutu ">= 4 haftalik kayit" buradan sinanir.

Tasarim: olcumler **kamera 0 intrinsics (fx, fy, cx, cy) + taban_0i**.
Kamera 0 referans oldugu icin mutlak T'si her zaman 0'dir ve bilgi tasimaz
(`calib.multiview` sozlesmesi); gercek geometrik degisim taban uzunluklarinda
ve kamera 0'in kendi ic parametrelerinde gorunur. Duzenek degismisse (kamera
sayisi/parametre kumesi) seriler karsilastirilamaz -- sessiz karsilastirma
yerine acik hata verilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from calib.io import Kabin


def tarih_coz(iso: str) -> datetime:
    """ISO 8601'i siralanabilir, saat dilimi bilinen tarihe cevir.

    Iki tuzak kapaniyor: 'Z' soneki `fromisoformat` icin normallestiriliyor, ve
    saat dilimi tasimayan kayit UTC kabul ediliyor. Ikincisi olmazsa, kayitlarin
    bir kismi dilimli bir kismi dilimsizse siralama TypeError ile patlar --
    gercek kayit klasorlerinde karisik format sik gorulur.
    """
    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _olcumler(kabin: Kabin) -> dict[str, float]:
    """Kayittan izlenen parametreler (`montecarlo._olcumler` ile ayni sozlesme)."""
    if not kabin.Ks:
        raise ValueError("kabinde kamera yok; olcum yapilamaz")
    K = np.asarray(kabin.Ks[0], dtype=np.float64)
    m = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
         "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    for i in range(1, len(kabin.Ts)):
        m[f"taban_0{i}"] = float(np.linalg.norm(np.asarray(kabin.Ts[i])))
    return m


@dataclass
class SuruklenmeEgrisi:
    """Zaman sirali parametre serileri.

    `olcumler[p]`, `tarihler` ile ayni sirada k deger tasiyor. Parametre
    adlari ilk kayittan gelir; her kayit ayni kumeyi tasimak zorundadir.
    """

    tarihler: tuple[str, ...]
    olcumler: dict[str, list[float]]

    @property
    def n_nokta(self) -> int:
        return len(self.tarihler)

    def parametreler(self) -> tuple[str, ...]:
        return tuple(self.olcumler)

    def seri(self, param: str) -> np.ndarray:
        if param not in self.olcumler:
            raise KeyError(f"'{param}' izlenmiyor: {sorted(self.olcumler)}")
        return np.asarray(self.olcumler[param], dtype=np.float64)

    def sapma_yuzdesi(self, param: str, baz_indeks: int = 0) -> np.ndarray:
        """Her noktanin baz noktaya gore yuzde sapmasi.

        Sifir veya yakin-sifir parametrede yuzde tanimsizdir; cx/cy pratikte
        goruntu merkezi civarindadir ama genel guvenlik icin sifira yakin
        baz hata uretir.
        """
        dizi = self.seri(param)
        baz = float(dizi[baz_indeks])
        if abs(baz) < 1e-9:
            raise ValueError(f"'{param}' baz degeri sifira yakin ({baz}); "
                             "yuzde sapma tanimsiz")
        return (dizi - baz) / abs(baz) * 100.0

    def tarih_araligi_gun(self) -> float:
        """Ilk ve son olcum arasindaki gun sayisi (>= 0)."""
        if self.n_nokta < 2:
            return 0.0
        t = sorted(tarih_coz(x) for x in self.tarihler)
        return (t[-1] - t[0]).total_seconds() / 86400.0

    def en_buyuk_suruklenme(self) -> tuple[str, float]:
        """(parametre, bazdan en buyuk mutlak sapma yuzdesi). Hizli rapor icin."""
        en_iyi: tuple[str, float] | None = None
        for p in self.parametreler():
            try:
                boyut = float(np.max(np.abs(self.sapma_yuzdesi(p))))
            except ValueError:
                continue
            if en_iyi is None or boyut > en_iyi[1]:
                en_iyi = (p, boyut)
        if en_iyi is None:
            raise ValueError("yuzdeyle ifade edilebilir parametre yok")
        return en_iyi

    def ozet(self) -> dict:
        """JSON'a donebilir ozet: parametre basina ilk/son/sapma."""
        params = {}
        for p in self.parametreler():
            dizi = self.seri(p)
            try:
                sapma = self.sapma_yuzdesi(p)
                sapma_yuzde = float(sapma[-1])
                en_buyuk = float(np.max(np.abs(sapma)))
            except ValueError:
                sapma_yuzde = None
                en_buyuk = None
            params[p] = {"ilk": float(dizi[0]), "son": float(dizi[-1]),
                         "sapma_yuzde": sapma_yuzde, "en_buyuk_sapma_yuzde": en_buyuk}
        return {
            "n_nokta": self.n_nokta,
            "tarih_araligi_gun": self.tarih_araligi_gun(),
            "kapi_2_4_hafta": self.yeterli_kayit(),
            "parametreler": params,
        }

    def yeterli_kayit(self, min_nokta: int = 4, min_gun: float = 21.0) -> bool:
        """Kapi 2 olcutu: ">= 4 haftalik suruklenme verisi".

        Yorum: haftalik ritimde en az **4 oturum** (nokta) ve bunun kapsadigi
        en az 21 gun. Dort haftalik oturum serisi 3 haftayi kapsar; PROJE-PLANI
        da nokta sayisiyla konusur: "her hafta ... 25+ noktali bir egri".
        """
        return self.n_nokta >= min_nokta and self.tarih_araligi_gun() >= min_gun


def egri_olustur(kabinler: list[Kabin]) -> SuruklenmeEgrisi:
    """Kayitlari zamana dizer, parametre serilerini uretir.

    Siralama tarihe gore yapilir; cagiranin listeyi sirali vermesi gerekmez.
    Duzenek uyumsuzlugu (kamera sayisi ya da izlenen parametre kumesi farkli)
    hata verir -- boyle bir degisiklik sonrasi seriler karsilastirilamaz.
    """
    if not kabinler:
        raise ValueError("suruklenme egrisi icin en az bir kayit gerekli")
    sirali = sorted(kabinler, key=lambda k: tarih_coz(k.meta.tarih_iso))

    ilk = _olcumler(sirali[0])
    n_kamera = len(sirali[0].Ks)
    tarihler, seriler = [], {p: [] for p in ilk}
    for kabin in sirali:
        if len(kabin.Ks) != n_kamera:
            raise ValueError(
                f"duzenek degisti: {kabin.meta.tarih_iso} tarihli kayit "
                f"{len(kabin.Ks)} kamera, ilki {n_kamera} -- seriler karsilastirilamaz")
        olcum = _olcumler(kabin)
        if set(olcum) != set(ilk):
            raise ValueError(
                f"izlenen parametre kumesi degisti: {kabin.meta.tarih_iso} -- "
                "muhtemelen kamera sayisi/yerlesimi degmis; yeni egri baslatin")
        tarihler.append(kabin.meta.tarih_iso)
        for p, v in olcum.items():
            seriler[p].append(v)
    return SuruklenmeEgrisi(tarihler=tuple(tarihler), olcumler=seriler)
