"""Egzersiz profili ve karar kapisi. Esikler arastirma baslangic ayaridir.

Geometri motoru yalniz olcer. Bu katman belirsizlik/gorunurluk/gorev kapsamini
urun kararindan ayirir. Bilinmeyen belirsizlik kesin kullanici cevabi uretmez.
"""
from dataclasses import dataclass

import numpy as np

from eval.form import Karar, VARSAYILAN_ESIKLER, form_degerlendir


@dataclass(frozen=True)
class EgzersizProfili:
    ad: str = "squat"
    surum: str = "squat-arastirma-v1"
    kapsam: tuple[str, ...] = tuple(VARSAYILAN_ESIKLER)
    evreler: tuple[str, ...] = ("inis", "dip", "cikis")
    min_kapsama: float = 0.8
    kusur_suresi_s: float = 0.2
    baslama_fleksiyon: float = 25.0
    dip_fleksiyon: float = 65.0
    bitis_fleksiyon: float = 15.0
    max_bosluk_s: float = 0.25
    # Tamamlanan tekrarin en kisa suresi: tek karelik fleksiyon sicramasi tekrar
    # sayilmaz (bagimsiz dogrulama, 25 Eylul). Kusur icin istenen sureklilikle
    # ayni (0,2 s); gercek squat 2-5 s surer, esik yalnizca artigi eler.
    min_tekrar_suresi_s: float = 0.2

    def __post_init__(self):
        if self.ad != "squat" or not self.kapsam or set(self.kapsam) - set(VARSAYILAN_ESIKLER):
            raise ValueError("desteklenmeyen egzersiz/kusur")
        if not self.evreler or set(self.evreler) - {"inis", "dip", "cikis"}:
            raise ValueError("gecersiz evreler")
        if not 0 < self.min_kapsama <= 1:
            raise ValueError("min_kapsama 0..1 olmali")
        if not 0 < self.bitis_fleksiyon < self.baslama_fleksiyon < self.dip_fleksiyon < 180:
            raise ValueError("fleksiyon esikleri sirali olmali")
        if any(not np.isfinite(x) or x <= 0 for x in (self.kusur_suresi_s, self.max_bosluk_s,
                                              self.min_tekrar_suresi_s)):
            raise ValueError("sureler pozitif sonlu olmali")


SQUAT = EgzersizProfili()
ACIKLAMALAR = {
    "diz_valgusu_sag": "Sag dizde ice yonelme",
    "diz_valgusu_sol": "Sol dizde ice yonelme",
    "kalca_hizasi": "Omuz-kalca eksenleri arasinda yan egim",
    "govde_rotasyonu": "Omuz-kalca eksenleri arasinda burulma",
}


def duzenek_degerlendir(kalib, gozlemler, *, sigma_px, esik_px=8.0, kosullu_izin=False):
    """Saglam ucgenleme -> ayni kabul edilen goruslerden kovaryans -> karar.

sigma_px tespit katkisidir. Kalibrasyon/senkron dahil toplam belirsizlik
olmadigindan varsayilan kullanici kapisi kesin karar vermez.
"""
    from pose3d.iskelet import iskelet_ucgenle
    iskelet = iskelet_ucgenle(kalib, gozlemler, saglam=True,
                             esik_px=esik_px, sigma_px=sigma_px)
    return iskelet, kare_degerlendir(iskelet, kosullu_izin=kosullu_izin)


def diz_fleksiyonu(iskelet):
    """Sag/sol diz fleksiyonunun ortalamasi; iki taraf da gozlenmis olmali."""
    acilar = []
    for taraf in ("sag", "sol"):
        ids = [iskelet.tanim.indeks(f"{taraf}_{e}") for e in ("kalca", "diz", "ayak_bilegi")]
        if not iskelet.gorunur[ids].all():
            return None
        h, k, a = iskelet.noktalar[ids]
        u, v = h-k, a-k
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm < 1e-12:
            return None
        acilar.append(180 - np.degrees(np.arccos(np.clip(u @ v / norm, -1, 1))))
    return float(np.mean(acilar))


def kare_degerlendir(iskelet, *, profil=SQUAT, kosullu_izin=False):
    """Ayni cikti semasiyla acik kapsam ve belirsizlik durumu.

kosullu_izin sadece arastirma koludur; fiziksel olarak guvenilir denmez.
Izleme/evre kapisi tekrar katmaninda uygulanir, klinik karar uretilmez.
"""
    neden = None
    if iskelet.kovaryans is None:
        neden = "belirsizlik_yok"
    elif iskelet.birim != "metre":
        neden = "metrik_belirsizlik_yok"
    elif iskelet.belirsizlik_durumu != "kalibre" and not kosullu_izin:
        neden = "belirsizlik_dogrulanmadi"
    rapor = form_degerlendir(iskelet, konum_belirsizligi_m=(
        iskelet.kovaryans if neden is None else None))
    olcumler = {}
    for ad in profil.kapsam:
        o = rapor.olcumler[ad]
        olcumler[ad] = {
            "karar": str(o.karar if neden is None else Karar.BELIRSIZ),
            "aci_derece": o.deger if np.isfinite(o.deger) else None,
            "neden": neden or ("eksik_veya_gecersiz_olcum" if o.karar == Karar.BELIRSIZ else None),
            "aciklama": ACIKLAMALAR[ad],
        }
    return {"profil": profil.surum, "olcumler": olcumler,
            "fleksiyon_derece": diz_fleksiyonu(iskelet),
            "evre_olcumu": "geometrik_aday_fiziksel_dogrulama_yok",
            "belirsizlik_durumu": iskelet.belirsizlik_durumu,
            "belirsizlik_kaynagi": iskelet.belirsizlik_kaynagi,
            "kapsam": list(profil.kapsam), "klinik_dogrulama": False}
