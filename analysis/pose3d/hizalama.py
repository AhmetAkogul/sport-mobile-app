"""Rijit hizalama (Kabsch) -- iki 3B nokta kumesini ayni cerceveye getir.

Neden gerekli: kalibrasyon sonucu kamera 0'i referans alir, ama **kestirilen**
kamera 0 ile **gercek** kamera 0 ayni cerceve degildir; ic parametrelerdeki kucuk
hata bile tum yeniden yapilandirmayi rijit olarak kaydirir ve dondurur.

Sonuc: mutlak 3B konumu yer gercegiyle dogrudan karsilastirmak yanlis bir olcumdur.
Bir kalibrasyon sisteminin urettigi anlamli buyuklukler **cerceveden bagimsiz**
olanlardir: mesafeler, acilar, sekil. `PROJE-PLANI.md`'nin Kapi 2 olcutunu
"bilinen mesafe duzenegi" uzerinden tanimlamasinin sebebi tam da bu.

Bu modul iki isi ayirir:
  * `rijit_hizala` -- cerceve farkini kaldirir, geriye **sekil bozulmasi** kalir
  * olcek destegi  -- olcek hatasini ayri bir sayi olarak raporlar

Yontem: Kabsch/Umeyama. Kapali formda en kucuk kareler; iterasyon yok.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hizalama:
    """Kaynak -> hedef rijit donusum ve artik sekil hatasi."""

    R: np.ndarray            # (3,3) rotasyon
    t: np.ndarray            # (3,) oteleme
    olcek: float             # 1.0 = olcek hatasi yok
    rms_mm: float            # hizalama sonrasi artik -- gercek sekil bozulmasi
    en_buyuk_mm: float

    def __post_init__(self) -> None:
        """Alan denetimi (H.2).

        `rijit_hizala` dogru uretiyor ama disaridan kurulan (test, pickle, elle)
        bir Hizalama bozuk olabilir ve `uygula` sessizce sacma sonuc verir.
        """
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(-1)
        if self.R.shape != (3, 3):
            raise ValueError(f"R (3,3) olmali, {self.R.shape} geldi")
        if self.t.shape != (3,):
            raise ValueError(f"t (3,) olmali, {self.t.shape} geldi")
        if not np.isfinite(self.olcek) or self.olcek <= 0:
            raise ValueError(f"olcek pozitif olmali, {self.olcek} geldi")
        if self.rms_mm < 0 or self.en_buyuk_mm < 0:
            raise ValueError("artik olculeri negatif olamaz")

    def uygula(self, noktalar: np.ndarray) -> np.ndarray:
        a = np.asarray(noktalar, dtype=np.float64)
        if a.ndim == 0 or a.shape[-1] != 3:
            raise ValueError(f"noktalar (..., 3) olmali, {a.shape} geldi")
        n = a.reshape(-1, 3)
        return (self.olcek * (self.R @ n.T)).T + self.t


def rijit_hizala(
    kaynak: np.ndarray,
    hedef: np.ndarray,
    olcek_serbest: bool = False,
) -> Hizalama:
    """Kaynak noktalari hedefe rijit olarak oturt (Kabsch/Umeyama).

    `olcek_serbest=False` (varsayilan): yalnizca donme ve oteleme. Artik hata
    hem olcek hatasini hem sekil bozulmasini icerir.
    `olcek_serbest=True`: olcek de cozulur. Artik hata **sadece** sekil
    bozulmasidir; olcek hatasi `olcek` alaninda ayrica raporlanir.
    """
    A = np.asarray(kaynak, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(hedef, dtype=np.float64).reshape(-1, 3)
    if A.shape != B.shape:
        raise ValueError(f"nokta kumeleri ayni sekilde olmali: {A.shape} vs {B.shape}")
    if len(A) < 3:
        raise ValueError("rijit hizalama icin en az 3 nokta gerekli")

    A_ort, B_ort = A.mean(0), B.mean(0)
    A0, B0 = A - A_ort, B - B_ort

    # Dejenere girdi: butun noktalar ayni yerdeyse donme tanimsizdir. SVD bos
    # bir matriste keyfi bir R uretir ve olcek 1.0'a duser; sonuc anlamsiz ama
    # gecerli gorunur. "3 nokta var" kontrolu bunu yakalamaz (H.1).
    if float((A0 ** 2).sum()) < 1e-12 or float((B0 ** 2).sum()) < 1e-12:
        raise ValueError(
            "nokta kumesi dejenere: butun noktalar ayni yerde, donme tanimsiz")

    U, S, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])                 # yansimayi engelle
    R = Vt.T @ D @ U.T

    if olcek_serbest:
        var = float((A0 ** 2).sum())
        olcek = float((S * np.array([1.0, 1.0, d])).sum() / var) if var else 1.0
    else:
        olcek = 1.0

    t = B_ort - olcek * (R @ A_ort)
    artik = B - ((olcek * (R @ A.T)).T + t)
    uzunluk = np.linalg.norm(artik, axis=1) * 1000.0
    return Hizalama(R=R, t=t, olcek=olcek,
                    rms_mm=float(np.sqrt((uzunluk ** 2).mean())),
                    en_buyuk_mm=float(uzunluk.max()))
