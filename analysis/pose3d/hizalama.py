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

    def uygula(self, noktalar: np.ndarray) -> np.ndarray:
        n = np.asarray(noktalar, dtype=np.float64).reshape(-1, 3)
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
