"""Monte Carlo belirsizlik: kalibrasyon parametreleri ne kadar guvenilir?

`calibrateMultiview` size tek bir parametre kumesi verir, o kumenin **belirsizligini
vermez.** Tez iddiasinin omurgasi tam burada: sorumuz nasil kalibre edilecegi degil,
sonuca ne zaman guvenilebilecegi.

Yontem: ayni fiziksel duzenegi N kez, bagimsiz tespit gurultusuyle yeniden
kalibre et ve parametre sacilimini olc. Sentetik sahnede gurultuyu biz koydugumuz
icin beklenen davranis bilinir (gurultu artarsa sacilim artar), dolayisiyla
belirsizlik tahmininin kendisi burada sinanabilir.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from calib.multiview import Kalibrasyon, kalibre_et


@dataclass
class Sacilim:
    """Bir parametrenin N tekrardaki dagilimi."""

    ad: str
    gercek: float | None
    ortalama: float
    std: float
    en_kucuk: float
    en_buyuk: float

    @property
    def sapma(self) -> float | None:
        """Ortalamanin gercek degerden yuzde sapmasi."""
        if self.gercek in (None, 0):
            return None
        return abs(self.ortalama - self.gercek) / abs(self.gercek) * 100.0


@dataclass
class KoşuSonucu:
    n_tekrar: int
    basarili: int
    rms_ortalama: float
    sacilimlar: dict[str, Sacilim]

    def sozluk(self) -> dict:
        return {
            "n_tekrar": self.n_tekrar,
            "basarili": self.basarili,
            "rms_ortalama": self.rms_ortalama,
            "parametreler": {k: asdict(v) | {"sapma_yuzde": v.sapma}
                             for k, v in self.sacilimlar.items()},
        }


def _olcumler(kalib: Kalibrasyon) -> dict[str, float]:
    """Takip edilen parametreler. Kamera 0 referans oldugu icin onun T'si her zaman 0."""
    K = kalib.Ks[0]
    olcum = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
             "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    for i, taban in enumerate(kalib.taban_uzunluklari()[1:], start=1):
        olcum[f"taban_0{i}"] = taban
    return olcum


def sacilim_olc(
    sahne_ureteci: Callable[[int], object],
    n_tekrar: int = 20,
    gercekler: dict[str, float] | None = None,
) -> KoşuSonucu:
    """N bagimsiz gurultu gerceklemesi uzerinde kalibre et, parametre sacilimini dondur.

    `sahne_ureteci(seed)` her cagrildiginda ayni fiziksel duzenegi, bagimsiz
    gurultuyle uretmeli. Yakinsamayan kosular sessizce atilmaz; `basarili` sayisi
    raporlanir, cunku yakinsama oraninin kendisi bir bulgudur.
    """
    if n_tekrar < 2:
        raise ValueError("sacilim icin en az 2 tekrar gerekli")
    toplanan: dict[str, list[float]] = {}
    rmsler: list[float] = []

    for tekrar in range(n_tekrar):
        s = sahne_ureteci(tekrar)
        try:
            k = kalibre_et(s.obj_noktalari, s.img_noktalari,
                           [c.boyut for c in s.kameralar], s.mask)
        except Exception:
            continue
        if not np.isfinite(k.rms_px):
            continue
        rmsler.append(k.rms_px)
        for ad, deger in _olcumler(k).items():
            toplanan.setdefault(ad, []).append(deger)

    if len(rmsler) < 2:
        raise RuntimeError(f"{n_tekrar} tekrarda yalnizca {len(rmsler)} kosu yakinsadi")

    gercekler = gercekler or {}
    sacilimlar = {}
    for ad, degerler in toplanan.items():
        a = np.asarray(degerler, dtype=float)
        sacilimlar[ad] = Sacilim(
            ad=ad, gercek=gercekler.get(ad),
            ortalama=float(a.mean()), std=float(a.std(ddof=1)),
            en_kucuk=float(a.min()), en_buyuk=float(a.max()),
        )
    return KoşuSonucu(n_tekrar=n_tekrar, basarili=len(rmsler),
                      rms_ortalama=float(np.mean(rmsler)), sacilimlar=sacilimlar)
