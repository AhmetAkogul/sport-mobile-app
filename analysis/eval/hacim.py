"""Calisma hacmi hata haritasi: olcum dogrulugu mekanda nasil degisiyor?

`PROJE-PLANI.md` Faz 2: "Calisma hacminin farkli noktalarinda hata haritasi
(yakin/uzak, merkez/kenar)". Bilinen mesafe deneyi tek bir noktada -- hacmin
merkezinde -- yapildi; oradaki sonuc kenarlar icin gecerli degildir.

Olculen buyuklugun **cerceveden bagimsiz** olmasi sart. Mutlak 3B konumu yer
gercegiyle karsilastirmak yanlistir (bkz. `pose3d/hizalama.py`), cunku kestirilen
kamera 0 ile gercek kamera 0 ayni cerceve degil. Bu yuzden hacmin her noktasina
**bilinen uzunlukta bir prob** yerlestirilir ve sistemin olctugu uzunlugun
hatasi raporlanir -- serit metreyle dogrulanabilir bir buyukluk.

Ikinci cikti gorunurluk haritasi: bir nokta iki kameradan az gorulurse
triangulation yapilamaz. Kamera yerlesimi kararinin dogrudan girdisi budur;
hata haritasi guzel gorunse bile gorunmeyen bolge kullanilamaz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib.multiview import Kalibrasyon
from calib.synthetic import Kamera, nokta_gozlemleri
from pose3d.triangulate import mesafe, ucgenle

# Prob: hacmin her noktasina konan, bilinen uzunlukta kisa cubuk.
# Insan ekleminden eklemine mesafeler bu mertebede (onkol ~25 cm).
VARSAYILAN_PROB_M = 0.20

# Probun uc yonu: hata yone gore degisir (derinlik ekseni en zayif olan).
PROB_YONLERI = (
    ("x", np.array([1.0, 0.0, 0.0])),
    ("y", np.array([0.0, 1.0, 0.0])),
    ("z", np.array([0.0, 0.0, 1.0])),
)


@dataclass
class HacimHaritasi:
    """Izgara uzerinde hata ve gorunurluk."""

    x_m: np.ndarray               # (nx,) yatay eksen degerleri
    z_m: np.ndarray               # (nz,) derinlik ekseni degerleri
    yukseklik_m: float
    hata_mm: np.ndarray           # (nz, nx) olculemeyen hucre NaN
    gorunurluk: np.ndarray        # (nz, nx) probu tam goren kamera sayisi
    prob_m: float

    @property
    def olculebilir_oran(self) -> float:
        return float(np.isfinite(self.hata_mm).mean())

    def ozet(self) -> dict:
        gecerli = self.hata_mm[np.isfinite(self.hata_mm)]
        if gecerli.size == 0:
            raise ValueError("hicbir izgara noktasi olculemedi")
        return {
            "prob_m": self.prob_m,
            "yukseklik_m": self.yukseklik_m,
            "izgara": [int(self.hata_mm.shape[1]), int(self.hata_mm.shape[0])],
            "olculebilir_oran": self.olculebilir_oran,
            "hata_ortalama_mm": float(gecerli.mean()),
            "hata_medyan_mm": float(np.median(gecerli)),
            "hata_p95_mm": float(np.percentile(gecerli, 95)),
            "hata_en_buyuk_mm": float(gecerli.max()),
            "ortalama_gorunurluk": float(self.gorunurluk.mean()),
        }


def _prob_hatasi(
    kalib: Kalibrasyon,
    kameralar: list[Kamera],
    merkez: np.ndarray,
    prob_m: float,
    gurultu_px: float,
    seed: int,
) -> tuple[float, int]:
    """Tek izgara noktasinda prob uzunlugu hatasi (mm) ve goren kamera sayisi.

    Uc yonde olculur ve ortalamasi alinir; tek yon yaniltici olur cunku derinlik
    eksenindeki hata yanal eksendekinden buyuktur.
    """
    hatalar, gorunurlukler = [], []
    for _, yon in PROB_YONLERI:
        uclar = np.array([merkez - yon * prob_m / 2, merkez + yon * prob_m / 2])
        gozlemler = nokta_gozlemleri(kameralar, uclar, gurultu_px=gurultu_px, seed=seed)
        ortak = set(gozlemler[0]) & set(gozlemler[1])
        gorunurlukler.append(len(ortak))
        if len(ortak) < 2:
            continue
        try:
            a = ucgenle(kalib, {i: gozlemler[0][i] for i in ortak})
            b = ucgenle(kalib, {i: gozlemler[1][i] for i in ortak})
        except ValueError:
            continue
        if not (a.gecerli and b.gecerli):
            continue
        hatalar.append(abs(mesafe(a, b) - prob_m) * 1000.0)

    gorunurluk = int(min(gorunurlukler)) if gorunurlukler else 0
    if not hatalar:
        return float("nan"), gorunurluk
    return float(np.mean(hatalar)), gorunurluk


def hacim_haritasi(
    kalib: Kalibrasyon,
    kameralar: list[Kamera],
    x_araligi: tuple[float, float] = (-1.5, 1.5),
    z_araligi: tuple[float, float] = (-1.5, 1.5),
    yukseklik_m: float = 0.0,
    adim_m: float = 0.25,
    prob_m: float = VARSAYILAN_PROB_M,
    gurultu_px: float = 0.25,
    seed: int = 0,
) -> HacimHaritasi:
    """Yatay (X) x derinlik (Z) duzleminde hata ve gorunurluk haritasi.

    Koordinatlar **dunya** sisteminde: kameralar orijine bakan bir yay uzerinde,
    Z ileri, X yana, Y yukari (`calib.synthetic.rig_yay`).

    Varsayilan aralik orijinin etrafidir, cunku `rig_yay` kameralari orijine
    baktirir ve calisma hacmi onlarin birlestigi yerdir. Z burada kameraya olan
    mesafe degil, dunya eksenidir: kameralar Z<0 tarafinda durur.
    """
    if adim_m <= 0:
        raise ValueError("adim pozitif olmali")
    x = np.arange(x_araligi[0], x_araligi[1] + 1e-9, adim_m)
    z = np.arange(z_araligi[0], z_araligi[1] + 1e-9, adim_m)
    hata = np.full((len(z), len(x)), np.nan)
    gorunurluk = np.zeros((len(z), len(x)), dtype=int)

    for iz, zz in enumerate(z):
        for ix, xx in enumerate(x):
            merkez = np.array([float(xx), yukseklik_m, float(zz)])
            h, g = _prob_hatasi(kalib, kameralar, merkez, prob_m, gurultu_px,
                                seed + iz * 1000 + ix)
            hata[iz, ix] = h
            gorunurluk[iz, ix] = g

    return HacimHaritasi(x_m=x, z_m=z, yukseklik_m=yukseklik_m, hata_mm=hata,
                         gorunurluk=gorunurluk, prob_m=prob_m)


def kullanilabilir_bolge(harita: HacimHaritasi, esik_mm: float = 10.0) -> dict:
    """Hangi bolge Kapi 2 olcutunu (varsayilan 10 mm) sagliyor?

    "Sistem 10 mm dogrulukta" demek yeterli degil; **nerede** 10 mm oldugunu
    soylemek gerekir. Veri toplama sirasinda denegin duracagi yer buna gore secilir.
    """
    gecerli = np.isfinite(harita.hata_mm)
    gecen = gecerli & (harita.hata_mm < esik_mm)
    hucre_alani = float(np.diff(harita.x_m).mean() * np.diff(harita.z_m).mean()) \
        if len(harita.x_m) > 1 and len(harita.z_m) > 1 else 0.0
    z_gecen = harita.z_m[gecen.any(axis=1)] if gecen.any() else np.array([])
    return {
        "esik_mm": esik_mm,
        "gecen_hucre": int(gecen.sum()),
        "olculebilir_hucre": int(gecerli.sum()),
        "toplam_hucre": int(gecerli.size),
        "gecen_oran": float(gecen.sum() / gecerli.size),
        "yaklasik_alan_m2": float(gecen.sum() * hucre_alani),
        "derinlik_araligi_m": [float(z_gecen.min()), float(z_gecen.max())]
        if z_gecen.size else None,
    }
