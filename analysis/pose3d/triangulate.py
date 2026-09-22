"""Coklu gorusten 3B nokta kestirimi (triangulation).

Kalibrasyon "kamera nerede" sorusunu cevaplar; triangulation "nokta nerede"
sorusunu. Duzenegin yer gercegi urettigi yer burasi: `PROJE-PLANI.md` Kapi 2
olcutu, 2 m mesafede bilinen bir nesnede 3B konum hatasinin 10 mm'nin altinda
olmasini istiyor.

Yontem: DLT (Direct Linear Transform). Her kamera bir noktaya iki dogrusal
kisit koyar; N kameradan gelen 2N kisit en kucuk kareler ile cozulur. Iki kamera
yeterlidir ama daha fazlasi hem gurultuyu bastirir hem de bir kamera noktayi
gormedigi (ortulme) durumda hatti ayakta tutar.

Onemli: DLT dogrusal pinhole modeli varsayar. Bozulma (distortion) **once
giderilmeli**, aksi halde hata sessizce sizar. `ucgenle` bunu kendisi yapar;
dogrudan `_dlt` cagiran olursa sorumluluk onundur.

OpenCV'nin `triangulatePoints` fonksiyonu yalnizca **iki** goruse calisir; coklu
kamera duzenegimizin butun kameralarini kullanabilmek icin DLT burada yazildi.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calib.multiview import Kalibrasyon


def projeksiyon_matrisi(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = K [R | t], (3,4)."""
    return np.asarray(K, float) @ np.hstack([np.asarray(R, float).reshape(3, 3),
                                             np.asarray(t, float).reshape(3, 1)])


def kalibrasyondan_projeksiyonlar(kalib: Kalibrasyon) -> list[np.ndarray]:
    """Kalibrasyon sonucundan kamera basina projeksiyon matrisi.

    Rs/Ts kamera 0'a gore oldugu icin cikan 3B noktalar da **kamera 0
    koordinat sisteminde** olur. Olcek board'un fiziksel kare kenarindan gelir,
    yani metre.
    """
    return [projeksiyon_matrisi(K, R, T)
            for K, R, T in zip(kalib.Ks, kalib.Rs, kalib.Ts)]


def bozulma_gider(noktalar: np.ndarray, K: np.ndarray, bozulma: np.ndarray) -> np.ndarray:
    """Piksel noktalarini bozulmadan arindirip yine piksel uzayina dondur."""
    n = np.asarray(noktalar, dtype=np.float64).reshape(-1, 1, 2)
    duz = cv2.undistortPoints(n, np.asarray(K, float), np.asarray(bozulma, float), P=K)
    return duz.reshape(-1, 2)


def _dlt(projeksiyonlar: list[np.ndarray], noktalar: np.ndarray) -> np.ndarray:
    """Tek nokta icin DLT. noktalar: (M,2), projeksiyonlar: M adet (3,4)."""
    satirlar = []
    for P, (x, y) in zip(projeksiyonlar, noktalar):
        satirlar.append(x * P[2] - P[0])
        satirlar.append(y * P[2] - P[1])
    A = np.asarray(satirlar, dtype=np.float64)
    # Kosullanma icin olcekle: satir normlari cok farkliysa SVD sayisal olarak bozulur.
    normlar = np.linalg.norm(A, axis=1, keepdims=True)
    normlar[normlar == 0] = 1.0
    _, _, Vt = np.linalg.svd(A / normlar)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:                      # sonsuzdaki nokta -- kesisme yok
        return np.full(3, np.nan)
    return X[:3] / X[3]


@dataclass
class Ucgenleme:
    """Bir noktanin 3B kestirimi ve ne kadar guvenilir oldugu."""

    nokta: np.ndarray            # (3,) kamera 0 koordinatinda, metre
    goren_kamera: int
    artik_px: float              # ortalama yeniden izdusum artigi

    @property
    def gecerli(self) -> bool:
        return bool(np.isfinite(self.nokta).all())


def ucgenle(
    kalib: Kalibrasyon,
    gozlemler: dict[int, np.ndarray],
    bozulma_giderildi: bool = False,
) -> Ucgenleme:
    """Bir 3B noktayi, onu goren kameralarin 2B gozlemlerinden kestir.

    `gozlemler`: {kamera_indeksi: (2,) piksel}. En az iki kamera gerekir --
    tek kameradan derinlik cikmaz, bu projenin tum tezi de zaten bu.
    """
    if len(gozlemler) < 2:
        raise ValueError(
            f"triangulation icin en az 2 gorus gerekli, {len(gozlemler)} verildi")

    indeksler = sorted(gozlemler)
    P_hepsi = kalibrasyondan_projeksiyonlar(kalib)
    Ps, noktalar = [], []
    for i in indeksler:
        p = np.asarray(gozlemler[i], dtype=np.float64).ravel()
        if p.shape != (2,):
            raise ValueError(f"kamera {i} gozlemi (2,) olmali, {p.shape} geldi")
        if not bozulma_giderildi:
            p = bozulma_gider(p, kalib.Ks[i], kalib.bozulmalar[i])[0]
        Ps.append(P_hepsi[i])
        noktalar.append(p)

    X = _dlt(Ps, np.asarray(noktalar))
    if not np.isfinite(X).all():
        return Ucgenleme(nokta=X, goren_kamera=len(indeksler), artik_px=float("nan"))

    Xh = np.append(X, 1.0)
    artiklar = []
    for P, p in zip(Ps, noktalar):
        izd = P @ Xh
        if abs(izd[2]) < 1e-12:
            artiklar.append(np.inf)
            continue
        artiklar.append(float(np.linalg.norm(izd[:2] / izd[2] - p)))
    return Ucgenleme(nokta=X, goren_kamera=len(indeksler),
                     artik_px=float(np.mean(artiklar)))


def ucgenle_toplu(
    kalib: Kalibrasyon,
    gozlem_listesi: list[dict[int, np.ndarray]],
    bozulma_giderildi: bool = False,
) -> list[Ucgenleme]:
    """Birden cok nokta icin triangulation. Gormeyen kamera sozlukte bulunmaz."""
    return [ucgenle(kalib, g, bozulma_giderildi) for g in gozlem_listesi]


def mesafe(a: Ucgenleme | np.ndarray, b: Ucgenleme | np.ndarray) -> float:
    """Iki 3B nokta arasi Oklid mesafesi (metre)."""
    pa = a.nokta if isinstance(a, Ucgenleme) else np.asarray(a, float)
    pb = b.nokta if isinstance(b, Ucgenleme) else np.asarray(b, float)
    return float(np.linalg.norm(pa - pb))


def bilinen_mesafe_hatasi(
    kestirimler: list[Ucgenleme],
    gercek_mesafeler: list[tuple[int, int, float]],
) -> dict:
    """Bilinen mesafe duzenegi degerlendirmesi -- juri demosunun temeli.

    `gercek_mesafeler`: (i, j, metre) uclulari; uzerinde olculu isaretler olan
    sert bir cubuk/levhadan gelir ve serit metreyle dogrulanabilir. Sistemin
    olctugu mesafe ile gercek mesafe farki, 3B dogrulugun dogrudan kanitidir.
    """
    if not gercek_mesafeler:
        raise ValueError("en az bir bilinen mesafe gerekli")
    hatalar_mm = []
    satirlar = []
    for i, j, gercek_m in gercek_mesafeler:
        olculen = mesafe(kestirimler[i], kestirimler[j])
        hata_mm = (olculen - gercek_m) * 1000.0
        hatalar_mm.append(abs(hata_mm))
        satirlar.append({"i": i, "j": j, "gercek_mm": gercek_m * 1000.0,
                         "olculen_mm": olculen * 1000.0, "hata_mm": hata_mm})
    h = np.asarray(hatalar_mm)
    return {
        "n": len(h),
        "ortalama_mutlak_hata_mm": float(h.mean()),
        "rms_hata_mm": float(np.sqrt((h ** 2).mean())),
        "en_buyuk_hata_mm": float(h.max()),
        "kapi_2_gecti": bool(h.max() < 10.0),      # PROJE-PLANI.md Kapi 2 olcutu
        "olcumler": satirlar,
    }
