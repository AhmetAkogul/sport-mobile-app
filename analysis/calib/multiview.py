"""cv2.calibrateMultiview sarmalayicisi.

Bu modulun varlik sebebi, OpenCV cagrisinin **sozlesmesini tek yerde sabitlemek.**
Ampirik olarak dogrulanan imza (OpenCV 5.0.0):

    calibrateMultiview(objPoints, imagePoints, imageSize, detectionMask, models,
                       Ks, distortions, Rs, Ts[, flagsForIntrinsics[, flags[, criteria]]])
        -> retval, Ks, distortions, Rs, Ts

Girdi bicimleri (deneyerek bulundu, belgelerde yazmiyor):

- `objPoints`   : kare basina (N,3) float32 -- board yerel koordinatinda
- `imagePoints` : [kamera][kare] listesi, her biri (N,1,2) float32.
                  Gormeyen kamera icin **bos (0,1,2) dizi konur, atlanmaz** --
                  yoksa kare indeksleri kayar ve mask anlamsizlasir.
- `imageSize`   : kamera basina (genislik, yukseklik)
- `detectionMask`: (kamera x kare) uint8
- `models`      : kamera basina CALIB_MODEL_PINHOLE / CALIB_MODEL_FISHEYE, uint8 dizi
- `Ks`, `distortions`: giris-cikis; onceden ayrilmis liste verilmeli

**Dis parametre anlami (dogrulandi):** donen `Rs[i]` ve `Ts[i]`, kamera i'nin
**kamera 0'a gore** konumudur; kamera 0 icin R birim, T sifirdir. Yani duzenek
kamera 0 merkezli bir koordinat sisteminde cikar; dunya olcegi board'un fiziksel
kare kenarindan gelir.

OpenCV rotasyonlari **Rodrigues vektoru (3,)** olarak donduruyor, matris olarak degil.
Bu modul sinirda (3,3) matrise cevirir ki ust katmanlar bicim tahmini yapmasin.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from eval.metrics import ReprojectionError, reprojection_error


@dataclass
class Kalibrasyon:
    """Kalibrasyon sonucu. Uzunluk birimi board'un kare kenariyla ayni (metre)."""

    rms_px: float
    Ks: list[np.ndarray]
    bozulmalar: list[np.ndarray]
    Rs: list[np.ndarray]          # kamera 0'a gore rotasyon (3,3)
    Ts: list[np.ndarray]          # kamera 0'a gore oteleme (3,1)

    @property
    def n_kamera(self) -> int:
        return len(self.Ks)

    def odak(self, kamera: int = 0) -> tuple[float, float]:
        K = self.Ks[kamera]
        return float(K[0, 0]), float(K[1, 1])

    def taban_uzunluklari(self) -> list[float]:
        """Kamera 0'a olan mesafeler -- duzenegin olcegini gozle kontrol etmek icin."""
        return [float(np.linalg.norm(np.asarray(T))) for T in self.Ts]


def _rotasyon_matrisi(R) -> np.ndarray:
    """Rodrigues vektoru veya matris gelebilir -> her zaman (3,3) matris dondur."""
    R = np.asarray(R, dtype=np.float64)
    if R.size == 3:
        return cv2.Rodrigues(R.reshape(3, 1))[0]
    if R.size == 9:
        return R.reshape(3, 3)
    raise ValueError(f"rotasyon 3 veya 9 elemanli olmali, {R.size} geldi")


def _dogrula(obj_noktalari, img_noktalari, mask) -> None:
    n_kamera, n_kare = mask.shape
    if len(obj_noktalari) != n_kare:
        raise ValueError(
            f"objPoints {len(obj_noktalari)} kare, mask {n_kare} kare gosteriyor")
    if len(img_noktalari) != n_kamera:
        raise ValueError(
            f"imagePoints {len(img_noktalari)} kamera, mask {n_kamera} kamera gosteriyor")
    for ci, kamera in enumerate(img_noktalari):
        if len(kamera) != n_kare:
            raise ValueError(f"kamera {ci} icin {len(kamera)} kare var, {n_kare} bekleniyor")
    if mask.dtype != np.uint8:
        raise ValueError("detectionMask uint8 olmali (OpenCV CV_8UC1 bekliyor)")
    # Mask ile gercek nokta varligi celisirse kalibrasyon sessizce bozulur; burada yakala.
    for ci, kamera in enumerate(img_noktalari):
        for kare, p in enumerate(kamera):
            var = len(p) > 0
            if bool(mask[ci, kare]) != var:
                raise ValueError(
                    f"mask[{ci},{kare}]={mask[ci, kare]} ama nokta sayisi {len(p)} -- "
                    "mask ile imagePoints celisiyor")


def kalibre_et(
    obj_noktalari: list[np.ndarray],
    img_noktalari: list[list[np.ndarray]],
    boyutlar: list[tuple[int, int]],
    mask: np.ndarray,
    balik_gozu: bool = False,
) -> Kalibrasyon:
    """Coklu kamera kalibrasyonu. Girdiler `calib.detect.hazirla` ciktisiyla uyumlu."""
    _dogrula(obj_noktalari, img_noktalari, mask)
    n_kamera = mask.shape[0]
    if len(boyutlar) != n_kamera:
        raise ValueError("her kamera icin bir goruntu boyutu gerekli")

    model = cv2.CALIB_MODEL_FISHEYE if balik_gozu else cv2.CALIB_MODEL_PINHOLE
    modeller = np.array([model] * n_kamera, dtype=np.uint8)
    Ks = [np.eye(3) for _ in range(n_kamera)]
    bozulmalar = [np.zeros(5) for _ in range(n_kamera)]

    rms, Ks, bozulmalar, Rs, Ts = cv2.calibrateMultiview(
        obj_noktalari, img_noktalari, list(boyutlar), mask, modeller, Ks, bozulmalar,
        None, None,
    )
    return Kalibrasyon(
        rms_px=float(rms),
        Ks=[np.asarray(K, dtype=np.float64) for K in Ks],
        bozulmalar=[np.asarray(d, dtype=np.float64).ravel() for d in bozulmalar],
        Rs=[_rotasyon_matrisi(R) for R in Rs],
        Ts=[np.asarray(T, dtype=np.float64).reshape(3, 1) for T in Ts],
    )


def kamera_yeniden_izdusum(
    kalib: Kalibrasyon,
    kamera: int,
    obj_noktalari: list[np.ndarray],
    img_noktalari: list[list[np.ndarray]],
    mask: np.ndarray,
    board_pozlari: list[tuple[np.ndarray, np.ndarray]],
) -> ReprojectionError:
    """Tek kameranin yeniden izdusum hatasi, `eval.metrics` ile olculur.

    Dikkat -- iki farkli "mask" var ve karistirilmasi kolay:
      * `detectionMask` : (kamera x kare), **uint8**, OpenCV sozlesmesi
      * eval maskesi    : nokta basina, **bool**, `reprojection_error` sozlesmesi
    Ayni sey degiller. Burada kare secimi detectionMask'ten gelir, metrige verilen
    noktalar zaten suzulmustur, dolayisiyla eval maskesi kullanilmaz.
    """
    gozlenen, izdusen = [], []
    R_bagil, T_bagil = kalib.Rs[kamera], kalib.Ts[kamera]
    for kare, (Rb, tb) in enumerate(board_pozlari):
        if not mask[kamera, kare]:
            continue
        R = R_bagil @ Rb
        t = R_bagil @ tb + T_bagil
        p, _ = cv2.projectPoints(
            obj_noktalari[kare].astype(np.float64), cv2.Rodrigues(R)[0], t,
            kalib.Ks[kamera], kalib.bozulmalar[kamera],
        )
        izdusen.append(p.reshape(-1, 2))
        gozlenen.append(img_noktalari[kamera][kare].reshape(-1, 2))
    if not gozlenen:
        raise ValueError(f"kamera {kamera} hicbir karede board gormemis")
    return reprojection_error(np.vstack(gozlenen), np.vstack(izdusen))
