"""CMU Panoptic Studio verisini proje sozlesmelerine ceviren adaptor.

Panoptic bicimi (toolbox README ve dosyalarin kendisinden dogrulandi, 25 Eylul):

- Kalibrasyon `calibration_<dizi>.json`: kamera basina `K`, `distCoef` (5),
  `R` ve `t` -- **dunya -> kamera** donusumu, birim **santimetre**.
- 3B iskelet `hdPose3d_stage1_coco19/body3DScene_<kare:08d>.json`: kisi basina
  `joints19` = 19 x (x, y, z, guven), dunya koordinatinda, santimetre. Dosya
  numarasi HD video kare numarasidir. Y ekseni asagiyi gosterir.
- COCO19 sirasi: 0 boyun, 1 burun, 2 kalca merkezi, 3-5 sol omuz/dirsek/bilek,
  6-8 sol kalca/diz/ayak bilegi, 9-11 sag omuz/dirsek/bilek, 12-14 sag
  kalca/diz/ayak bilegi, 15-18 goz/kulak. Sol/sag kisinin kendi soludur.

Proje sozlesmesi (`calib.multiview.Kalibrasyon`): `Rs`/`Ts` **kamera 0'a
gore**, birim **metre**. Donusum: R_goreli = R_i R_0^T, T_goreli = t_i - R_goreli t_0.
Boylece bizim ucgenlememiz kamera 0 cercevesinde 3B verir; Panoptic'in 3B
iskeleti de karsilastirma icin ayni cerceveye tasinir (`dunyadan_kamera0`).
"""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calib.multiview import Kalibrasyon
from pose3d.iskelet import REFERANS_ISKELET

CM = 0.01

# REFERANS_ISKELET eklemi -> COCO19 indeksi. Boyun, iki hattaki gibi iki omzun
# ortalamasindan turetilir (Panoptic'in anatomik boyun noktasi degil): boylece
# karsilastirma MediaPipe/RTMPose'un boyun tanimiyla ayni seyi olcer.
COCO19 = {
    "sol_omuz": 3, "sol_dirsek": 4, "sol_bilek": 5, "sol_kalca": 6, "sol_diz": 7,
    "sol_ayak_bilegi": 8, "sag_omuz": 9, "sag_dirsek": 10, "sag_bilek": 11,
    "sag_kalca": 12, "sag_diz": 13, "sag_ayak_bilegi": 14,
}
_TURETILMIS = {"boyun": (3, 9)}


@dataclass(frozen=True)
class PanopticKalibrasyon:
    kalib: Kalibrasyon          # kamera 0'a gore, metre
    R0: np.ndarray              # dunya -> kamera 0 (3,3)
    t0_m: np.ndarray            # (3,1) metre
    adlar: tuple[str, ...]
    boyutlar: tuple[tuple[int, int], ...]


def kalibrasyon_oku(yol: str | Path, kamera_adlari: list[str]) -> PanopticKalibrasyon:
    """Secilen kameralari (ilk ad kamera 0 olur) proje sozlesmesine cevir."""
    if len(kamera_adlari) < 2 or len(set(kamera_adlari)) != len(kamera_adlari):
        raise ValueError("en az iki farkli kamera adi gerekli")
    kameralar = {c["name"]: c for c in json.loads(Path(yol).read_text())["cameras"]}
    eksik = [a for a in kamera_adlari if a not in kameralar]
    if eksik:
        raise ValueError(f"kalibrasyonda olmayan kamera: {eksik}")
    secili = [kameralar[a] for a in kamera_adlari]
    R0 = np.asarray(secili[0]["R"], float)
    t0 = np.asarray(secili[0]["t"], float).reshape(3, 1) * CM
    Ks, bozulmalar, Rs, Ts = [], [], [], []
    for c in secili:
        R = np.asarray(c["R"], float)
        t = np.asarray(c["t"], float).reshape(3, 1) * CM
        R_g = R @ R0.T
        Ks.append(np.asarray(c["K"], float))
        bozulmalar.append(np.asarray(c["distCoef"], float).ravel())
        Rs.append(R_g)
        Ts.append(t - R_g @ t0)
    # rms_px Panoptic'te yayimlanmiyor: NaN, "bilinmiyor" demektir (uydurulmaz).
    kalib = Kalibrasyon(rms_px=float("nan"), Ks=Ks, bozulmalar=bozulmalar, Rs=Rs, Ts=Ts,
                        modeller=tuple("pinhole" for _ in secili))
    boyutlar = tuple((int(c["resolution"][0]), int(c["resolution"][1])) for c in secili)
    return PanopticKalibrasyon(kalib, R0, t0, tuple(kamera_adlari), boyutlar)


def dunyadan_kamera0(noktalar_m: np.ndarray, pk: PanopticKalibrasyon) -> np.ndarray:
    """Dunya (metre) -> kamera 0 cercevesi (metre). NaN korunur."""
    P = np.asarray(noktalar_m, float).reshape(-1, 3)
    return (pk.R0 @ P.T + pk.t0_m).T


def kimlikli_iskeletler_oku(tar_yolu: str | Path,
                            kareler: list[int]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Kare -> [(Panoptic kisi kimligi, (19, 4) [x, y, z (metre), guven])].

    Dizilerin bir kismi dosyalari `hdPose3d_stage1_coco19/hd/` altinda tutar;
    eslesme dosya adiyla yapilir.
    """
    istenen = {f"body3DScene_{k:08d}.json": k for k in kareler}
    sonuc: dict[int, list[tuple[int, np.ndarray]]] = {k: [] for k in kareler}
    with tarfile.open(tar_yolu) as tar:
        for uye in tar:
            kare = istenen.get(uye.name.rsplit("/", 1)[-1])
            if kare is None or not uye.name.startswith("hdPose3d_stage1_coco19/"):
                continue
            dosya = tar.extractfile(uye)
            if dosya is None:
                continue
            veri = json.loads(dosya.read())
            for govde in veri.get("bodies", []):
                j = np.asarray(govde["joints19"], float).reshape(19, 4)
                j[:, :3] *= CM
                sonuc[kare].append((int(govde.get("id", -1)), j))
    return sonuc


def iskeletler_oku(tar_yolu: str | Path, kareler: list[int]) -> dict[int, list[np.ndarray]]:
    """Kare -> kisi basina (19, 4) dizi [x, y, z (metre), guven]. Kayit yoksa bos liste."""
    return {k: [j for _, j in v] for k, v in kimlikli_iskeletler_oku(tar_yolu, kareler).items()}


def kamera_oku(yol: str | Path, ad: str) -> dict:
    """Tek kameranin ham Panoptic kaydi (K, distCoef, R, t [cm], resolution)."""
    for c in json.loads(Path(yol).read_text())["cameras"]:
        if c["name"] == ad:
            return c
    raise ValueError(f"kalibrasyonda olmayan kamera: {ad}")


def izdusur(noktalar_m: np.ndarray, kamera: dict) -> np.ndarray:
    """Dunya noktalarini (metre) kameranin ozgun piksellerine izdusurur (bozulma dahil).

    Kameranin arkasinda kalan nokta NaN olur.
    """
    import cv2
    P = np.asarray(noktalar_m, float).reshape(-1, 3) / CM
    R = np.asarray(kamera["R"], float)
    t = np.asarray(kamera["t"], float).reshape(3, 1)
    z = (R @ P.T + t)[2]
    px, _ = cv2.projectPoints(P.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t,
                              np.asarray(kamera["K"], float),
                              np.asarray(kamera["distCoef"], float).ravel())
    px = px.reshape(-1, 2)
    px[~(z > 0)] = np.nan
    return px


# COCO19 -> TAM_VUCUT govde (0..16): burun, gozler, kulaklar, omuz..ayak bilegi.
COCO19_TAM_VUCUT = (1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14)


def referansa_tasi(joints19: np.ndarray, guven_esigi: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(19, 4) -> REFERANS_ISKELET sirasinda (13, 3) metre + gorunur maskesi.

    Panoptic'in guveni esigin altindaki eklem NaN ve gorunmez olur (uydurulmaz).
    """
    n = len(REFERANS_ISKELET)
    P = np.full((n, 3), np.nan)
    gorunur = np.zeros(n, bool)
    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        idx = _TURETILMIS.get(ad, (COCO19.get(ad),))
        secim = joints19[list(idx)]
        if (secim[:, 3] >= guven_esigi).all():
            P[i] = secim[:, :3].mean(axis=0)
            gorunur[i] = True
    return P, gorunur
