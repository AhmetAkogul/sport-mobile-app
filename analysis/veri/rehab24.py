"""REHAB24-6 verisini proje sozlesmesine ceviren adaptor.

Bicim (Zenodo 13305826 aciklamasi, `joints_names.txt`, `Segmentation.txt` ve
dosyalarin kendisinden dogrulandi, 25 Eylul):

- `3d_joints/Ex<k>/<video>-30fps.npy`: (kare, 26, 4); ilk uc sutun konum,
  dorduncu hep 1 (homojen). Iskelet OptiTrack Motive'in 41 marker'dan kurdugu
  26 eklem; birim **metre** (uyluk ~0,41 m, kalca yuksekligi ~0,98 m), y yukari.
  Sol/sag kisinin kendi soludur (Motive adlandirmasi).
- `Segmentation.csv` (`;` ayracli): tekrar basina video, kisi, ilk/son kare
  (30 fps dizinine gore), kamera 17'ye gore yon, `mocap_erroneous`, `correctness`.
- Squat Ex6'dir. `correctness` = 0 tekrarlar **tek bir kusur degildir**:
  fizyoterapist her kisiye farkli hata yaptirdi. Valgus kuralinin dogrudan
  sinamasi degil; dogru tekrarlarda yanlis alarm orani ise dogrudan olculur.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B

# REFERANS eklemi -> REHAB24-6 26 eklem indeksi. Omuz, Motive'in `*Arm`
# (glenohumeral) noktasi; `*Shoulder` klavikuladir. Boyun, diger adaptorler gibi
# iki omzun ortalamasi (Motive'in Neck noktasi degil).
EKLEMLER = {
    "sol_omuz": 7, "sol_dirsek": 8, "sol_bilek": 9,
    "sag_omuz": 12, "sag_dirsek": 13, "sag_bilek": 14,
    "sol_kalca": 16, "sol_diz": 17, "sol_ayak_bilegi": 18,
    "sag_kalca": 21, "sag_diz": 22, "sag_ayak_bilegi": 23,
}
_TURETILMIS = {"boyun": (7, 12)}
SQUAT = 6


@dataclass(frozen=True)
class Tekrar:
    video: str
    tekrar_no: int
    kisi: str
    ilk: int
    son: int
    yon: str               # kamera 17'ye gore: front / half-profile / profile
    mocap_hatali: bool
    dogru: bool


def kareyi_cevir(kare_26x4: np.ndarray) -> Iskelet3B:
    """(26, 3|4) -> REFERANS_ISKELET sirasinda Iskelet3B (metre)."""
    Q = np.asarray(kare_26x4, float)[:, :3]
    n = len(REFERANS_ISKELET)
    P = np.full((n, 3), np.nan)
    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        idx = _TURETILMIS.get(ad, (EKLEMLER.get(ad),))
        P[i] = Q[list(idx)].mean(axis=0)
    gorunur = np.isfinite(P).all(axis=1)
    P[~gorunur] = np.nan
    return Iskelet3B(tanim=REFERANS_ISKELET, noktalar=P, gorunur=gorunur,
                     goren_kamera=gorunur.astype(int) * 16, artik_px=np.full(n, np.nan),
                     birim="metre", cerceve="rehab24_mocap_dunya",
                     kaynak="REHAB24-6 OptiTrack Motive 26 eklem")


def tekrarlar_oku(csv_yolu: str | Path, egzersiz: int = SQUAT) -> list[Tekrar]:
    with open(csv_yolu, newline="", encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f, delimiter=";"))
    return [
        Tekrar(video=r["video_id"], tekrar_no=int(r["repetition_number"]),
               kisi=r["person_id"], ilk=int(r["first_frame"]), son=int(r["last_frame"]),
               yon=r["cam17_orientation"], mocap_hatali=r["mocap_erroneous"] == "1",
               dogru=r["correctness"] == "1")
        for r in satirlar if int(r["exercise_id"]) == egzersiz
    ]


def tekrar_kareleri(kok: str | Path, t: Tekrar, egzersiz: int = SQUAT) -> np.ndarray:
    """Tekrarin (kare, 26, 4) dizisi; son kare dahil (Segmentation.txt)."""
    dizi = np.load(Path(kok) / f"3d_joints/Ex{egzersiz}/{t.video}-30fps.npy")
    if not 0 <= t.ilk <= t.son < len(dizi):
        raise ValueError(f"{t.video} tekrar {t.tekrar_no}: kare {t.ilk}-{t.son}, dizi {len(dizi)}")
    return dizi[t.ilk:t.son + 1]


@dataclass(frozen=True)
class KameraKestirimi:
    K: np.ndarray           # (3, 3)
    R: np.ndarray           # (3, 3) mocap dunyasi -> kamera
    t: np.ndarray           # (3,)  metre
    rms_px: float           # geri izdusum hatasi
    n_nokta: int


def kamera_kestir(X: np.ndarray, x: np.ndarray) -> KameraKestirimi:
    """3B mocap noktalari ile 2B izdusumlerinden pinhole kamera (normalize DLT).

    REHAB24-6'nin 2B verisi, yazarlarin **basit pinhole** modeliyle 3B'den
    izdusurulmustur (Zenodo aciklamasi); bozulma yoktur. Bu yuzden DLT kesin
    cozumdur ve `rms_px` ~0 cikmalidir: cikmazsa ya eslesme ya da model
    varsayimi yanlistir. Kalibrasyon, kare basina degil kamera basina bir kez
    kestirilir (0070: test pozuna kare basina oturtma yok).
    """
    X = np.asarray(X, float).reshape(-1, 3)
    x = np.asarray(x, float).reshape(-1, 2)
    gecerli = np.isfinite(X).all(axis=1) & np.isfinite(x).all(axis=1)
    X, x = X[gecerli], x[gecerli]
    if len(X) < 6:
        raise ValueError(f"DLT icin en az 6 nokta gerekli, {len(X)} var")

    def _normalize(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = P.mean(axis=0)
        s = np.sqrt(P.shape[1]) / np.mean(np.linalg.norm(P - m, axis=1))
        T = np.eye(P.shape[1] + 1)
        T[:-1, :-1] *= s
        T[:-1, -1] = -s * m
        return (np.c_[P, np.ones(len(P))] @ T.T), T

    Xn, TX = _normalize(X)
    xn, Tx = _normalize(x)
    A = np.zeros((2 * len(X), 12))
    A[0::2, 0:4] = Xn
    A[0::2, 8:12] = -xn[:, [0]] * Xn
    A[1::2, 4:8] = Xn
    A[1::2, 8:12] = -xn[:, [1]] * Xn
    P = np.linalg.inv(Tx) @ np.linalg.svd(A)[2][-1].reshape(3, 4) @ TX

    import cv2
    # P'nin olcegi isaretsizdir: det(M) > 0 secilir, RQ'nun isaret belirsizligi
    # K'nin kosegenini pozitife cekerek giderilir (K D, D R ayni P'yi verir).
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    K, R, C = cv2.decomposeProjectionMatrix(P)[:3]
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    K = K / K[2, 2]
    C = (C[:3] / C[3]).ravel()
    if np.linalg.det(R) < 0 or np.median((X - C) @ R[2]) <= 0:
        raise ValueError("DLT ayrisimi gecersiz (sol elli donme ya da noktalar kamera arkasinda)")
    t = -R @ C
    izd = (K @ (R @ X.T + t[:, None])).T
    izd = izd[:, :2] / izd[:, 2:]
    rms = float(np.sqrt(np.mean(np.sum((izd - x) ** 2, axis=1))))
    return KameraKestirimi(K=K, R=R, t=t, rms_px=rms, n_nokta=len(X))
