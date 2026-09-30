"""Taylor (2000) tek goruntu 3B: kemik uzunlugundan goreli derinlik.

C. J. Taylor, "Reconstruction of Articulated Objects from Point
Correspondences in a Single Uncalibrated Image", CVIU 80(3), 2000.

Yontem. Olcekli ortografik izdusumde bir kemigin goruntu boyu l, gercek boyu L
ve olcek s icin derinlik farki

    dZ = ± sqrt(L^2 - (l / s)^2)

olur. `s`, butun kemiklerin fiziksel olarak mumkun oldugu en kucuk degerdir
(s >= l/L her kemik icin); Taylor'in onerdigi alt sinir budur. `tek_gorus_3b`
(s = f L / l, kemik goruntu duzlemine paralel varsayimi) kameraya donuk kemikte
derinligi patlatir; burada ayni kemik buyuk dZ ile dogru uzunlugunu korur.

Kalan belirsizlik **isaret**tir: her kemik kameraya mi yaklasiyor, uzaklasiyor
mu. Isaret disaridan verilir (`isaret_kaynagi`, kamera yonunde Z'si anlamli
herhangi bir 3B: or. MediaPipe `pose_world`, ya da ust sinir icin referans).
Verilmezse +1 (uzaklasir) alinir ve sayisi `ek["isaretsiz_kemik"]`e yazilir.

Koordinatlar: pinhole normalize (x = (u-cx)/fx); kok derinligi Z0 = 1/s,
eklem X = x Z, Y = y Z (perspektif geri izdusum; kemik boyu yaklasik korunur).
Cikti telefon kamerasi cercevesinde, metre.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from mono.phone import UzunlukOnculeri
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B, IskeletTanimi, eslestir
from pose3d.pose2d import Poz2B

KOK_EKLEM = "boyun"


def _agac(tanim: IskeletTanimi, kok: str) -> list[tuple[str, str]]:
    """Baglantilardan kokten yayilan agac (ebeveyn, cocuk); dongu kenarlari atlanir."""
    komsu: dict[str, list[str]] = {e: [] for e in tanim.eklemler}
    for a, b in tanim.baglantilar:
        komsu[a].append(b)
        komsu[b].append(a)
    gorulen, sira, kuyruk = {kok}, [], deque([kok])
    while kuyruk:
        e = kuyruk.popleft()
        for c in komsu[e]:
            if c not in gorulen:
                gorulen.add(c)
                sira.append((e, c))
                kuyruk.append(c)
    return sira


def _uzunluk(onculer: UzunlukOnculeri, a: str, b: str) -> float | None:
    L = onculer.get((a, b), onculer.get((b, a)))
    return float(L) if L is not None and np.isfinite(L) and L > 0 else None


def taylor_3b(
    poz: Poz2B,
    K: np.ndarray,
    uzunluklar_m: UzunlukOnculeri,
    isaret_kaynagi: np.ndarray | None = None,
    tanim: IskeletTanimi = REFERANS_ISKELET,
) -> Iskelet3B:
    K = np.asarray(K, float)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K sonlu (3, 3) olmali")
    n = len(tanim)
    harita = eslestir(poz.iskelet, tanim)
    uv = np.full((n, 2), np.nan)
    for j, k in enumerate(harita):
        if k >= 0 and poz.gorunur[k]:
            uv[j] = poz.noktalar[k]
    xy = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]]
    gor = np.isfinite(xy).all(axis=1)
    idx = tanim.indeks

    agac = [(a, b, _uzunluk(uzunluklar_m, a, b)) for a, b in _agac(tanim, KOK_EKLEM)]
    kullanilir = [(a, b, L) for a, b, L in agac if L is not None and gor[idx(a)] and gor[idx(b)]]
    bos = Iskelet3B(tanim=tanim, noktalar=np.full((n, 3), np.nan), gorunur=np.zeros(n, bool),
                    goren_kamera=np.zeros(n, int), artik_px=np.full(n, np.nan),
                    cerceve="telefon_kamera", kaynak="taylor2000")
    if not gor[idx(KOK_EKLEM)] or not kullanilir:
        return bos
    # Olcek: her kemigin mumkun oldugu en kucuk s (Taylor'in alt siniri).
    s = max(float(np.linalg.norm(xy[idx(a)] - xy[idx(b)])) / L for a, b, L in kullanilir)
    if not np.isfinite(s) or s <= 0:
        return bos

    Z = np.full(n, np.nan)
    Z[idx(KOK_EKLEM)] = 1.0 / s
    isaretsiz = 0
    for a, b, L in agac:                       # BFS sirasi: ebeveyn once cozulur
        ia, ib = idx(a), idx(b)
        if L is None or not (np.isfinite(Z[ia]) and gor[ib]):
            continue
        boy = float(np.linalg.norm(xy[ia] - xy[ib]))
        dz = float(np.sqrt(max(L * L - (boy / s) ** 2, 0.0)))
        if isaret_kaynagi is not None and np.isfinite(isaret_kaynagi[[ia, ib], 2]).all():
            isaret = 1.0 if isaret_kaynagi[ib, 2] >= isaret_kaynagi[ia, 2] else -1.0
        else:
            isaret, isaretsiz = 1.0, isaretsiz + 1
        Z[ib] = Z[ia] + isaret * dz

    gorunur = np.isfinite(Z) & gor
    P = np.full((n, 3), np.nan)
    P[gorunur] = np.c_[xy[gorunur] * Z[gorunur, None], Z[gorunur]]
    return Iskelet3B(tanim=tanim, noktalar=P, gorunur=gorunur,
                     goren_kamera=gorunur.astype(int), artik_px=np.full(n, np.nan),
                     cerceve="telefon_kamera", kaynak="taylor2000",
                     ek={"olcek": s, "isaretsiz_kemik": isaretsiz,
                         "isaret_kaynagi": None if isaret_kaynagi is None else "verildi"})
