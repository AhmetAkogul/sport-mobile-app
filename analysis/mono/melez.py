"""Melez iskelet: goruntu duzlemi 2B noktalardan, derinlik MediaPipe dunya iskeletinden (0041).

MediaPipe dunya iskeleti (kalca merkezli, kamera eksenli, metre) yalniz **goreli
derinlik** icin kullanilir. s*M + T'nin izdusumu 2B noktalara en kucuk kareler
ile (saglam kayip) oturtulur; her eklemin Z'si oturtulmus iskeletten, X ve Y'si
2B noktanin geri izdusumunden gelir. REHAB24 squat, onden: valgus artigi saglam
SD 9,47 -> 5,99 derece, +4 derecelik kayma kalkiyor
(`scripts/deney_rehab24_melez_iskelet.py`). Kamera ic parametresi K gerekir.
"""

from __future__ import annotations

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET

_I = REFERANS_ISKELET.indeks
OTURT = tuple(_I(e) for e in ("sag_omuz", "sol_omuz", "sag_kalca", "sol_kalca", "sag_diz",
                              "sol_diz", "sag_ayak_bilegi", "sol_ayak_bilegi"))


def oturt(M: np.ndarray, p2: np.ndarray, g: np.ndarray, K: np.ndarray,
          eklemler=OTURT, en_az: int = 5) -> np.ndarray | None:
    """(s, Tx, Ty, Tz): s*M + T izdusumu 2B'ye en yakin; yetersiz nokta ya da
    kameranin arkasi -> None. M (N, 3) metre, p2 (N, 2) piksel, g (N,) gorunur."""
    from scipy.optimize import least_squares
    j = [k for k in eklemler if g[k] and np.isfinite(p2[k]).all() and np.isfinite(M[k]).all()]
    if len(j) < en_az:
        return None
    f, c = np.array([K[0, 0], K[1, 1]]), np.asarray(K[:2, 2], float)
    Mj, uj = M[j], p2[j]
    z0 = f[1] * np.ptp(Mj[:, 1]) / max(np.ptp(uj[:, 1]), 1.0)
    x0 = np.array([1.0, *((uj.mean(0) - c) * z0 / f), z0])

    def artik(x):
        P = x[0] * Mj + x[1:]
        return ((P[:, :2] / P[:, 2:3]) * f + c - uj).ravel()
    r = least_squares(artik, x0, loss="soft_l1", f_scale=10.0)
    if not r.success or r.x[3] <= 0.1 or r.x[0] <= 0:
        return None
    return r.x


def melez_iskelet(M: np.ndarray, p2: np.ndarray, g: np.ndarray,
                  K: np.ndarray) -> np.ndarray | None:
    """(N, 3) kamera cercevesinde melez iskelet (metre) ya da None."""
    x = oturt(M, p2, g, K)
    if x is None:
        return None
    f, c = np.array([K[0, 0], K[1, 1]]), np.asarray(K[:2, 2], float)
    P = x[0] * np.asarray(M, float) + x[1:]
    H = P.copy()
    for k in range(len(M)):
        if g[k] and np.isfinite(p2[k]).all() and np.isfinite(P[k, 2]):
            H[k, :2] = (p2[k] - c) / f * P[k, 2]
    return H
