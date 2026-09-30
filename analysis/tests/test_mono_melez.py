"""Melez iskelet (0041): bilinen kamera ve iskelette geri kurulum."""

import numpy as np
import pytest

from mono.melez import melez_iskelet, oturt
from pose3d.iskelet import REFERANS_ISKELET

pytest.importorskip("scipy")
K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])


def _iskelet(rng):
    M = rng.normal(0, 0.25, (len(REFERANS_ISKELET), 3))
    M[:, 1] = np.linspace(-0.8, 0.9, len(M))             # boy boyunca yayilsin
    return M


def _izdusur(P):
    return (P[:, :2] / P[:, 2:3]) * [K[0, 0], K[1, 1]] + K[:2, 2]


def _olcekle(H, P):
    """Tek goruste mutlak olcek-derinlik belirsiz: H'yi P'ye en iyi olcekle."""
    return H * (np.sum(H * P) / np.sum(H * H))


def test_gurultusuz_oturtma_sekli_olcek_disinda_bulur():
    M = _iskelet(np.random.default_rng(0))
    s, T = 1.1, np.array([0.2, -0.1, 3.0])
    P = s * M + T
    p2 = _izdusur(P)
    x = oturt(M, p2, np.ones(len(M), bool), K)
    assert x[0] / x[3] == pytest.approx(s / T[2], rel=1e-3)   # yalniz oran belirli
    H = melez_iskelet(M, p2, np.ones(len(M), bool), K)
    np.testing.assert_allclose(_olcekle(H, P), P, atol=2e-3)


def test_goruntu_duzlemi_2b_den_gelir():
    """Derinlik ayni, 2B'de dizi kaydirinca melez iskelette yalniz X,Y kayar."""
    M = _iskelet(np.random.default_rng(1))
    P = M + [0, 0, 3.0]
    p2 = _izdusur(P)
    diz = REFERANS_ISKELET.indeks("sag_diz")
    p2k = p2.copy()
    p2k[diz, 0] += 20.0
    H = _olcekle(melez_iskelet(M, p2k, np.ones(len(M), bool), K), P)
    assert H[diz, 0] - P[diz, 0] == pytest.approx(20.0 / 1000 * P[diz, 2], rel=0.1)
    oteki = [k for k in range(len(M)) if k != diz]
    np.testing.assert_allclose(H[oteki], P[oteki], atol=0.02)


def test_yetersiz_nokta_none():
    M = _iskelet(np.random.default_rng(2))
    g = np.zeros(len(M), bool)
    assert melez_iskelet(M, _izdusur(M + [0, 0, 3]), g, K) is None
