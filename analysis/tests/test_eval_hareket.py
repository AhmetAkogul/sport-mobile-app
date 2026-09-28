"""Hareket tanima ozellikleri ve canli tahmin sarmali."""

import numpy as np
import pytest

from eval.hareket import (HareketTanima, aynala, kare_ozellikleri, pencere_ozellikleri,
                          pencereler)
from pose3d.tam_vucut import TAM_VUCUT

IX = TAM_VUCUT.indeks


def _ayakta(T=20, olcek=1.0, kaydir=(0.0, 0.0)):
    P = np.full((T, 65, 2), np.nan)
    ref = {"burun": (0, -70), "sol_omuz": (15, -50), "sag_omuz": (-15, -50),
           "sol_dirsek": (18, -25), "sag_dirsek": (-18, -25), "sol_bilek": (20, 0),
           "sag_bilek": (-20, 0), "sol_kalca": (10, 0), "sag_kalca": (-10, 0),
           "sol_diz": (10, 40), "sag_diz": (-10, 40), "sol_ayak_bilegi": (10, 80),
           "sag_ayak_bilegi": (-10, 80), "sol_topuk": (8, 85), "sag_topuk": (-8, 85),
           "sol_bas_parmak": (20, 88), "sag_bas_parmak": (-20, 88)}
    for ad, xy in ref.items():
        P[:, IX(ad)] = np.array(xy) * olcek + kaydir
    return P, np.isfinite(P).all(-1)


def test_ozellik_olcek_ve_konumdan_bagimsiz():
    P, G = _ayakta()
    Q, H = _ayakta(olcek=3.0, kaydir=(500, 300))
    np.testing.assert_allclose(kare_ozellikleri(P, G), kare_ozellikleri(Q, H), atol=1e-9)


def test_ayakta_acilar_ve_govde_egimi():
    P, G = _ayakta()
    F = kare_ozellikleri(P, G)
    diz = F[0, 17 * 2 + 6]                          # sol diz acisi (ACILAR sirasi 6)
    assert diz == pytest.approx(180, abs=1)
    assert F[0, 42] == pytest.approx(0, abs=1)      # govde dik (34 konum + 8 aci)
    assert F[0, 43] == pytest.approx(1)             # govde boyu / medyan


def test_eksik_nokta_nan_ve_pencere_eksik_orani():
    P, G = _ayakta()
    G[:, IX("sol_diz")] = False
    F = kare_ozellikleri(P, G)
    assert np.isnan(F[0, 17 * 2 + 6])
    w = pencere_ozellikleri(F)
    assert w.shape == (4 * F.shape[1] + 1,) and 0 < w[-1] < 1


def test_aynala_sol_sag_takas_eder():
    P, G = _ayakta()
    Q, H = aynala(P, G, genislik=101)
    np.testing.assert_allclose(Q[0, IX("sol_omuz")], [100 - P[0, IX("sag_omuz"), 0],
                                                      P[0, IX("sag_omuz"), 1]])
    # simetrik pozda acilar aynalamadan etkilenmez
    np.testing.assert_allclose(kare_ozellikleri(Q, H)[:, 34:42], kare_ozellikleri(P, G)[:, 34:42])


def test_pencereler_tam_ve_kaydirmali():
    ws = pencereler(45, kare_hizi=10)
    assert ws[0] == (0, 20) and ws[1] == (5, 25) and ws[-1][1] <= 45
    assert pencereler(10, kare_hizi=10) == []


class _SahteModel:
    classes_ = np.array([0, 6])

    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.array([self.p])


def test_canli_tahmin_esik_ve_yumusatma():
    P, G = _ayakta(T=25)
    t = np.arange(25) / 10.0
    ht = HareketTanima(_SahteModel([0.1, 0.9]), esik=0.6)
    assert ht.tahmin(1, t[:5], P[:5], G[:5]) is None          # 2 s dolmadi
    ad, p = ht.tahmin(1, t, P, G)
    assert ad == "squat" and p == pytest.approx(0.9)
    ht.model = _SahteModel([0.9, 0.1])
    ad, p = ht.tahmin(1, t, P, G)                             # 0.5*0.9+0.5*0.1 yumusak
    assert ad == "belirsiz" and p == pytest.approx(0.5)
    ht.unut(1)
    assert ht.tahmin(1, t, P, G)[0] == "yok"
