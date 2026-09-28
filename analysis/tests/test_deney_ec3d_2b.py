"""deney_ec3d_2b: 2B KASR, FPPA ve surekli tepe."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_ec3d_2b.py"
_spec = importlib.util.spec_from_file_location("deney_ec3d_2b", _YOL)
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)


def _bacaklar(diz_ice=0.0):
    P = {}
    for (h, k, a), x in zip(zip(e2.KALCA, e2.DIZ, e2.AYAK), (-50.0, 50.0)):
        P[h] = np.array([x, 0.0])
        P[k] = np.array([x - np.sign(x) * diz_ice, 100.0])
        P[a] = np.array([x, 200.0])
    return P


def test_kasr_duz_bir_ice_kucuk():
    assert e2.kasr(_bacaklar()) == 1.0
    assert e2.kasr(_bacaklar(diz_ice=20)) == 0.6


def test_fppa_duz_sifir_bukulunce_artar():
    assert abs(e2.fppa(_bacaklar())) < 1e-9
    assert e2.fppa(_bacaklar(diz_ice=20)) > 20


def test_eksik_eklem_nan():
    P = _bacaklar()
    del P[e2.DIZ[0]]
    assert np.isnan(e2.kasr(P))
    assert np.isfinite(e2.fppa(P))          # diger bacak yeter


def test_surekli_en_buyuk():
    v = np.array([0, 5, 5, 5, 5, 5, 5, 9, np.nan, 9], float)
    assert e2.surekli_en_buyuk(v, 6) == 5.0
    assert e2.surekli_en_buyuk(v[:5], 6) == -np.inf


def test_diz_sapmasi_2b_ice_pozitif_disa_negatif():
    P = _bacaklar()
    P[22], P[19] = np.array([-50.0, 220.0]), np.array([50.0, 220.0])   # ayak uclari duz
    assert abs(e2.diz_sapmasi_2b(P)) < 1e-9
    assert e2.diz_sapmasi_2b({**P, **_bacaklar(diz_ice=20)}) > 0
    disa = _bacaklar(diz_ice=-20)
    assert e2.diz_sapmasi_2b({**P, **{k: v for k, v in disa.items() if k in e2.DIZ}}) < 0


def test_diz_sapmasi_3b_duzleme_uzaklik():
    X = np.zeros((25, 3))
    for (h, k, a, t), x in zip(zip(e2.KALCA, e2.DIZ, e2.AYAK, e2.AYAK_UCU), (-0.1, 0.1)):
        X[h], X[k], X[a], X[t] = [x, 0, 1.0], [x, 0.1, 0.5], [x, 0, 0.0], [x, 0.15, 0.0]
    assert abs(e2.diz_sapmasi_3b(X.T)) < 1e-9
    X[e2.DIZ[0], 0] += 0.05                                  # sag diz ice (+x)
    assert e2.diz_sapmasi_3b(X.T) > 0
