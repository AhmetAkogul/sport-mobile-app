"""deney_rehab24_tanim: dikey kestirimi ve AUC."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_tanim.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_tanim", _YOL)
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)


def test_dikey_ayak_bileginden_boyuna():
    P = np.zeros((3, 13, 3))
    P[:, dt.I("boyun")] = [0, 0, 1.5]
    P[:, dt.I("sag_ayak_bilegi")] = [-0.1, 0, 0]
    P[:, dt.I("sol_ayak_bilegi")] = [0.1, 0, 0]
    np.testing.assert_allclose(dt.dikey_kestir(P), [0, 0, 1])


def test_sag_sol_fleksiyon_duz_bacak_sifir():
    P = np.zeros((13, 3))
    for taraf, x in (("sag", -0.1), ("sol", 0.1)):
        P[dt.I(f"{taraf}_kalca")] = [x, 0, 1.0]
        P[dt.I(f"{taraf}_diz")] = [x, 0, 0.5]
        P[dt.I(f"{taraf}_ayak_bilegi")] = [x, 0, 0.0]
    a, b = dt._sag_sol_fleksiyon(P)
    assert abs(a) < 1e-6 and abs(b) < 1e-6


def test_auc_nan_atlar():
    assert dt.auc([2.0, np.nan], [1.0]) == 1.0
    assert dt.auc([np.nan], [1.0]) is None
