"""deney_rehab24_lunge_telefon: telefon iskeletinin REHAB24 26 eklem duzenine yerlesimi."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_lunge_telefon.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_lunge_telefon", _YOL)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)


def test_yerlesim_ve_ayak_ucu():
    govde = np.arange(13 * 3, dtype=float).reshape(13, 3)
    ayak = 100 + np.arange(12, dtype=float).reshape(4, 3)
    X = lt.telefon_iskeleti(govde, ayak)
    assert X.shape == (26, 3)
    np.testing.assert_array_equal(X[22], govde[lt.I("sag_diz")])
    np.testing.assert_array_equal(X[17], govde[lt.I("sol_diz")])
    np.testing.assert_array_equal(X[24], ayak[1])          # sag ayak ucu
    np.testing.assert_array_equal(X[19], ayak[3])          # sol ayak ucu
    np.testing.assert_array_equal(X[0], (govde[lt.I("sag_kalca")] + govde[lt.I("sol_kalca")]) / 2)
    assert np.isnan(X[5]).all()                            # kullanilmayan eklem


def test_2b_de_calisir():
    X = lt.telefon_iskeleti(np.ones((13, 2)), np.zeros((4, 2)))
    assert X.shape == (26, 2)
