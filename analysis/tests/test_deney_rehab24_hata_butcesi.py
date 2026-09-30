"""deney_rehab24_hata_butcesi: kok merkezleme, olcek esleme, melez iskelet."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_hata_butcesi.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_hata_butcesi", _YOL)
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)


def _P(rng):
    return rng.normal(0, 0.3, (13, 3))


def test_kok_merkezle_kalca_ortasini_sifirlar():
    P = hb.kok_merkezle(_P(np.random.default_rng(0)))
    np.testing.assert_allclose((P[hb.I("sag_kalca")] + P[hb.I("sol_kalca")]) / 2, 0, atol=1e-12)


def test_olcek_esle_bacak_oranini_geri_verir():
    Q = _P(np.random.default_rng(1))
    np.testing.assert_allclose(hb.olcek_esle(Q * 0.5, Q), Q)


def test_melez_xy_ve_z_kaynaklarini_ayirir():
    rng = np.random.default_rng(2)
    A, B = _P(rng), _P(rng)
    M = hb.melez(A, B)
    np.testing.assert_array_equal(M[:, :2], A[:, :2])
    np.testing.assert_array_equal(M[:, 2], B[:, 2])


def test_ayni_iskelette_valgus_farki_sifir():
    rng = np.random.default_rng(3)
    P = hb.kok_merkezle(_P(rng))
    v = hb.valgus(P)
    assert np.allclose(hb.valgus(hb.melez(P, P)), v, equal_nan=True)


def test_ozet():
    o = hb._ozet([1.0, -3.0, np.nan, 2.0])
    assert o["n"] == 3 and o["medyan"] == 1.0 and o["mutlak_medyan"] == pytest.approx(2.0)
    assert hb._ozet([np.nan]) is None
