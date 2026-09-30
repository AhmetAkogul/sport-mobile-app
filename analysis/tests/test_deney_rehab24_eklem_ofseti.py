"""deney_rehab24_eklem_ofseti: govde cercevesinde ofset ogrenme ve uygulama."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_eklem_ofseti.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_eklem_ofseti", _YOL)
eo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eo)


def _donme(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_ofset_govde_cercevesinde_ogrenilip_donuk_iskelete_uygulanir():
    rng = np.random.default_rng(0)
    o = rng.normal(0, 0.02, (13, 3))            # govde cercevesinde gercek ofset
    kayit = []
    for a in (0.1, 0.5, 1.2):
        R = _donme(a)
        M = rng.normal(0, 0.3, (13, 3))
        G = eo.ofset_uygula(M, R, o)
        kayit.append({"dilim": 0, "ofset": (G - M) @ R})
    tablo = eo.ofset_ogren(kayit)
    np.testing.assert_allclose(tablo[0], o, atol=1e-12)
    R = _donme(0.8)
    M = rng.normal(0, 0.3, (13, 3))
    np.testing.assert_allclose(eo.ofset_uygula(M, R, tablo[0]) - M, o @ R.T, atol=1e-12)


def test_ofset_ogren_min_ornek_ve_anahtar():
    k = [{"hucre": (0, 1), "ofset": np.ones((13, 3))}] * 2
    assert eo.ofset_ogren(k, "hucre", 3) == {}
    assert (0, 1) in eo.ofset_ogren(k, "hucre", 2)


def test_dilim_sinirlari():
    assert eo.dilim(0.0) == 0 and eo.dilim(45.0) == 1 and eo.dilim(90.0) == 2
    assert eo.dilim(-1.0) == -1 and eo.dilim(95.0, eo.FLEKS_DILIMLERI) == 3
