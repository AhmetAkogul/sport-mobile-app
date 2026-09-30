"""deney_rehab24_belirsizlik: hucreleme, dayanikli SD ve belirsizlikli tekrar karari."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_belirsizlik.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_belirsizlik", _YOL)
bz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bz)
G = bz.GEREKLI


def test_hucre_aralik_disi_ve_nan_eksi_bir():
    h = bz.hucre(np.array([10.0, 50.0, np.nan, 95.0]), np.array([20.0, 100.0, 20.0, 20.0]))
    assert h[0] == 0 and h[1] == 2 * (len(bz.FLEKS_SINIR) - 1) + 3
    assert h[2] == -1 and h[3] == -1


def test_dayanikli_sd_normal_icin_sd():
    x = np.random.default_rng(0).normal(0, 3.0, 20000)
    assert bz.dayanikli_sd(x) == pytest.approx(3.0, rel=0.03)


def test_sigma_ogren_az_ornekli_hucre_genele_duser():
    rng = np.random.default_rng(1)
    art = np.concatenate([rng.normal(0, 1, 5000), rng.normal(0, 10, 10)])
    hc = np.concatenate([np.zeros(5000, int), np.ones(10, int)])
    tablo, genel = bz.sigma_ogren(art, hc)
    assert tablo[0] == pytest.approx(1.0, rel=0.05) and tablo[1] == genel


def _dizi(deger, T=30):
    return np.full((T, 2), float(deger)), np.ones((T, 2), bool)


@pytest.mark.parametrize("deger,sigma,beklenen", [
    (20.0, 2.0, "kusurlu"),      # 20 - 3,3 > 10
    (0.0, 2.0, "dogru"),         # 0 + 3,3 < 10
    (9.0, 2.0, "belirsiz"),      # band esigi kesiyor
    (20.0, 8.0, "belirsiz"),     # 20 - 13,2 < 10
])
def test_belirsizlikli_karar(deger, sigma, beklenen):
    v, ok = _dizi(deger)
    assert bz.belirsiz_karar(v, ok, np.full(len(v), sigma), 10.0) == beklenen


def test_z_sifirda_yalin_esik():
    v, ok = _dizi(10.5)
    assert bz.belirsiz_karar(v, ok, np.full(len(v), 5.0), 10.0, z=0.0) == "kusurlu"


def test_gecersiz_kareler_pencereyi_keser():
    v, ok = _dizi(0.0, T=G + 1)
    ok[G // 2] = False
    assert bz.belirsiz_karar(v, ok, np.full(len(v), 1.0), 10.0) == "belirsiz"
