"""deney_ec3d_lunge: ondeki bacak, diz onde ve derinlik olculeri."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_ec3d_lunge.py"
_spec = importlib.util.spec_from_file_location("deney_ec3d_lunge", _YOL)
el = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(el)
Z = np.array([0.0, 0.0, 1.0])


def _lunge(diz_ileri=0.0, on="sag"):
    """3B BODY_25, z yukari, ileri +y. Ondeki bacak ayagi y=0,4'te, dizi dik acida."""
    X = np.full((25, 3), np.nan)
    for ad, (h, k, a, t, tp) in el.BACAK.items():
        x = -0.1 if ad == "sag" else 0.1
        if ad == on:
            X[a], X[tp], X[t] = [x, 0.4, 0.0], [x, 0.35, 0.0], [x, 0.55, 0.0]
            X[k] = [x, 0.4 + diz_ileri, 0.4]
            X[h] = [x, 0.0 + diz_ileri, 0.4]
        else:
            X[a], X[tp], X[t] = [x, -0.4, 0.0], [x, -0.45, 0.0], [x, -0.3, 0.0]
            X[k], X[h] = [x, -0.2, 0.1], [x, 0.0, 0.4]
    return X


@pytest.mark.parametrize("on", ["sag", "sol"])
def test_ondeki_bacak_ve_derinlik(on):
    diz_onde, derinlik = el.kare_olculeri(_lunge(on=on), Z)
    assert derinlik == pytest.approx(90.0, abs=1e-6)
    assert diz_onde == pytest.approx((0.4 - 0.55) / 0.4)      # diz ayak ucunun gerisinde


def test_diz_ayak_ucunu_gecince_pozitif():
    diz_onde, _ = el.kare_olculeri(_lunge(diz_ileri=0.25), Z)
    assert diz_onde == pytest.approx((0.65 - 0.55) / np.hypot(0.25, 0.4))
    assert diz_onde > 0


def test_eksik_ayak_nan():
    X = _lunge()
    X[22] = np.nan
    assert all(np.isnan(el.kare_olculeri(X, Z)))
