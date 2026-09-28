"""deney_rehab24_lunge_tanim: mocap lunge olculeri ve tekrar skoru."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_lunge_tanim.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_lunge_tanim", _YOL)
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)
Z = np.array([0.0, 0.0, 1.0])


def _iskelet(diz_ileri=0.0):
    X = np.zeros((26, 3))
    X[0], X[3] = [0, 0, 0.9], [0, 0, 1.5]
    for ad, (h, k, a, t) in rl.BACAK.items():
        x = -0.1 if ad == "sag" else 0.1
        if ad == "sag":                                   # ondeki bacak
            X[a], X[t] = [x, 0.4, 0.0], [x, 0.55, 0.0]
            X[k], X[h] = [x, 0.4 + diz_ileri, 0.4], [x, 0.0 + diz_ileri, 0.4]
        else:
            X[a], X[t] = [x, -0.4, 0.0], [x, -0.25, 0.0]
            X[k], X[h] = [x, -0.2, 0.1], [x, 0.0, 0.4]
    return X


def test_ondeki_bacak_diz_onde_ve_derinlik():
    o = rl.kare_olculeri(_iskelet(), Z)
    assert o["derin"] == pytest.approx(90.0) and o["sig"] == pytest.approx(-90.0)
    assert o["diz_onde"] == pytest.approx(-0.15 / 0.4)
    assert o["valgus"] == pytest.approx(0.0, abs=1e-9)
    assert rl.kare_olculeri(_iskelet(diz_ileri=0.25), Z)["diz_onde"] > 0


def test_sig_skoru_en_derin_anin_eksisi():
    kk = [{"sig": -f} for f in (10, 60, 60, 60, 60, 60, 60, 20)]
    assert rl.tekrar_skoru(kk, "sig") == pytest.approx(-60.0)
