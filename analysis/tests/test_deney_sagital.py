"""deney_sagital: Spearman yardimcisi."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_sagital.py"
_spec = importlib.util.spec_from_file_location("deney_sagital", _YOL)
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)


def test_spearman_monoton_ve_ters():
    x = np.arange(10.0)
    assert sg.spearman(x, x ** 3) == pytest.approx(1.0)
    assert sg.spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_az_ornek_none_nan_atlar():
    assert sg.spearman([1, 2, 3], [1, 2, 3]) is None
    x = np.array([1, 2, 3, 4, 5, np.nan, 7.0])
    assert sg.spearman(x, x) == pytest.approx(1.0)
