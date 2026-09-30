"""deney_fitness_aqa_valgus: esikten bagimsiz skor ve AUC."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eval.form import Karar

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_fitness_aqa_valgus.py"
_spec = importlib.util.spec_from_file_location("deney_fitness_aqa_valgus", _YOL)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)
tg = fa.tg
GEREKLI = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))
ESIK = tg.VARSAYILAN_ESIKLER[tg.VALGUS[0]].deger


@pytest.mark.parametrize("tohum", range(40))
def test_skor_esigi_asar_ancak_ve_ancak_kural_kusurlu_der(tohum):
    rng = np.random.default_rng(tohum)
    T = int(rng.integers(3, 40))
    deg = rng.normal(ESIK - 2, 6, (T, 2))
    deg[rng.random((T, 2)) < 0.1] = np.nan
    kk = [tuple(Karar.BELIRSIZ if (not np.isfinite(d) or rng.random() < 0.05)
                else (Karar.KUSURLU if d > ESIK else Karar.DOGRU) for d in kare) for kare in deg]
    kayma = float(rng.normal(0, 5))
    karar = tg.tekrar_karari(tg.kaydirilmis_kararlar(kk, deg, kayma))
    assert (karar == "kusurlu") == (fa.surekli_tepe(deg, kk, kayma, GEREKLI) > ESIK)


def test_skor_kesintisiz_seriyi_ister():
    deg = np.full((GEREKLI + 2, 2), -20.0)
    deg[: GEREKLI, 0] = 30.0
    kk = [(Karar.DOGRU, Karar.DOGRU)] * len(deg)
    assert fa.surekli_tepe(deg, kk, 0.0, GEREKLI) == 30.0
    kk[GEREKLI // 2] = (Karar.BELIRSIZ, Karar.DOGRU)       # seri kesildi
    assert fa.surekli_tepe(deg, kk, 0.0, GEREKLI) == -20.0
    assert fa.surekli_tepe(deg[:GEREKLI - 1], kk[:GEREKLI - 1], 0.0, GEREKLI) == -np.inf


def test_auc():
    assert fa.auc([3, 4], [1, 2]) == 1.0
    assert fa.auc([1, 2], [3, 4]) == 0.0
    assert fa.auc([2], [2]) == 0.5
    assert fa.auc([-np.inf, 5], [0]) == 0.5
    assert fa.auc([], [1]) is None
