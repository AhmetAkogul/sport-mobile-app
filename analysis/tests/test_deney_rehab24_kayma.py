"""deney_rehab24_tek_gorus: isaretli valgus farki ve LOPO kayma duzeltmesi."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eval.form import VARSAYILAN_ESIKLER, Karar

_YOL = Path(__file__).resolve().parent.parent / "scripts/deney_rehab24_tek_gorus.py"
_spec = importlib.util.spec_from_file_location("deney_rehab24_tek_gorus", _YOL)
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)

ESIK = VARSAYILAN_ESIKLER["diz_valgusu_sag"].deger
D, K, B = Karar.DOGRU, Karar.KUSURLU, Karar.BELIRSIZ


def _yalin(deg):
    return [tuple(B if not np.isfinite(d) else (K if d > ESIK else D) for d in kare)
            for kare in deg]


def test_sifir_kayma_orijinal_kararlari_aynen_uretir():
    deg = np.array([[ESIK - 1, ESIK + 1], [ESIK + 5, -20.0], [np.nan, ESIK]])
    kk = _yalin(deg)
    assert tg.kaydirilmis_kararlar(kk, deg, 0.0) == kk


def test_kayma_esigi_asan_degeri_geri_ceker():
    deg = np.array([[ESIK + 4, ESIK + 12]])
    assert tg.kaydirilmis_kararlar(_yalin(deg), deg, 5.0) == [(D, K)]


def test_belirsiz_kare_duzeltmeyle_karar_kazanmaz():
    # Orijinali BELIRSIZ (or. 0014 imkansiz geometri) ama degeri sonlu olan kare.
    deg = np.array([[ESIK + 30, -5.0]])
    assert tg.kaydirilmis_kararlar([(B, B)], deg, 0.0) == [(B, B)]
    assert tg.kaydirilmis_kararlar([(B, D)], deg, -20.0) == [(B, K)]


def test_lopo_test_kisisini_disarida_birakir():
    ogeler = [("a", np.full((3, 2), 10.0)), ("a", np.full((1, 2), 10.0)),
              ("b", np.full((2, 2), 2.0)), ("c", np.array([[4.0, np.nan]]))]
    k = tg.lopo_kayma(ogeler)
    assert k[0] == k[1] == pytest.approx((2.0 * 4 + 4.0) / 5)   # b + c, a haric
    assert k[2] == pytest.approx((10.0 * 8 + 4.0) / 9)          # a + c
    assert k[3] == pytest.approx((10.0 * 8 + 2.0 * 4) / 12)     # a + b


def test_lopo_tek_kiside_duzeltme_yok():
    assert tg.lopo_kayma([("a", np.ones((2, 2))), ("a", np.ones((1, 2)))]) == [0.0, 0.0]


def test_ayrisim_taraflari_ayri_seri_sayar():
    # Sag +10, sol -10 sabit: her seri icinde gurultu yok. Taraflar birlestirilse
    # tekrar ici SD 10 cikardi; dogru ayrisim 0 verir, fark tekrarlar arasindadir.
    f = np.column_stack([np.full(20, 10.0), np.full(20, -10.0)])
    a = tg.kayma_ayristir([("a", f), ("b", f.copy())])
    assert a["kayma"] == 0.0
    assert a["kayma_taraf"] == [10.0, -10.0]
    assert a["tekrar_ici_sd"] == 0.0
    assert a["tekrarlar_arasi_sd"] == pytest.approx(10.0)
    assert a["kisi_arasi_sd"] == 0.0


def test_ayrisim_kisi_farkini_ve_gurultuyu_ayirir():
    rng = np.random.default_rng(0)
    ogeler = [(k, m + rng.normal(0, 2.0, (4000, 2))) for k, m in (("a", 5.0), ("b", 15.0))]
    a = tg.kayma_ayristir(ogeler)
    assert a["kayma"] == pytest.approx(10.0, abs=0.1)
    assert a["kisi_arasi_sd"] == pytest.approx(5.0, abs=0.1)
    assert a["tekrar_ici_sd"] == pytest.approx(2.0, abs=0.1)


def test_ayrisim_bos_ya_da_hepsi_nan():
    assert tg.kayma_ayristir([("a", np.full((3, 2), np.nan))]) is None


def _govde(aci_derece, sirt=False):
    """Kalca ve omuz hatti X-Z duzleminde verilen acida; geri kalan eklemler NaN."""
    from pose3d.iskelet import REFERANS_ISKELET as R
    a = np.radians(aci_derece)
    yon = np.array([np.cos(a), 0.0, np.sin(a)]) * (-1 if sirt else 1)
    P = np.full((len(R), 3), np.nan)
    for sag, sol, y in (("sag_kalca", "sol_kalca", 0.0), ("sag_omuz", "sol_omuz", -0.5)):
        P[R.indeks(sag)] = -0.15 * yon + [0, y, 3]
        P[R.indeks(sol)] = 0.15 * yon + [0, y, 3]
    return P


@pytest.mark.parametrize("aci", [0.0, 30.0, 45.0, 90.0])
def test_govde_acisi_onden_profile(aci):
    assert tg.govde_acisi(_govde(aci)) == pytest.approx(aci, abs=1e-9)
    assert tg.govde_acisi(_govde(aci, sirt=True)) == pytest.approx(aci, abs=1e-9)


def test_govde_acisi_tek_cift_yeter_hic_yoksa_nan():
    from pose3d.iskelet import REFERANS_ISKELET as R
    P = _govde(60.0)
    P[R.indeks("sag_omuz")] = np.nan
    assert tg.govde_acisi(P) == pytest.approx(60.0)
    P[R.indeks("sol_kalca")] = np.nan
    assert np.isnan(tg.govde_acisi(P))


def test_lopo_aci_kayma_dogruyu_test_kisisi_olmadan_ogrenir():
    # Kayma = 2 + 0,1 * aci; test kisisinin kendi farki yaniltici (+100) olsa da
    # tahmin yalnizca diger kisilerden gelir.
    ogeler = [(k, np.full((5, 2), 2 + 0.1 * a), a)
              for k, a in (("a", 0.0), ("b", 45.0), ("c", 90.0), ("b", 30.0))]
    ogeler.append(("d", np.full((5, 2), 100.0), 60.0))
    k = tg.lopo_aci_kayma(ogeler)
    assert k[4] == pytest.approx(2 + 0.1 * 60.0)
    assert tg.lopo_aci_kayma(ogeler[:4])[1] == pytest.approx(2 + 0.1 * 45.0)


def test_lopo_aci_kayma_aci_yoksa_ortalamaya_duser():
    ogeler = [("a", np.full((2, 2), 4.0), np.nan), ("b", np.full((2, 2), 6.0), 10.0),
              ("c", np.full((2, 2), 8.0), 10.0)]
    k = tg.lopo_aci_kayma(ogeler)
    assert k[0] == pytest.approx(7.0)     # aci bilinmiyor
    assert k[1] == pytest.approx(6.0)     # egitimde tek sonlu aci (c): ortalama (4+8)/2
    assert tg.lopo_aci_kayma([("a", np.ones((1, 2)), 5.0)]) == [0.0]
