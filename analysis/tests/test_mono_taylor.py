"""mono.taylor: Taylor (2000) kemik uzunluklu goreli derinlik."""
import numpy as np
import pytest

from eval.aci_taramasi import kemik_onculeri
from eval.form import sentetik_durus
from mono.phone import tek_gorus_3b
from mono.taylor import taylor_3b
from pose3d.iskelet import REFERANS_ISKELET as R
from pose3d.iskelet import Iskelet3B
from pose3d.pose2d import Poz2B

K = np.array([[2000.0, 0, 960], [0, 2000.0, 540], [0, 0, 1]])


def _comelmis(uzaklik=4.0):
    """Onden bakan kameraya donuk bukulmus bacaklar (uyluk kameraya dogru)."""
    d = sentetik_durus()
    P = d.noktalar.copy()
    P[:, 1] *= -1                                  # y asagi (kamera ekseni)
    for t in ("sag", "sol"):
        h = P[R.indeks(f"{t}_kalca")]
        k = h + 0.425 * np.array([0.0, np.cos(np.radians(60)), -np.sin(np.radians(60))])
        P[R.indeks(f"{t}_diz")] = k
        P[R.indeks(f"{t}_ayak_bilegi")] = k + 0.425 * np.array(
            [0.0, np.cos(np.radians(60)), np.sin(np.radians(60))])
    P[:, 2] += uzaklik
    return Iskelet3B(tanim=R, noktalar=P, gorunur=np.ones(len(R), bool),
                     goren_kamera=np.ones(len(R), int), artik_px=np.full(len(R), np.nan))


def _poz(P):
    uv = (K @ P.T).T
    uv = uv[:, :2] / uv[:, 2:]
    return Poz2B(iskelet=R, noktalar=uv, guven=np.ones(len(R)), gorunur=np.ones(len(R), bool),
                 tespit=True, goruntu_boyutu=(1920, 1080))


def _kok_goreli_hata_mm(A, B):
    return float(np.mean(np.linalg.norm((A - A.mean(0)) - (B - B.mean(0)), axis=1)) * 1000)


def test_kameraya_donuk_uylukta_taylor_derinligi_korur():
    """tek_gorus_3b (s = fL/l) burada patlar; Taylor + dogru isaret santimetre duzeyinde."""
    isk = _comelmis()
    on = kemik_onculeri(isk)
    t = taylor_3b(_poz(isk.noktalar), K, on, isaret_kaynagi=isk.noktalar)
    g = tek_gorus_3b(_poz(isk.noktalar), K, on)
    assert t.gorunur.all() and t.ek["isaretsiz_kemik"] == 0
    hata_t = _kok_goreli_hata_mm(t.noktalar, isk.noktalar)
    hata_g = _kok_goreli_hata_mm(g.noktalar[g.gorunur], isk.noktalar[g.gorunur])
    assert hata_t < 40.0                         # zayif perspektif yaklasiklik payi
    assert hata_g > 5 * hata_t


def test_kemik_uzunluklari_yaklasik_korunur():
    isk = _comelmis()
    on = kemik_onculeri(isk)
    t = taylor_3b(_poz(isk.noktalar), K, on, isaret_kaynagi=isk.noktalar)
    for (a, b), L in on.items():
        assert t.uzunluk(a, b) == pytest.approx(L, rel=0.08)


def test_isaret_yoksa_sayilir_ve_gorunmeyen_eklem_uydurulmaz():
    isk = _comelmis()
    poz = _poz(isk.noktalar)
    g = poz.gorunur.copy()
    g[R.indeks("sol_ayak_bilegi")] = False
    P = poz.noktalar.copy()
    P[~g] = np.nan
    poz = Poz2B(iskelet=R, noktalar=P, guven=g.astype(float), gorunur=g, tespit=True,
                goruntu_boyutu=(1920, 1080))
    t = taylor_3b(poz, K, kemik_onculeri(isk))
    assert t.ek["isaretsiz_kemik"] > 0
    assert not t.gorunur[R.indeks("sol_ayak_bilegi")]
    assert np.isnan(t.noktalar[R.indeks("sol_ayak_bilegi")]).all()
