"""Tam vucut (65 nokta) sozlesmesi: RTMW ve MediaPipe baglantilari."""

from types import SimpleNamespace

import numpy as np
import pytest

from pose3d.iskelet import REFERANS_ISKELET
from pose3d.tam_vucut import (COCO_WB_INDEKS, GRUPLAR, MP_POZ_INDEKS, TAM_VUCUT, bos_poz,
                              coco_wholebody_poz, el_ata, mediapipe_tam_vucut, referansa_indir)


def test_iskelet_65_nokta_ve_gruplar_ortusmez():
    assert len(TAM_VUCUT) == 65
    hepsi = [i for ix in GRUPLAR.values() for i in ix]
    assert sorted(hepsi) == list(range(65))
    assert len(COCO_WB_INDEKS) == 65 and len(MP_POZ_INDEKS) == 23
    # yuz noktasi yok (KVKK): COCO-WholeBody 23..90 alinmaz
    assert not set(COCO_WB_INDEKS.tolist()) & set(range(23, 91))
    for taraf in ("sol", "sag"):
        assert (f"{taraf}_bilek", f"{taraf}_el_kok") in TAM_VUCUT.baglantilar
        assert f"{taraf}_serce_4" in TAM_VUCUT.eklemler


def test_coco_wholebody_yuzu_atlar_eli_dogru_yere_koyar():
    P = np.stack([np.arange(133.0), np.arange(133.0) * 2], 1)
    s = np.full(133, 5.0)
    s[100] = 0.1                                   # sol elde tek dusuk skor
    poz = coco_wholebody_poz(P, s, (640, 480), model="t", esik=1.0)
    assert poz.iskelet is TAM_VUCUT and poz.tespit
    assert poz.noktalar[23, 0] == 91               # sol el kok = COCO-WB 91
    assert poz.noktalar[64, 0] == 132              # sag serce ucu = COCO-WB 132
    i = 23 + (100 - 91)
    assert not poz.gorunur[i] and np.isnan(poz.noktalar[i]).all()
    assert poz.guven.max() <= 1.0 and poz.ek["ham_skor"][0] == 5.0
    with pytest.raises(ValueError):
        coco_wholebody_poz(P[:17], s[:17], (640, 480), model="t", esik=1.0)


def _mp_poz(omuz_px=100.0):
    P = np.zeros((33, 2))
    P[:, 0], P[:, 1] = 300, 200
    P[11], P[12] = [300 + omuz_px / 2, 200], [300 - omuz_px / 2, 200]   # sol, sag omuz
    P[15], P[16] = [420, 300], [180, 300]                               # sol, sag bilek
    P[29], P[31] = [330, 600], [340, 620]                               # sol topuk, ayak ucu
    return P, np.full(33, 0.9)


def _el(kok, avuc=30.0):
    el = np.tile(np.asarray(kok, float), (21, 1))
    el[1:] += np.linspace(5, avuc * 2, 20)[:, None] * [0, 1]
    el[9] = np.asarray(kok) + [0, avuc]            # orta parmak koku
    return el


def test_mediapipe_eli_bilege_gore_atar_etikete_bakmaz():
    P, g = _mp_poz()
    sag_el, sol_el = _el([185, 310]), _el([415, 310])
    poz = mediapipe_tam_vucut(P, g, [sag_el, sol_el], (640, 480), model="t")
    np.testing.assert_allclose(poz.noktalar[GRUPLAR["sol_el"][0]], [415, 310])
    np.testing.assert_allclose(poz.noktalar[GRUPLAR["sag_el"][0]], [185, 310])
    assert poz.ek["atanan_el"] == ["sag", "sol"]
    # ayak: sol topuk (MP 29) ve sol bas parmak (MP 31); serce parmak MP'de yok
    np.testing.assert_allclose(poz.noktalar[TAM_VUCUT.indeks("sol_topuk")], [330, 600])
    np.testing.assert_allclose(poz.noktalar[TAM_VUCUT.indeks("sol_bas_parmak")], [340, 620])
    assert not poz.gorunur[TAM_VUCUT.indeks("sol_serce_parmak")]


def test_bilekten_uzak_ve_cokmus_el_atanmaz():
    P, g = _mp_poz()
    uzak = _el([600, 50])                          # hicbir bilege yakin degil
    cokmus = _el([418, 302], avuc=2.0)             # omzun %2'si: sahte tespit
    poz = mediapipe_tam_vucut(P, g, [uzak, cokmus], (640, 480), model="t")
    assert not poz.gorunur[list(GRUPLAR["sol_el"] + GRUPLAR["sag_el"])].any()
    assert poz.ek["elenen_el"] == 1 and poz.ek["bulunan_el"] == 2


def test_gorunmeyen_bilege_el_atanmaz():
    P, g = _mp_poz()
    g[15] = 0.1                                    # sol bilek gorunmuyor
    poz = mediapipe_tam_vucut(P, g, [_el([415, 310])], (640, 480), model="t")
    assert not poz.gorunur[list(GRUPLAR["sol_el"])].any()


def test_el_ata_iki_el_ayni_bilege_gitmez():
    bilekler = np.array([[0.0, 0.0], [100.0, 0.0]])
    atama = el_ata(bilekler, [np.array([1.0, 0]), np.array([2.0, 0])], en_fazla=150)
    assert atama == {0: "sol", 1: "sag"}


def test_referansa_indir_boyun_omuz_ortasi():
    P = np.stack([np.arange(133.0), np.zeros(133)], 1)
    poz = referansa_indir(coco_wholebody_poz(P, np.full(133, 5.0), (640, 480),
                                             model="t", esik=1.0))
    assert poz.iskelet is REFERANS_ISKELET
    assert poz.noktalar[REFERANS_ISKELET.indeks("boyun"), 0] == pytest.approx(5.5)
    assert poz.noktalar[REFERANS_ISKELET.indeks("sol_diz"), 0] == 13
    assert poz.gorunur.all()


def test_bos_poz_tespitsiz():
    poz = bos_poz((10, 10), "t")
    assert not poz.tespit and np.isnan(poz.noktalar).all()


def test_mediapipe_el_kirpintisi_ozgun_piksele_doner():
    # El modeli kirpintida (0.5, 0.5) dondururse ozgun goruntude kirpinti merkezine duser.
    from mono.mediapipe_model import EL_GIRIS, MediaPipeEstimator
    est = MediaPipeEstimator.__new__(MediaPipeEstimator)
    est._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB="rgb"), Image=lambda **kw: kw["data"])
    boyutlar = []

    def detect(img):
        boyutlar.append(img.shape)
        return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=0.5, y=0.5)] * 21])
    est._el = SimpleNamespace(detect=detect)
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(33)]
    lm[13], lm[15] = SimpleNamespace(x=0.40, y=0.5), SimpleNamespace(x=0.50, y=0.5)  # sol kol
    lm[14], lm[16] = SimpleNamespace(x=0.60, y=0.5), SimpleNamespace(x=0.60, y=0.5)  # sag: onkol 0
    eller = est._eller(np.zeros((100, 200, 3), np.uint8), lm, 200, 100)
    assert boyutlar == [(EL_GIRIS, EL_GIRIS, 3)]          # onkolu olmayan kol atlandi
    # merkez = bilek + 0.45 * (bilek - dirsek) = 100 + 9 = 109 (x), 50 (y); yuvarlama payi 1 px
    np.testing.assert_allclose(eller[0][0], [109, 50], atol=1.0)


def test_canli_aynala_ve_ciz():
    from mono.canli import aynala, iskelet_ciz
    P = np.stack([np.linspace(10, 90, 133), np.linspace(10, 90, 133)], 1)
    poz = coco_wholebody_poz(P, np.full(133, 5.0), (100, 100), model="t", esik=1.0)
    ay = aynala(poz)
    assert ay.noktalar[0, 0] == pytest.approx(99 - poz.noktalar[0, 0])
    assert ay.iskelet is TAM_VUCUT and ay.noktalar is not poz.noktalar
    out = iskelet_ciz(np.zeros((100, 100, 3), np.uint8), poz)
    assert out.any()


def test_rtmw_kisi_basina_poz_ve_kutu(monkeypatch):
    from mono.rtmw_model import RTMWEstimator
    est = RTMWEstimator.__new__(RTMWEstimator)
    est._closed, est.threshold, est.model_id = False, 1.0, "rtmw-test"
    est._detector = lambda im: np.array([[0, 0, 10, 10], [20, 20, 40, 40]], float)
    est._pose = lambda im, bboxes: (np.zeros((len(bboxes), 133, 2)), np.full((len(bboxes), 133), 5.0))
    im = np.zeros((50, 50, 3), np.uint8)
    kisiler = est.kisiler(im)
    assert len(kisiler) == 2 and kisiler[1].ek["kutu"] == [20, 20, 40, 40]
    assert all(k.iskelet is TAM_VUCUT for k in kisiler)
    # verilen kutularla dedektor atlanir; bos kutu listesi bos doner
    assert len(est.kisiler(im, kutular=[[1, 1, 5, 5]])) == 1
    assert est.kisiler(im, kutular=[]) == []
    est.close()
    with pytest.raises(RuntimeError):
        est.kisiler(im)
