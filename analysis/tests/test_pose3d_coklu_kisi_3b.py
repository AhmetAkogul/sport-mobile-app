"""Coklu kamera cok kisi: epipolar eslestirme, ucgenleme, 3B kimlik (sentetik)."""

import cv2
import numpy as np

from calib.multiview import Kalibrasyon
from pose3d.coklu_kisi_3b import (Takip3B, epipolar_maliyet, kisileri_esle, kisileri_ucgenle,
                                  pelvis, temel_matris)
from pose3d.pose2d import Poz2B
from pose3d.tam_vucut import TAM_VUCUT

K = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]])


def _kalib():
    Rs, Ts = [], []
    for aci in (0, 35, -35):                      # kamera 0 merkezde, digerleri yanlarda
        R = cv2.Rodrigues(np.array([0, np.radians(aci), 0]))[0]
        c = np.array([4 * np.sin(np.radians(aci)), 0, -4 + 4 * (1 - np.cos(np.radians(aci)))])
        Rs.append(R)
        Ts.append((-R @ c).reshape(3, 1))
    return Kalibrasyon(rms_px=0.0, Ks=[K] * 3, bozulmalar=[np.zeros(5)] * 3, Rs=Rs, Ts=Ts)


def _kisi(x0, rng):
    X = rng.normal(0, 0.3, (65, 3))
    X[:, 1] = rng.uniform(-0.9, 0.9, 65)
    X[:, 0] += x0
    return X


def _izdusur(kalib, k, X):
    p, _ = cv2.projectPoints(X, cv2.Rodrigues(kalib.Rs[k])[0], kalib.Ts[k], kalib.Ks[k],
                             kalib.bozulmalar[k])
    return p.reshape(-1, 2)


def _poz(P):
    return Poz2B(TAM_VUCUT, P, np.ones(65), np.ones(65, bool), True, (1280, 720), model="t")


def test_temel_matris_dogru_esleri_sifira_goturur():
    kalib, rng = _kalib(), np.random.default_rng(0)
    X = _kisi(0, rng)
    F = temel_matris(kalib, 0, 1)
    a, b = _izdusur(kalib, 0, X), _izdusur(kalib, 1, X)
    assert epipolar_maliyet(F, a, b) < 1e-6
    assert epipolar_maliyet(F, a, _izdusur(kalib, 1, _kisi(0.8, rng))) > 5
    assert epipolar_maliyet(F, a[:3], b[:3]) == np.inf


def test_iki_kisi_uc_kamera_eslesir_ve_ucgenlenir():
    kalib, rng = _kalib(), np.random.default_rng(1)
    A, B = _kisi(-0.6, rng), _kisi(0.6, rng)
    kameralar = {}
    for k in range(3):
        pozlar = [_poz(_izdusur(kalib, k, A)), _poz(_izdusur(kalib, k, B))]
        kameralar[k] = pozlar[::-1] if k == 1 else pozlar          # sira kameraya gore degisir
    kumeler = kisileri_esle(kalib, kameralar)
    assert sorted(sorted(k.items()) for k in kumeler) == [[(0, 0), (1, 1), (2, 0)],
                                                           [(0, 1), (1, 0), (2, 1)]]
    isk = kisileri_ucgenle(kalib, kameralar, kumeler)
    hedef = {0: A, 1: B}
    for kume, s in zip(kumeler, isk):
        np.testing.assert_allclose(s.noktalar, hedef[kume[0]], atol=1e-6)


def test_tek_kamerada_gorulen_kisi_kume_olmaz():
    kalib, rng = _kalib(), np.random.default_rng(2)
    A = _kisi(0, rng)
    kameralar = {0: [_poz(_izdusur(kalib, 0, A))], 1: [], 2: []}
    assert kisileri_esle(kalib, kameralar) == []


def test_takip3b_kimlik_surer_ve_uzaktaki_yeni_kimlik_alir():
    rng = np.random.default_rng(3)

    class _Isk:
        def __init__(self, P):
            self.noktalar = P
    A = _kisi(0, rng)
    t = Takip3B(kapi_m=0.5)
    k1 = [k for k, _ in t.guncelle([_Isk(A), _Isk(A + [2, 0, 0])])]
    k2 = [k for k, _ in t.guncelle([_Isk(A + [2.05, 0, 0]), _Isk(A + [0.05, 0, 0])])]
    assert k1 == [1, 2] and k2 == [2, 1]
    k3 = [k for k, _ in t.guncelle([_Isk(A + [5, 0, 0])])]
    assert k3 == [3]
    assert np.isfinite(pelvis(_Isk(A))).all()


def test_bolunen_kisi_birlesir_ve_pelvis_yedegi():
    from pose3d.coklu_kisi_3b import yakin_kumeleri_birlestir
    kalib, rng = _kalib(), np.random.default_rng(4)
    A = _kisi(0, rng)
    kameralar = {k: [_poz(_izdusur(kalib, k, A))] for k in range(3)}

    class _Isk:                      # ikinci kumenin (tek basina) kaba 3B tahmini
        noktalar = A + 0.05
    kumeler = [{0: 0, 1: 0}, {2: 0}]
    isk = kisileri_ucgenle(kalib, kameralar, kumeler[:1]) + [_Isk]
    kumeler, isk = yakin_kumeleri_birlestir(kalib, kameralar, kumeler, isk)
    assert len(kumeler) == 1 and set(kumeler[0]) == {0, 1, 2}
    np.testing.assert_allclose(isk[0].noktalar, A, atol=1e-6)
    B = A.copy()
    B[[TAM_VUCUT.indeks("sol_kalca"), TAM_VUCUT.indeks("sag_kalca")]] = np.nan

    class _Kalcasiz:
        noktalar = B
    assert np.isfinite(pelvis(_Kalcasiz)).all()
