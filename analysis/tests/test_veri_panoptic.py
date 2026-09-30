"""veri.panoptic: Panoptic bicimi -> proje sozlesmesi (kamera 0'a gore, metre).

Sentetik bir mini dizi (2 kamera, 1 kare) uretilir; gercek veri varsa ayrica
gercek dosyalarla ayni kontrol yapilir.
"""
import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose3d.triangulate import ucgenle
from veri.panoptic import (
    dunyadan_kamera0, iskeletler_oku, kalibrasyon_oku, referansa_tasi,
)

GERCEK = Path(__file__).resolve().parent.parent / "data/dis/cmu_panoptic/171204_pose1"


def _kamera(ad, aci, t_cm):
    R = cv2.Rodrigues(np.array([0.0, np.radians(aci), 0.0]))[0]
    return {"name": ad, "type": "hd", "resolution": [1920, 1080],
            "K": [[1600.0, 0, 960], [0, 1600.0, 540], [0, 0, 1]],
            "distCoef": [-0.2, 0.18, 0.0, 0.0, 0.04], "R": R.tolist(), "t": [[v] for v in t_cm]}


@pytest.fixture
def mini(tmp_path):
    kal = {"cameras": [_kamera("00_00", 0, [0, 100, 300]), _kamera("00_01", 30, [-50, 100, 320])]}
    (tmp_path / "calib.json").write_text(json.dumps(kal))
    rng = np.random.default_rng(0)
    j = np.zeros((19, 4))
    j[:, :3] = rng.uniform(-40, 40, (19, 3)) + [0, -90, 0]        # cm
    j[:, 3] = 0.8
    j[5, 3] = 0.01                                                 # dusuk guven: sol bilek
    govde = {"id": 0, "joints19": j.ravel().tolist()}
    veri = json.dumps({"version": 0.7, "bodies": [govde]}).encode()
    with tarfile.open(tmp_path / "p.tar", "w") as tar:
        bilgi = tarfile.TarInfo("hdPose3d_stage1_coco19/body3DScene_00000007.json")
        bilgi.size = len(veri)
        tar.addfile(bilgi, io.BytesIO(veri))
    return tmp_path, kal, j


def test_donusum_izdusumu_panoptic_ile_ayni_ve_ucgenleme_geri_kazanir(mini):
    d, kal, j = mini
    pk = kalibrasyon_oku(d / "calib.json", ["00_00", "00_01"])
    P, g = referansa_tasi(iskeletler_oku(d / "p.tar", [7])[7][0])
    Pc0 = dunyadan_kamera0(P[g], pk)
    for ci, c in enumerate(kal["cameras"]):
        p1, _ = cv2.projectPoints(P[g] * 100, cv2.Rodrigues(np.array(c["R"]))[0],
                                  np.array(c["t"], float), np.array(c["K"]), np.array(c["distCoef"]))
        p2, _ = cv2.projectPoints(Pc0, cv2.Rodrigues(pk.kalib.Rs[ci])[0], pk.kalib.Ts[ci],
                                  pk.kalib.Ks[ci], pk.kalib.bozulmalar[ci])
        assert np.abs(p1 - p2).max() < 1e-4
    for n in range(len(Pc0)):
        goz = {}
        for ci in range(2):
            p, _ = cv2.projectPoints(Pc0[n:n + 1], cv2.Rodrigues(pk.kalib.Rs[ci])[0],
                                     pk.kalib.Ts[ci], pk.kalib.Ks[ci], pk.kalib.bozulmalar[ci])
            goz[ci] = p.reshape(2)
        assert np.linalg.norm(ucgenle(pk.kalib, goz).nokta - Pc0[n]) < 1e-6


def test_birim_metre_dusuk_guven_gorunmez_boyun_omuz_ortalamasi(mini):
    d, _, j = mini
    P, g = referansa_tasi(iskeletler_oku(d / "p.tar", [7])[7][0])
    from pose3d.iskelet import REFERANS_ISKELET as R
    assert not g[R.indeks("sol_bilek")] and np.isnan(P[R.indeks("sol_bilek")]).all()
    np.testing.assert_allclose(P[R.indeks("sag_diz")], j[13, :3] / 100)
    np.testing.assert_allclose(P[R.indeks("boyun")], (j[3, :3] + j[9, :3]) / 200)


def test_kayitsiz_kare_bos_ve_hatali_kamera_reddedilir(mini):
    d, _, _ = mini
    assert iskeletler_oku(d / "p.tar", [8]) == {8: []}
    with pytest.raises(ValueError, match="olmayan kamera"):
        kalibrasyon_oku(d / "calib.json", ["00_00", "99_99"])
    with pytest.raises(ValueError, match="iki farkli"):
        kalibrasyon_oku(d / "calib.json", ["00_00", "00_00"])


@pytest.mark.skipif(not (GERCEK / "calibration_171204_pose1.json").exists(),
                    reason="CMU Panoptic 171204_pose1 indirilmemis")
def test_gercek_panoptic_kalibrasyonu_31_hd_kamera():
    pk = kalibrasyon_oku(GERCEK / "calibration_171204_pose1.json", ["00_00", "00_01", "00_02"])
    assert pk.boyutlar[0] == (1920, 1080)
    np.testing.assert_allclose(pk.kalib.Rs[0], np.eye(3), atol=1e-9)
    # Panoptic R'leri 10 haneyle yazilmis: R R^T - I ~ 1e-10, kamera 0 otelemesi
    # ~2e-10 m kalinti tasir (olculdu, 25 Eylul); nanometrenin altinda, zararsiz.
    np.testing.assert_allclose(pk.kalib.Ts[0], 0, atol=1e-9)
