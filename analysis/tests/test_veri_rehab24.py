"""veri.rehab24: 26 eklem -> REFERANS eslemesi ve tekrar okuma (sentetik; gercek veri gerekmez)."""
import numpy as np
import pytest

from pose3d.iskelet import REFERANS_ISKELET as R
from veri.rehab24 import EKLEMLER, kareyi_cevir, tekrar_kareleri, tekrarlar_oku

_BASLIK = ("video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;"
           "cam17_orientation;mocap_erroneous;exercise_subtype;lights_on;"
           "extra_person_in_cam17;extra_person_in_cam18;correctness\n")


def test_eklem_eslemesi_ve_boyun_omuz_ortalamasi():
    kare = np.concatenate([np.arange(78, dtype=float).reshape(26, 3), np.ones((26, 1))], axis=1)
    isk = kareyi_cevir(kare)
    np.testing.assert_allclose(isk.noktalar[R.indeks("sol_diz")], kare[EKLEMLER["sol_diz"], :3])
    np.testing.assert_allclose(isk.noktalar[R.indeks("boyun")], kare[[7, 12], :3].mean(axis=0))
    assert isk.gorunur.all()
    assert (isk.birim, isk.cerceve) == ("metre", "rehab24_mocap_dunya")


def test_squat_tekrarlari_ve_kare_araligi(tmp_path):
    (tmp_path / "Segmentation.csv").write_text(
        _BASLIK
        + "PM_1;1;6;3;2;4;front;0;;1;0;0;1\n"
        + "PM_1;2;6;3;5;9;half-profile;1;;1;0;0;0\n"
        + "PM_2;1;1;3;0;1;front;0;right arm;1;0;0;1\n", encoding="utf-8")
    (tmp_path / "3d_joints/Ex6").mkdir(parents=True)
    np.save(tmp_path / "3d_joints/Ex6/PM_1-30fps.npy", np.arange(8)[:, None, None] * np.ones((8, 26, 4)))
    tek = tekrarlar_oku(tmp_path / "Segmentation.csv")
    assert [(t.tekrar_no, t.dogru, t.mocap_hatali, t.yon) for t in tek] == [
        (1, True, False, "front"), (2, False, True, "half-profile")]
    kareler = tekrar_kareleri(tmp_path, tek[0])
    assert kareler[:, 0, 0].tolist() == [2, 3, 4]      # son kare dahil
    with pytest.raises(ValueError, match="kare"):
        tekrar_kareleri(tmp_path, tek[1])              # 9 > dizi sonu (7)


def test_dlt_bilinen_kamerayi_geri_bulur():
    from veri.rehab24 import kamera_kestir
    rng = np.random.default_rng(3)
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    aci = np.radians(35.0)
    R = np.array([[np.cos(aci), 0, -np.sin(aci)], [0, 1, 0], [np.sin(aci), 0, np.cos(aci)]])
    t = np.array([0.2, -0.9, 4.0])
    X = rng.uniform([-1, 0, -1], [1, 1.8, 1], size=(200, 3))
    izd = (K @ (R @ X.T + t[:, None])).T
    x = izd[:, :2] / izd[:, 2:]
    k = kamera_kestir(X, x)
    np.testing.assert_allclose(k.K, K, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(k.R, R, atol=1e-8)
    np.testing.assert_allclose(k.t, t, atol=1e-7)
    assert k.rms_px < 1e-6 and k.n_nokta == 200
