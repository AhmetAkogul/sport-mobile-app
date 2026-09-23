"""calib.synthetic kabul testleri: sentetik sahne gercekci ve tekrar uretilebilir mi."""
import numpy as np
import pytest

from calib.board import BoardSpec
from calib.synthetic import POZ_DAGILIMLARI, Kamera, rig_yay, sahne_uret

SPEC = BoardSpec(5, 7)


def test_rig_kameralari_merkeze_bakiyor():
    for kam in rig_yay():
        # merkeze dogru bakis: kameranin ileri ekseni (-merkez) yonunu gostermeli
        ileri = kam.R[2]
        hedefe = -kam.merkez / np.linalg.norm(kam.merkez)
        assert float(ileri @ hedefe) > 0.999


def test_rig_yaricapi_ve_yuksekligi_uyuyor():
    """yaricap yatay duzlemdeki (XZ) yaricap; yukseklik Y'de eklenir."""
    for kam in rig_yay(yaricap_m=2.6, yukseklik_m=0.5):
        yatay = float(np.hypot(kam.merkez[0], kam.merkez[2]))
        assert yatay == pytest.approx(2.6, abs=1e-9)
        assert kam.merkez[1] == pytest.approx(0.5, abs=1e-9)


def test_ayni_seed_ayni_sahne():
    """Tekrar uretilebilirlik: Kapi 1 olcutu 'ayni veriyle iki kosu ayni sonuc'."""
    a = sahne_uret(SPEC, n_kare=6, gurultu_px=0.4, seed=7)
    b = sahne_uret(SPEC, n_kare=6, gurultu_px=0.4, seed=7)
    assert np.array_equal(a.mask, b.mask)
    for ka, kb in zip(a.img_noktalari, b.img_noktalari):
        for pa, pb in zip(ka, kb):
            assert np.array_equal(pa, pb)


def test_farkli_seed_farkli_gurultu():
    a = sahne_uret(SPEC, n_kare=6, gurultu_px=0.4, seed=1)
    b = sahne_uret(SPEC, n_kare=6, gurultu_px=0.4, seed=2)
    assert not np.array_equal(a.img_noktalari[0][0], b.img_noktalari[0][0])


def test_mask_ile_nokta_varligi_tutarli():
    """Gormeyen kamera icin bos dizi konur; hiza bozulmaz."""
    s = sahne_uret(SPEC, n_kare=10, dagilim="genis", seed=3)
    for ci, kamera in enumerate(s.img_noktalari):
        assert len(kamera) == s.mask.shape[1]
        for kare, p in enumerate(kamera):
            assert (len(p) > 0) == bool(s.mask[ci, kare])
            if len(p) == 0:
                assert p.shape == (0, 1, 2)


def test_kadraj_disi_kare_gormedi_sayilir():
    """Cok yakin ve genis dagilimda bazi kareler tasar -> mask'te 0 gorulmeli."""
    kameralar = rig_yay(yaricap_m=0.9, boyut=(320, 240))
    s = sahne_uret(SPEC, kameralar=kameralar, n_kare=25, dagilim="genis", seed=5)
    assert s.gorunurluk < 1.0, "bu kadar yakin rigde her kare sigmamali"
    assert s.mask.dtype == np.uint8


def test_gurultusuz_sahne_tam_izdusum():
    s = sahne_uret(SPEC, n_kare=4, gurultu_px=0.0, seed=1)
    assert s.gurultu_px == 0.0
    assert all(p.dtype == np.float32 for kamera in s.img_noktalari for p in kamera)


def test_poz_dagilimlari_farkli_cesitlilik_uretiyor():
    dar = POZ_DAGILIMLARI["dar"].pozlar(30, seed=1)
    genis = POZ_DAGILIMLARI["genis"].pozlar(30, seed=1)
    def yayilim(pozlar):
        return np.std([t.ravel() for _, t in pozlar], axis=0).sum()

    assert yayilim(genis) > yayilim(dar) * 3


# --- B.5.1: kamera arkasi noktalar (dis inceleme bulgusu) --------------------

def test_kamera_arkasindaki_nokta_gorulmus_sayilmaz():
    """Kameranin arkasindaki nokta kadraja dusse bile gozlem uretmemeli.

    `cv2.projectPoints` negatif Z'li noktalari da izdusurur ve sonuc cogu zaman
    kadrajin **icine** duser -- 1 m arkadaki bir nokta tam merkeze gelir. Yalnizca
    kadraj kontrolu yapan bir suzgec bunu "gorulmus" sayar ve sentetik veriye
    sessizce cop girer. Dis inceleme B.5.1.
    """
    import numpy as np
    from calib.synthetic import nokta_gozlemleri, rig_yay

    kam = rig_yay(odak_px=900.0)[0]
    ileri = -kam.merkez / np.linalg.norm(kam.merkez)
    arkada = kam.merkez - ileri * 1.0

    z_cam = float((kam.R @ arkada + kam.t.ravel())[2])
    assert z_cam < 0                      # gercekten arkada

    gozlem = nokta_gozlemleri([kam], arkada.reshape(1, 3))[0]
    assert 0 not in gozlem


def test_board_kamera_arkasina_duserse_kare_gorulmus_sayilmaz():
    """Ayni tuzak sahne uretiminde: arkadaki board kare `mask=1` almamali.

    Kamera board'a **sirti donuk** kuruluyor (R=I, t=[0,0,-3]): board orijin
    civarinda, kamera onu arkasinda birakiyor. Izdusum yine kadraj merkezine
    duser, yani yalnizca kadraj bakan bir suzgec butun kareleri "gordu" sayar.
    """
    import numpy as np
    from calib.board import BoardSpec
    from calib.synthetic import sahne_uret

    kameralar = [Kamera(
        K=np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
        R=np.eye(3), t=np.array([[0.0], [0.0], [-3.0]]), boyut=(1280, 720),
    )]
    s = sahne_uret(BoardSpec(5, 7), kameralar=kameralar, n_kare=5,
                   gurultu_px=0.0, dagilim="dar", seed=1)
    assert int(s.mask.sum()) == 0
