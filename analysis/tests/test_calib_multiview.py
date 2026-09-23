"""calib.multiview kabul testleri.

KOD-PLANI.md Adim 2 kabul olcutu: gurultusuz sentetik veride geri kazanilan odak
uzakligi hatasi < %0.1.
"""
import numpy as np
import pytest

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import rig_yay, sahne_uret

SPEC = BoardSpec(5, 7)
GERCEK_ODAK = 900.0


@pytest.fixture(scope="module")
def gurultusuz():
    kameralar = rig_yay(odak_px=GERCEK_ODAK)
    s = sahne_uret(SPEC, kameralar=kameralar, n_kare=20, gurultu_px=0.0,
                   dagilim="genis", seed=1)
    k = kalibre_et(s.obj_noktalari, s.img_noktalari,
                   [c.boyut for c in s.kameralar], s.mask)
    return s, k


def test_odak_uzakligi_binde_birden_iyi_geri_bulunuyor(gurultusuz):
    """KABUL TESTI: hata < %0.1."""
    _, k = gurultusuz
    for kamera in range(k.n_kamera):
        fx, fy = k.odak(kamera)
        for deger in (fx, fy):
            hata = abs(deger - GERCEK_ODAK) / GERCEK_ODAK * 100
            assert hata < 0.1, f"kamera {kamera}: {deger:.3f} px, hata {hata:.4f}%"


def test_gurultusuz_rms_sifira_yakin(gurultusuz):
    _, k = gurultusuz
    assert k.rms_px < 1e-3


def test_dis_parametreler_kamera_sifira_gore(gurultusuz):
    """Sozlesme: kamera 0 referans -- R birim, T sifir."""
    _, k = gurultusuz
    assert np.allclose(k.Rs[0], np.eye(3), atol=1e-6)
    assert np.allclose(k.Ts[0], 0.0, atol=1e-6)
    assert k.taban_uzunluklari()[0] == pytest.approx(0.0, abs=1e-9)


def test_taban_uzunluklari_gercek_geometriyi_veriyor(gurultusuz):
    """Duzenek olcegi board'un fiziksel kare kenarindan gelir; mm mertebesinde tutmali."""
    s, k = gurultusuz
    merkezler = [c.merkez for c in s.kameralar]
    for i, taban in enumerate(k.taban_uzunluklari()):
        gercek = float(np.linalg.norm(merkezler[i] - merkezler[0]))
        assert taban == pytest.approx(gercek, abs=2e-3), f"kamera {i}"


def test_rotasyon_her_zaman_matris(gurultusuz):
    """OpenCV Rodrigues vektoru donduruyor; sinirda matrise cevrilmeli."""
    _, k = gurultusuz
    for R in k.Rs:
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_mask_ile_nokta_celisirse_hata():
    s = sahne_uret(SPEC, n_kare=6, seed=1)
    bozuk = s.mask.copy()
    bozuk[0, 0] = 0                       # nokta var ama mask gormedi diyor
    with pytest.raises(ValueError, match="celisiyor"):
        kalibre_et(s.obj_noktalari, s.img_noktalari,
                   [c.boyut for c in s.kameralar], bozuk)


def test_mask_dtype_zorunlu():
    s = sahne_uret(SPEC, n_kare=6, seed=1)
    with pytest.raises(ValueError, match="uint8"):
        kalibre_et(s.obj_noktalari, s.img_noktalari,
                   [c.boyut for c in s.kameralar], s.mask.astype(bool))


def test_kare_sayisi_uyusmazsa_hata():
    s = sahne_uret(SPEC, n_kare=6, seed=1)
    with pytest.raises(ValueError, match="kare"):
        kalibre_et(s.obj_noktalari[:-1], s.img_noktalari,
                   [c.boyut for c in s.kameralar], s.mask)


def test_boyut_sayisi_uyusmazsa_hata():
    s = sahne_uret(SPEC, n_kare=6, seed=1)
    with pytest.raises(ValueError, match="goruntu boyutu"):
        kalibre_et(s.obj_noktalari, s.img_noktalari, [(1280, 720)], s.mask)


def test_fisheye_dort_parametreli_bozulma_istiyor():
    """OpenCV fisheye modeli 4 parametre bekler, pinhole 5 (B.4.1).

    Bes eleman verildiginde OpenCV ya sessizce sonuncuyu yok sayar ya da hata
    verir; her iki durumda da davranis belirsizdir.
    """
    from calib.multiview import bozulma_uzunlugu

    assert bozulma_uzunlugu(balik_gozu=True) == 4
    assert bozulma_uzunlugu(balik_gozu=False) == 5


def test_kalibrasyon_liste_uzunluklari_dogrulaniyor():
    """Kalibrasyon yapicisi tutarsiz listeleri kabul etmemeli (B.4.3)."""
    import numpy as np
    import pytest
    from calib.multiview import Kalibrasyon

    with pytest.raises(ValueError):
        Kalibrasyon(rms_px=0.3, Ks=[np.eye(3), np.eye(3)],
                    bozulmalar=[np.zeros(5)], Rs=[np.eye(3)], Ts=[np.zeros((3, 1))])

    with pytest.raises(ValueError, match="Ks"):
        Kalibrasyon(rms_px=0.3, Ks=[np.zeros((2, 2))], bozulmalar=[np.zeros(5)],
                    Rs=[np.eye(3)], Ts=[np.zeros((3, 1))])
