"""calib.detect kabul testleri.

Asil odak detectionMask: yanlis mask hata vermez, sessizce bozuk kalibrasyon uretir.
"""
import numpy as np
import pytest

from calib.board import BoardSpec, goruntu_uret
from calib.detect import (
    VARSAYILAN_MIN_KOSE,
    Tespit,
    dedektor_kur,
    dizin_tara,
    hazirla,
    kose_bul,
)

SPEC = BoardSpec(5, 7)


@pytest.fixture(scope="module")
def temiz_tespit():
    g = goruntu_uret(SPEC, dpi=150)
    t = kose_bul(g, dedektor_kur(SPEC))
    assert t is not None
    return t


def test_bilinen_goruntude_beklenen_kose_sayisi(temiz_tespit):
    """KABUL TESTI: bilinen goruntude beklenen kose sayisi +-0."""
    assert temiz_tespit.kose_sayisi == SPEC.inner_corners
    assert temiz_tespit.idler.dtype == np.int32
    assert temiz_tespit.koseler.shape == (SPEC.inner_corners, 1, 2)


def test_bos_goruntude_tespit_yok():
    bos = np.full((600, 400), 255, dtype=np.uint8)
    assert kose_bul(bos, dedektor_kur(SPEC)) is None


def test_min_kose_esigi_uygulanir(temiz_tespit):
    """Esigin uzerine cikilamayan kare 'gormedi' sayilmali."""
    g = goruntu_uret(SPEC, dpi=150)
    d = dedektor_kur(SPEC)
    assert kose_bul(g, d, min_kose=SPEC.inner_corners) is not None
    assert kose_bul(g, d, min_kose=SPEC.inner_corners + 1) is None


def _tespit(idler):
    """Verilen kose kimlikleriyle yapay tespit."""
    idler = np.asarray(idler, dtype=np.int32)
    koseler = np.stack([idler, idler], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    return Tespit(koseler=koseler, idler=idler, marker_sayisi=len(idler) // 2)


def test_mask_opencv_sozlesmesine_uyuyor():
    """detectionMask: (kamera x kare), CV_8UC1 -> numpy uint8."""
    tam = list(range(SPEC.inner_corners))
    tespitler = [[_tespit(tam), _tespit(tam)], [_tespit(tam), _tespit(tam)]]
    obj, img, mask = hazirla(tespitler, SPEC)
    assert mask.dtype == np.uint8
    assert mask.shape == (2, 2)
    assert mask.tolist() == [[1, 1], [1, 1]]
    assert len(obj) == 2 and len(img) == 2


def test_gormeyen_kamera_maskte_sifir():
    tam = list(range(SPEC.inner_corners))
    # kamera 1 ikinci kareyi gormedi
    tespitler = [[_tespit(tam), _tespit(tam)], [_tespit(tam), None]]
    obj, img, mask = hazirla(tespitler, SPEC)
    assert mask.tolist() == [[1, 1], [1, 0]]
    # gormeyen kamera icin bos nokta dizisi konur, hiza bozulmaz
    assert img[1][1].shape == (0, 1, 2)


def test_objpoints_ve_imagepoints_birebir_esleser():
    """Kare basina tek nokta kumesi olmali; aksi halde kalibrasyon coker."""
    a = list(range(0, 20))
    b = list(range(10, 24))          # ortak: 10..19 -> 10 kose
    tespitler = [[_tespit(a)], [_tespit(b)]]
    obj, img, mask = hazirla(tespitler, SPEC)
    assert len(obj) == 1
    assert obj[0].shape == (10, 3)
    for kamera in img:
        assert kamera[0].shape == (10, 1, 2)
    assert mask.tolist() == [[1], [1]]


def test_ortak_kose_yetersizse_kare_dusurulur():
    tespitler = [[_tespit(range(0, 8))], [_tespit(range(20, 24))]]  # ortak yok
    obj, img, mask = hazirla(tespitler, SPEC)
    assert obj == [] and mask.shape == (2, 0)


def test_kameralarda_kare_sayisi_farkliysa_hata():
    tam = list(range(SPEC.inner_corners))
    with pytest.raises(ValueError, match="kare sayisi farkli"):
        hazirla([[_tespit(tam), _tespit(tam)], [_tespit(tam)]], SPEC)


def test_dizin_tara_orani_hesaplar(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "01.png"), goruntu_uret(SPEC, dpi=150))
    cv2.imwrite(str(tmp_path / "02.png"), np.full((600, 400), 255, dtype=np.uint8))
    r = dizin_tara(tmp_path, SPEC)
    assert r["kare_sayisi"] == 2
    assert r["tespit_edilen"] == 1
    assert r["tespit_orani"] == 0.5
    assert r["min_kose"] == VARSAYILAN_MIN_KOSE
