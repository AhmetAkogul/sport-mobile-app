"""calib/ girdi dogrulamasi -- dis inceleme B.1.1, B.1.2, B.1.3, B.2.1, B.2.3,
B.3.3, B.4.2, B.5.2. Her bozuk girdi anlasilir bir ValueError ile kurulumda
durmali; OpenCV'nin icinde anlamsiz bir hataya donusmemeli."""
import json

import numpy as np
import pytest

from calib.board import BoardSpec, goruntu_uret, pdf_yaz, sozluk_al
from calib.detect import hazirla, kose_bul
from calib.io import _matris_yaz, _vektor_yaz
from calib.multiview import _dogrula
from calib.synthetic import sahne_uret


@pytest.mark.parametrize("ayar", [
    {"square_mm": 0.0, "marker_mm": -5.0},
    {"square_mm": -60.0, "marker_mm": -70.0},
    {"square_mm": float("nan")},
])
def test_board_olculeri_pozitif_olmali(ayar):
    with pytest.raises(ValueError, match="pozitif"):
        BoardSpec(**ayar)


@pytest.mark.parametrize("ad", ["CharucoBoard", "DICT_YOK_BOYLE", 42])
def test_sozluk_adi_gercek_bir_sozluk_olmali(ad):
    with pytest.raises(ValueError, match="bilinmeyen ArUco"):
        sozluk_al(ad)


def test_gecerli_sozluk_alinir():
    assert sozluk_al("DICT_5X5_100") is not None


@pytest.mark.parametrize("dpi", [0, -300, 300.0])
def test_dpi_pozitif_tamsayi(dpi):
    with pytest.raises(ValueError, match="dpi"):
        goruntu_uret(BoardSpec(), dpi=dpi)


def test_pdf_yan_dosyasi_atomik_ve_gecerli_json(tmp_path):
    cikti = pdf_yaz(BoardSpec(), tmp_path / "board.pdf", dpi=72)
    yan = cikti.with_suffix(".json")
    assert json.loads(yan.read_text())["dpi"] == 72
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="kenar_mm"):
        pdf_yaz(BoardSpec(), tmp_path / "b2.pdf", dpi=72, kenar_mm=-1.0)


@pytest.mark.parametrize("esik", [0, -5, 3, 2.5])
def test_min_kose_en_az_dort(esik):
    with pytest.raises(ValueError, match="min_kose"):
        kose_bul(np.zeros((10, 10), np.uint8), dedektor=None, min_kose=esik)


def test_bos_sahne_reddedilir():
    with pytest.raises(ValueError, match="hic kare yok"):
        hazirla([[], []], BoardSpec())


@pytest.mark.parametrize("ayar,mesaj", [
    ({"n_kare": 0}, "n_kare"),
    ({"gurultu_px": -0.5}, "gurultu_px"),
])
def test_sahne_uret_parametreleri(ayar, mesaj):
    with pytest.raises(ValueError, match=mesaj):
        sahne_uret(**ayar)


def test_kayitta_nan_yazilmaz():
    with pytest.raises(ValueError, match="sonlu"):
        _matris_yaz(np.array([[1.0, 0, 0], [0, np.nan, 0], [0, 0, 1]]))
    with pytest.raises(ValueError, match="sonlu"):
        _vektor_yaz(np.array([0.0, np.inf, 0.0, 0.0, 0.0]))


def test_mask_sekli_ve_degerleri_dogrulanir():
    obj = [np.zeros((4, 3), np.float32)]
    img = [[np.zeros((4, 2), np.float32)]]
    with pytest.raises(ValueError, match="2 boyutlu"):
        _dogrula(obj, img, np.ones(1, np.uint8))
    with pytest.raises(ValueError, match="0/1"):
        _dogrula(obj, img, np.full((1, 1), 5, np.uint8))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        _dogrula([np.zeros((4, 2), np.float32)], img, np.ones((1, 1), np.uint8))


def test_meta_tarihi_iso_olmali():
    from calib.io import Meta
    with pytest.raises(ValueError, match="ISO 8601"):
        Meta(tarih_iso="gecen sali")
    Meta(tarih_iso="2026-10-19T09:00:00")          # dilimsiz: UTC kabul edilir


@pytest.mark.parametrize("ayar,mesaj", [
    ({"K": np.eye(2)}, "K"),
    ({"R": np.diag([1.0, 1.0, -1.0])}, "donme"),
    ({"t": np.zeros(2)}, "t 3"),
    ({"boyut": (0, 480)}, "boyut"),
])
def test_kamera_dogrulanir(ayar, mesaj):
    from calib.synthetic import Kamera
    temel = dict(K=np.array([[900.0, 0, 320], [0, 900.0, 240], [0, 0, 1]]),
                 R=np.eye(3), t=np.zeros((3, 1)), boyut=(640, 480))
    with pytest.raises(ValueError, match=mesaj):
        Kamera(**{**temel, **ayar})
