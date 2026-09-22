"""calib.board kabul testleri.

KOD-PLANI.md kabul olcutu: "Uretilen board tekrar tespit edilince ayni kose sayisi."
"""
import numpy as np
import pytest

from calib.board import BoardSpec, board_kur, goruntu_uret, pdf_yaz
from calib.detect import dedektor_kur, kose_bul


def test_spec_gecersiz_marker_reddedilir():
    with pytest.raises(ValueError):
        BoardSpec(5, 7, square_mm=40, marker_mm=40)   # marker >= kare
    with pytest.raises(ValueError):
        BoardSpec(2, 7)                                # cok az kare


def test_ic_kose_sayisi_opencv_ile_ayni():
    """Bizim hesabimiz OpenCV'nin model noktalariyla uyusmali.

    Uyusmazsa objPoints ile imagePoints farkli uzunlukta olur ve kalibrasyon
    anlasilmaz bir hatayla duser.
    """
    spec = BoardSpec(5, 7)
    assert spec.inner_corners == len(board_kur(spec).getChessboardCorners()) == 24


def test_fiziksel_olcek_mm_cinsinden_dogru():
    spec = BoardSpec(5, 7, square_mm=60)
    assert (spec.width_mm, spec.height_mm) == (300.0, 420.0)
    # OpenCV metre bekliyor; donusum tek yerde yapiliyor
    assert board_kur(spec).getSquareLength() == pytest.approx(0.060, abs=1e-6)


def test_goruntu_uret_dpi_oranini_korur():
    spec = BoardSpec(5, 7, square_mm=60)
    g = goruntu_uret(spec, dpi=300)
    beklenen_g = round(300.0 * 300 / 25.4)
    beklenen_y = round(420.0 * 300 / 25.4)
    assert g.shape == (beklenen_y, beklenen_g)
    assert g.dtype == np.uint8


@pytest.mark.parametrize("kare", [(5, 7), (4, 6), (7, 5)])
def test_uretilen_board_geri_tespit_edilir(kare):
    """KABUL TESTI: kendi urettigimiz board'da tum ic koseler bulunmali."""
    spec = BoardSpec(kare[0], kare[1])
    g = goruntu_uret(spec, dpi=150)
    t = kose_bul(g, dedektor_kur(spec))
    assert t is not None, "kendi urettigimiz board tespit edilemedi"
    assert t.kose_sayisi == spec.inner_corners


def test_pdf_ve_spec_yazilir(tmp_path):
    spec = BoardSpec(5, 7)
    yol = pdf_yaz(spec, tmp_path / "board.pdf", dpi=100)
    assert yol.exists() and yol.stat().st_size > 0
    yan = yol.with_suffix(".json")
    assert yan.exists(), "hangi board'la kalibre edildigi kaydedilmeli"
    assert "square_mm" in yan.read_text()


@pytest.mark.parametrize("olcu,beklenen", [
    ((210, 297), "A4"),
    ((274, 390), "A3"),
    ((324, 460), "A2"),     # varsayilan 5x7 / 60 mm board bu gruba duser
    ((390, 274), "A3"),     # yatay da kabul edilmeli
    ((1200, 1500), "A0 ustu -- plot/afis baskisi gerekir"),
])
def test_kagit_onerisi(olcu, beklenen):
    """Board %100 olcekte basilmak zorunda; kagit kucukse yazici kucultur."""
    from calib.board import kagit_oner
    assert kagit_oner(*olcu) == beklenen
