"""pose3d.triangulate kabul testleri.

PROJE-PLANI.md Kapi 2 olcutu: 2 m mesafede bilinen nesnede 3B konum hatasi < 10 mm.
Burada sentetik esdegeri sinanir; fiziksel dogrulama kameralar gelince yapilacak.
"""
import numpy as np
import pytest

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import (
    dunyadan_kamera0, nokta_gozlemleri, olculu_cubuk, rig_yay, sahne_uret,
)
from pose3d.triangulate import (
    bilinen_mesafe_hatasi, mesafe, projeksiyon_matrisi, ucgenle, ucgenle_toplu,
)

SPEC = BoardSpec(5, 7)


def kalibrasyon(kameralar, gurultu=0.0, seed=1):
    s = sahne_uret(SPEC, kameralar=kameralar, n_kare=20, gurultu_px=gurultu,
                   dagilim="genis", seed=seed)
    return kalibre_et(s.obj_noktalari, s.img_noktalari,
                      [c.boyut for c in kameralar], s.mask)


@pytest.fixture(scope="module")
def duzenek():
    kameralar = rig_yay(odak_px=900.0)
    return kameralar, kalibrasyon(kameralar)


def test_projeksiyon_matrisi_bicimi():
    P = projeksiyon_matrisi(np.eye(3), np.eye(3), np.zeros((3, 1)))
    assert P.shape == (3, 4)
    assert np.allclose(P[:, :3], np.eye(3))


def test_gurultusuzde_konum_hatasi_mikron_mertebesinde(duzenek):
    """KABUL TESTI: gurultusuz veride triangulation neredeyse tam."""
    kameralar, kalib = duzenek
    noktalar, _ = olculu_cubuk(uzunluk_m=1.0, n_isaret=5)
    kest = ucgenle_toplu(kalib, nokta_gozlemleri(kameralar, noktalar, seed=2))
    gercek = dunyadan_kamera0(kameralar, noktalar)
    for k, g in zip(kest, gercek):
        assert k.gecerli
        assert np.linalg.norm(k.nokta - g) * 1000 < 0.1     # mm


def test_bilinen_mesafe_kapi_2_olcutu(duzenek):
    """KABUL TESTI: bilinen mesafe duzeneginde hata < 10 mm (Kapi 2)."""
    kameralar, kalib = duzenek
    noktalar, gercek_mesafeler = olculu_cubuk(uzunluk_m=1.0, n_isaret=5)
    kest = ucgenle_toplu(kalib, nokta_gozlemleri(kameralar, noktalar, seed=2))
    r = bilinen_mesafe_hatasi(kest, gercek_mesafeler)
    assert r["kapi_2_gecti"]
    assert r["en_buyuk_hata_mm"] < 1.0
    assert r["n"] == 10        # 5 isaretten C(5,2) cift


def test_yeniden_izdusum_artigi_kucuk(duzenek):
    kameralar, kalib = duzenek
    noktalar, _ = olculu_cubuk(n_isaret=3)
    for k in ucgenle_toplu(kalib, nokta_gozlemleri(kameralar, noktalar, seed=2)):
        assert k.artik_px < 0.01


def test_tek_gorusten_triangulation_olmaz(duzenek):
    """Tek kameradan derinlik cikmaz -- projenin tum tezi zaten bu."""
    _, kalib = duzenek
    with pytest.raises(ValueError, match="en az 2 gorus"):
        ucgenle(kalib, {0: np.array([640.0, 360.0])})


def test_bozuk_gozlem_bicimi_reddedilir(duzenek):
    _, kalib = duzenek
    with pytest.raises(ValueError, match=r"\(2,\)"):
        ucgenle(kalib, {0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0])})


def test_iki_kamera_da_yeterli(duzenek):
    """Ortulme durumunda iki gorus kalsa bile hat ayakta kalmali."""
    kameralar, kalib = duzenek
    noktalar, _ = olculu_cubuk(n_isaret=3)
    tam = nokta_gozlemleri(kameralar, noktalar, seed=2)
    iki = [{k: v for k, v in g.items() if k in (0, 3)} for g in tam]
    gercek = dunyadan_kamera0(kameralar, noktalar)
    for g, gercek_nokta in zip(ucgenle_toplu(kalib, iki), gercek):
        assert g.goren_kamera == 2
        assert np.linalg.norm(g.nokta - gercek_nokta) * 1000 < 1.0


def test_daha_cok_kamera_gurultude_daha_iyi():
    """Gurultu altinda 4 kamera, 2 kameradan daha kesin olmali."""
    kameralar = rig_yay(odak_px=900.0)
    kalib = kalibrasyon(kameralar)
    noktalar, _ = olculu_cubuk(uzunluk_m=1.0, n_isaret=5)
    gercek = dunyadan_kamera0(kameralar, noktalar)

    def ortalama_hata(kamera_seti, tohum_sayisi=12):
        hatalar = []
        for t in range(tohum_sayisi):
            goz = nokta_gozlemleri(kameralar, noktalar, gurultu_px=0.5, seed=100 + t)
            secili = [{k: v for k, v in g.items() if k in kamera_seti} for g in goz]
            for k, gn in zip(ucgenle_toplu(kalib, secili), gercek):
                if k.gecerli:
                    hatalar.append(np.linalg.norm(k.nokta - gn) * 1000)
        return float(np.mean(hatalar))

    assert ortalama_hata((0, 1, 2, 3)) < ortalama_hata((0, 3))


def test_mesafe_yardimcisi_simetrik(duzenek):
    kameralar, kalib = duzenek
    noktalar, _ = olculu_cubuk(n_isaret=2)
    a, b = ucgenle_toplu(kalib, nokta_gozlemleri(kameralar, noktalar, seed=2))
    assert mesafe(a, b) == pytest.approx(mesafe(b, a))
    assert mesafe(a, b) == pytest.approx(1.0, abs=1e-3)


def test_bilinen_mesafe_bos_liste_reddedilir(duzenek):
    _, kalib = duzenek
    with pytest.raises(ValueError, match="en az bir"):
        bilinen_mesafe_hatasi([], [])


def test_olculu_cubuk_gecerli_uretim():
    noktalar, mesafeler = olculu_cubuk(uzunluk_m=1.2, n_isaret=4)
    assert noktalar.shape == (4, 3)
    assert len(mesafeler) == 6
    uzun = max(m for _, _, m in mesafeler)
    assert uzun == pytest.approx(1.2)
    with pytest.raises(ValueError, match="en az 2"):
        olculu_cubuk(n_isaret=1)


# --- dis inceleme T.1 / T.3 / T.6 --------------------------------------------

def test_gecersiz_kamera_indeksi_reddediliyor(duzenek):
    """Sozlukte olmayan kamera indeksi IndexError yerine net hata vermeli (T.1).

    `iskelet_ucgenle` ucgenle'nin ValueError'unu yutuyor; IndexError ise
    yutulmuyor ve hatti cokertiyor. Ikisi de kotu: biri sessiz, digeri
    anlasilmaz. Sinir kontrolu basta yapilmali.
    """
    import numpy as np
    import pytest
    from pose3d.triangulate import ucgenle

    _, kalib = duzenek
    n = len(kalib.Ks)
    gozlem = {0: np.array([100.0, 100.0]), n + 5: np.array([200.0, 200.0])}
    with pytest.raises(ValueError, match="kamera indeksi"):
        ucgenle(kalib, gozlem)


def test_artik_px_sonsuz_olmaz(duzenek):
    """Kamera duzlemine dusen nokta icin artik inf degil NaN olmali (T.3).

    inf bir sayi gibi ortalamaya giriyor ve butun artigi inf yapiyor;
    downstream kod bunu esikle karsilastirinca sessizce "cok kotu" diye
    yorumluyor. "Hesaplanamadi" demek dogrusu.
    """
    import numpy as np
    from pose3d.triangulate import Ucgenleme

    u = Ucgenleme(nokta=np.zeros(3), goren_kamera=2, artik_px=float("inf"))
    assert not np.isinf(u.artik_px)
    assert np.isnan(u.artik_px)


def test_ucgenleme_sekil_dogrulamasi():
    """Ucgenleme dogrulamasiz bir sinifti; bozuk sekil sessizce yayiliyordu (T.6)."""
    import numpy as np
    import pytest
    from pose3d.triangulate import Ucgenleme

    with pytest.raises(ValueError, match="nokta"):
        Ucgenleme(nokta=np.zeros(2), goren_kamera=2, artik_px=0.1)
    with pytest.raises(ValueError, match="kovaryans"):
        Ucgenleme(nokta=np.zeros(3), goren_kamera=2, artik_px=0.1,
                  kovaryans=np.zeros((2, 2)))
