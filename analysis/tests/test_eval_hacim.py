"""eval.hacim kabul testleri.

PROJE-PLANI.md Faz 2: calisma hacminin farkli noktalarinda hata haritasi.
Kritik nokta, olculen buyuklugun cerceveden bagimsiz olmasi: harita mutlak
konumu degil, bilinen uzunluktaki probun olculen uzunlugunu degerlendirir.
"""
import numpy as np
import pytest

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import rig_yay, sahne_uret
from eval.hacim import (
    PROB_YONLERI, VARSAYILAN_PROB_M, hacim_haritasi, kullanilabilir_bolge,
)

SPEC = BoardSpec(5, 7)


@pytest.fixture(scope="module")
def duzenek():
    kameralar = rig_yay(odak_px=900.0)
    s = sahne_uret(SPEC, kameralar=kameralar, n_kare=20, gurultu_px=0.0,
                   dagilim="genis", seed=8080)
    kalib = kalibre_et(s.obj_noktalari, s.img_noktalari,
                       [c.boyut for c in kameralar], s.mask)
    return kalib, kameralar


@pytest.fixture(scope="module")
def harita(duzenek):
    kalib, kameralar = duzenek
    return hacim_haritasi(kalib, kameralar, adim_m=0.5, gurultu_px=0.0, seed=1)


def test_izgara_bicimi_tutarli(harita):
    assert harita.hata_mm.shape == (len(harita.z_m), len(harita.x_m))
    assert harita.gorunurluk.shape == harita.hata_mm.shape
    assert harita.prob_m == VARSAYILAN_PROB_M


def test_uc_yonde_olculuyor():
    """Tek yon yaniltir; derinlik ekseni yanal eksenden zayiftir."""
    assert len(PROB_YONLERI) == 3
    eksenler = {ad for ad, _ in PROB_YONLERI}
    assert eksenler == {"x", "y", "z"}


def test_gurultusuzde_hata_ihmal_edilebilir(duzenek):
    """Gurultu ve kalibrasyon hatasi yoksa prob uzunlugu neredeyse tam olculmeli."""
    kalib, kameralar = duzenek
    h = hacim_haritasi(kalib, kameralar, x_araligi=(-0.5, 0.5), z_araligi=(2.0, 3.0),
                       adim_m=0.5, gurultu_px=0.0, seed=2)
    gecerli = h.hata_mm[np.isfinite(h.hata_mm)]
    assert gecerli.size > 0
    assert gecerli.max() < 0.5      # mm


def test_gurultu_artinca_hata_artar(duzenek):
    kalib, kameralar = duzenek
    ortak = dict(x_araligi=(-0.5, 0.5), z_araligi=(2.0, 3.0), adim_m=0.5, seed=3)
    dusuk = hacim_haritasi(kalib, kameralar, gurultu_px=0.1, **ortak)
    yuksek = hacim_haritasi(kalib, kameralar, gurultu_px=1.0, **ortak)
    assert yuksek.ozet()["hata_ortalama_mm"] > dusuk.ozet()["hata_ortalama_mm"]


def test_uzakta_hata_buyur(duzenek):
    """Derinlik arttikca ayni piksel gurultusu daha buyuk mm hatasina donusur."""
    kalib, kameralar = duzenek
    ortak = dict(x_araligi=(-0.25, 0.25), adim_m=0.5, gurultu_px=0.5, seed=4)
    yakin = hacim_haritasi(kalib, kameralar, z_araligi=(1.5, 2.0), **ortak)
    uzak = hacim_haritasi(kalib, kameralar, z_araligi=(4.0, 4.5), **ortak)
    assert uzak.ozet()["hata_ortalama_mm"] > yakin.ozet()["hata_ortalama_mm"]


def test_gorunmeyen_bolge_olculemez_isaretlenir(duzenek):
    """Kameralarin arkasi iki kameradan az gorulur -> NaN ve gorunurluk < 2."""
    kalib, kameralar = duzenek
    h = hacim_haritasi(kalib, kameralar, x_araligi=(-1.0, 1.0),
                       z_araligi=(-4.0, -3.0), adim_m=0.5, gurultu_px=0.0, seed=5)
    assert np.isnan(h.hata_mm).all()
    assert (h.gorunurluk < 2).all()
    assert h.olculebilir_oran == 0.0
    with pytest.raises(ValueError, match="olculemedi"):
        h.ozet()


def test_kullanilabilir_bolge_esige_duyarli(harita):
    gevsek = kullanilabilir_bolge(harita, esik_mm=1000.0)
    siki = kullanilabilir_bolge(harita, esik_mm=0.001)
    assert gevsek["gecen_hucre"] >= siki["gecen_hucre"]
    assert gevsek["gecen_oran"] <= 1.0
    assert gevsek["toplam_hucre"] == harita.hata_mm.size


def test_ozet_serilestirilebilir(harita):
    import json
    metin = json.dumps(harita.ozet(), allow_nan=False)
    assert "hata_p95_mm" in metin and "olculebilir_oran" in metin


def test_gecersiz_adim_reddedilir(duzenek):
    kalib, kameralar = duzenek
    with pytest.raises(ValueError, match="adim pozitif"):
        hacim_haritasi(kalib, kameralar, adim_m=0.0)


def test_genis_izgara_tohum_cakismasi_reddedilir(duzenek):
    """Dis inceleme H.3: 1000+ hucre genisliginde iki hucre ayni tohumu alirdi."""
    kalib, kameralar = duzenek
    with pytest.raises(ValueError, match="tohumlari cakisir"):
        hacim_haritasi(kalib, kameralar, adim_m=0.001, x_araligi=(-0.5, 0.5),
                       z_araligi=(0.0, 0.0))


def test_kismen_olculen_hucre_olculmus_sayilmaz(duzenek, monkeypatch):
    """Dis inceleme H.1: uc prob yonunden biri bile olculemezse hucre NaN."""
    import eval.hacim as hacim

    gercek = hacim.nokta_gozlemleri
    sayac = {"n": 0}

    def ucuncu_yonu_gizle(kameralar, uclar, **k):
        sayac["n"] += 1
        g = gercek(kameralar, uclar, **k)
        return [dict(), dict()] if sayac["n"] % 3 == 0 else g

    monkeypatch.setattr(hacim, "nokta_gozlemleri", ucuncu_yonu_gizle)
    kalib, kameralar = duzenek
    h = hacim_haritasi(kalib, kameralar, adim_m=1.0, x_araligi=(0.0, 0.0),
                       z_araligi=(0.0, 0.0), gurultu_px=0.0, seed=1)
    assert np.isnan(h.hata_mm).all()


def test_azalan_aralik_reddedilir(duzenek):
    """Dis inceleme H.4: azalan aralik sessizce bos izgara uretirdi."""
    kalib, kameralar = duzenek
    with pytest.raises(ValueError, match="artan"):
        hacim_haritasi(kalib, kameralar, x_araligi=(1.0, -1.0))
