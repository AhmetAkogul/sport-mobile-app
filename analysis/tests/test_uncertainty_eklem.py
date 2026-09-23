"""uncertainty.eklem kabul testleri.

Sinanan sey bir sayinin dogrulugu degil, **davranisin dogrulugu**: belirsizlik
gurultuyle buyuyor mu, kamera sayisiyla kuculuyor mu, ve hesaplanamadiginda
sifir yerine NaN mi donuyor. Sonuncusu kritik: sifir donseydi hesaplanamayan
eklem kusursuz olcum gibi gorunur ve form karari yanlis yerde kesinlesirdi.
"""
import numpy as np
import pytest

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import rig_yay, sahne_uret
from eval.form import Karar, form_degerlendir
from pose3d.iskelet import REFERANS_ISKELET, iskelet_ucgenle
from pose3d.pose2d import sentetik_poz
from uncertainty.eklem import (
    belirsizlik_ozeti, eklem_belirsizligi, eklem_kovaryanslari,
)

ISK = REFERANS_ISKELET
DURUS = {
    "boyun": (0, 0.55, 0), "sag_omuz": (-0.18, 0.45, 0),
    "sag_dirsek": (-0.22, 0.18, 0.05), "sag_bilek": (-0.24, -0.08, 0.08),
    "sol_omuz": (0.18, 0.45, 0), "sol_dirsek": (0.22, 0.18, 0.05),
    "sol_bilek": (0.24, -0.08, 0.08), "sag_kalca": (-0.11, 0, 0),
    "sag_diz": (-0.12, -0.42, 0.02), "sag_ayak_bilegi": (-0.12, -0.85, 0),
    "sol_kalca": (0.11, 0, 0), "sol_diz": (0.12, -0.42, 0.02),
    "sol_ayak_bilegi": (0.12, -0.85, 0),
}
P3 = np.array([DURUS[e] for e in ISK.eklemler])


@pytest.fixture(scope="module")
def duzenek():
    kameralar = rig_yay(odak_px=900.0)
    s = sahne_uret(BoardSpec(5, 7), kameralar=kameralar, n_kare=20,
                   gurultu_px=0.0, dagilim="genis", seed=8080)
    kalib = kalibre_et(s.obj_noktalari, s.img_noktalari,
                       [c.boyut for c in kameralar], s.mask)
    return kalib, kameralar


def _gozlemler(kameralar, ortulu=(), kamera_sayisi=None):
    secili = kameralar if kamera_sayisi is None else kameralar[:kamera_sayisi]
    return {i: sentetik_poz(kam, P3, ISK, gurultu_px=0.0, ortulu=ortulu,
                            seed=i, kamera_indeksi=i)
            for i, kam in enumerate(secili)}


def _iskelet(kalib, gozlemler):
    return iskelet_ucgenle(kalib, gozlemler, ISK)


# --- temel davranis ----------------------------------------------------------

def test_gorunur_eklemler_icin_belirsizlik_uretiliyor(duzenek):
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    sigmalar = eklem_belirsizligi(kalib, isk, goz, sigma_px=1.0)

    assert sigmalar.shape == (len(ISK),)
    assert np.isfinite(sigmalar[isk.gorunur]).all()
    assert (sigmalar[isk.gorunur] > 0).all()


def test_belirsizlik_gurultuyle_dogru_orantili(duzenek):
    """Cov = sigma_px^2 (J^T J)^-1 oldugu icin sigma, sigma_px ile dogrusal."""
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    bir = eklem_belirsizligi(kalib, isk, goz, sigma_px=1.0)
    iki = eklem_belirsizligi(kalib, isk, goz, sigma_px=2.0)
    m = isk.gorunur
    assert np.allclose(iki[m], 2.0 * bir[m], rtol=1e-9)


def test_daha_cok_kamera_belirsizligi_dusuruyor(duzenek):
    """Dort kamera iki kameradan daha kesin olmali."""
    kalib, kameralar = duzenek
    iki_goz = _gozlemler(kameralar, kamera_sayisi=2)
    dort_goz = _gozlemler(kameralar)
    iki = eklem_belirsizligi(kalib, _iskelet(kalib, iki_goz), iki_goz)
    dort = eklem_belirsizligi(kalib, _iskelet(kalib, dort_goz), dort_goz)
    m = np.isfinite(iki) & np.isfinite(dort)
    assert m.any()
    assert np.median(dort[m]) < np.median(iki[m])


def test_belirsizlik_makul_mertebede(duzenek):
    """1 px tespit gurultusunde eklem belirsizligi milimetre mertebesinde olmali.

    Hata butcesi deneyi ayni rejimde birkac mm veriyordu; buyuklugu tutmayan bir
    kestirim, karar katmanina yanlis olcek tasir.
    """
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    mm = eklem_belirsizligi(kalib, isk, goz, sigma_px=1.0)[isk.gorunur] * 1000.0
    assert 0.1 < float(np.median(mm)) < 50.0


# --- eksik veri --------------------------------------------------------------

def test_gorunmeyen_eklem_nan_doner(duzenek):
    """Sifir degil NaN: hesaplanamayan eklem kusursuz olcum gibi gorunmemeli."""
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar, ortulu=("sag_diz", "sag_ayak_bilegi"))
    isk = _iskelet(kalib, goz)
    sigmalar = eklem_belirsizligi(kalib, isk, goz)
    for eklem in ("sag_diz", "sag_ayak_bilegi"):
        j = ISK.indeks(eklem)
        assert not isk.gorunur[j]
        assert np.isnan(sigmalar[j])


def test_gorunurluk_karari_yeniden_verilmiyor(duzenek):
    """Modul kendi gorunurluk mantigini kurmamali; iskeletinkini izlemeli."""
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar, ortulu=("sol_bilek",))
    isk = _iskelet(kalib, goz)
    sigmalar = eklem_belirsizligi(kalib, isk, goz)
    assert np.isfinite(sigmalar[isk.gorunur]).all()
    assert np.isnan(sigmalar[~isk.gorunur]).all()


def test_kovaryans_simetrik_ve_pozitif_tanimli(duzenek):
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    for kov in eklem_kovaryanslari(kalib, isk, goz, sigma_px=1.0):
        if kov is None:
            continue
        assert np.allclose(kov, kov.T, atol=1e-18)
        assert (np.linalg.eigvalsh(kov) > 0).all()


def test_kovaryans_anizotropik(duzenek):
    """Derinlik yonu yanal yonlerden kotu olmali; skalara indirgemenin bedeli bu.

    Ozdegerlerin birbirine esit ciktigi bir sonuc, Jacobian'in yanlis
    kuruldugunun isareti olurdu.
    """
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    oranlar = [
        float(np.linalg.eigvalsh(k).max() / np.linalg.eigvalsh(k).min())
        for k in eklem_kovaryanslari(kalib, isk, goz) if k is not None
    ]
    assert oranlar
    assert max(oranlar) > 2.0


# --- form katmanina baglanti -------------------------------------------------

def test_form_katmanina_dogrudan_verilebiliyor(duzenek):
    """A6'nin amaci: gercek hat, form katmanini belirsizlikle besleyebilmeli."""
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    sigmalar = eklem_belirsizligi(kalib, isk, goz, sigma_px=1.0)
    rapor = form_degerlendir(isk, konum_belirsizligi_m=np.nan_to_num(sigmalar))
    assert set(rapor.olcumler)
    assert all(o.karar in tuple(Karar) for o in rapor.olcumler.values())


def test_buyuk_gurultu_karari_belirsizlestiriyor(duzenek):
    """Belirsizlik buyudukce karar BELIRSIZ'e kaymali -- zincir uctan uca calismali."""
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    az = eklem_belirsizligi(kalib, isk, goz, sigma_px=0.1)
    cok = eklem_belirsizligi(kalib, isk, goz, sigma_px=200.0)
    a = form_degerlendir(isk, konum_belirsizligi_m=np.nan_to_num(az))
    b = form_degerlendir(isk, konum_belirsizligi_m=np.nan_to_num(cok))

    def belirsiz(rapor):
        return sum(o.karar is Karar.BELIRSIZ for o in rapor.olcumler.values())

    assert belirsiz(b) > belirsiz(a)


# --- rapor -------------------------------------------------------------------

def test_ozet_alanlari(duzenek):
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar, ortulu=("sag_bilek",))
    isk = _iskelet(kalib, goz)
    ozet = belirsizlik_ozeti(eklem_belirsizligi(kalib, isk, goz), isk)
    assert ozet["n_eklem"] == len(ISK)
    assert ozet["n_belirsizlik_kestirildi"] <= ozet["n_gorunur"]
    assert ozet["en_kotu_eklem"] in ISK.eklemler


def test_gecersiz_sigma_hata(duzenek):
    kalib, kameralar = duzenek
    goz = _gozlemler(kameralar)
    isk = _iskelet(kalib, goz)
    with pytest.raises(ValueError, match="sigma_px pozitif"):
        eklem_belirsizligi(kalib, isk, goz, sigma_px=0.0)
