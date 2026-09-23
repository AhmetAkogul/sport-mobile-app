"""uncertainty.montecarlo kabul testleri.

KOD-PLANI.md Adim 2: "gurultu seviyesi <-> parametre varyansi egrisi".
Kabul olcutu monotonluk: gurultu artarsa sacilim artmali. Sureç rastgele oldugu
icin esikler genis, tohumlar sabit -- test kirilgan olmamali.
"""
import pytest

from calib.board import BoardSpec
from calib.synthetic import rig_yay, sahne_uret
from uncertainty.montecarlo import sacilim_olc

SPEC = BoardSpec(5, 7)
KAMERALAR = rig_yay(odak_px=900.0)


def kosu(gurultu, dagilim="genis", n_tekrar=8, n_kare=16):
    def uretec(seed):
        return sahne_uret(SPEC, kameralar=KAMERALAR, n_kare=n_kare,
                          gurultu_px=gurultu, dagilim=dagilim, seed=5000 + seed)
    return sacilim_olc(uretec, n_tekrar=n_tekrar,
                       gercekler={"fx": 900.0, "fy": 900.0})


@pytest.fixture(scope="module")
def dusuk():
    return kosu(0.1)


@pytest.fixture(scope="module")
def yuksek():
    return kosu(1.0)


def test_en_az_iki_tekrar_gerekli():
    with pytest.raises(ValueError, match="en az 2"):
        kosu(0.1, n_tekrar=1)


def test_takip_edilen_parametreler_var(dusuk):
    for ad in ("fx", "fy", "cx", "cy", "taban_01", "taban_02", "taban_03"):
        assert ad in dusuk.sacilimlar


def test_yakinsama_orani_raporlanir(dusuk):
    """Yakinsamayan kosu sessizce atilmaz; sayisi bulgunun parcasi."""
    assert dusuk.basarili <= dusuk.n_tekrar
    assert dusuk.basarili >= 2


def test_gurultu_artarsa_sacilim_artar(dusuk, yuksek):
    """KABUL TESTI: monotonluk."""
    assert yuksek.sacilimlar["fx"].std > dusuk.sacilimlar["fx"].std
    assert yuksek.rms_ortalama > dusuk.rms_ortalama


def test_gurultusuzde_sacilim_yok_denecek_kadar_kucuk():
    r = kosu(0.0, n_tekrar=4)
    assert r.sacilimlar["fx"].std < 0.5
    assert r.sacilimlar["fx"].sapma < 0.1     # kabul olcutu: %0.1


def test_poz_cesitliligi_belirsizligi_dusurur():
    """Ayni gurultuyle genis poz dagilimi, dar dagilimdan daha kesin sonuc verir.

    Bu, veri toplama protokolunu belirleyen bulgu: board'u cesitli acilarda ve
    mesafelerde gezdirmek, kod kalitesinden daha cok fark yaratiyor.
    """
    dar = kosu(0.3, dagilim="dar", n_tekrar=10)
    genis = kosu(0.3, dagilim="genis", n_tekrar=10)
    assert genis.sacilimlar["fx"].std < dar.sacilimlar["fx"].std


def test_sapma_gercek_verilmezse_none():
    def uretec(seed):
        return sahne_uret(SPEC, kameralar=KAMERALAR, n_kare=12,
                          gurultu_px=0.2, seed=seed)
    r = sacilim_olc(uretec, n_tekrar=4)
    assert r.sacilimlar["fx"].sapma is None
    assert r.sacilimlar["fx"].gercek is None


def test_sozluk_serilestirilebilir(dusuk):
    import json
    metin = json.dumps(dusuk.sozluk(), allow_nan=False)
    assert "parametreler" in metin and "sapma_yuzde" in metin
