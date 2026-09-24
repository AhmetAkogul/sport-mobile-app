"""eval.protokol: 0070'in paydalari, agirliklandirmasi, bolmesi ve kilidi.

Her test 0070'teki bir cumlenin kod karsiligini sinar.
"""
import pytest

from eval.protokol import (
    BELIRSIZ, DOGRU, KUSURLU, Oge, Sayim, kisi_agirlikli, kisi_bolmesi,
    kisi_bootstrap, manifest_kilidi,
)


def _sayim():
    return Sayim({
        (KUSURLU, KUSURLU): 6, (KUSURLU, DOGRU): 2, (KUSURLU, BELIRSIZ): 2,
        (DOGRU, DOGRU): 3, (DOGRU, KUSURLU): 1, (DOGRU, None): 1,
        (BELIRSIZ, DOGRU): 4, (None, KUSURLU): 1,
    })


def test_paydalar_0070_tanimina_uyar():
    s = _sayim()
    assert (s.M, s.R, s.N) == (20, 5, 15)       # referans BELIRSIZ/None -> R
    assert (s.D, s.C, s.W, s.U) == (12, 9, 3, 3)  # telefon None -> U, N kuculmez


def test_oranlar_ve_sinif_dengeli():
    o = _sayim().oranlar()
    assert o["A"] == pytest.approx(9 / 12) and o["K"] == pytest.approx(12 / 15)
    assert o["Y"] == pytest.approx(o["A"] * o["K"])
    # kusurlu sinifi: C/D = 6/8, D/N = 8/10; dogru sinifi: 3/4, 4/5
    assert o["A_dengeli"] == pytest.approx((6 / 8 + 3 / 4) / 2)
    assert o["K_dengeli"] == pytest.approx((8 / 10 + 4 / 5) / 2)


def test_tanimsiz_oran_none_dir_asla_0_ya_da_1_degil():
    hic_karar_yok = Sayim({(KUSURLU, BELIRSIZ): 5}).oranlar()
    assert hic_karar_yok["A"] is None and hic_karar_yok["K"] == 0.0
    tek_sinif = Sayim({(KUSURLU, KUSURLU): 3}).oranlar()
    assert tek_sinif["A_dengeli"] is None          # tek sinifli sonuc genel basari degil
    assert Sayim({}).oranlar()["K"] is None


def test_gecersiz_etiket_reddedilir():
    with pytest.raises(ValueError, match="gecersiz"):
        Sayim({("iyi", DOGRU): 1})


def test_uzun_oturum_daha_agir_degil():
    """0070 §2: oturum oranlari esit, sonra kisiler esit agirlikla."""
    ogeler = ([Oge("k1", "o1", KUSURLU, KUSURLU)] * 90       # uzun, hep dogru
              + [Oge("k1", "o2", KUSURLU, DOGRU)] * 10        # kisa, hep yanlis
              + [Oge("k2", "o1", DOGRU, DOGRU)] * 5)
    o = kisi_agirlikli(ogeler)
    # k1: (1.0 + 0.0)/2 = 0.5 ; k2: 1.0 -> 0.75 (havuzlanmis olsa 95/105)
    assert o["A"] == pytest.approx(0.75)
    assert o["n_kisi"] == 2


def test_hic_karar_vermeyen_kisinin_A_si_none_K_si_sifir():
    o = kisi_agirlikli([Oge("k1", "o1", KUSURLU, BELIRSIZ), Oge("k2", "o1", DOGRU, DOGRU)])
    assert o["null_kisi"]["A"] == 1
    assert o["K"] == pytest.approx(0.5)            # (0 + 1)/2


def test_bootstrap_ikiden_az_kiside_dogrulanmadi():
    assert kisi_bootstrap([Oge("k1", "o1", DOGRU, DOGRU)])["durum"] == "dogrulanmadi"


def test_bootstrap_tekrarlanabilir_ve_araligi_verir():
    ogeler = [Oge(f"k{i}", "o1", KUSURLU, KUSURLU if i % 3 else DOGRU) for i in range(6)]
    a = kisi_bootstrap(ogeler, tekrar=200)
    b = kisi_bootstrap(ogeler, tekrar=200)
    assert a == b and a["tohum"] == 70
    alt, ust = a["aralik_95"]["A"]
    assert 0.0 <= alt <= ust <= 1.0


def test_bolme_deterministik_ayrik_ve_boyutlu():
    kisiler = [f"p{i:02d}" for i in range(10)]
    b = kisi_bolmesi(kisiler)
    assert b == kisi_bolmesi(list(reversed(kisiler)))     # sira girdiden bagimsiz
    assert (len(b["test"]), len(b["dogrulama"]), len(b["egitim"])) == (2, 2, 6)
    assert set(b["test"]).isdisjoint(b["dogrulama"]) and set(b["test"]).isdisjoint(b["egitim"])
    assert b["genelleme_kabulu_mumkun"]


def test_az_kisiyle_genelleme_kabulu_verilmez():
    assert not kisi_bolmesi([f"p{i}" for i in range(5)])["genelleme_kabulu_mumkun"]
    with pytest.raises(ValueError, match="benzersiz"):
        kisi_bolmesi(["a", "a", "b"])


def test_manifest_kilidi_anahtar_sirasindan_bagimsiz_icerige_duyarli():
    m1 = {"kisiler": ["p1", "p2"], "kosul": {"aci": [0, 45]}}
    m2 = {"kosul": {"aci": [0, 45]}, "kisiler": ["p1", "p2"]}
    assert manifest_kilidi(m1) == manifest_kilidi(m2)
    assert manifest_kilidi(m1) != manifest_kilidi({**m1, "kisiler": ["p1", "p3"]})


@pytest.mark.parametrize("girdi,beklenen", [
    ((0.90, 0.95, 0.85, 0.88, 12), "basarili"),
    ((0.78, 0.95, 0.70, 0.90, 12), "basarisiz"),     # nokta hedefin altinda
    ((0.90, 0.75, 0.85, 0.70, 3), "basarisiz"),      # A yuksek K dusuk: basari degil
    ((0.90, 0.92, 0.62, 0.70, 12), "dogrulanmadi"),  # aralik hedefi kesiyor
    ((0.90, 0.92, 0.62, 0.70, 2), "on_kanit"),       # az kisi: basari degil, on kanit
    ((None, 0.92, None, 0.70, 12), "dogrulanmadi"),  # sinif yok
])
def test_hedef_hukmu_0070_ve_eki(girdi, beklenen):
    from eval.protokol import hedef_hukmu
    assert hedef_hukmu(*girdi) == beklenen
