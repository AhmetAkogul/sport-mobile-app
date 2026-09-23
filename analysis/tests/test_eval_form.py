"""eval.form kabul testleri.

Odak noktasi iki sey: (1) bilinen bir kusur acisi verildiginde katman ayni
sayiyi geri okuyor mu, (2) okuyamadigi durumda **susuyor mu**. Ikincisi daha
onemli -- sessizce "dogru form" demek, tezin olcmeye calistigi hatayi gizler.
"""
import json

import numpy as np
import pytest

from eval.form import (
    VARSAYILAN_ESIKLER, Esik, Karar, form_degerlendir, govde_cercevesi,
    sentetik_durus,
)


def _deger(rapor, ad):
    return rapor.olcumler[ad].deger


def _karar(rapor, ad):
    return rapor.olcumler[ad].karar


# --- olcumun dogrulugu -------------------------------------------------------

@pytest.mark.parametrize("aci", [5.0, 10.0, 15.0, 25.0])
def test_valgus_acisi_birebir_geri_okunur(aci):
    """Sentetik durusa verilen valgus acisi olcumden aynen cikmali."""
    rapor = form_degerlendir(sentetik_durus(valgus_sag=aci))
    assert _deger(rapor, "diz_valgusu_sag") == pytest.approx(aci, abs=1e-6)
    # Kusur tek bacakta; digeri temiz kalmali.
    assert _deger(rapor, "diz_valgusu_sol") == pytest.approx(0.0, abs=1e-9)


def test_iki_bacak_bagimsiz_olculur():
    rapor = form_degerlendir(sentetik_durus(valgus_sag=14.0, valgus_sol=3.0))
    assert _deger(rapor, "diz_valgusu_sag") == pytest.approx(14.0, abs=1e-6)
    assert _deger(rapor, "diz_valgusu_sol") == pytest.approx(3.0, abs=1e-6)
    assert _karar(rapor, "diz_valgusu_sag") is Karar.KUSURLU
    assert _karar(rapor, "diz_valgusu_sol") is Karar.DOGRU


@pytest.mark.parametrize("aci", [-12.0, -4.0, 6.0, 12.0])
def test_kalca_hizasi_birebir_geri_okunur(aci):
    rapor = form_degerlendir(sentetik_durus(kalca_hizasi=aci))
    assert _deger(rapor, "kalca_hizasi") == pytest.approx(aci, abs=1e-6)


@pytest.mark.parametrize("aci", [-25.0, -10.0, 10.0, 30.0])
def test_govde_rotasyonu_birebir_geri_okunur(aci):
    rapor = form_degerlendir(sentetik_durus(govde_rotasyonu=aci))
    assert _deger(rapor, "govde_rotasyonu") == pytest.approx(aci, abs=1e-6)


# --- karar -------------------------------------------------------------------

def test_dogru_durus_hicbir_kusur_uretmez():
    rapor = form_degerlendir(sentetik_durus())
    assert rapor.genel_karar is Karar.DOGRU
    assert rapor.kusurlar == ()


def test_varus_valgus_esigini_tetiklemez():
    """Diz disa acilirsa (negatif deger) valgus kusuru **verilmemeli**.

    Tek yonlu esik budur: varus ayri bir kusurdur ve bu katmanda olculmez;
    isaretli deger raporda durur ama karar DOGRU kalir.
    """
    rapor = form_degerlendir(sentetik_durus(valgus_sag=-20.0))
    assert _deger(rapor, "diz_valgusu_sag") == pytest.approx(-20.0, abs=1e-6)
    assert _karar(rapor, "diz_valgusu_sag") is Karar.DOGRU


def test_esik_disaridan_degistirilebilir():
    iskelet = sentetik_durus(valgus_sag=6.0)
    assert _karar(form_degerlendir(iskelet), "diz_valgusu_sag") is Karar.DOGRU
    siki = {"diz_valgusu_sag": Esik(4.0, tek_yonlu=True)}
    assert _karar(form_degerlendir(iskelet, esikler=siki),
                  "diz_valgusu_sag") is Karar.KUSURLU


def test_genel_karar_en_kotuyu_alir():
    rapor = form_degerlendir(sentetik_durus(govde_rotasyonu=40.0))
    assert rapor.genel_karar is Karar.KUSURLU
    assert rapor.kusurlar == ("govde_rotasyonu",)


# --- kamera durusundan bagimsizlik ------------------------------------------

def test_olcumler_kamera_durusundan_bagimsiz():
    """Ayni durus, farkli kamera koordinatinda ayni sayiyi vermeli.

    3B noktalar kamera 0'a goredir ve yer cekimi yonu bilinmez; olcumler
    iskeletten kurulan govde cercevesinde yapildigi icin donme ve otelemeye
    duyarsiz olmalidir. Bu sinanmazsa, kamera yerlesimi degistiginde form
    karari sessizce kayar.
    """
    kusurlar = dict(valgus_sag=13.0, kalca_hizasi=-6.0, govde_rotasyonu=11.0)
    duz = form_degerlendir(sentetik_durus(**kusurlar))

    rng = np.random.default_rng(4242)
    for _ in range(5):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))          # isaret belirsizligini sabitle
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1                    # sag-elli donme olsun
        M = np.eye(4)
        M[:3, :3] = Q
        M[:3, 3] = rng.normal(scale=3.0, size=3)
        donuk = form_degerlendir(sentetik_durus(**kusurlar, donusum=M))
        for ad in duz.olcumler:
            assert _deger(donuk, ad) == pytest.approx(_deger(duz, ad), abs=1e-6)


def test_govde_cercevesi_ortonormal():
    c = govde_cercevesi(sentetik_durus(kalca_hizasi=9.0))
    assert c is not None
    for v in (c.yanal, c.yukari, c.ileri):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
    assert c.yanal @ c.yukari == pytest.approx(0.0, abs=1e-12)
    assert np.cross(c.yanal, c.yukari) @ c.ileri == pytest.approx(1.0, abs=1e-12)


# --- eksik eklem -------------------------------------------------------------

def test_eksik_eklem_uydurulmaz():
    """Gorunmeyen diz, valgus karari verdirtmemeli; diger olcumler surmeli."""
    rapor = form_degerlendir(sentetik_durus(valgus_sag=20.0, gorunmez=("sag_diz",)))
    olcum = rapor.olcumler["diz_valgusu_sag"]
    assert olcum.karar is Karar.BELIRSIZ
    assert not olcum.hesaplandi
    assert "sag_diz" in olcum.eksik_eklemler
    assert _karar(rapor, "govde_rotasyonu") is Karar.DOGRU
    assert rapor.genel_karar is Karar.BELIRSIZ


def test_cerceve_eklemi_eksikse_hicbir_olcum_yapilmaz():
    """Boyun gorunmuyorsa govde cercevesi kurulamaz; hepsi BELIRSIZ olmali."""
    iskelet = sentetik_durus(valgus_sag=20.0, gorunmez=("boyun",))
    assert govde_cercevesi(iskelet) is None
    rapor = form_degerlendir(iskelet)
    assert {o.karar for o in rapor.olcumler.values()} == {Karar.BELIRSIZ}
    assert all("boyun" in o.eksik_eklemler for o in rapor.olcumler.values())


# --- belirsizlik -------------------------------------------------------------

def test_belirsizlik_esigi_kesince_karar_verilmez():
    """Esige yakin bir kusur, buyuk eklem belirsizligiyle karara baglanmamali.

    Tezin iddiasinin kod karsiligi: olcum hatasi karar sinirindan buyukse o
    kare hakkinda "dogru" da "kusurlu" da denemez.
    """
    iskelet = sentetik_durus(valgus_sag=11.0)          # esik 10 derece
    kesin = form_degerlendir(iskelet, konum_belirsizligi_m=0.0005)
    belirsiz = form_degerlendir(iskelet, konum_belirsizligi_m=0.02)
    assert kesin.olcumler["diz_valgusu_sag"].karar is Karar.KUSURLU
    assert belirsiz.olcumler["diz_valgusu_sag"].karar is Karar.BELIRSIZ
    assert (belirsiz.olcumler["diz_valgusu_sag"].belirsizlik
            > kesin.olcumler["diz_valgusu_sag"].belirsizlik)


def test_belirsizlik_eklem_konumuyla_buyur():
    iskelet = sentetik_durus(valgus_sag=5.0)
    kucuk = form_degerlendir(iskelet, konum_belirsizligi_m=0.002)
    buyuk = form_degerlendir(iskelet, konum_belirsizligi_m=0.02)
    for ad in kucuk.olcumler:
        assert buyuk.olcumler[ad].belirsizlik > kucuk.olcumler[ad].belirsizlik


def test_belirsizlik_tekrarlanabilir():
    """Ayni tohum ayni sayiyi vermeli; `make reproduce` buna dayanir."""
    iskelet = sentetik_durus(valgus_sol=8.0)
    a = form_degerlendir(iskelet, konum_belirsizligi_m=0.01, seed=7)
    b = form_degerlendir(iskelet, konum_belirsizligi_m=0.01, seed=7)
    c = form_degerlendir(iskelet, konum_belirsizligi_m=0.01, seed=8)
    assert (a.olcumler["diz_valgusu_sol"].belirsizlik
            == b.olcumler["diz_valgusu_sol"].belirsizlik)
    assert (a.olcumler["diz_valgusu_sol"].belirsizlik
            != c.olcumler["diz_valgusu_sol"].belirsizlik)


def test_belirsizlik_verilmezse_nan_kalir():
    rapor = form_degerlendir(sentetik_durus(valgus_sag=12.0))
    olcum = rapor.olcumler["diz_valgusu_sag"]
    assert np.isnan(olcum.belirsizlik)
    assert olcum.karar is Karar.KUSURLU        # yalin esik karsilastirmasi


def test_eklem_basina_belirsizlik_verilebilir():
    """Ucgenlemeden gelen eklem basina 1-sigma dizisi kabul edilmeli."""
    iskelet = sentetik_durus(valgus_sag=11.0)
    sigma = np.full(len(iskelet.tanim), 0.001)
    sigma[iskelet.tanim.indeks("sag_diz")] = 0.03      # yalniz sag diz gurultulu
    rapor = form_degerlendir(iskelet, konum_belirsizligi_m=sigma)
    assert rapor.olcumler["diz_valgusu_sag"].karar is Karar.BELIRSIZ
    assert (rapor.olcumler["diz_valgusu_sag"].belirsizlik
            > rapor.olcumler["diz_valgusu_sol"].belirsizlik)


@pytest.mark.parametrize("bozuk", [-0.01, np.full(3, 0.01), np.nan])
def test_gecersiz_belirsizlik_hata_verir(bozuk):
    with pytest.raises(ValueError):
        form_degerlendir(sentetik_durus(), konum_belirsizligi_m=bozuk)


# --- rapor sozlesmesi --------------------------------------------------------

def test_ozet_json_yazilabilir():
    """Rapor ozeti NaN icermeden JSON'a gitmeli; eval/report.py boyle tuketiyor."""
    rapor = form_degerlendir(sentetik_durus(valgus_sag=12.0, gorunmez=("sol_diz",)),
                             konum_belirsizligi_m=0.001)
    metin = json.dumps(rapor.ozet(), ensure_ascii=False, allow_nan=False)
    geri = json.loads(metin)
    assert geri["genel_karar"] == "kusurlu"
    assert geri["olcumler"]["diz_valgusu_sol"]["deger_derece"] is None
    assert geri["olcumler"]["diz_valgusu_sol"]["eksik_eklemler"] == ["sol_diz"]


def test_esikler_raporda_tasinir():
    rapor = form_degerlendir(sentetik_durus())
    for ad, esik in VARSAYILAN_ESIKLER.items():
        assert rapor.olcumler[ad].esik == esik.deger


# --- dis inceleme F.1 / F.2: hesaplanamayan belirsizlik kesin karar uretmez ----

@pytest.mark.parametrize("n", [0, 1])
def test_tek_ornekle_belirsizlik_istenirse_hata(n):
    """Tek ornekten sapma cikmaz; NaN'a dusup yalin esige donmemeli."""
    with pytest.raises(ValueError, match="n_ornek en az 2"):
        form_degerlendir(sentetik_durus(valgus_sag=11.0),
                         konum_belirsizligi_m=0.01, n_ornek=n)


def test_hesaplanamayan_belirsizlik_karar_uretmez():
    """Belirsizlik istendi ama NaN: sonuc BELIRSIZ, yalin esik degil."""
    from eval.form import _karar_ver
    esik = VARSAYILAN_ESIKLER["diz_valgusu_sag"]
    assert _karar_ver(25.0, esik, float("nan"), 2.0) is Karar.BELIRSIZ
    # istenmediyse (None) yalin esik: acik bir kusur kusurlu kalir
    assert _karar_ver(25.0, esik, None, 2.0) is Karar.KUSURLU
    # sifir belirsizlik gecerli bir olcumdur: yalin esik
    assert _karar_ver(25.0, esik, 0.0, 2.0) is Karar.KUSURLU


def test_ornekleri_cogunlukla_tanimsiz_olcum_belirsiz_kalir(monkeypatch):
    """Monte Carlo orneklerinin %5'inden fazlasi NaN ise sapma guvenilmez."""
    import eval.form as form

    gercek = form._tum_olcumler
    sayac = {"n": 0}

    def yarisi_nan(noktalar, tanim):
        sayac["n"] += 1
        d = gercek(noktalar, tanim)
        if sayac["n"] % 2 == 0:
            d = {**d, "diz_valgusu_sag": float("nan")}
        return d

    monkeypatch.setattr(form, "_tum_olcumler", yarisi_nan)
    rapor = form_degerlendir(sentetik_durus(valgus_sag=25.0), konum_belirsizligi_m=0.001)
    assert rapor.olcumler["diz_valgusu_sag"].karar is Karar.BELIRSIZ
    assert np.isnan(rapor.olcumler["diz_valgusu_sag"].belirsizlik)


# --- dis inceleme YC.5: eklem basina kovaryans --------------------------------

def test_izotrop_kovaryans_skaler_sigma_ile_bit_bit_ayni():
    """sigma^2 * I kovaryansi, skaler sigma ile ayni ornekleri uretmeli."""
    iskelet = sentetik_durus(valgus_sag=11.0)
    n = len(iskelet.tanim)
    skaler = form_degerlendir(iskelet, konum_belirsizligi_m=0.01)
    kov = form_degerlendir(iskelet, konum_belirsizligi_m=np.tile(1e-4 * np.eye(3), (n, 1, 1)))
    for ad in skaler.olcumler:
        assert kov.olcumler[ad].belirsizlik == skaler.olcumler[ad].belirsizlik
        assert kov.olcumler[ad].karar is skaler.olcumler[ad].karar


def test_yalniz_derinlik_eksenindeki_belirsizlik_onden_aciyi_az_etkiler():
    """Onden bakan telefonda hata derinlikte (Z); valgus goruntu duzleminde.

    Ayni buyuklukte belirsizlik yalniz Z'de ise valgus belirsizligi, izotrop
    durumdakinden kucuk cikmali -- izotrop sigma karari gereksiz yere susturur.
    """
    iskelet = sentetik_durus(valgus_sag=14.0)
    n = len(iskelet.tanim)
    s = 0.03
    izotrop = form_degerlendir(iskelet, konum_belirsizligi_m=s)
    yalniz_z = form_degerlendir(
        iskelet, konum_belirsizligi_m=np.tile(np.diag([0.0, 0.0, s * s]), (n, 1, 1)))
    ad = "diz_valgusu_sag"
    assert yalniz_z.olcumler[ad].belirsizlik < izotrop.olcumler[ad].belirsizlik


def test_eklem_kovaryanslari_listesi_dogrudan_verilebilir():
    """uncertainty.eklem.eklem_kovaryanslari ciktisi (liste, gorunmezde None)."""
    iskelet = sentetik_durus(valgus_sag=11.0)
    liste = [1e-4 * np.eye(3) for _ in iskelet.tanim.eklemler]
    rapor = form_degerlendir(iskelet, konum_belirsizligi_m=liste)
    assert np.isfinite(rapor.olcumler["diz_valgusu_sag"].belirsizlik)


@pytest.mark.parametrize("bozuk,mesaj", [
    (np.tile(np.array([[1e-4, 1e-5, 0], [0, 1e-4, 0], [0, 0, 1e-4]]), (13, 1, 1)), "simetrik"),
    (np.tile(np.diag([1e-4, -1e-4, 1e-4]), (13, 1, 1)), "yari tanimli"),
    (np.full((13, 3, 3), np.nan), "sonlu"),
    (np.zeros((13, 2, 2)), r"\(N, 3, 3\)"),
])
def test_bozuk_kovaryans_reddedilir(bozuk, mesaj):
    with pytest.raises(ValueError, match=mesaj):
        form_degerlendir(sentetik_durus(), konum_belirsizligi_m=bozuk)
