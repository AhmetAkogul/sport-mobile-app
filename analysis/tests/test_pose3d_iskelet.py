"""pose3d.iskelet ve pose3d.pose2d kabul testleri.

Odak: HAZIR-TEKNOLOJILER.md madde 2 ve 3'teki sessiz hata tuzaklari.
Eklem sirasi esit varsayilmamali, eksik eklem uydurulmamali, goruntu olcegi
geri alinmali.
"""
import numpy as np
import pytest

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import dunyadan_kamera0, rig_yay, sahne_uret
from pose3d.iskelet import (
    REFERANS_ISKELET, Iskelet3B, IskeletTanimi, eslestir, iskelet_ucgenle,
)
from pose3d.pose2d import (
    BIRIM_DONUSUM, Donusum, Poz2B, letterbox_donusumu, sentetik_poz,
)

ISK = REFERANS_ISKELET
DURUS = {
    "boyun": (0, 0.55, 0), "sag_omuz": (-0.18, 0.45, 0), "sag_dirsek": (-0.22, 0.18, 0.05),
    "sag_bilek": (-0.24, -0.08, 0.08), "sol_omuz": (0.18, 0.45, 0),
    "sol_dirsek": (0.22, 0.18, 0.05), "sol_bilek": (0.24, -0.08, 0.08),
    "sag_kalca": (-0.11, 0, 0), "sag_diz": (-0.12, -0.42, 0.02),
    "sag_ayak_bilegi": (-0.12, -0.85, 0), "sol_kalca": (0.11, 0, 0),
    "sol_diz": (0.12, -0.42, 0.02), "sol_ayak_bilegi": (0.12, -0.85, 0),
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


# --- iskelet tanimi ---

def test_tekrarli_eklem_ismi_reddedilir():
    with pytest.raises(ValueError, match="benzersiz"):
        IskeletTanimi("bozuk", ("diz", "diz"))


def test_baglantida_bilinmeyen_eklem_reddedilir():
    with pytest.raises(ValueError, match="bilinmeyen eklem"):
        IskeletTanimi("bozuk", ("diz",), (("diz", "omuz"),))


def test_referans_iskelet_yuz_noktasi_icermez():
    """KVKK: yuz verisi toplamiyoruz, iskelette de bulunmamali."""
    yasak = ("goz", "burun", "kulak", "agiz", "yuz")
    assert not any(y in e for e in ISK.eklemler for y in yasak)


def test_bilinmeyen_eklem_sorulunca_hata():
    with pytest.raises(KeyError, match="bu iskelette yok"):
        ISK.indeks("kuyruk")


# --- eklem eslestirme: indeks esitligi varsayilmaz ---

def test_eslesmeyen_eklem_eksi_bir():
    yabanci = IskeletTanimi("yabanci", ("sag_diz", "bilinmeyen"))
    harita = eslestir(yabanci, ISK)
    assert harita[ISK.indeks("sag_diz")] == 0
    assert harita[ISK.indeks("boyun")] == -1


def test_isim_esleme_tablosuyla_cevrilir():
    """Baska modelin isimleri farkliysa tablo verilir, indeks varsayilmaz."""
    mp = IskeletTanimi("mp", ("RIGHT_KNEE", "LEFT_KNEE"))
    harita = eslestir(mp, ISK, {"sag_diz": "RIGHT_KNEE", "sol_diz": "LEFT_KNEE"})
    assert harita[ISK.indeks("sag_diz")] == 0
    assert harita[ISK.indeks("sol_diz")] == 1
    assert harita[ISK.indeks("boyun")] == -1


# --- Poz2B sozlesmesi ---

def test_eklem_sayisi_uyusmazsa_hata():
    with pytest.raises(ValueError, match="eklem sirasi esit varsayilamaz"):
        Poz2B(iskelet=ISK, noktalar=np.zeros((5, 2)), guven=np.ones(5),
              gorunur=np.ones(5, bool), goruntu_boyutu=(1280, 720))


def test_gecersiz_uzay_reddedilir():
    n = len(ISK)
    with pytest.raises(ValueError, match="uzay"):
        Poz2B(iskelet=ISK, noktalar=np.zeros((n, 2)), guven=np.ones(n),
              gorunur=np.ones(n, bool), goruntu_boyutu=(1280, 720), uzay="baska")


def test_guven_esikleme_diziyi_kisaltmaz():
    """Nokta silinmez, maskelenir -- yoksa indeksler kayar."""
    n = len(ISK)
    poz = Poz2B(iskelet=ISK, noktalar=np.zeros((n, 2)),
                guven=np.linspace(0, 1, n), gorunur=np.ones(n, bool),
                goruntu_boyutu=(1280, 720))
    suzulmus = poz.guven_esikle(0.5)
    assert len(suzulmus.noktalar) == n
    assert suzulmus.gorunur.sum() < n
    assert (suzulmus.guven[suzulmus.gorunur] >= 0.5).all()


# --- koordinat donusumu ---

def test_letterbox_geri_donusu_tam():
    d = letterbox_donusumu((1280, 720), (640, 640))
    noktalar = np.array([[0.0, 0.0], [1279.0, 719.0], [640.0, 360.0]])
    assert np.allclose(d.geri(d.ileri(noktalar)), noktalar, atol=1e-9)


def test_letterbox_en_boy_orani_korunur():
    d = letterbox_donusumu((1280, 720), (640, 640))
    assert d.olcek == pytest.approx(0.5)
    assert d.ofset[0] == pytest.approx(0.0)
    assert d.ofset[1] == pytest.approx(140.0)     # (640 - 720*0.5)/2


def test_gecersiz_boyut_reddedilir():
    with pytest.raises(ValueError, match="pozitif"):
        letterbox_donusumu((0, 720), (640, 640))


def test_sifir_olcekli_donusum_geri_alinmaz():
    with pytest.raises(ValueError, match="olcek sifir"):
        Donusum(olcek=0.0, ofset=np.zeros(2)).geri([[1.0, 2.0]])


def test_model_uzayindaki_poz_ozgune_cevrilir():
    n = len(ISK)
    d = letterbox_donusumu((1280, 720), (640, 640))
    ozgun = np.tile([640.0, 360.0], (n, 1))
    poz = Poz2B(iskelet=ISK, noktalar=d.ileri(ozgun), guven=np.ones(n),
                gorunur=np.ones(n, bool), goruntu_boyutu=(1280, 720), uzay="model")
    cevrilmis = poz.ozgun_uzaya(d)
    assert cevrilmis.uzay == "ozgun"
    assert np.allclose(cevrilmis.noktalar, ozgun, atol=1e-9)
    # zaten ozgun uzaydaysa dokunulmamali
    assert cevrilmis.ozgun_uzaya(d) is cevrilmis


# --- ucdan uca 3B iskelet ---

def test_gurultusuzde_eklemler_geri_bulunur(duzenek):
    kalib, kameralar = duzenek
    pozlar = {i: sentetik_poz(k, P3, seed=i, kamera_indeksi=i)
              for i, k in enumerate(kameralar)}
    isk = iskelet_ucgenle(kalib, pozlar)
    gercek = dunyadan_kamera0(kameralar, P3)
    assert isk.gorunur.all()
    hata = np.linalg.norm(isk.noktalar - gercek, axis=1) * 1000
    assert hata.max() < 1.0      # mm


def test_ortulu_eklem_digerlerinden_ucgenlenir(duzenek):
    """Bir kamera bir eklemi gormuyorsa digerleri isi surdurur."""
    kalib, kameralar = duzenek
    pozlar = {i: sentetik_poz(k, P3, seed=i, kamera_indeksi=i,
                              ortulu=("sag_diz",) if i == 0 else ())
              for i, k in enumerate(kameralar)}
    isk = iskelet_ucgenle(kalib, pozlar)
    j = ISK.indeks("sag_diz")
    assert isk.gorunur[j]
    assert isk.goren_kamera[j] == 3


def test_iki_gorusun_altindaki_eklem_uydurulmaz(duzenek):
    """Eksik eklem NaN kalir; tahmin edilmez."""
    kalib, kameralar = duzenek
    pozlar = {i: sentetik_poz(k, P3, seed=i, kamera_indeksi=i,
                              ortulu=("sag_bilek",) if i > 0 else ())
              for i, k in enumerate(kameralar)}
    isk = iskelet_ucgenle(kalib, pozlar)
    j = ISK.indeks("sag_bilek")
    assert not isk.gorunur[j]
    assert np.isnan(isk.noktalar[j]).all()
    assert isk.goren_kamera[j] == 1
    assert np.isnan(isk.uzunluk("sag_dirsek", "sag_bilek"))


def test_uzuv_uzunlugu_dogru_olculuyor(duzenek):
    kalib, kameralar = duzenek
    pozlar = {i: sentetik_poz(k, P3, gurultu_px=0.25, seed=i, kamera_indeksi=i)
              for i, k in enumerate(kameralar)}
    isk = iskelet_ucgenle(kalib, pozlar)
    gercek = float(np.linalg.norm(P3[ISK.indeks("sag_kalca")] - P3[ISK.indeks("sag_diz")]))
    assert isk.uzunluk("sag_kalca", "sag_diz") == pytest.approx(gercek, abs=5e-3)


def test_farkli_iskeletli_kameralar_karismaz(duzenek):
    """Bir kamera baska tanim kullaniyorsa isimle eslenir, indeksle degil."""
    kalib, kameralar = duzenek
    ters = IskeletTanimi("ters", tuple(reversed(ISK.eklemler)))
    pozlar = {}
    for i, k in enumerate(kameralar):
        poz = sentetik_poz(k, P3, seed=i, kamera_indeksi=i)
        if i == 0:
            sira = [ISK.indeks(e) for e in ters.eklemler]
            poz = Poz2B(iskelet=ters, noktalar=poz.noktalar[sira],
                        guven=poz.guven[sira], gorunur=poz.gorunur[sira],
                        goruntu_boyutu=poz.goruntu_boyutu, kamera=i)
        pozlar[i] = poz
    isk = iskelet_ucgenle(kalib, pozlar)
    gercek = dunyadan_kamera0(kameralar, P3)
    hata = np.linalg.norm(isk.noktalar - gercek, axis=1) * 1000
    assert hata.max() < 1.0, "isim eslemesi yapilmazsa eklemler karisirdi"


def test_sentetik_poz_eklem_sayisini_dogrular(duzenek):
    _, kameralar = duzenek
    with pytest.raises(ValueError, match="3B iskelet"):
        sentetik_poz(kameralar[0], P3[:5])


def test_iskelet3b_uzunluk_dogrulamasi():
    with pytest.raises(ValueError, match="uzunlugu"):
        Iskelet3B(tanim=ISK, noktalar=np.zeros((3, 3)), gorunur=np.ones(3, bool),
                  goren_kamera=np.zeros(3, int), artik_px=np.zeros(3))
