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
              gorunur=np.ones(5, bool), tespit=True, goruntu_boyutu=(1280, 720))


def test_gecersiz_uzay_reddedilir():
    n = len(ISK)
    with pytest.raises(ValueError, match="uzay"):
        Poz2B(iskelet=ISK, noktalar=np.zeros((n, 2)), guven=np.ones(n),
              gorunur=np.ones(n, bool), tespit=True, goruntu_boyutu=(1280, 720),
              uzay="baska")


def test_uzay_sonradan_degistirilemez():
    """Sozlesme alani kurulumdan sonra mutasyona kapali (dis inceleme P.1)."""
    from dataclasses import FrozenInstanceError
    n = len(ISK)
    poz = Poz2B(iskelet=ISK, noktalar=np.zeros((n, 2)), guven=np.ones(n),
                gorunur=np.ones(n, bool), tespit=True, goruntu_boyutu=(1280, 720))
    with pytest.raises(FrozenInstanceError):
        poz.uzay = "bozuk"
    with pytest.raises(FrozenInstanceError):
        poz.kare = 3


def test_tespit_false_ile_gorunur_eklem_celiskili():
    n = len(ISK)
    with pytest.raises(ValueError, match="tespit=False"):
        Poz2B(iskelet=ISK, noktalar=np.zeros((n, 2)), guven=np.ones(n),
              gorunur=np.ones(n, bool), tespit=False, goruntu_boyutu=(1280, 720))


def test_guven_esikleme_diziyi_kisaltmaz():
    """Nokta silinmez, maskelenir -- yoksa indeksler kayar."""
    n = len(ISK)
    poz = Poz2B(iskelet=ISK, noktalar=np.zeros((n, 2)),
                guven=np.linspace(0, 1, n), gorunur=np.ones(n, bool), tespit=True,
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


def test_sifir_olcekli_donusum_kurulamaz():
    """olcek==0 artik kurulum aninda reddedilir; ileri() sessizce cokertmez."""
    with pytest.raises(ValueError, match="olcek"):
        Donusum(olcek=0.0, ofset=np.zeros(2))


def test_yanlis_sekilli_ofset_reddedilir():
    with pytest.raises(ValueError, match="ofset"):
        Donusum(olcek=1.0, ofset=np.zeros(3))


def test_dort_kanalli_nokta_sessizce_kabul_edilmez():
    """reshape(-1, 2) (N, 4) girdiyi (2N, 2) yapip hata vermezdi."""
    with pytest.raises(ValueError, match="son ekseni 2"):
        BIRIM_DONUSUM.ileri(np.zeros((3, 4)))


def test_model_uzayindaki_poz_ozgune_cevrilir():
    n = len(ISK)
    d = letterbox_donusumu((1280, 720), (640, 640))
    ozgun = np.tile([640.0, 360.0], (n, 1))
    poz = Poz2B(iskelet=ISK, noktalar=d.ileri(ozgun), guven=np.ones(n),
                gorunur=np.ones(n, bool), tespit=True, goruntu_boyutu=(1280, 720),
                uzay="model")
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
                        tespit=poz.tespit,
                        goruntu_boyutu=poz.goruntu_boyutu, kamera=i)
        pozlar[i] = poz
    isk = iskelet_ucgenle(kalib, pozlar)
    gercek = dunyadan_kamera0(kameralar, P3)
    hata = np.linalg.norm(isk.noktalar - gercek, axis=1) * 1000
    assert hata.max() < 1.0, "isim eslemesi yapilmazsa eklemler karisirdi"


def test_kamera_arkasindaki_iskelet_gorunmez_sayilir():
    """cv2.projectPoints negatif Z'li noktayi da izdusurur ve sonuc genelde
    kadrajin ICINE duser (1 m arkadaki nokta tam merkeze gelir). Yalnizca kadraj
    filtresi kullanilirsa cop veri 'goruldu' sayilir (dis inceleme P.2).
    """
    from calib.synthetic import Kamera
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    arkasinda = Kamera(K=K, R=np.eye(3), t=np.array([[0.0], [0.0], [-1.4]]),
                       boyut=(1280, 720))
    poz = sentetik_poz(arkasinda, P3)
    assert not poz.gorunur.any()
    assert poz.tespit is False


def test_kadraj_disindaki_iskelet_tespitsiz_ve_koordinatsizdir():
    """Kamera onunde ama kadraj disinda: tespit yok, koordinat da yok.

    Onceden `tespit=gorunur.any()` sonlu koordinatla birlesip Poz2B kurulurken
    "tespit=False iken koordinat tasinamaz" hatasiyla cokuyordu.
    """
    from calib.synthetic import Kamera
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    kam = Kamera(K=K, R=np.eye(3), t=np.array([[50.0], [0.0], [3.0]]), boyut=(1280, 720))
    poz = sentetik_poz(kam, P3)
    assert poz.tespit is False
    assert np.isnan(poz.noktalar).all()


def test_tamamen_ortulu_iskelet_tespitli_ama_gorunmezdir(duzenek):
    """Kisi kadrajda ama her eklem ortulu: tespit var, gorunur eklem yok."""
    _, kameralar = duzenek
    poz = sentetik_poz(kameralar[0], P3, ortulu=REFERANS_ISKELET.eklemler)
    assert poz.tespit is True
    assert not poz.gorunur.any()


def test_negatif_gurultu_reddedilir(duzenek):
    _, kameralar = duzenek
    with pytest.raises(ValueError, match="gurultu_px negatif"):
        sentetik_poz(kameralar[0], P3, gurultu_px=-0.5)


def test_sentetik_poz_eklem_sayisini_dogrular(duzenek):
    _, kameralar = duzenek
    with pytest.raises(ValueError, match="3B iskelet"):
        sentetik_poz(kameralar[0], P3[:5])


def test_iskelet3b_sekil_dogrulamasi():
    with pytest.raises(ValueError, match="bekleniyor"):
        Iskelet3B(tanim=ISK, noktalar=np.zeros((3, 3)), gorunur=np.ones(3, bool),
                  goren_kamera=np.zeros(3, int), artik_px=np.zeros(3))


def test_iskelet3b_2b_dizi_3b_yerine_gecmez():
    """len() kontrolu (N,2) diziyi kabul ediyordu: sessiz veri bozulmasi (I.1)."""
    n = len(ISK)
    with pytest.raises(ValueError, match=r"\(13, 3\) bekleniyor"):
        Iskelet3B(tanim=ISK, noktalar=np.zeros((n, 2)), gorunur=np.ones(n, bool),
                  goren_kamera=np.zeros(n, int), artik_px=np.zeros(n))


def test_iskelet3b_gorunmez_eklem_deger_tasimaz():
    n = len(ISK)
    noktalar = np.zeros((n, 3))
    gorunur = np.ones(n, bool)
    gorunur[0] = False                 # NaN olmali, 0.0 degil
    with pytest.raises(ValueError, match="NaN beklenir"):
        Iskelet3B(tanim=ISK, noktalar=noktalar, gorunur=gorunur,
                  goren_kamera=np.zeros(n, int), artik_px=np.zeros(n))


def test_iskelet3b_gorunur_eklem_nan_olamaz():
    n = len(ISK)
    noktalar = np.zeros((n, 3))
    noktalar[2] = np.nan
    with pytest.raises(ValueError, match="sonlu olmayan"):
        Iskelet3B(tanim=ISK, noktalar=noktalar, gorunur=np.ones(n, bool),
                  goren_kamera=np.zeros(n, int), artik_px=np.zeros(n))


def test_baglanti_kendine_ve_tekrarli_reddedilir():
    with pytest.raises(ValueError, match="kendisine"):
        IskeletTanimi("bozuk", ("diz", "ayak"), (("diz", "diz"),))
    with pytest.raises(ValueError, match="tekrarli"):
        IskeletTanimi("bozuk", ("diz", "ayak"),
                      (("diz", "ayak"), ("ayak", "diz")))
