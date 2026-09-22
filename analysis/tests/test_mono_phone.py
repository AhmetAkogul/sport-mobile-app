"""mono.phone kabul testleri.

KOD-PLANI.md modul haritasindaki kabul olcutu: "Ayni videodan iki kez ayni
cikti". Ek olarak HAZIR-TEKNOLOJILER sozlesmesinin kod tarafindaki denetimleri
(uzay geri donusumu, eklem sirasinin isimle eslesmesi) ve tek gorus baseline'in
paralel duzlemde kapali-form dogrulugu burada sinanir.
"""
from dataclasses import replace

import numpy as np
import pytest

from calib.synthetic import Kamera
from mono.phone import (
    KareKaydi,
    hatti_kostur,
    sentetik_kestirici,
    tek_gorus_3b,
)
from pose3d.iskelet import REFERANS_ISKELET, IskeletTanimi
from pose3d.pose2d import sentetik_poz

K = np.array([[700.0, 0.0, 640.0],
              [0.0, 700.0, 360.0],
              [0.0, 0.0, 1.0]])
KAMERA = Kamera(K=K, R=np.eye(3), t=np.zeros((3, 1)), boyut=(1280, 720))


def _poz_3b(kayma_x: float = 0.0) -> np.ndarray:
    """Bilinen 3B iskelet; tum eklemler z=3.0 duzleminde (kamera duzlemine paralel).

    Kamera R=I, t=0 oldugu icin kamera koordinati = dunya koordinati; derinlik
    her eklem icin tam 3.0'dir.
    """
    xy = {
        "boyun": (0.00, 0.35),
        "sag_omuz": (-0.20, 0.30), "sag_dirsek": (-0.26, 0.10), "sag_bilek": (-0.30, -0.08),
        "sol_omuz": (0.20, 0.30), "sol_dirsek": (0.26, 0.10), "sol_bilek": (0.30, -0.08),
        "sag_kalca": (-0.10, -0.05), "sag_diz": (-0.11, -0.48), "sag_ayak_bilegi": (-0.11, -0.90),
        "sol_kalca": (0.10, -0.05), "sol_diz": (0.11, -0.48), "sol_ayak_bilegi": (0.11, -0.90),
    }
    return np.array([[xy[ad][0] + kayma_x, xy[ad][1], 3.0]
                     for ad in REFERANS_ISKELET.eklemler])


@pytest.fixture()
def onculer() -> dict:
    """Kemik uzunlugu onculeri: yer gercegi iskeletten hesaplanir."""
    X = _poz_3b()
    return {(a, b): float(np.linalg.norm(X[REFERANS_ISKELET.indeks(a)]
                                         - X[REFERANS_ISKELET.indeks(b)]))
            for a, b in REFERANS_ISKELET.baglantilar}


@pytest.fixture()
def iskeletler_3b() -> list[np.ndarray]:
    return [_poz_3b(kayma_x=0.02 * i) for i in range(5)]


def _kareler(n: int = 5) -> list[KareKaydi]:
    return [KareKaydi(kare=i, image_bgr=np.zeros((720, 1280, 3), np.uint8),
                      zaman_ms=33.0 * i) for i in range(n)]


# ---------------------------------------------------------------------------
# Kabul testi: paralel duzlemde tek gorus kestirimi kapali-form olarak dogru
# ---------------------------------------------------------------------------

def test_paralel_duzlemde_derinlik_tam_bulunuyor(onculer):
    """Kamera duzlemine paralel pozda s = f*L/l tam derinligi verir; hatasizda
    yeniden yapilandirma yer gercegine esit olmali."""
    gercek = _poz_3b()
    poz = sentetik_poz(KAMERA, gercek, REFERANS_ISKELET, 0.0)
    s = tek_gorus_3b(poz, K, onculer)
    assert s.gorunur.all()
    np.testing.assert_allclose(s.noktalar, gercek, atol=1e-9)


def test_kemik_uzunlugu_ciktiyi_bagimsiz_doldurmaz(onculer):
    """Yalniz kalca-diz onculu verilmisse yalniz o kemicin uclari gorunur kalir;
    diger eklemler uydurulmaz."""
    X = _poz_3b()
    poz = sentetik_poz(KAMERA, X, REFERANS_ISKELET, 0.0)
    L = float(np.linalg.norm(X[7] - X[8]))          # sag_kalca - sag_diz
    s = tek_gorus_3b(poz, K, {("sag_kalca", "sag_diz"): L})
    assert s.gorunur[7] and s.gorunur[8]
    assert not s.gorunur[0] and not s.gorunur[10]   # boyun, sol_kalca
    assert s.goren_kamera[7] == 1                   # tek gorus
    np.testing.assert_allclose(s.noktalar[8, 2], 3.0, atol=1e-9)


def test_baglantisiz_eklem_uydurulmaz():
    """Gorunse bile hicbir onculu kemige bagli olmayan eklem gorunmez isaretlenir."""
    tanim = IskeletTanimi("uc", ("a", "b", "c"), (("a", "b"),))
    X = np.array([[0.0, 0.0, 3.0], [0.2, 0.0, 3.0], [0.5, 0.3, 3.0]])
    poz = sentetik_poz(KAMERA, X, tanim, 0.0)
    s = tek_gorus_3b(poz, K, {("a", "b"): 0.2}, tanim=tanim)
    assert s.gorunur[:2].all() and not s.gorunur[2]
    assert np.isnan(s.noktalar[2]).all()


def test_sifir_uzunluklu_izdusum_derinlik_vermez():
    tanim = IskeletTanimi("iki", ("a", "b"), (("a", "b"),))
    X = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    poz = sentetik_poz(KAMERA, X, tanim, 0.0)
    s = tek_gorus_3b(poz, K, {("a", "b"): 0.2}, tanim=tanim)
    assert not s.gorunur.any()


def test_eklem_sirasi_isimle_eslesir(onculer):
    """Model eklemleri baska sirada tasiyorsa sonuc degismemeli (HAZIR madde 2)."""
    karisik_sira = tuple(reversed(REFERANS_ISKELET.eklemler))
    model_tanim = IskeletTanimi("karisik", karisik_sira)
    gercek = _poz_3b()
    poz_ref = sentetik_poz(KAMERA, gercek, REFERANS_ISKELET, 0.0)
    permutasyon = [REFERANS_ISKELET.indeks(ad) for ad in karisik_sira]
    poz_model = replace(poz_ref,
                        iskelet=model_tanim,
                        noktalar=poz_ref.noktalar[permutasyon],
                        guven=poz_ref.guven[permutasyon],
                        gorunur=poz_ref.gorunur[permutasyon])
    a = tek_gorus_3b(poz_ref, K, onculer)
    b = tek_gorus_3b(poz_model, K, onculer)
    np.testing.assert_allclose(b.noktalar, a.noktalar, atol=1e-12)


# ---------------------------------------------------------------------------
# Girdi denetimleri
# ---------------------------------------------------------------------------

def test_odak_pozitif_olmali(onculer):
    poz = sentetik_poz(KAMERA, _poz_3b(), REFERANS_ISKELET, 0.0)
    bozuk_k = np.array([[0.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="odak"):
        tek_gorus_3b(poz, bozuk_k, onculer)


def test_gecersiz_oncu_uzunlugu_reddedilir(onculer):
    poz = sentetik_poz(KAMERA, _poz_3b(), REFERANS_ISKELET, 0.0)
    with pytest.raises(ValueError, match="pozitif"):
        tek_gorus_3b(poz, K, {("boyun", "sag_omuz"): -1.0})


# ---------------------------------------------------------------------------
# Hat: kayit -> 2B poz -> 3B
# ---------------------------------------------------------------------------

def test_hat_uctan_uca_letterbox_geri_donusuyle(iskeletler_3b, onculer):
    """Letterbox'li adaptor ciktisi ozgun uzaya geri dondurulmus olmali (HAZIR madde 3)."""
    kestirici = sentetik_kestirici(KAMERA, iskeletler_3b, model_boyutu=(640, 480))
    sonuc = hatti_kostur(_kareler(), kestirici, K, onculer)
    assert sonuc.tanim is REFERANS_ISKELET
    assert all(p.uzay == "ozgun" for p in sonuc.pozlar)
    o = sonuc.ozet()
    assert o["n_kare"] == 5
    assert o["zamani_bilinen_kare"] == 5
    assert o["modeller"] == ["sentetik-baseline"]
    assert o["iskelet"] == REFERANS_ISKELET.ad
    assert o["ortalama_gorunur_oran"] == pytest.approx(1.0)
    assert set(o["eklem_gorunurluk"].values()) == {1.0}


def test_ayni_girdiyle_iki_kosu_ayni_cikti(iskeletler_3b, onculer):
    """KABUL TESTI (KOD-PLANI): ayni videodan iki kez ayni cikti."""
    # Kestirici durumludur (kare sayaci); gercek bir yeniden kosu da modeli
    # bastan kosturur, dolayisiyla iki cagri de taze kestiriciyle yapilir.
    def kos():
        return hatti_kostur(_kareler(), sentetik_kestirici(KAMERA, iskeletler_3b),
                            K, onculer)
    bir, iki = kos(), kos()
    assert bir.ozet() == iki.ozet()
    for a, b in zip(bir.iskeletler, iki.iskeletler):
        np.testing.assert_array_equal(a.noktalar, b.noktalar)
        np.testing.assert_array_equal(a.gorunur, b.gorunur)


def test_guven_esigi_noktayi_silmez_maskeler(iskeletler_3b, onculer):
    kestirici = sentetik_kestirici(KAMERA, iskeletler_3b)
    sonuc = hatti_kostur(_kareler(), kestirici, K, onculer, guven_esigi=0.95)
    assert sonuc.ozet()["ortalama_gorunur_oran"] < 1.0
    maskeli_var = any((~s.gorunur).any() for s in sonuc.iskeletler)
    assert maskeli_var
    # Maskeli eklem icin 2B nokta dizide durur; yalnizca 3B uydurulmaz.
    assert len(sonuc.pozlar[0].noktalar) == len(REFERANS_ISKELET)


def test_model_uzayindaki_poz_reddedilir(iskeletler_3b, onculer):
    """Geri donusum yapilmamis model-uzayi poz hat'a giremez -- sessiz hata engeli."""
    ic_kestirici = sentetik_kestirici(KAMERA, iskeletler_3b)

    def bozuk_kestirici(image_bgr):
        return replace(ic_kestirici(image_bgr), uzay="model")

    with pytest.raises(ValueError, match="ozgun"):
        hatti_kostur(_kareler(1), bozuk_kestirici, K, onculer)


def test_iki_kanalli_goruntu_reddedilir(onculer):
    kestirici = sentetik_kestirici(KAMERA, [_poz_3b()])
    kareler = [KareKaydi(kare=0, image_bgr=np.zeros((720, 1280)), zaman_ms=None)]
    with pytest.raises(ValueError, match="3 kanalli"):
        hatti_kostur(kareler, kestirici, K, onculer)


def test_kestirici_kare_sayisi_asilursa_hata(iskeletler_3b, onculer):
    kestirici = sentetik_kestirici(KAMERA, iskeletler_3b)
    with pytest.raises(ValueError, match="bitti"):
        hatti_kostur(_kareler(6), kestirici, K, onculer)


def test_bos_kare_listesi(onculer):
    sonuc = hatti_kostur([], sentetik_kestirici(KAMERA, []), K, onculer)
    o = sonuc.ozet()
    assert o["n_kare"] == 0
    assert o["zamani_bilinen_kare"] == 0
    assert all(v == 0.0 for v in o["eklem_gorunurluk"].values())
