"""eval.senkron kabul testleri.

En onemlisi `test_ortak_kayma_hata_uretmez`: hata kaymanin kendisinden degil,
kameralar **arasindaki** farktan dogar. Bu ayrim yanlis kurulursa donanim
gereksinimi gereksiz yere sikilasir -- hepsi birlikte gec tetiklenen bir sistem
kusursuzdur, yalnizca zamani otelenmistir.
"""
import json

import numpy as np
import pytest

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import rig_yay, sahne_uret
from eval.senkron import KaymaSonucu, azami_kayma_ms, kayma_hatasi, kayma_taramasi


@pytest.fixture(scope="module")
def duzenek():
    kameralar = rig_yay(odak_px=900.0)
    s = sahne_uret(BoardSpec(5, 7), kameralar=kameralar, n_kare=20,
                   gurultu_px=0.0, dagilim="genis", seed=8080)
    kalib = kalibre_et(s.obj_noktalari, s.img_noktalari,
                       [c.boyut for c in kameralar], s.mask)
    return kalib, kameralar


NOKTA = np.array([0.1, 0.0, -0.05])
HIZ = np.array([1.0, 0.0, 0.0])          # 1 m/s yanal


# --- fizik -------------------------------------------------------------------

def test_kayma_yoksa_hata_yok(duzenek):
    kalib, kameralar = duzenek
    h = kayma_hatasi(kalib, kameralar, NOKTA, HIZ, np.zeros(len(kameralar)))
    assert h is not None
    assert h * 1000.0 < 0.1          # mm


def test_ortak_kayma_hata_uretmez(duzenek):
    """Butun kameralar ayni anda gec tetiklenirse hata olusmamali.

    Hata kameralar arasindaki **farktan** dogar. Bu test gecmezse donanim
    gereksinimi yanlis hesaplanir: ortak gecikme zararsizdir.
    """
    kalib, kameralar = duzenek
    ortak = np.full(len(kameralar), 0.050)        # hepsi 50 ms gec
    h = kayma_hatasi(kalib, kameralar, NOKTA, HIZ, ortak)
    assert h is not None
    assert h * 1000.0 < 0.1


def test_durgun_nokta_kaymadan_etkilenmez(duzenek):
    """Hareket yoksa senkron kaymasi hata uretmez -- hata hiz x zamandir."""
    kalib, kameralar = duzenek
    kaymalar = np.array([0.0, 0.01, -0.01, 0.02])[:len(kameralar)]
    h = kayma_hatasi(kalib, kameralar, NOKTA, np.zeros(3), kaymalar)
    assert h is not None
    assert h * 1000.0 < 0.1


def test_hata_hizla_dogru_orantili(duzenek):
    """Hiz iki katina cikarsa hata da iki katina cikmali (p = v * dt)."""
    kalib, kameralar = duzenek
    kaymalar = np.array([0.0, 0.0, 0.0, 0.010])[:len(kameralar)]
    kaymalar = kaymalar - kaymalar.mean()
    tek = kayma_hatasi(kalib, kameralar, NOKTA, HIZ, kaymalar)
    cift = kayma_hatasi(kalib, kameralar, NOKTA, 2.0 * HIZ, kaymalar)
    assert tek is not None and cift is not None
    assert cift == pytest.approx(2.0 * tek, rel=0.05)


def test_hata_kaymayla_artiyor(duzenek):
    kalib, kameralar = duzenek
    sonuclar = kayma_taramasi(kalib, kameralar, [0.0, 10.0, 30.0], [1.0],
                              rejimler=("tek_kamera",), n_ornek=40)
    p95 = [s.p95_mm for s in sorted(sonuclar, key=lambda s: s.kayma_ms)]
    assert p95[0] < p95[1] < p95[2]


# --- rejimler ----------------------------------------------------------------

def test_iki_rejim_de_taraniyor(duzenek):
    kalib, kameralar = duzenek
    sonuclar = kayma_taramasi(kalib, kameralar, [10.0], [1.0], n_ornek=30)
    assert {s.rejim for s in sonuclar} == {"tek_kamera", "dagilmis"}


def test_bilinmeyen_rejim_hata_verir(duzenek):
    kalib, kameralar = duzenek
    with pytest.raises(ValueError, match="bilinmeyen rejim"):
        kayma_taramasi(kalib, kameralar, [10.0], [1.0],
                       rejimler=("yok",), n_ornek=5)


# --- donanim gereksinimi -----------------------------------------------------

def test_azami_kayma_esikten_bulunur(duzenek):
    """Kapi 2'nin 10 mm butcesi zaman cinsinden bir sinira cevrilmeli."""
    kalib, kameralar = duzenek
    sonuclar = kayma_taramasi(kalib, kameralar, [0.0, 10.0, 20.0, 40.0], [1.0],
                              rejimler=("tek_kamera",), n_ornek=60)
    sinir = azami_kayma_ms(sonuclar, "tek_kamera", 1.0, esik_mm=10.0)
    assert sinir is not None
    assert 0.0 < sinir < 40.0


def test_azami_kayma_hic_asilmazsa_taranan_en_buyugu_doner(duzenek):
    kalib, kameralar = duzenek
    sonuclar = kayma_taramasi(kalib, kameralar, [0.0, 2.0], [0.1],
                              rejimler=("tek_kamera",), n_ornek=20)
    assert azami_kayma_ms(sonuclar, "tek_kamera", 0.1, esik_mm=10.0) == 2.0


def test_azami_kayma_ilk_noktada_asiliyorsa_none():
    """Tolerans yoksa sayi uydurulmamali."""
    sonuclar = [
        KaymaSonucu(kayma_ms=5.0, hiz_m_s=1.0, rejim="tek_kamera",
                    ortalama_mm=50.0, medyan_mm=50.0, p95_mm=80.0,
                    n_ornek=10, n_basarisiz=0),
    ]
    assert azami_kayma_ms(sonuclar, "tek_kamera", 1.0, esik_mm=10.0) is None


def test_azami_kayma_bilinmeyen_kosul_hata():
    with pytest.raises(ValueError, match="sonuc icermiyor"):
        azami_kayma_ms([], "tek_kamera", 1.0)


# --- rapor -------------------------------------------------------------------

def test_basarisiz_ornek_sayiliyor(duzenek):
    """Ucgenlenemeyen ornek sessizce sifir hata sayilmamali."""
    kalib, kameralar = duzenek
    sonuclar = kayma_taramasi(kalib, kameralar, [10.0], [1.0],
                              rejimler=("tek_kamera",), n_ornek=25)
    s = sonuclar[0]
    assert s.n_ornek + s.n_basarisiz == 25


def test_ozet_sozluk_dondurur(duzenek):
    kalib, kameralar = duzenek
    s = kayma_taramasi(kalib, kameralar, [10.0], [1.0],
                       rejimler=("tek_kamera",), n_ornek=20)[0]
    geri = json.loads(json.dumps(s.ozet(), ensure_ascii=False, allow_nan=False))
    assert geri["rejim"] == "tek_kamera"
    assert geri["kayma_ms"] == 10.0


def test_tekrarlanabilir(duzenek):
    kalib, kameralar = duzenek
    a = kayma_taramasi(kalib, kameralar, [15.0], [1.0], n_ornek=25, seed=7)
    b = kayma_taramasi(kalib, kameralar, [15.0], [1.0], n_ornek=25, seed=7)
    assert [x.ozet() for x in a] == [x.ozet() for x in b]
