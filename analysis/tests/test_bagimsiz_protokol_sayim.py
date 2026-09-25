"""Bagimsiz dogrulama — 0070 sayim ve kisi bootstrap sozlesmeleri.

`eval/protokol` uzerinde (misyon 2. maddenin eval yarisi). Kaynak:
`docs/kararlar/0070` §1 (paydalar/oranlar) ve §2 (kisi kumesi bootstrap).

Mevcut `tests/test_eval_protokol.py` sabit bir matris ve varsayilan tohumla
calisir; burada **rastgele uretilmis** matrislerde degismezler ve tohum
determinizmi sinanir. Uretim kodu degismez; bulgu yalniz basarisiz testle
gosterilir. `hypothesis` kurulu degil, tohumlanmis numpy kullanilir.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.protokol import (
    BELIRSIZ,
    BOOTSTRAP_TOHUM,
    DOGRU,
    KUSURLU,
    Oge,
    Sayim,
    kisi_agirlikli,
    kisi_bootstrap,
)

ETIKETLER = (DOGRU, KUSURLU, BELIRSIZ, None)


def _rastgele_sayim(rng, ust=40):
    n = int(rng.integers(0, ust))
    ciftler = [(ETIKETLER[int(rng.integers(0, 4))], ETIKETLER[int(rng.integers(0, 4))])
               for _ in range(n)]
    return Sayim.ciftlerden(ciftler)


def _kacirma(s: Sayim) -> int:
    return s.matris.get((KUSURLU, DOGRU), 0)


def _yanlis_alarm(s: Sayim) -> int:
    return s.matris.get((DOGRU, KUSURLU), 0)


# --- §1 payda ve oran degismezleri -------------------------------------------

def test_bagimsiz_yanlis_kacirma_arti_yanlis_alarm():
    """W = D - C, tanim geregi kacirma + yanlis alarm olmali (0070 §1)."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        s = _rastgele_sayim(rng)
        assert s.W == _kacirma(s) + _yanlis_alarm(s), s.matris
        assert s.W >= 0


def test_bagimsiz_payda_ozdeslikleri_rastgele_matrislerde():
    """M = R + N, D + U = N, C + W = D her matriste tutmali (0070 §1)."""
    rng = np.random.default_rng(12)
    for _ in range(300):
        s = _rastgele_sayim(rng)
        assert s.M == s.R + s.N
        assert s.D + s.U == s.N
        assert s.C + s.W == s.D
        assert s.M == sum(s.matris.values())
        # Referansi kesin olmayan her oge R'ye, telefonu kesin olmayan her oge U'ya girer.
        assert s.R == s._topla(lambda r, t: r not in (DOGRU, KUSURLU))
        assert s.U == s._topla(lambda r, t: r in (DOGRU, KUSURLU)
                               and t not in (DOGRU, KUSURLU))


def test_bagimsiz_A_K_Y_tanimlari_rastgele_matrislerde():
    """A = C/D, K = D/N, Y = C/N; tanimsizsa None (0070 §1)."""
    rng = np.random.default_rng(13)
    for _ in range(300):
        s = _rastgele_sayim(rng)
        o = s.oranlar()
        assert o["A"] == (s.C / s.D if s.D else None)
        assert o["K"] == (s.D / s.N if s.N else None)
        assert o["Y"] == (s.C / s.N if s.N else None)
        assert o["W_N"] == (s.W / s.N if s.N else None)
        assert o["U_N"] == (s.U / s.N if s.N else None)
        # Y = A x K, iki oran da tanimliyken (0070 §1).
        if s.D and s.N:
            assert o["Y"] == pytest.approx(o["A"] * o["K"], abs=1e-12)


def test_bagimsiz_oran_daima_00_ile_10_arasi_veya_none():
    rng = np.random.default_rng(14)
    for _ in range(200):
        o = _rastgele_sayim(rng).oranlar()
        for ad in ("A", "K", "Y", "W_N", "U_N"):
            v = o[ad]
            assert v is None or 0.0 <= v <= 1.0, (ad, v)


# --- §1 sinir durumlari -------------------------------------------------------

def test_bagimsiz_bos_sayimda_hicbir_oran_tanimli_degil():
    o = Sayim({}).oranlar()
    assert (o["M"], o["R"], o["N"], o["D"], o["C"], o["W"], o["U"]) == (0,) * 7
    for ad in ("A", "K", "Y", "W_N", "U_N", "A_dengeli", "K_dengeli"):
        assert o[ad] is None, ad
    assert o["matris"] == {}


def test_bagimsiz_hepsi_eksik_kararda_A_none_K_sifir():
    """D = 0 ama N > 0: A tanimsiz, kapsama 0 (asla None degil)."""
    o = Sayim.ciftlerden([(KUSURLU, None)] * 5).oranlar()
    assert o["A"] is None and o["Y"] == 0.0 and o["K"] == 0.0
    assert o["W_N"] == 0.0 and o["U_N"] == 1.0


def test_bagimsiz_hic_gecerli_referans_yoksa_N_sifir_ve_oranlar_none():
    """R = M ise N = 0: K ve Y tanimsiz olmali (0070 §1)."""
    o = Sayim.ciftlerden([(None, DOGRU), (BELIRSIZ, KUSURLU), (None, None)]).oranlar()
    assert (o["M"], o["R"], o["N"]) == (3, 3, 0)
    assert o["K"] is None and o["Y"] is None and o["A"] is None
    assert o["A_dengeli"] is None and o["K_dengeli"] is None


def test_bagimsiz_tek_sinifli_matriste_dengeli_olcut_none():
    """Tek referans sinifi varsa sinif-dengeli A ve K genel basari sayilmaz."""
    o = Sayim.ciftlerden([(KUSURLU, KUSURLU)] * 4).oranlar()
    assert o["A"] == 1.0 and o["A_dengeli"] is None
    assert o["K_dengeli"] is None


def test_bagimsiz_ayni_cift_tekrarlari_sayilir():
    s = Sayim.ciftlerden([(KUSURLU, DOGRU)] * 7 + [(DOGRU, DOGRU)])
    assert s.matris[(KUSURLU, DOGRU)] == 7
    assert s.M == 8 and s.C == 1


# --- §2 kisi bootstrap: tohum determinizmi ------------------------------------

def _heterojen_ogeler(n=15, oturum=1):
    """Kisiler arasinda farkli basari oranlari olan kume (bootstrap icin)."""
    return [Oge(f"k{i:02d}", f"o{j}", KUSURLU, KUSURLU if (i + j) % 3 else DOGRU)
            for i in range(n) for j in range(oturum)]


def test_bagimsiz_bootstrap_ayni_tohum_birebir_ayni_aralik():
    ogeler = _heterojen_ogeler()
    a = kisi_bootstrap(ogeler, tekrar=300, tohum=BOOTSTRAP_TOHUM)
    b = kisi_bootstrap(ogeler, tekrar=300, tohum=BOOTSTRAP_TOHUM)
    assert a == b
    assert a["tohum"] == BOOTSTRAP_TOHUM and a["durum"] == "hesaplandi"


def test_bagimsiz_bootstrap_farkli_tohum_farkli_aralik():
    """Tohum degisince aralik degismeli; aksi halde tohum etkisiz demektir."""
    ogeler = _heterojen_ogeler()
    a = kisi_bootstrap(ogeler, tekrar=300, tohum=70)
    b = kisi_bootstrap(ogeler, tekrar=300, tohum=71)
    c = kisi_bootstrap(ogeler, tekrar=300, tohum=12345)
    assert a != b and a != c, "tohum araligi degistirmiyor"


def test_bagimsiz_bootstrap_girdi_sirasindan_bagimsiz():
    """Ayni kume karisik verilirse ayni tohumla birebir ayni aralik cikmali."""
    ogeler = _heterojen_ogeler(n=12, oturum=2)
    rng = np.random.default_rng(21)
    karisik = list(ogeler)
    rng.shuffle(karisik)
    assert (kisi_bootstrap(ogeler, tekrar=200, tohum=70)
            == kisi_bootstrap(karisik, tekrar=200, tohum=70))


def test_bagimsiz_bootstrap_ikiden_az_kiside_dogrulanmadi():
    bos = kisi_bootstrap([])
    tek = kisi_bootstrap([Oge("k1", "o1", DOGRU, DOGRU)])
    assert bos["durum"] == "dogrulanmadi" and bos["n_kisi"] == 0
    assert tek["durum"] == "dogrulanmadi" and tek["n_kisi"] == 1
    assert "aralik_95" not in tek


def test_bagimsiz_bootstrap_aralik_sirali_ve_bir_kisi_hic_silinmez():
    """Bir kisinin butun oturumlari birlikte tasinmali; araliklar sirali."""
    ogeler = _heterojen_ogeler(n=10, oturum=3)
    b = kisi_bootstrap(ogeler, tekrar=400, tohum=70)
    assert b["n_kisi"] == 10
    for ad, aralik in b["aralik_95"].items():
        assert aralik is None or aralik[0] <= aralik[1], ad


def test_bagimsiz_bootstrap_tanimsiz_tekrarlar_sayilir_ve_gizlenmez():
    """Hic D'si olmayan kume: A her cekiliste tanimsiz, sayisi yazilmali (0070 §2)."""
    ogeler = [Oge(f"k{i}", "o1", KUSURLU, None) for i in range(4)]
    b = kisi_bootstrap(ogeler, tekrar=50, tohum=70)
    assert b["aralik_95"]["A"] is None
    assert b["tanimsiz_tekrar"]["A"] == 50
    # Kapsama tanimli: D = 0 ama N > 0.
    assert b["aralik_95"]["K"] == [0.0, 0.0]
    assert b["tanimsiz_tekrar"]["K"] == 0


def test_bagimsiz_kisi_agirlikli_bos_kume():
    o = kisi_agirlikli([])
    assert o["n_kisi"] == 0
    assert o["A"] is None and o["K"] is None and o["Y"] is None
    assert o["null_kisi"] == {"A": 0, "K": 0, "Y": 0}


def test_bagimsiz_uzun_oturum_kisiyi_daha_agir_yapmaz():
    """Bir kisinin uzun oturumu, oranlari oturum sayisiyla esit agirlikta."""
    ogeler = ([Oge("k1", "o1", KUSURLU, KUSURLU)] * 300      # uzun, hep dogru
              + [Oge("k1", "o2", KUSURLU, DOGRU)] * 3        # kisa, hep yanlis
              + [Oge("k2", "o1", DOGRU, DOGRU)] * 10)
    assert kisi_agirlikli(ogeler)["A"] == pytest.approx(0.75)
