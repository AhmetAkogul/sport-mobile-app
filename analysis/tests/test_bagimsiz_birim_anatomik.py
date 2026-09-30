"""Bagimsiz dogrulama — anatomik oran ve birim sozlesmeleri (misyon 2d-2e).

Bu dosya **uretim kodunu degistirmez**. Kaynak sozlesmeler:
`docs/kararlar/0014-bukulmeye-dayanikli-valgus.md` (Karar 2: uyluk/baldir
orani [0,5; 2,0] disindaysa valgus NaN, karar BELIRSIZ) ve `docs/kararlar/0072`
(§1: normalize iskelet metrik islemlerden gecmez).

`hypothesis` ortamda kurulu degil; rastgelelik tohumlanmis numpy ile uretilir.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eval.egzersiz import kare_degerlendir
from eval.form import REFERANS_ISKELET, Karar, form_degerlendir, sentetik_durus
from pose3d.iskelet import Iskelet3B
from uncertainty.eklem import eklem_kovaryanslari

IDX = REFERANS_ISKELET.indeks


def _deger(rapor, ad: str) -> float:
    return rapor.olcumler[ad].deger


def _oranli_durus(oran: float, baldir_m: float = 0.4) -> Iskelet3B:
    """Sag bacagi duz (valgus 0) ve uyluk/baldir orani **tam** `oran` olan durus.

    Diz kalca-ayak bilegi dogrusunun uzerindedir, dolayisiyla geometrik valgus
    tam 0'dir; tek serbest degisken orandir.
    """
    temel = sentetik_durus()
    P = temel.noktalar.copy()
    uyluk_m = oran * baldir_m
    h = P[IDX("sag_kalca")].copy()
    P[IDX("sag_diz")] = h + np.array([0.0, -uyluk_m, 0.0])
    P[IDX("sag_ayak_bilegi")] = P[IDX("sag_diz")] + np.array([0.0, -baldir_m, 0.0])
    return dataclasses.replace(temel, noktalar=P)


def _normalize(iskelet: Iskelet3B) -> Iskelet3B:
    return dataclasses.replace(iskelet, birim="normalize")


# --- (d) anatomik oran -------------------------------------------------------

@pytest.mark.parametrize("oran", [0.49, 0.499999, 2.000001, 2.01])
def test_bagimsiz_anatomik_oran_disinda_nan_ve_belirsiz(oran):
    """Aralik disindaki oranda olcum yok ve karar BELIRSIZ olmali (0014)."""
    o = form_degerlendir(_oranli_durus(oran)).olcumler["diz_valgusu_sag"]
    assert not np.isfinite(o.deger), f"oran {oran}: beklenen NaN, gelen {o.deger}"
    assert o.karar is Karar.BELIRSIZ, f"oran {oran}: karar {o.karar}"


@pytest.mark.parametrize("oran", [0.6, 0.8, 1.0, 1.25, 1.4, 1.8, 1.999])
def test_bagimsiz_anatomik_oran_ici_gecerli(oran):
    """Aralik icindeki oranda olcum verilmeli (duz bacak -> 0 derece)."""
    o = form_degerlendir(_oranli_durus(oran)).olcumler["diz_valgusu_sag"]
    assert np.isfinite(o.deger), f"oran {oran}: olcum atildi"
    assert o.deger == pytest.approx(0.0, abs=1e-9)
    assert o.karar is Karar.DOGRU


@pytest.mark.parametrize("oran", [0.5, 2.0])
def test_bagimsiz_anatomik_oran_sinirlari_dahil(oran):
    """0014 araligi **kapali**: tam 0,5 ve 2,0 olcum vermeli.

    Tam sinirda karar `BELIRSIZ` ise gecerli bir geometri sessizce atilmis
    demektir; bu, oranin nokta hesabinda yuvarlanmasiyla olur.
    """
    o = form_degerlendir(_oranli_durus(oran)).olcumler["diz_valgusu_sag"]
    assert np.isfinite(o.deger), (
        f"oran {oran} kapali sinirda; beklenen sonlu deger, gelen NaN "
        f"(karar {o.karar})")


def test_bagimsiz_anatomik_oran_sinir_komsulugu_tutarli():
    """Sinirdan cok az icerisi gecmeli, cok az disarisi elinmeli."""
    iceri = form_degerlendir(_oranli_durus(1e-6 + 0.5)).olcumler["diz_valgusu_sag"]
    disari = form_degerlendir(_oranli_durus(0.5 - 1e-6)).olcumler["diz_valgusu_sag"]
    assert np.isfinite(iceri.deger)
    assert not np.isfinite(disari.deger)


def test_bagimsiz_anatomik_oran_iki_tarafi_icin_kapali_olcum():
    """Ust sinir da ayni sekilde kapali davranmali (alt sinirla tutarli)."""
    alt = form_degerlendir(_oranli_durus(0.5)).olcumler["diz_valgusu_sag"]
    ust = form_degerlendir(_oranli_durus(2.0)).olcumler["diz_valgusu_sag"]
    assert np.isfinite(alt.deger) == np.isfinite(ust.deger), (
        f"alt sinir sonlu={np.isfinite(alt.deger)}, "
        f"ust sinir sonlu={np.isfinite(ust.deger)}: sinirlar tutarsiz")


# --- (e) birim sozlesmesi ----------------------------------------------------

def test_bagimsiz_normalize_uzunluk_hata_verir():
    """Normalleştirilmis iskelette metrik `uzunluk` reddedilmeli (0072 §1)."""
    with pytest.raises(ValueError):
        _normalize(sentetik_durus()).uzunluk("sag_kalca", "sag_diz")


def test_bagimsiz_normalize_eklem_kovaryanslari_hata_verir():
    """Normalleştirilmis iskelette metre cinsinden kovaryans uretilememeli."""
    with pytest.raises(ValueError):
        eklem_kovaryanslari(None, _normalize(sentetik_durus()), {})


def test_bagimsiz_normalize_belirsizlikli_form_degerlendirme_hata_verir():
    """Belirsizlik metre cinsindendir; normalize iskelete verilememeli."""
    with pytest.raises(ValueError):
        form_degerlendir(_normalize(sentetik_durus()), konum_belirsizligi_m=0.003)


def test_bagimsiz_normalize_eksen_basina_kovaryansla_da_hata_verir():
    """(N,3,3) kovaryans yolu da metre denetiminden gecmeli."""
    n = len(REFERANS_ISKELET)
    kov = np.tile(np.eye(3) * 1e-6, (n, 1, 1))
    with pytest.raises(ValueError):
        form_degerlendir(_normalize(sentetik_durus()), konum_belirsizligi_m=list(kov))


def test_bagimsiz_normalize_salt_aci_hesabi_calisir_ve_metreyle_ayni():
    """Normalize iskelette yalniz aci hesabi calismali ve metreyle ayni olmali."""
    temel = sentetik_durus(valgus_sag=12.0, valgus_sol=-6.0)
    metre = form_degerlendir(temel)
    normalize = form_degerlendir(_normalize(temel))
    for ad in ("diz_valgusu_sag", "diz_valgusu_sol", "kalca_hizasi", "govde_rotasyonu"):
        assert _deger(normalize, ad) == pytest.approx(_deger(metre, ad), abs=1e-9), ad


def test_bagimsiz_normalize_olcek_serbest_wrapper_de_ayni():
    """Ayni durus 100 kat buyutulunce normalize sonuc degismemeli."""
    temel = sentetik_durus(valgus_sag=12.0)
    buyuk = dataclasses.replace(temel, noktalar=temel.noktalar * 100.0)
    assert _deger(form_degerlendir(_normalize(buyuk)), "diz_valgusu_sag") == \
        pytest.approx(_deger(form_degerlendir(_normalize(temel)), "diz_valgusu_sag"),
                      abs=1e-9)


def test_bagimsiz_normalize_egzersiz_kapisi_kesin_karar_vermez():
    """Belirsizlikli karar kapisi normalize iskelette BELIRSIZ demeli, cokmemeli."""
    n = len(REFERANS_ISKELET)
    kov = np.tile(np.eye(3) * 1e-6, (n, 1, 1))
    iskelet = dataclasses.replace(
        _normalize(sentetik_durus(valgus_sag=12.0)), kovaryans=kov,
        belirsizlik_durumu="kosullu", belirsizlik_kaynagi="tespit_gurultusu")
    rapor = kare_degerlendir(iskelet)
    olcum = rapor["olcumler"]["diz_valgusu_sag"]
    assert olcum["karar"] == str(Karar.BELIRSIZ)
    assert olcum["neden"] == "metrik_belirsizlik_yok"


def test_bagimsiz_normalize_metrik_islem_sozlesmesi_boolean():
    """Sozlesme cagri bazli olmali: ayni iskelet aci icin gecerli, uzunluk icin degil."""
    iskelet = _normalize(sentetik_durus(valgus_sag=12.0))
    assert np.isfinite(_deger(form_degerlendir(iskelet), "diz_valgusu_sag"))
    with pytest.raises(ValueError):
        iskelet.uzunluk("sag_kalca", "sag_diz")
