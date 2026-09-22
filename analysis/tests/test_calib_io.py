"""calib.io kabul testleri.

KOD-PLANI.md modul haritasindaki kabul olcutu: yaz-oku turu **birebir ayni veriyi**
doner. Buna ek olarak: bicim surumu uyumsuzlugu reddedilir, yazim atomiktir,
kayit olcege uygun boyutta kalir.
"""
import json

import numpy as np
import pytest

from calib.io import (
    BICIM_SURUMU,
    Kabin,
    Meta,
    kalibrasyondan,
    kalibrasyona,
    oku,
    yaz,
)
from calib.multiview import Kalibrasyon
from calib.synthetic import rig_yay, sahne_uret

from calib.board import BoardSpec

SPEC = BoardSpec(5, 7)


@pytest.fixture(scope="module")
def kalibrasyon() -> Kalibrasyon:
    s = sahne_uret(SPEC, kameralar=rig_yay(odak_px=900.0), n_kare=8,
                   gurultu_px=0.0, dagilim="genis", seed=3)
    from calib.multiview import kalibre_et
    return kalibre_et(s.obj_noktalari, s.img_noktalari,
                      [c.boyut for c in s.kameralar], s.mask)


@pytest.fixture()
def kabin(kalibrasyon) -> Kabin:
    return kalibrasyondan(
        kalibrasyon, n_kare=8, gorunurluk=0.75, surum="test-commit",
        notlar="deneme", etiketler={"duzenek": "yay-4"})


# ---------------------------------------------------------------------------
# Kabul testi: yaz-oku turu birebir ayni veri
# ---------------------------------------------------------------------------

def test_yaz_oku_turu_birebir(kabin, tmp_path):
    yol = yaz(kabin, tmp_path / "kabin.json")
    geri = oku(yol)
    a, b = kabin.to_sozluk(), geri.to_sozluk()
    assert a == b, "sozluk duzeyinde tur birebir degil"


def test_parametreler_birebir(kabin, tmp_path):
    geri = oku(yaz(kabin, tmp_path / "kabin.json"))
    for ad in ("Ks", "bozulmalar", "Rs", "Ts"):
        for x, y in zip(getattr(kabin, ad), getattr(geri, ad)):
            np.testing.assert_array_equal(x, y)


def test_rms_birebir(kabin, tmp_path):
    geri = oku(yaz(kabin, tmp_path / "kabin.json"))
    assert geri.meta.rms_px == kabin.meta.rms_px


def test_kalibrasyona_turu_parametreleri_korur(kabin, tmp_path):
    geri = kalibrasyona(oku(yaz(kabin, tmp_path / "kabin.json")))
    for x, y in zip(kabin.Ks, geri.Ks):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(kabin.Rs, geri.Rs):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(kabin.Ts, geri.Ts):
        np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------------------
# Meta ve surum damgasi
# ---------------------------------------------------------------------------

def test_meta_alanlari_tur_gecisi(kabin, tmp_path):
    geri = oku(yaz(kabin, tmp_path / "kabin.json"))
    assert geri.meta.surum == "test-commit"
    assert geri.meta.notlar == "deneme"
    assert geri.meta.n_kare == 8
    assert geri.meta.gorunurluk == pytest.approx(0.75)
    assert geri.meta.etiketler == {"duzenek": "yay-4"}
    assert geri.meta.bicim_surumu == BICIM_SURUMU
    assert geri.meta.n_kamera == kabin.meta.n_kamera


def test_tarih_iso_utc_biciminde(kalibrasyon):
    k = kalibrasyondan(kalibrasyon)
    assert k.meta.tarih_iso.endswith("+00:00")
    assert "T" in k.meta.tarih_iso


def test_tarih_elle_verilebilir(kalibrasyon):
    k = kalibrasyondan(kalibrasyon, tarih_iso="2026-09-22T12:00:00+00:00")
    assert k.meta.tarih_iso == "2026-09-22T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Uyumsuz bicim reddedilir -- sessiz cozum yok
# ---------------------------------------------------------------------------

def test_uyumsuz_bicim_surumu_reddedilir(kabin, tmp_path):
    yol = yaz(kabin, tmp_path / "kabin.json")
    sozluk = json.loads(yol.read_text(encoding="utf-8"))
    sozluk["bicim_surumu"] = BICIM_SURUMU + 1
    bozuk = tmp_path / "bozuk.json"
    bozuk.write_text(json.dumps(sozluk), encoding="utf-8")
    with pytest.raises(ValueError, match="bicim_surumu"):
        oku(bozuk)


def test_liste_uzunluklari_uyusmazsa_hata(kabin):
    kabin.Rs = kabin.Rs[:-1]
    with pytest.raises(ValueError, match="uyusmuyor"):
        kabin._dogrula()


def test_meta_n_kamera_uyusmazsa_hata(kalibrasyon):
    k = kalibrasyondan(kalibrasyon)
    k.meta.n_kamera = 99
    with pytest.raises(ValueError, match="kamera"):
        k._dogrula()


# ---------------------------------------------------------------------------
# Atomik yazim ve dosya hijyeni
# ---------------------------------------------------------------------------

def test_yazim_atomik_gecici_dosya_birakmaz(kabin, tmp_path):
    yol = yaz(kabin, tmp_path / "kabin.json")
    kalan = sorted(p.name for p in tmp_path.iterdir())
    assert kalan == ["kabin.json"], f"gecici dosya kalmis: {kalan}"
    assert yol.exists()


def test_ara_dizini_otomatik_olusturulur(kabin, tmp_path):
    yol = yaz(kabin, tmp_path / "a" / "b" / "kabin.json")
    assert yol.exists()


def test_json_metin_olarak_okunabilir(kabin, tmp_path):
    yol = yaz(kabin, tmp_path / "kabin.json")
    veri = json.loads(yol.read_text(encoding="utf-8"))
    assert set(veri) == {"bicim_surumu", "meta", "Ks", "bozulmalar", "Rs", "Ts"}
    assert isinstance(veri["Ks"][0][0][0], str), "sayilar metin cinsinden yazilmali"


def test_kayit_kucuk_kalir(kabin, tmp_path):
    yol = yaz(kabin, tmp_path / "kabin.json")
    boyut_kb = yol.stat().st_size / 1024
    assert boyut_kb < 64, f"kayit cok buyuk: {boyut_kb:.1f} KB"


# ---------------------------------------------------------------------------
# fisheye (4 parametreli) bozulma vektoru
# ---------------------------------------------------------------------------

def test_dort_parametreli_bozulma_turu():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    geri = Kabin.from_sozluk({
        "bicim_surumu": BICIM_SURUMU,
        "meta": {"tarih_iso": "2026-09-22T00:00:00+00:00"},
        "Ks": [[[ "1.0", "0.0", "0.0"], ["0.0", "1.0", "0.0"], ["0.0", "0.0", "1.0"]]],
        "bozulmalar": [[repr(float(x)) for x in v]],
        "Rs": [[[ "1.0", "0.0", "0.0"], ["0.0", "1.0", "0.0"], ["0.0", "0.0", "1.0"]]],
        "Ts": [[[ "0.0"], ["0.0"], ["0.0"]]],
    })
    np.testing.assert_array_equal(geri.bozulmalar[0], np.append(v, 0.0))
