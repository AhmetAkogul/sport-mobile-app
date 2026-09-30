"""uncertainty.drift kabul testleri.

Suruklenme egrisi kalibrasyon kayitlarindan (`calib.io`) zaman serisi uretir;
veri diske girmeden once serinin kendisi sinanir. Kapi 2 olcutu (">= 4 haftalik
kayit") ve duzenek degisiminin sessiz karsilastirmasinin engellenmesi burada.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from calib.io import Kabin, kalibrasyondan, yaz
from calib.multiview import Kalibrasyon
from uncertainty.drift import egri_olustur, _olcumler


def _kabin(tarih: str, fx=900.0, fy=900.0, taban=2.0, n_kamera=4,
           surum="", notlar="") -> Kabin:
    """Kucuk ama gecerli bir kalibrasyon kaydi uretir."""
    def kamera_i(i):
        return np.array([[fx + 0.01 * i, 0.0, 640.0],
                         [0.0, fy + 0.01 * i, 360.0],
                         [0.0, 0.0, 1.0]])

    return kalibrasyondan(
        Kalibrasyon(
            rms_px=0.3,
            Ks=[kamera_i(i) for i in range(n_kamera)],
            bozulmalar=[np.zeros(5) for _ in range(n_kamera)],
            Rs=[np.eye(3)] + [np.eye(3) for _ in range(n_kamera - 1)],
            Ts=[np.zeros((3, 1))] + [np.array([[taban * i], [0.0], [0.0]])
                                     for i in range(1, n_kamera)],
        ),
        n_kare=10, gorunurluk=0.9, surum=surum, notlar=notlar, tarih_iso=tarih)


def _haftalik(n: int, fx_surunme: float = 0.0) -> list[Kabin]:
    """n haftalik kayit; fx her hafta fx_surunme yüzde kayar."""
    simdiki = datetime(2026, 9, 22, tzinfo=timezone.utc)
    kayitlar = []
    for w in range(n):
        t = (simdiki + timedelta(weeks=w)).isoformat()
        kayitlar.append(_kabin(t, fx=900.0 * (1 + fx_surunme * w / 100.0)))
    return kayitlar


# ---------------------------------------------------------------------------
# Temel seri uretimi
# ---------------------------------------------------------------------------

def test_kayitlar_zaman_siraya_dizilir():
    """Cagiran sirasiz verir; egri tarihe gore dizilir."""
    kayitlar = _haftalik(4)
    ters = list(reversed(kayitlar))
    e = egri_olustur(ters)
    assert e.tarihler == tuple(k.meta.tarih_iso for k in kayitlar)
    assert e.n_nokta == 4


def test_olcumler_kamera0_intrinsics_ve_tabanlar():
    k = _kabin("2026-09-22T00:00:00+00:00", fx=905.0, taban=2.5)
    m = _olcumler(k)
    assert m["fx"] == pytest.approx(905.0)
    assert m["fy"] == pytest.approx(900.0)
    assert m["cx"] == pytest.approx(640.0)
    assert m["taban_01"] == pytest.approx(2.5)
    assert m["taban_03"] == pytest.approx(7.5)
    assert "taban_00" not in m, "kamera 0 referans; T=0 bilgi tasimaz"


def test_fx_serisi_kaymayi_tasiyor():
    e = egri_olustur(_haftalik(5, fx_surunme=0.2))
    seri = e.seri("fx")
    beklenen = [900.0 * (1 + 0.2 * w / 100.0) for w in range(5)]
    np.testing.assert_allclose(seri, beklenen, rtol=1e-12)
    sapma = e.sapma_yuzdesi("fx")
    np.testing.assert_allclose(sapma, [0.2 * w for w in range(5)], rtol=1e-12)


def test_baz_noktasi_secilebilir():
    """Surunme varken baz secimi sonucu degistirir: baz haftada sapma sifir,
    oncesi negatif, sonrasi pozitif. (Surunmesiz seride bu test bos gecerdi.)"""
    e = egri_olustur(_haftalik(4, fx_surunme=0.2))
    seri = e.seri("fx")
    sapma = e.sapma_yuzdesi("fx", baz_indeks=2)
    np.testing.assert_allclose(sapma, (seri - seri[2]) / abs(seri[2]) * 100, rtol=1e-12)
    assert sapma[2] == 0.0 and sapma[0] < 0.0 < sapma[3]


def test_taban_uzunlugu_taban_uzunluktan_gelir():
    """taban_0i, Ts[i]'nin normundan gelir (dunya/otel degil -- kamera 0 referans)."""
    k = _kabin("2026-09-22T00:00:00+00:00", taban=1.7)
    m = _olcumler(k)
    assert m["taban_02"] == pytest.approx(3.4)


# ---------------------------------------------------------------------------
# Hata yollari -- sessiz karsilastirma engelleri
# ---------------------------------------------------------------------------

def test_bos_liste_reddedilir():
    with pytest.raises(ValueError, match="en az bir"):
        egri_olustur([])


def test_duzenek_degisti_reddedilir():
    kayitlar = _haftalik(3)
    kayitlar.append(_kabin("2026-10-20T00:00:00+00:00", n_kamera=3))
    with pytest.raises(ValueError, match="karsilastirilamaz"):
        egri_olustur(kayitlar)


def test_ikinci_kayitta_sifir_baz_degil_seri_uretilir():
    """cx yalnizca ikinci kayitta 0 ise baz (ilk) hala gecerli; seri uretilir.
    Yuzde sapma baz noktaya gore tanimli, dolayisiyla bu gecerli bir seri."""
    kayitlar = _haftalik(2)
    kayitlar[1].Ks[0][0, 2] = 0.0               # ikinci kayitta cx = 0 (baz degil)
    e = egri_olustur(kayitlar)
    assert e.n_nokta == 2
    np.testing.assert_allclose(e.seri("cx"), [640.0, 0.0])


def test_sifir_baz_parametrede_sapma_tanimsiz():
    """Baz (ilk) kayitta cx = 0: yuzde tanimsiz, acik hata."""
    kayitlar = _haftalik(2)
    kayitlar[0].Ks[0][0, 2] = 0.0               # baz kayitta cx = 0
    e = egri_olustur(kayitlar)
    with pytest.raises(ValueError, match="sifira yakin"):
        e.sapma_yuzdesi("cx")


def test_bilinmeyen_parametre_anahtari():
    e = egri_olustur(_haftalik(2))
    with pytest.raises(KeyError):
        e.seri("olmayan")


# ---------------------------------------------------------------------------
# Kapi 2 olcutu ve ozet
# ---------------------------------------------------------------------------

def test_dort_haftalik_kayit_kapi_2_olcutunu_saglar():
    e = egri_olustur(_haftalik(4))
    assert e.yeterli_kayit() and e.ozet()["kapi_2_4_hafta"]


def test_uc_haftalik_kayit_yetersiz():
    """3 oturum 14 gun kapsar; olcutu saglamaz."""
    e = egri_olustur(_haftalik(3))
    assert not e.yeterli_kayit()


def test_dort_nokta_ama_dar_aralik_yetersiz():
    """4 nokta 10 gun icinde sikistirilmis: haftalik ritim yok, gecmez."""
    from datetime import datetime, timedelta, timezone as tz
    t0 = datetime(2026, 9, 22, tzinfo=tz.utc)
    kayitlar = [_kabin((t0 + timedelta(days=3 * i)).isoformat()) for i in range(4)]
    assert not egri_olustur(kayitlar).yeterli_kayit()


def test_tarih_araligi_gun():
    e = egri_olustur(_haftalik(5))
    assert e.tarih_araligi_gun() == pytest.approx(28.0)


def test_en_buyuk_suruklenme_fx_kaymasini_bulur():
    e = egri_olustur(_haftalik(5, fx_surunme=0.5))
    p, v = e.en_buyuk_suruklenme()
    assert p == "fx" and v == pytest.approx(2.0)


def test_en_buyuk_suruklenme_hepsi_sabitse_sifir():
    e = egri_olustur(_haftalik(3))
    p, v = e.en_buyuk_suruklenme()
    assert v == pytest.approx(0.0)


def test_ozet_json_uyumlu():
    import json
    e = egri_olustur(_haftalik(5, fx_surunme=0.1))
    s = e.ozet()
    json.dumps(s, allow_nan=False)              # NaN/Inf tasimamali
    assert s["n_nokta"] == 5
    assert s["parametreler"]["fx"]["sapma_yuzde"] == pytest.approx(0.4)
    assert s["parametreler"]["cx"]["sapma_yuzde"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Disk turu: kayitlari yaz, geri oku, egriyi uret
# ---------------------------------------------------------------------------

def test_disk_uzerinden_tur(tmp_path):
    """Gercek akis: yaz -> oku -> egri. Tarihler ve degerler korunur."""
    kayitlar = _haftalik(4, fx_surunme=0.2)
    yollar = []
    for i, k in enumerate(kayitlar):
        yollar.append(yaz(k, tmp_path / f"kal_{i}.json"))
    geri = [oku_yolu(p) for p in yollar]
    e = egri_olustur(geri)
    assert e.n_nokta == 4
    np.testing.assert_allclose(e.seri("fx"), [900.0 * (1 + 0.2 * w / 100.0)
                                              for w in range(4)], rtol=1e-12)


def oku_yolu(p):
    from calib.io import oku
    return oku(p)
