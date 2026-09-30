"""uncertainty.haftalik kabul testleri.

Kapi 2'nin ">= 4 haftalik veri" satiri burada sinaniyor. En onemli test
`test_ayni_gun_dort_kayit_gecmez`: sayarak bakan bir kontrol o vakayi gecirir
ve kapi yanlis yerde acilir. Suruklenme zamanla olusur, kayit sayisiyla degil.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from calib.io import kalibrasyondan, yaz
from calib.multiview import Kalibrasyon
from uncertainty.haftalik import (
    haftalik_grupla, kapi2_degerlendir, kapi2_kontrol, kayitlari_yukle,
)

BASLANGIC = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)   # pazartesi


def _kabin(tarih: str, n_kamera: int = 4):
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    return kalibrasyondan(
        Kalibrasyon(
            rms_px=0.3,
            Ks=[K.copy() for _ in range(n_kamera)],
            bozulmalar=[np.zeros(5) for _ in range(n_kamera)],
            Rs=[np.eye(3) for _ in range(n_kamera)],
            Ts=[np.zeros((3, 1))] + [np.array([[2.0 * i], [0.0], [0.0]])
                                     for i in range(1, n_kamera)],
        ),
        n_kare=10, gorunurluk=0.9, tarih_iso=tarih)


def _gunlerde(*gunler: float):
    """Baslangictan verilen gun ofsetlerinde kayitlar."""
    return [_kabin((BASLANGIC + timedelta(days=g)).isoformat()) for g in gunler]


# --- olcutun kalbi -----------------------------------------------------------

def test_ayni_gun_dort_kayit_gecmez():
    """Dort kayit dort hafta degildir.

    Ayni gun alinmis dort kalibrasyon sayarak bakan bir kontrolden gecer ama
    suruklenme hakkinda hicbir sey soylemez. Bu testin gecmesi, Kapi 2'nin
    sayiyla degil zamanla sinandiginin kanitidir.
    """
    durum = kapi2_degerlendir(haftalik_grupla(_gunlerde(0, 0.2, 0.4, 0.6)))
    assert not durum.gecti
    assert durum.seri.n_kayit == 4
    assert durum.seri.n_hafta == 1
    assert any("hafta sayisi" in e for e in durum.eksikler)


def test_haftada_bir_dort_kayit_gecer():
    durum = kapi2_degerlendir(haftalik_grupla(_gunlerde(0, 7, 14, 21, 28)))
    assert durum.gecti
    assert durum.eksikler == ()
    assert durum.seri.n_hafta == 5


def test_kapsam_kisa_ise_gecmez():
    """Dort farkli takvim haftasi 22 gune sigabilir; kapsam olcutu bunu yakalar.

    Pazar gunu baslayip ertesi pazartesi devam eden bir seri, hafta sayarak
    bakildiginda dolu gorunur ama fiilen uc haftadan kisadir.
    """
    pazar = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
    kabinler = [_kabin((pazar + timedelta(days=g)).isoformat())
                for g in (0, 1, 8, 15, 22)]
    durum = kapi2_degerlendir(haftalik_grupla(kabinler))
    assert durum.seri.n_hafta >= 4
    assert not durum.gecti
    assert any("kapsam" in e for e in durum.eksikler)


def test_buyuk_bosluk_gecmez():
    """Dort hafta ve yeterli kapsam, ama arada 28 gun bosluk var."""
    durum = kapi2_degerlendir(haftalik_grupla(_gunlerde(0, 7, 14, 42)))
    assert durum.seri.n_hafta == 4
    assert durum.seri.kapsam_gun >= 28.0
    assert not durum.gecti
    assert any("bosluk" in e for e in durum.eksikler)


def test_eksikler_ayri_ayri_raporlanir():
    """Tek bir 'gecmedi' yetmez; hangi kosulun dustugu ayri yazmali."""
    durum = kapi2_degerlendir(haftalik_grupla(_gunlerde(0, 0.1)))
    assert len(durum.eksikler) >= 2


def test_olcut_disaridan_degistirilebilir():
    seri = haftalik_grupla(_gunlerde(0, 7))
    assert not kapi2_degerlendir(seri).gecti
    gevsek = kapi2_degerlendir(seri, en_az_hafta=2, en_az_kapsam_gun=7.0,
                               en_fazla_bosluk_gun=10.0)
    assert gevsek.gecti


# --- seri ciktisi ------------------------------------------------------------

def test_bos_haftalar_raporlanir():
    """Ilk ile son arasindaki kayitsiz takvim haftalari gorunur olmali."""
    seri = haftalik_grupla(_gunlerde(0, 7, 28))
    assert seri.n_hafta == 3
    assert len(seri.bos_haftalar) == 2


def test_ayni_haftadaki_kayitlar_tek_haftada_toplanir():
    seri = haftalik_grupla(_gunlerde(0, 1, 2, 7))
    assert seri.n_kayit == 4
    assert seri.n_hafta == 2
    assert len(seri.haftalar[0].kayitlar) == 3


def test_yil_siniri_dogru_gruplanir():
    """ISO hafta numarasi yil basinda sarmalanir; seri kirilmamali."""
    aralik = datetime(2026, 12, 28, 9, 0, tzinfo=timezone.utc)
    kabinler = [_kabin((aralik + timedelta(days=g)).isoformat())
                for g in (0, 7, 14, 21, 28)]
    seri = haftalik_grupla(kabinler)
    assert seri.n_hafta == 5
    assert len({h.yil for h in seri.haftalar}) == 2       # iki takvim yili
    assert seri.bos_haftalar == ()


def test_karisik_saat_dilimi_siralamayi_bozmaz():
    """Gercek klasorde 'Z', '+03:00' ve dilimsiz tarih yan yana bulunur."""
    kabinler = [
        _kabin("2026-10-05T09:00:00Z"),
        _kabin("2026-10-12T12:00:00+03:00"),
        _kabin("2026-10-19T09:00:00"),          # dilimsiz -- UTC kabul edilir
        _kabin("2026-10-26T09:00:00Z"),
        _kabin("2026-11-02T09:00:00Z"),
    ]
    seri = haftalik_grupla(kabinler)
    assert seri.n_hafta == 5
    assert seri.bos_haftalar == ()
    assert kapi2_degerlendir(seri).gecti


def test_kayit_yoksa_hata():
    with pytest.raises(ValueError):
        haftalik_grupla([])


# --- klasorden yukleme -------------------------------------------------------

def test_klasorden_yuklenir_ve_siralanir(tmp_path):
    """Kayitlar sirasiz yazilsa da tarihe gore dizilmeli."""
    for g in (14, 0, 7):
        yaz(_kabin((BASLANGIC + timedelta(days=g)).isoformat()),
            tmp_path / f"kayit_{g:02d}.json")
    kabinler = kayitlari_yukle(tmp_path)
    assert len(kabinler) == 3
    tarihler = [k.meta.tarih_iso for k in kabinler]
    assert tarihler == sorted(tarihler)


def test_uctan_uca_kapi2_kontrolu(tmp_path):
    for g in (0, 7, 14, 21, 28):
        yaz(_kabin((BASLANGIC + timedelta(days=g)).isoformat()),
            tmp_path / f"kayit_{g:02d}.json")
    durum = kapi2_kontrol(tmp_path)
    assert durum.gecti
    assert durum.seri.n_hafta == 5


def test_bozuk_dosya_sessizce_atlanmaz(tmp_path):
    """Okunamayan kayit, adiyla birlikte hata vermeli.

    Sessizce atlanirsa Kapi 2 eksik veriyle gecmis olur -- projenin en kritik
    kapisi yanlis yerde acilir.
    """
    yaz(_kabin(BASLANGIC.isoformat()), tmp_path / "iyi.json")
    (tmp_path / "bozuk.json").write_text("{ bu json degil", encoding="utf-8")
    with pytest.raises(ValueError, match="bozuk.json"):
        kayitlari_yukle(tmp_path)


def test_olmayan_klasor_hata(tmp_path):
    with pytest.raises(ValueError, match="kayit klasoru yok"):
        kayitlari_yukle(tmp_path / "yok")


def test_bos_klasor_hata(tmp_path):
    with pytest.raises(ValueError, match="eslesen kayit yok"):
        kayitlari_yukle(tmp_path)


# --- rapor -------------------------------------------------------------------

def test_ozet_json_yazilabilir():
    durum = kapi2_degerlendir(haftalik_grupla(_gunlerde(0, 7, 14, 42)))
    geri = json.loads(json.dumps(durum.ozet(), ensure_ascii=False, allow_nan=False))
    assert geri["gecti"] is False
    assert geri["olcut"]["en_az_hafta"] == 4
    assert len(geri["seri"]["haftalar"]) == 4
