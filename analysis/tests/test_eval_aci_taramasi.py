"""eval.aci_taramasi kabul testleri.

Harness'in isi bir egri uretmek degil, **dogru egriyi** uretmek. Sinanan sey
su: telefon kararinin yer gercegi karariyla karsilastirilmasi acidan bagimsiz
olarak dogru sayiliyor mu, gozlemsiz aci sessizce basari sayiliyor mu, ve
belirsizlik bilgisi verildiginde katman yanlis karar yerine susuyor mu.
"""
import json

import numpy as np
import pytest

from eval.aci_taramasi import aci_taramasi, kemik_onculeri, sentetik_poz_ureteci
from eval.form import sentetik_durus

ACILAR = (0.0, 30.0, 60.0, 90.0)
VALGUS = "diz_valgusu_sag"


def _duruslar():
    """Esigin iki yanina dagilmis duruslar: bir kismi kusurlu, bir kismi degil."""
    return [sentetik_durus(valgus_sag=float(v)) for v in range(0, 22, 2)]


def _sayim(sonuc, aci, ad=VALGUS):
    return next(n for n in sonuc.noktalar if n.aci_derece == aci).sayimlar[ad]


def _hata(sonuc, aci, ad=VALGUS):
    return next(n for n in sonuc.noktalar if n.aci_derece == aci).aci_hatasi_derece[ad]


# --- egrinin sekli -----------------------------------------------------------

def test_onden_bakista_karar_dogru():
    """Frontal acida, gurultusuz, tek gorus kestirimi kusuru bulmali.

    Bu testin gecmemesi harness'in degil hattin bozuk oldugunu soyler: onden
    bakarken valgus goruntu duzleminde durur, derinlik belirsizligi karari
    etkilemez.
    """
    uret, K = sentetik_poz_ureteci(gurultu_px=0.0)
    sonuc = aci_taramasi(_duruslar(), [0.0], uret, K)
    assert _sayim(sonuc, 0.0).dogruluk == 1.0


def test_dogruluk_yandan_bakista_dusuyor():
    """Tezin manset iddiasinin kod karsiligi: aci arttikca karar bozulur."""
    uret, K = sentetik_poz_ureteci(gurultu_px=0.0)
    sonuc = aci_taramasi(_duruslar(), ACILAR, uret, K)
    assert _sayim(sonuc, 0.0).dogruluk > _sayim(sonuc, 90.0).dogruluk


def test_aci_hatasi_yandan_bakista_buyuyor():
    """Karar bozulmadan once olculen aci bozulur; ikisi ayri ayri izlenmeli."""
    uret, K = sentetik_poz_ureteci(gurultu_px=0.0)
    sonuc = aci_taramasi(_duruslar(), ACILAR, uret, K)
    assert _hata(sonuc, 90.0) > 10.0 * _hata(sonuc, 0.0)


def test_sayimlar_toplami_durus_sayisina_esit():
    uret, K = sentetik_poz_ureteci()
    duruslar = _duruslar()
    sonuc = aci_taramasi(duruslar, ACILAR, uret, K)
    for n in sonuc.noktalar:
        for sayim in n.sayimlar.values():
            assert sayim.toplam == len(duruslar)


# --- sozlesme ----------------------------------------------------------------

def test_poz_kaynagi_disaridan_verilir():
    """Gercek model takilinca degisecek tek sey bu uretec olmali.

    Kabul olcutu "model takilinca tek komutla egri" diyor; sozlesme buysa
    sentetik izdusumden gercek kayda gecis kod degisikligi gerektirmez.
    """
    cagrilar = []
    gercek_uret, K = sentetik_poz_ureteci(gurultu_px=0.0)

    def sahte_uret(aci, iskelet):
        cagrilar.append(aci)
        return gercek_uret(aci, iskelet)

    duruslar = _duruslar()
    aci_taramasi(duruslar, ACILAR, sahte_uret, K)
    assert len(cagrilar) == len(ACILAR) * len(duruslar)
    assert set(cagrilar) == set(ACILAR)


def test_gozlemsiz_aci_basari_sayilmaz():
    """Gozlem yoksa dogruluk yukselmemeli -- eksik veri sessizce iyi gorunmesin."""
    gercek_uret, K = sentetik_poz_ureteci(gurultu_px=0.0)

    def bos(aci, iskelet):
        return None if aci == 90.0 else gercek_uret(aci, iskelet)

    sonuc = aci_taramasi(_duruslar(), ACILAR, bos, K)
    sayim = _sayim(sonuc, 90.0)
    assert sayim.gozlemsiz == len(_duruslar())
    assert sayim.dogruluk == 0.0
    assert np.isnan(_hata(sonuc, 90.0))


def test_durus_verilmezse_hata():
    uret, K = sentetik_poz_ureteci()
    with pytest.raises(ValueError):
        aci_taramasi([], ACILAR, uret, K)


def test_kemik_onculeri_yer_gercegi_uzunluklarini_verir():
    durus = sentetik_durus(valgus_sag=8.0)
    onculer = kemik_onculeri(durus)
    assert onculer
    for (a, b), uzunluk in onculer.items():
        assert uzunluk == pytest.approx(durus.uzunluk(a, b), abs=1e-12)


def test_tekrarlanabilir():
    """Ayni tohum ayni egriyi vermeli; deney kaydi buna dayaniyor."""
    a, _ = sentetik_poz_ureteci(gurultu_px=2.0, seed=5)
    b, K = sentetik_poz_ureteci(gurultu_px=2.0, seed=5)
    duruslar = _duruslar()
    assert (aci_taramasi(duruslar, ACILAR, a, K).ozet()
            == aci_taramasi(duruslar, ACILAR, b, K).ozet())


# --- belirsizligin degeri ----------------------------------------------------

def test_belirsizlik_bilinince_yanlis_karar_yerine_susuluyor():
    """Projenin iddiasi: belirsizligini bilen katman yanlis karar vermez.

    Ayni aci taramasi iki kez kosuluyor. Birincisinde telefon kendi kestirim
    hatasini bilmiyor ve esigi yalin karsilastiriyor -- telefon uygulamalarinin
    yaptigi sey. Ikincisinde ayni hata belirsizlik olarak veriliyor. Beklenen:
    yanlis karar sayisi dusuyor, yerini BELIRSIZ aliyor.
    """
    duruslar = _duruslar()
    uret, K = sentetik_poz_ureteci(gurultu_px=0.0)
    bilmeyen = aci_taramasi(duruslar, [90.0], uret, K)

    uret2, K2 = sentetik_poz_ureteci(gurultu_px=0.0)
    bilen = aci_taramasi(duruslar, [90.0], uret2, K2, konum_belirsizligi_m=0.05)

    assert _sayim(bilen, 90.0).yanlis < _sayim(bilmeyen, 90.0).yanlis
    assert _sayim(bilen, 90.0).belirsiz > _sayim(bilmeyen, 90.0).belirsiz


# --- rapor -------------------------------------------------------------------

def test_ozet_json_yazilabilir():
    uret, K = sentetik_poz_ureteci(gurultu_px=1.0)
    sonuc = aci_taramasi(_duruslar(), ACILAR, uret, K)
    geri = json.loads(json.dumps(sonuc.ozet(), ensure_ascii=False, allow_nan=False))
    assert len(geri["noktalar"]) == len(ACILAR)
    assert VALGUS in geri["noktalar"][0]["olcumler"]


def test_egri_cizim_icin_hazir():
    uret, K = sentetik_poz_ureteci()
    sonuc = aci_taramasi(_duruslar(), ACILAR, uret, K)
    x, y = sonuc.egri(VALGUS)
    assert x == list(ACILAR)
    assert len(y) == len(ACILAR)
