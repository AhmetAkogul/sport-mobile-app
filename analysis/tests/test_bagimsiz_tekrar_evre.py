"""Bagimsiz dogrulama — tekrar/evre katmani ve egzersiz karar kapisi.

`eval/tekrar.SquatTakip` ve `eval/egzersiz` sozlesmeleri (misyon 2. maddenin
eval yarisi). Kaynak: `docs/kararlar/0072` §4 ve `eval/egzersiz.SQUAT`.

Bu dosya **uretim kodunu degistirmez**; bulgu yalniz basarisiz testle gosterilir.
`hypothesis` ortamda kurulu degil, seriler determinist ve elle kurulur.

Sozlesmeler:
  - Sentetik squat serisinde tekrar sayisi dogru cikar.
  - Tek karelik kusur sicramasi kusur sayilmaz (ardisik >= 0,2 s gerekir).
  - Yarim tekrar ve eksik kareler dogru isaretlenir (doldurulmaz).
  - Sinirlar: bos seri, tek kare, hepsi eksik, tum tekrarlar yarim.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eval.egzersiz import SQUAT, diz_fleksiyonu, kare_degerlendir
from eval.form import sentetik_durus
from eval.tekrar import SquatTakip
from pose3d.iskelet import REFERANS_ISKELET

IDX = REFERANS_ISKELET.indeks
FPS = 30.0

# Ayakta baslangic kareleri (evre "ayakta" olmadan tekrar baslamaz).
_AYAKTA = 5

# Gercekci bir tekrar: inis -> dip -> cikis -> bitis. Bitis karesi f<=15.
_REP = [30.0, 45.0, 60.0, 70.0, 75.0, 75.0, 70.0, 55.0, 40.0, 25.0, 10.0, 0.0, 0.0]
# _REP icinde ilk inis karesi ve son bitis karesinin mutlak dizinleri.
_REP_BASI = _AYAKTA
_REP_SONU = _AYAKTA + len(_REP) - 1


def _kare(fleksiyon, karar="dogru"):
    return {
        "profil": SQUAT.surum,
        "olcumler": {ad: {"karar": karar} for ad in SQUAT.kapsam},
        "fleksiyon_derece": fleksiyon,
    }


def _kostur(dizi, fps=FPS):
    """(fleksiyon, karar) dizisini SquatTakip'e verir; kapanan tekrarlari doner."""
    takip = SquatTakip()
    dt = 1.0 / fps
    for i, (f, karar) in enumerate(dizi):
        takip.ekle(i * dt, _kare(f, karar))
    return takip.bitir()


def _seri(tekrar=1, ek_karar=None):
    """`tekrar` adet tam tekrardan olusan seri; `ek_karar` dizini->karar."""
    dizi = [(0.0, "dogru")] * _AYAKTA
    for _ in range(tekrar):
        dizi += [(f, "dogru") for f in _REP]
    if ek_karar:
        dizi = [(f, ek_karar.get(i, k)) for i, (f, k) in enumerate(dizi)]
    return dizi


def _kusurlu_seri(ilk, son, tekrar=1):
    """Tekrar icinde [ilk, son) dizinlerini kusurlu yapar (mutlak dizin)."""
    return _seri(tekrar, {i: "kusurlu" for i in range(ilk, son)})


# --- (1) tekrar sayisi --------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_bagimsiz_tam_tekrar_sayisi_dogru(n):
    sonuclar = _kostur(_seri(n))
    assert [r["tekrar"] for r in sonuclar] == list(range(1, n + 1))
    assert all(r["tamamlandi"] and r["neden"] == "tamamlandi" for r in sonuclar)
    assert all(r["karar"] == "dogru" for r in sonuclar)
    assert all(min(r["kapsama"].values()) == pytest.approx(1.0) for r in sonuclar)


def test_bagimsiz_evre_dongusu_bir_tekrarda_bir_kez_doner():
    """Ayni tekrar iki kez sayilmamali: sonuc listesi uzunlugu tam tekrar sayisi."""
    assert len(_kostur(_seri(4))) == 4


def test_bagimsiz_tekrar_tam_olmayan_seri_sayim_yapmaz():
    """Duz ayakta duran seri tekrar uretmemeli."""
    assert _kostur([(0.0, "dogru")] * 30) == []


# --- (2) tek karelik sicrama --------------------------------------------------

def test_bagimsiz_tek_kare_kusur_sayilmaz():
    """Tek karelik kusur sicramasi kusur sayilmamali (0072 §4: >= 0,2 s)."""
    r = _kostur(_kusurlu_seri(6, 7))[0]
    assert r["tamamlandi"] and r["karar"] == "dogru"
    assert r["kusurlar"] == []
    assert all(v == pytest.approx(0.0) for v in r["kusur_suresi_s"].values())


@pytest.mark.parametrize("uzunluk", [1, 3, 5, 6])
def test_bagimsiz_kisa_kusur_serisi_kusur_sayilmaz(uzunluk):
    """0,2 s altinda kalan kesin kusur serisi kusur olmamali."""
    r = _kostur(_kusurlu_seri(6, 6 + uzunluk))[0]
    assert r["kusurlar"] == [], (
        f"{uzunluk} kare = {(uzunluk - 1) / FPS:.4f} s; beklenen kusur yok")


def test_bagimsiz_surekli_kusur_kusur_sayilir():
    """0,2 s'yi asan kesin ve ardisik kusur gizlenmemeli (0072 §4)."""
    r = _kostur(_kusurlu_seri(6, 15))[0]
    assert set(r["kusurlar"]) == set(SQUAT.kapsam)
    assert r["karar"] == "kusurlu"
    assert max(r["kusur_suresi_s"].values()) >= SQUAT.kusur_suresi_s


def test_bagimsiz_iki_kesinlik_kesilince_seri_sifirlanir():
    """Araya kesin olmayan kare girerse tekrar eden kosu bolunur (0072 §4)."""
    dizi = _kusurlu_seri(6, 15)
    dizi[10] = (dizi[10][0], "belirsiz")          # iki kusur kosusunu boler
    r = _kostur(dizi)[0]
    assert r["kusurlar"] == []
    assert max(r["kusur_suresi_s"].values()) < SQUAT.kusur_suresi_s


# --- (3) yarim tekrar ve eksik kareler ---------------------------------------

def test_bagimsiz_yarim_tekrar_kayit_sonu():
    """Kayit ortasinda biten tekrar eksik isaretlenmeli, kaybolmamali."""
    dizi = [(0.0, "dogru")] * _AYAKTA + [(30.0, "dogru"), (50.0, "dogru"), (70.0, "dogru")]
    r = _kostur(dizi)
    assert len(r) == 1
    assert r[0]["tamamlandi"] is False and r[0]["neden"] == "kayit_sonu"
    assert r[0]["karar"] == "belirsiz"


def test_bagimsiz_yarim_tekrar_dip_gozlenmedi():
    """Dibe inmeden geri cikilan tekrar tam sayilmamali."""
    dizi = [(0.0, "dogru")] * _AYAKTA + [(30.0, "dogru"), (40.0, "dogru"), (20.0, "dogru"),
                                         (10.0, "dogru")]
    r = _kostur(dizi)
    assert len(r) == 1
    assert r[0]["tamamlandi"] is False
    assert r[0]["neden"] == "dip_evresi_gozlenmedi"


def test_bagimsiz_tum_tekrarlar_yarim_isaretlenir():
    """Hicbiri dibi gormeyen tekrarlar: sayi dogru, hepsi yarim."""
    yarim = [(0.0, "dogru")] * _AYAKTA + [(30.0, "dogru"), (40.0, "dogru"),
                                          (20.0, "dogru"), (10.0, "dogru")] * 3
    r = _kostur(yarim)
    assert [x["tekrar"] for x in r] == [1, 2, 3]
    assert all(x["tamamlandi"] is False for x in r)
    assert all(x["neden"] == "dip_evresi_gozlenmedi" for x in r)
    assert all(x["karar"] == "belirsiz" for x in r)


def test_bagimsiz_eksik_kare_doldurulmaz_ve_tekrar_sayisini_bozmaz():
    """f=None kareleri ne enterpole edilir ne de tekrar sayisini degistirir."""
    temiz = _seri(2)
    seyrek = []
    for i, (f, k) in enumerate(temiz):
        seyrek.append((None, k) if i in (7, 12, 25) else (f, k))
    assert len(_kostur(seyrek)) == len(_kostur(temiz)) == 2


def test_bagimsiz_eksik_kare_kapsamayi_dusurur_ve_doldurulmaz():
    """Eksik kare hem kendi araligini hem sonrakini kapsam disi birakir.

    Tek eksik kare 10 aralikli tekrarda kapsamayi tam olarak %80'e indirir;
    esik `>=` oldugu icin karar hala olumlu kalabilir.
    """
    dizi = _seri(1)
    dizi[10] = (None, "dogru")
    r = _kostur(dizi)[0]
    assert r["tamamlandi"] is True
    assert min(r["kapsama"].values()) == pytest.approx(SQUAT.min_kapsama)
    assert r["karar"] == "dogru"
    # Ayni seri eksik kare olmadan tam kapsama verir (kare enterpole edilmiyor).
    assert min(_kostur(_seri(1))[0]["kapsama"].values()) == pytest.approx(1.0)


def test_bagimsiz_yeterince_eksik_kare_karari_belirsiz_yapar():
    """Kapsama %80'in altina dusunce olumlu karar verilmemeli (0072 §4)."""
    dizi = _seri(1)
    dizi[8] = (None, "dogru")
    dizi[10] = (None, "dogru")
    r = _kostur(dizi)[0]
    assert r["tamamlandi"] is True
    assert min(r["kapsama"].values()) < SQUAT.min_kapsama
    assert r["karar"] == "belirsiz"


def test_bagimsiz_zaman_boslugu_tekrari_keser():
    """0,25 s'den buyuk bosluk tekrari keser ve eksik isaretler (0072 §4)."""
    takip = SquatTakip()
    dt = 1.0 / FPS
    dizi = [(0.0, "dogru")] * _AYAKTA + [(30.0, "dogru"), (50.0, "dogru"), (70.0, "dogru")]
    for i, (f, k) in enumerate(dizi):
        takip.ekle(i * dt, _kare(f, k))
    takip.ekle(3.0, _kare(70.0))          # 0,25 s'den cok buyuk bosluk
    sonuclar = takip.bitir()
    assert len(sonuclar) == 1
    assert sonuclar[0]["tamamlandi"] is False
    assert sonuclar[0]["neden"] == "zaman_boslugu"


def test_bagimsiz_bosluk_bitis_zamani_bosluk_oncesinde_kalmali():
    """Kesilen tekrarin bitisi, gorulen son kareden sonra olamaz.

    `bitis_s` bosluk sonrasi kareyi yazarsa `sure` hic kayit olmayan zamani
    kapsar; kapsama da bu sisirilmis sureye bolunur.
    """
    takip = SquatTakip()
    dt = 1.0 / FPS
    dizi = [(0.0, "dogru")] * _AYAKTA + [(30.0, "dogru"), (50.0, "dogru"), (70.0, "dogru")]
    son_gozlenen = (len(dizi) - 1) * dt
    for i, (f, k) in enumerate(dizi):
        takip.ekle(i * dt, _kare(f, k))
    takip.ekle(3.0, _kare(70.0))
    r = takip.bitir()[0]
    assert r["bitis_s"] <= son_gozlenen + 1e-9, (
        f"bitis_s={r['bitis_s']} son gozlenen kare={son_gozlenen}")


# --- (4) sinir durumlari ------------------------------------------------------

def test_bagimsiz_bos_seri_tekrar_uretmez():
    assert _kostur([]) == []
    assert SquatTakip().bitir() == []


@pytest.mark.parametrize("f", [0.0, 30.0, 70.0, None])
def test_bagimsiz_tek_kare_tekrar_uretmez(f):
    """Ayakta baslangic gozlenmeden tekrar baslatilmaz (0072 §4)."""
    assert _kostur([(f, "dogru")]) == []


def test_bagimsiz_hepsi_eksik_kare_bos_sonuc():
    assert _kostur([(None, "dogru")] * 20) == []
    assert _kostur([(None, "kusurlu")] * 20) == []


def test_bagimsiz_yuksekten_baslayan_seri_baslangic_bekler():
    """Ilk karelerde fleksiyon yuksekse ayakta gorulene kadar tekrar yok."""
    dizi = [(70.0, "dogru")] * 5 + [(0.0, "dogru")] * _AYAKTA + [(f, "dogru") for f in _REP]
    assert len(_kostur(dizi)) == 1


def test_bagimsiz_tek_kare_dip_sicramasi_tekrar_uretmemeli():
    """Hic squat olmayan seride tek karelik fleksiyon sicramasi tekrar uretmemeli.

    Kusur icin ardisik 0,2 s kesinlik isteniyor (0072 §4), ama ayni sureklilik
    kurali tekrar bolutlemesinde yok: 33 ms'lik bir artik tek basina esigi asip
    donunce tam ve 'dogru' bir tekrar sayiliyor.
    """
    duz = [(0.0, "dogru")] * 30
    sicramali = duz[:]
    sicramali[10] = (70.0, "dogru")
    r = _kostur(sicramali)
    assert r == [], (
        f"hic squat olmayan seride tek karelik sicrama {len(r)} tekrar uretti: "
        f"{[x['karar'] for x in r]}")


def test_bagimsiz_bitir_idempotent():
    takip = SquatTakip()
    for i, (f, k) in enumerate(_seri(2)):
        takip.ekle(i / FPS, _kare(f, k))
    assert len(takip.bitir()) == len(takip.bitir()) == 2


# --- (5) girdi sozlesmesi -----------------------------------------------------

def test_bagimsiz_zaman_geri_giderse_hata():
    takip = SquatTakip()
    takip.ekle(1.0, _kare(0.0))
    with pytest.raises(ValueError, match="kesin artmali"):
        takip.ekle(0.5, _kare(0.0))


@pytest.mark.parametrize("bozuk", [
    {"profil": "baska-profil", "olcumler": {}, "fleksiyon_derece": 0.0},
    {"profil": SQUAT.surum, "olcumler": {}, "fleksiyon_derece": 0.0},
    {"profil": SQUAT.surum, "olcumler": {ad: {"karar": "dogru"} for ad in SQUAT.kapsam},
     "fleksiyon_derece": 200.0},
])
def test_bagimsiz_gecersiz_kare_reddedilir(bozuk):
    takip = SquatTakip()
    with pytest.raises(ValueError):
        takip.ekle(0.0, bozuk)


def test_bagimsiz_gecersiz_karar_reddedilir():
    takip = SquatTakip()
    with pytest.raises(ValueError, match="gecersiz karar"):
        takip.ekle(0.0, _kare(0.0, "muhtemelen"))


# --- (6) egzersiz katmani ile entegrasyon ------------------------------------

def _bukuk_bacak(fleksiyon_derece, uyluk=0.425, baldir=0.425):
    """Iki bacagi verilen fleksiyonda buker; `diz_fleksiyonu` bunu geri okumali."""
    temel = sentetik_durus()
    P = temel.noktalar.copy()
    for taraf in ("sag", "sol"):
        h = P[IDX(f"{taraf}_kalca")].copy()
        k = h + uyluk * np.array([0.0, -1.0, 0.0])
        a = k + baldir * np.array([0.0, -np.cos(np.radians(fleksiyon_derece)),
                                   np.sin(np.radians(fleksiyon_derece))])
        P[IDX(f"{taraf}_diz")] = k
        P[IDX(f"{taraf}_ayak_bilegi")] = a
    return dataclasses.replace(temel, noktalar=P)


@pytest.mark.parametrize("fleksiyon", [0.0, 30.0, 60.0, 90.0, 120.0])
def test_bagimsiz_diz_fleksiyonu_bukulmeyi_geri_okur(fleksiyon):
    assert diz_fleksiyonu(_bukuk_bacak(fleksiyon)) == pytest.approx(fleksiyon, abs=1e-6)


def test_bagimsiz_diz_fleksiyonu_eksik_eklemde_none():
    assert diz_fleksiyonu(sentetik_durus(gorunmez=("sag_kalca",))) is None
    assert diz_fleksiyonu(sentetik_durus(gorunmez=("sol_ayak_bilegi",))) is None


def test_bagimsiz_kare_degerlendir_cikisi_tekrar_katmanina_girer():
    """`kare_degerlendir` ciktisi SquatTakip sozlesmesini gecmeli (uctan uca)."""
    dizi = [(0.0, "dogru")] * _AYAKTA + [(f, "dogru") for f in _REP]
    takip = SquatTakip()
    dt = 1.0 / FPS
    for i, (f, _) in enumerate(dizi):
        takip.ekle(i * dt, kare_degerlendir(_bukuk_bacak(f)))
    sonuclar = takip.bitir()
    assert len(sonuclar) == 1
    assert sonuclar[0]["tamamlandi"] is True
    # Kovaryans yok: form karari verilmez, ama evre/tekrar sayimi calisir.
    assert sonuclar[0]["karar"] == "belirsiz"


def test_bagimsiz_kare_degerlendir_semasi():
    rapor = kare_degerlendir(sentetik_durus(valgus_sag=12.0))
    assert set(rapor) == {"profil", "olcumler", "fleksiyon_derece", "evre_olcumu",
                          "belirsizlik_durumu", "belirsizlik_kaynagi", "kapsam",
                          "klinik_dogrulama"}
    assert rapor["profil"] == SQUAT.surum
    assert set(rapor["olcumler"]) == set(SQUAT.kapsam)
    assert rapor["klinik_dogrulama"] is False
    for ad, olcum in rapor["olcumler"].items():
        assert set(olcum) == {"karar", "aci_derece", "neden", "aciklama"}
        assert olcum["karar"] in ("dogru", "kusurlu", "belirsiz")
        assert isinstance(olcum["aciklama"], str) and olcum["aciklama"]
    # Kovaryanssiz iskelette karar kesinlesmez, gerekce acik yazilir.
    assert rapor["olcumler"]["diz_valgusu_sag"]["neden"] == "belirsizlik_yok"
    assert rapor["olcumler"]["diz_valgusu_sag"]["aci_derece"] == pytest.approx(12.0, abs=1e-6)
