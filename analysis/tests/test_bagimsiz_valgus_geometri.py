"""Bagimsiz dogrulama — diz valgusu geometri sozlesmeleri (misyon 2a-2c).

Bu dosya **uretim kodunu degistirmez**; yalnizca sozlesmeleri kirilmaya
zorlar. Kaynak sozlesmeler: `docs/kararlar/0014-bukulmeye-dayanikli-valgus.md`
ve `eval/form.py`.

`hypothesis` bu ortamda kurulu degil (`requirements-dev.txt` yalnizca ruff ve
pytest-cov tasiyor), bu yuzden rastgelelik **tohumlanmis numpy** taramalariyla
uretilir: her kosu ayni girdiyi uretir ve basarisizlik tekrar edilebilir.

Sinanlar:
  (a) rijit donme + oteleme ve 0,5-2 kat olcek degisiminde valgus degismez.
  (b) ayna: koordinat aynasinda isaretli medial deger korunur; sag/sol etiket
      takasinda degerler yer degistirir.
  (c) bacak duzlemi icinde saf bukulmede valgus ~0; orta hattan kacis arttikca
      monoton artar.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eval.form import REFERANS_ISKELET, form_degerlendir, sentetik_durus

IDX = REFERANS_ISKELET.indeks

OLCUMLER = ("diz_valgusu_sag", "diz_valgusu_sol", "kalca_hizasi", "govde_rotasyonu")


def _deger(rapor, ad: str) -> float:
    return rapor.olcumler[ad].deger


def _sag_elli_donme(rng: np.random.Generator) -> np.ndarray:
    """Duzgun (Haar) rastgele donme; det=+1'e zorlanir."""
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _donusum(Q: np.ndarray, t: np.ndarray, olcek: float = 1.0) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = olcek * Q
    M[:3, 3] = t
    return M


def _bukuk_bacak(prof, fleksiyon: float, kacis: float = 0.0,
                 uyluk: float = 0.425, baldir: float = 0.425):
    """Iki bacagi bacak duzlemi icinde `fleksiyon` derece buker.

    Uyluk diktir (kalcadan asagi); baldiri oneri `fleksiyon` kadar cevirdigimiz
    icin dizdeki **anatomik bukulme** tam olarak bu aci olur (duz bacak = 0,
    dik acili bacak = 90). `kacis` dizi orta hatta kaydirir; sifirken diz
    tarafsiz bacak duzleminin tam icindedir ve (c)'ye gore valgus 0 olmalidir.
    """
    P = prof.noktalar.copy()
    for taraf in ("sag", "sol"):
        medial = 1.0 if taraf == "sag" else -1.0
        h = P[IDX(f"{taraf}_kalca")].copy()
        k = h + uyluk * np.array([0.0, -1.0, 0.0]) \
            + kacis * np.array([medial, 0.0, 0.0])
        a = k + baldir * np.array([0.0,
                                   -np.cos(np.radians(fleksiyon)),
                                   np.sin(np.radians(fleksiyon))])
        P[IDX(f"{taraf}_diz")] = k
        P[IDX(f"{taraf}_ayak_bilegi")] = a
    return dataclasses.replace(prof, noktalar=P)


# --- (a) degismezlik ---------------------------------------------------------

@pytest.mark.parametrize("tur", ["oteleme", "donme", "olcek", "hepsi"])
def test_bagimsiz_valgus_degisimsiz(tur):
    """Rijit donme/oteleme ve 0,5-2 kat olcek olcumu degistirmemeli."""
    temel = sentetik_durus(valgus_sag=12.0, valgus_sol=-7.0,
                           kalca_hizasi=4.0, govde_rotasyonu=-5.0)
    duz = form_degerlendir(temel)
    rng = np.random.default_rng(20260925)
    for _ in range(25):
        Q = _sag_elli_donme(rng)
        t = np.zeros(3)
        s = 1.0
        if tur in ("oteleme", "hepsi"):
            t = rng.normal(scale=3.0, size=3)
        if tur in ("donme", "hepsi"):
            pass  # Q zaten rastgele
        if tur == "oteleme":
            Q = np.eye(3)
        if tur in ("olcek", "hepsi"):
            s = rng.uniform(0.5, 2.0)
        if tur == "olcek":
            Q = np.eye(3)
        donuk = form_degerlendir(
            dataclasses.replace(
                temel,
                noktalar=(temel.noktalar @ (s * Q).T) + t))
        for ad in OLCUMLER:
            assert _deger(donuk, ad) == pytest.approx(_deger(duz, ad), abs=1e-6), \
                f"{tur}: {ad} degisti"


def test_bagimsiz_valgus_olcek_araligi_tam_taranir():
    """0,5 ve 2,0 dahil tum olcek araliginda valgus sabit kalmali."""
    temel = sentetik_durus(valgus_sag=9.0, valgus_sol=-3.0)
    beklenen = _deger(form_degerlendir(temel), "diz_valgusu_sag")
    olcekler = np.concatenate([np.linspace(0.5, 2.0, 31), [0.5, 2.0]])
    for s in olcekler:
        olcekli = dataclasses.replace(temel, noktalar=temel.noktalar * s)
        assert _deger(form_degerlendir(olcekli), "diz_valgusu_sag") == \
            pytest.approx(beklenen, abs=1e-9), f"olcek {s}"


def test_bagimsiz_valgus_bukuk_bacakta_da_degisimsiz():
    """Degismezlik dik durusa ozel olmamali: bukuk bacakta da gecmeli."""
    temel = _bukuk_bacak(sentetik_durus(), fleksiyon=70.0, kacis=0.05)
    duz = form_degerlendir(temel)
    rng = np.random.default_rng(7)
    for _ in range(10):
        Q = _sag_elli_donme(rng)
        donuk = form_degerlendir(
            dataclasses.replace(temel,
                                noktalar=temel.noktalar @ Q.T * rng.uniform(0.5, 2.0)))
        for ad in OLCUMLER:
            assert _deger(donuk, ad) == pytest.approx(_deger(duz, ad), abs=1e-6)


# --- (b) ayna ----------------------------------------------------------------

def test_bagimsiz_ayna_koordinatta_medial_isaret_korunur():
    """Koordinat aynasinda (x -> -x) etiketli noktanin medial sapmasi korunur.

    Bu, olcumun koordinat el sistemine bagli olmamasi demektir: kamera duzeni
    el degistirse de "ice cokme" teshisi ayni kalir.
    """
    M = np.diag([-1.0, 1.0, 1.0, 1.0])
    duz = sentetik_durus(valgus_sag=15.0, valgus_sol=-4.0)
    aynali = sentetik_durus(valgus_sag=15.0, valgus_sol=-4.0, donusum=M)
    for ad in ("diz_valgusu_sag", "diz_valgusu_sol", "kalca_hizasi"):
        a = _deger(form_degerlendir(aynali), ad)
        b = _deger(form_degerlendir(duz), ad)
        # kalca_hizasi ve govde_rotasyonu aynada isaret degistirebilir; yalniz
        # valgusun medial degeri korunmalidir.
        if ad.startswith("diz_valgusu"):
            assert a == pytest.approx(b, abs=1e-6), f"{ad}: {a} != {b}"


def test_bagimsiz_ayna_etiket_takasi_sag_sol_yer_degistirir():
    """Sag/sol etiketleri takas edilince iki valgus degeri yer degistirmeli.

    Bu, bir modelin sol/sag karistirmasi durumudur ve olcumun etiketi nereye
    bagladigini gosterir.
    """
    temel = sentetik_durus(valgus_sag=15.0, valgus_sol=-4.0)
    P = temel.noktalar.copy()
    Q = P.copy()
    for ad in REFERANS_ISKELET.eklemler:
        if ad.startswith("sag_"):
            Q[IDX(ad)] = P[IDX(ad.replace("sag_", "sol_"))]
        elif ad.startswith("sol_"):
            Q[IDX(ad)] = P[IDX(ad.replace("sol_", "sag_"))]
    takas = dataclasses.replace(temel, noktalar=Q)
    o = form_degerlendir(temel)
    t = form_degerlendir(takas)
    assert _deger(t, "diz_valgusu_sag") == pytest.approx(
        _deger(o, "diz_valgusu_sol"), abs=1e-6)
    assert _deger(t, "diz_valgusu_sol") == pytest.approx(
        _deger(o, "diz_valgusu_sag"), abs=1e-6)


def test_bagimsiz_ayna_ve_etiket_takasi_birlikte_de_takas_verir():
    """Ayna + etiket takasi (anatomik ayna) yine yer degistirmeli."""
    M = np.diag([-1.0, 1.0, 1.0, 1.0])
    temel = sentetik_durus(valgus_sag=15.0, valgus_sol=-4.0)
    aynali = sentetik_durus(valgus_sag=15.0, valgus_sol=-4.0, donusum=M)
    P = aynali.noktalar.copy()
    Q = P.copy()
    for ad in REFERANS_ISKELET.eklemler:
        if ad.startswith("sag_"):
            Q[IDX(ad)] = P[IDX(ad.replace("sag_", "sol_"))]
        elif ad.startswith("sol_"):
            Q[IDX(ad)] = P[IDX(ad.replace("sol_", "sag_"))]
    anatomik = dataclasses.replace(aynali, noktalar=Q)
    o = form_degerlendir(temel)
    a = form_degerlendir(anatomik)
    assert _deger(a, "diz_valgusu_sag") == pytest.approx(
        _deger(o, "diz_valgusu_sol"), abs=1e-6)
    assert _deger(a, "diz_valgusu_sol") == pytest.approx(
        _deger(o, "diz_valgusu_sag"), abs=1e-6)


# --- (c) bukulme -------------------------------------------------------------

@pytest.mark.parametrize("fleksiyon", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0,
                                       90.0, 105.0, 120.0, 130.0])
def test_bagimsiz_saf_bukulmede_valgus_sifir(fleksiyon):
    """Bacak duzlemi icinde bukulme valgusu tetiklememeli (0014)."""
    rapor = form_degerlendir(_bukuk_bacak(sentetik_durus(), fleksiyon))
    for ad in ("diz_valgusu_sag", "diz_valgusu_sol"):
        assert abs(_deger(rapor, ad)) < 1e-9, f"flex {fleksiyon}: {ad}={_deger(rapor, ad)}"


def test_bagimsiz_bukulme_uyluk_baldir_orani_bozmaz():
    """Saf bukulme orani 1,0'da tutmali; aksi halde (d) yanlis eler."""
    for fleksiyon in (0.0, 30.0, 60.0, 90.0, 130.0):
        prof = _bukuk_bacak(sentetik_durus(), fleksiyon)
        P = prof.noktalar
        uyluk = float(np.linalg.norm(P[IDX("sag_diz")] - P[IDX("sag_kalca")]))
        baldir = float(np.linalg.norm(P[IDX("sag_diz")] - P[IDX("sag_ayak_bilegi")]))
        assert uyluk / baldir == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("fleksiyon", [20.0, 45.0, 70.0, 90.0, 120.0])
def test_bagimsiz_kacis_arttikca_valgus_monoton_artar(fleksiyon):
    """Bukuk bacakta orta hattan kacis arttikca valgus monoton artmali."""
    kacislar = np.linspace(0.0, 0.20, 21)
    degerler = [
        _deger(form_degerlendir(_bukuk_bacak(sentetik_durus(), fleksiyon, e)),
               "diz_valgusu_sag")
        for e in kacislar
    ]
    assert all(np.isfinite(degerler)), f"flex {fleksiyon}: sonlu olmayan olcum"
    assert all(b > a for a, b in zip(degerler, degerler[1:])), \
        f"flex {fleksiyon}: monoton degil -> {degerler}"
    assert degerler[0] == pytest.approx(0.0, abs=1e-9)


def test_bagimsiz_kacis_monoton_rastgele_geometride():
    """Monotonluk tek bir b-uzunluguna bagli olmamali."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        fleksiyon = rng.uniform(0.0, 130.0)
        uyluk = rng.uniform(0.3, 0.6)
        baldir = uyluk / rng.uniform(0.6, 1.8)
        kacislar = np.linspace(0.0, 0.15, 11)
        degerler = [
            _deger(form_degerlendir(_bukuk_bacak(
                sentetik_durus(), fleksiyon, e, uyluk, baldir)),
                "diz_valgusu_sag")
            for e in kacislar
        ]
        assert all(np.isfinite(degerler)), \
            f"flex={fleksiyon} uyluk={uyluk} baldir={baldir}"
        assert all(b > a for a, b in zip(degerler, degerler[1:])), \
            f"flex={fleksiyon} uyluk={uyluk} baldir={baldir}: {degerler}"


def test_bagimsiz_kacis_isareti_valgus_yonunu_verir():
    """Orta hattan disa kacis (varus) negatif, ice kacis pozitif olmali."""
    disa = _deger(form_degerlendir(
        _bukuk_bacak(sentetik_durus(), 90.0, kacis=-0.08)), "diz_valgusu_sag")
    ice = _deger(form_degerlendir(
        _bukuk_bacak(sentetik_durus(), 90.0, kacis=0.08)), "diz_valgusu_sag")
    assert disa < 0.0 < ice
    assert ice == pytest.approx(-disa, rel=1e-6)
