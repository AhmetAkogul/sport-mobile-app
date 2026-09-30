"""Hareket formu olculeri: sentetik iskeletlerde bilinen degerler."""

import numpy as np
import pytest

from eval.hareket_formu import (auc, bacak_acisi, dirsek_bukulme, duzlem_disi, govde_yana,
                                kalca_cizgiden, kol_acisi, pelvis_egimi, surekli_en_buyuk,
                                tekrar_olculeri)
from pose3d.iskelet import REFERANS_ISKELET

J = REFERANS_ISKELET.indeks
U = np.array([0.0, 1.0, 0.0])            # yukari +y; kisinin solu +x; one z


def _ayakta():
    X = np.zeros((13, 3))
    konum = {"boyun": (0, 1.5, 0), "sol_omuz": (0.2, 1.5, 0), "sag_omuz": (-0.2, 1.5, 0),
             "sol_dirsek": (0.2, 1.2, 0), "sag_dirsek": (-0.2, 1.2, 0),
             "sol_bilek": (0.2, 0.95, 0), "sag_bilek": (-0.2, 0.95, 0),
             "sol_kalca": (0.1, 1.0, 0), "sag_kalca": (-0.1, 1.0, 0),
             "sol_diz": (0.1, 0.55, 0), "sag_diz": (-0.1, 0.55, 0),
             "sol_ayak_bilegi": (0.1, 0.1, 0), "sag_ayak_bilegi": (-0.1, 0.1, 0)}
    for ad, p in konum.items():
        X[J(ad)] = p
    return X[None]


def test_kol_yana_ve_one_kaldirma():
    X = _ayakta()
    X[0, J("sag_dirsek")] = X[0, J("sag_omuz")] + [-0.3, 0, 0]        # yana yatay
    a = kol_acisi(X, U, "sag")
    assert a[0] == pytest.approx(90)
    assert duzlem_disi(X, U, J("sag_omuz"), J("sag_dirsek"), a)[0] == pytest.approx(0, abs=1e-6)
    X[0, J("sag_dirsek")] = X[0, J("sag_omuz")] + [0, 0, 0.3]         # one yatay
    a = kol_acisi(X, U, "sag")
    assert duzlem_disi(X, U, J("sag_omuz"), J("sag_dirsek"), a)[0] == pytest.approx(90)
    # kol asagidayken duzlem disi tanimsiz (NaN)
    assert np.isnan(duzlem_disi(_ayakta(), U, J("sag_omuz"), J("sag_dirsek"),
                                kol_acisi(_ayakta(), U, "sag"))[0])


def test_dirsek_ve_bacak_acilari():
    X = _ayakta()
    assert dirsek_bukulme(X, "sol")[0] == pytest.approx(0, abs=1e-6)
    X[0, J("sol_bilek")] = X[0, J("sol_dirsek")] + [0, 0, 0.25]      # 90 derece bukuk
    assert dirsek_bukulme(X, "sol")[0] == pytest.approx(90)
    X[0, J("sag_diz")] = X[0, J("sag_kalca")] + [-0.45 * np.sin(np.radians(30)),
                                                   -0.45 * np.cos(np.radians(30)), 0]
    assert bacak_acisi(X, U, "sag")[0] == pytest.approx(30)


def test_govde_yana_ve_pelvis_egimi():
    X = _ayakta()
    assert govde_yana(X, U)[0] == pytest.approx(0, abs=1e-6)
    X[0, J("boyun")] = X[0, [J("sol_kalca"), J("sag_kalca")]].mean(0) + [0.5 * np.sin(np.radians(20)),
                                                                           0.5 * np.cos(np.radians(20)), 0]
    assert govde_yana(X, U)[0] == pytest.approx(20)
    X[0, J("sol_kalca"), 1] += 0.2 * np.tan(np.radians(10))
    assert pelvis_egimi(X, U)[0] == pytest.approx(10, abs=0.1)


def test_sinav_kalca_cizgisi():
    X = np.zeros((1, 13, 3))
    X[0, J("boyun")] = [0, 0.5, 0]
    X[0, [J("sol_ayak_bilegi"), J("sag_ayak_bilegi")]] = [[0.1, 0.1, 1.5], [-0.1, 0.1, 1.5]]
    X[0, [J("sol_kalca"), J("sag_kalca")]] = [[0.1, 0.2, 0.75], [-0.1, 0.2, 0.75]]  # cizginin altinda
    assert kalca_cizgiden(X, U)[0] < -0.05                           # sarkma negatif


def test_surekli_en_buyuk_tek_kare_sivrisini_yok_sayar():
    v = np.array([0, 0, 10, 0, 3, 3, 3, 0], float)
    assert surekli_en_buyuk(v, 3) == 3
    assert np.isnan(surekli_en_buyuk(v[:2], 3))


def test_tekrar_olculeri_anahtarlar_ve_hata():
    X = np.repeat(_ayakta(), 12, axis=0)
    assert set(tekrar_olculeri(X, U, 3, None, 30.0)) == {"kalca_sarkma", "kalca_pike", "sig",
                                                         "dirsek_acilma"}
    assert "bacak_one" in tekrar_olculeri(X, U, 4, "sol", 30.0)
    with pytest.raises(ValueError):
        tekrar_olculeri(X, U, 7, None, 30.0)


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1], [1]) == 0.5
    assert auc([np.nan], [1]) is None


def test_diz_onde_isareti_ve_diz_ice():
    from eval.hareket_formu import diz_ice, diz_onde
    X = _ayakta()
    # kisinin solu +x, yukari +y ise one +z (kuzeye bakanin solu bati); dizi one it
    X[0, J("sag_diz"), 2] = 0.2
    assert diz_onde(X, U, "sag")[0] == pytest.approx(0.2 / np.linalg.norm([0, 0.45, 0.2]), rel=1e-6)
    X = _ayakta()
    X[0, J("sag_diz"), 0] += 0.05                  # sag diz orta hatta (+x) dogru: ice
    assert diz_ice(X, U, "sag")[0] > 0
    X = _ayakta()
    X[0, J("sol_diz"), 0] -= 0.05                  # sol diz orta hatta (-x) dogru: ice
    assert diz_ice(X, U, "sol")[0] > 0


def test_squat_lunge_olculeri():
    X = np.repeat(_ayakta(), 12, axis=0)
    X[2:10, J("sol_diz"), 2] = 0.3                 # sol diz buk (one), 8 kare > 0,2 s
    o5 = tekrar_olculeri(X, U, 5, None, 30.0)
    o6 = tekrar_olculeri(X, U, 6, None, 30.0)
    assert o5["derin"] > 20 and o6["derin"] == pytest.approx(o5["derin"])
    assert o5["diz_onde"] > 0
    ayak = np.full((12, 4, 3), np.nan)
    ayak[:, 3] = X[:, J("sol_ayak_bilegi")] + [0, 0, 0.15]       # sol ayak ucu one bakar
    ayak[:, 1] = X[:, J("sag_ayak_bilegi")] + [0, 0, 0.15]
    o = tekrar_olculeri(X, U, 5, None, 30.0, ayak=ayak)
    assert o["diz_parmak_onde"] == pytest.approx((0.3 - 0.15) / np.linalg.norm([0, 0.45, 0.3]))


def test_tekrar_sayaci_nedensel_tepe():
    from eval.hareket_formu import TekrarSayaci, esikler_ogren
    s = TekrarSayaci(yukseklik=60, belirginlik=20, en_az_aralik_s=1.0)
    t = np.arange(0, 6, 0.1)
    v = 45 + 40 * np.sin(2 * np.pi * t / 2 - np.pi / 2)      # 2 s periyot, 5..85 derece
    v[25] = np.nan
    v[31] += 6                                               # tepe ustunde kucuk titreme
    biten = [r for r in (s.ekle(a, b) for a, b in zip(t, v)) if r]
    assert len(biten) == 3
    assert biten[0][0] == pytest.approx(0.0) and 1.0 < biten[0][1] < 2.0
    with pytest.raises(ValueError):
        TekrarSayaci(10, 0, 1)
    assert esikler_ogren(np.array([0, 10]), np.array([100]), np.array([2.0, 4.0])) == (
        pytest.approx(43.0), pytest.approx(28.5), pytest.approx(1.5))


def test_aktif_taraf_hareketten():
    from eval.hareket_formu import aktif_taraf
    X = np.repeat(_ayakta(), 10, axis=0)
    for i in range(10):                                           # sol kol yana kalkar
        a = np.radians(9 * i)
        X[i, J("sol_dirsek")] = X[i, J("sol_omuz")] + 0.3 * np.array([np.sin(a), -np.cos(a), 0])
    assert aktif_taraf(X, U, 1) == "sol"


def test_esik_kurali_ayiran_esigi_bulur():
    from eval.hareket_formu import esik_kurali
    x = [1, 2, 3, 10, 11, 12, np.nan]
    y = [False, False, False, True, True, True, True]
    t = esik_kurali(x, y)
    assert 3 <= t < 10
    assert esik_kurali([1, 2], [True, True]) == np.inf       # tek sinif: karar yok


def test_nedensel_medyan_gelecegi_gormez():
    from eval.hareket_formu import nedensel_medyan
    x = np.array([0.0, 0.0, 9.0, 0.0, np.nan, 5.0])
    y = nedensel_medyan(x, 3)
    assert y[2] == 0.0                     # tek sicrama bastirilir
    assert y[4] == 4.5                     # NaN atlanir: medyan(9, 0)
    assert np.isnan(nedensel_medyan(np.array([np.nan]), 3)[0])
    z = x.copy()
    z[5] = 100.0                           # gelecek kare degisse de onceki cikti ayni
    assert np.array_equal(nedensel_medyan(z, 3)[:5], y[:5], equal_nan=True)


def test_sayac_esikleri_dinlenme_ile_tepe_arasinda():
    from eval.hareket_formu import sayac_esikleri
    kare = np.arange(300)
    sinyal = np.zeros(300)
    gercek = [(50, 110), (150, 210)]
    for a, b in gercek:
        sinyal[a:b + 1] = np.sin(np.linspace(0, np.pi, b - a + 1)) * 10
    yuk, bel, aralik = sayac_esikleri([{"kare": kare, "gercek": gercek, "sinyal": sinyal}],
                                      kare_hizi=30.0)
    assert 0 < yuk < 10 and bel > 0
    assert aralik == pytest.approx(0.5 * 2.0)


def test_birlesik_kisi_disarida_ayrilan_veride_yuksek_auc():
    pytest.importorskip("sklearn")
    from eval.hareket_formu import birlesik_kisi_disarida
    r = np.random.default_rng(0)
    satir = [{"kisi": f"k{i % 4}", "yanlis": bool(i % 2), "a": (i % 2) * 2 + r.normal(0, 0.5),
              "b": r.normal()} for i in range(80)]
    sonuc = birlesik_kisi_disarida(satir, ["a", "b"])
    assert sonuc["auc"] > 0.9 and sonuc["n"] == 80


def _lunge_2b(diz_x):
    """Sag ayak onde (+x yonune bakiyor), sol arkada; goruntu pikseli, y asagi."""
    from pose3d.tam_vucut import TAM_VUCUT
    P = np.full((5, len(TAM_VUCUT), 2), np.nan)
    for t, bilek_x in (("sag", 200.0), ("sol", 0.0)):
        P[:, TAM_VUCUT.indeks(f"{t}_ayak_bilegi")] = [bilek_x, 400]
        P[:, TAM_VUCUT.indeks(f"{t}_bas_parmak")] = [bilek_x + 40, 410]
        P[:, TAM_VUCUT.indeks(f"{t}_diz")] = [bilek_x + (diz_x if t == "sag" else 0), 300]
    return P


def test_ondeki_bacak_2b_bukulmeye_degil_konuma_bakar():
    from eval.hareket_formu import ondeki_bacak_2b
    assert ondeki_bacak_2b(_lunge_2b(0.0)) == "sag"
    assert ondeki_bacak_2b(_lunge_2b(0.0)[:, :, ::-1] * 0) is None     # yon belirsiz


def test_diz_parmak_onde_2b_gecince_pozitif():
    from eval.hareket_formu import diz_parmak_onde_2b
    kaval = 100.0
    assert diz_parmak_onde_2b(_lunge_2b(60.0), "sag")[0] == pytest.approx((60 - 40) / np.hypot(60, kaval))
    assert diz_parmak_onde_2b(_lunge_2b(0.0), "sag")[0] < 0


def test_secilen_model_seyrek_veride_l1_secer_ve_sizmaz():
    pytest.importorskip("sklearn")
    from eval.hareket_formu import ADAY_MODELLER, birlesik_kisi_disarida, secilen_model
    r = np.random.default_rng(1)
    n = 120
    y = np.arange(n) % 2 == 1
    X = np.c_[y * 1.5 + r.normal(0, 0.6, n), r.normal(0, 1, (n, 15))]  # 1 bilgili + 15 gurultu
    kisi = np.array([f"k{i % 6}" for i in range(n)])
    m = secilen_model(X, y, kisi)
    assert m.aday_ in ADAY_MODELLER
    satir = [{"kisi": k, "yanlis": bool(v), **{f"o{j}": x[j] for j in range(16)}}
             for k, v, x in zip(kisi, y, X)]
    sonuc = birlesik_kisi_disarida(satir, [f"o{j}" for j in range(16)], secim=True)
    assert sonuc["auc"] > 0.8
