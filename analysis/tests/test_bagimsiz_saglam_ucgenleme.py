"""Bagimsiz dogrulama — saglam ucgenleme (aykiri gorus) ve hizalama hatti.

Kaynak sozlesmeler: `docs/kararlar/0072` §2 ve `pose3d/saglam.py`,
`pose3d/triangulate.py`, `pose3d/hizalama.py`.

Duzenek **projenin kendi** sentetik rigidir (`calib.synthetic.rig_yay`, 4 kamera,
+/-40 derece yay) ve yer gercegi bilinir; `Kalibrasyon` kamera 0'a gore
kurulur (bkz. `calib.multiview`), boylece `dunyadan_kamera0` ile dogrudan
karsilastirilabilir.

Sinanan sozlesmeler:
  - Tek kamera saptiginda o kamera elenir ve sonuc sapmasiz kalir.
  - Saptirma kademelendikce eleme esiginin nerede devreye girdigi olculur.
  - Iki (ve daha fazla) kamera birlikte saptiginda sonuc ya acikca reddedilir
    ya da sessizce bozulur; hangisi oldugu olculur.
  - Kabul/red kumesi, neden alani ve kovaryansin hangi goruslerden geldigi.
  - Sinirlar: 3 gorunum, 2 gorunum, 0/1 gozlem.

`hypothesis` kurulu degil; saptirmalar determinist ve acik degerlerdir.
"""
from __future__ import annotations

import numpy as np
import pytest

from calib.multiview import Kalibrasyon
from calib.synthetic import dunyadan_kamera0, rig_yay
from pose3d.hizalama import rijit_hizala
from pose3d.saglam import ucgenle_saglam
from pose3d.triangulate import ucgenle

KAMERALAR = rig_yay()
_R0 = KAMERALAR[0].R
_T0 = KAMERALAR[0].t.reshape(3)

# `rig_yay` dunya pozu uretir; `Kalibrasyon` sozlesmesi kamera 0'a GORE pozdur.
# Cevrilmezse triangulasyon cikti cercevesi dunya olur ve yer gercegiyle
# karsilastirma sabit bir kayma gosterir (olcum hatasi degil, cerceve hatasi).
_R_REL = [k.R @ _R0.T for k in KAMERALAR]
KALIB = Kalibrasyon(
    rms_px=0.0,
    Ks=[k.K.copy() for k in KAMERALAR],
    bozulmalar=[np.zeros(5) for _ in KAMERALAR],
    Rs=[R.copy() for R in _R_REL],
    Ts=[k.t.reshape(3) - _R_REL[i] @ _T0 for i, k in enumerate(KAMERALAR)],
)

ESIK_PX = 8.0
NOKTA = np.array([0.10, 0.30, 0.05])          # rigin calisma hacminde bir nokta


def _gercek(X=NOKTA) -> np.ndarray:
    return dunyadan_kamera0(KAMERALAR, X).reshape(3)


def _gozlemler(X=NOKTA, sapmalar=None, kameralar=None) -> dict:
    """Her kameraya izdusum; `sapmalar` {kamera: px} kadar kaydirilir."""
    obs = {}
    for i, k in enumerate(KAMERALAR):
        if kameralar is not None and i not in kameralar:
            continue
        q = k.K @ (k.R @ X + k.t.reshape(3))
        p = q[:2] / q[2]
        if sapmalar and i in sapmalar:
            p = p + np.array([sapmalar[i], sapmalar[i]], dtype=float)
        obs[i] = p
    return obs


def _hata_mm(sonuc, X=NOKTA) -> float:
    return float(np.linalg.norm(sonuc.nokta - _gercek(X))) * 1000.0


def _kostur(sapmalar=None, esik=ESIK_PX, kameralar=None, X=NOKTA):
    r = ucgenle_saglam(KALIB, _gozlemler(X, sapmalar, kameralar), esik_px=esik)
    return r, (None if r.sonuc is None else _hata_mm(r.sonuc, X))


# --- (1) temel: sapma yok -----------------------------------------------------

def test_bagimsiz_sapma_yoksa_tum_gorusler_kabul_ve_sifir_hata():
    r, hata = _kostur()
    assert r.kabul == (0, 1, 2, 3) and r.red == ()
    assert r.neden == "uzlasma"
    assert hata == pytest.approx(0.0, abs=1e-6)
    assert r.sonuc.artik_px == pytest.approx(0.0, abs=1e-9)


# --- (2) tek kamera saptiginda: elenir ve sonuc sapmasiz -----------------------

@pytest.mark.parametrize("kamera", [1, 2, 3])
@pytest.mark.parametrize("sapma", [50.0, 200.0, 1e4])
def test_bagimsiz_tek_kamera_elenir_ve_sonuc_sapmasiz(kamera, sapma):
    """Esik ustu saptiran kamera kabul kumesinden cikar, konum bozulmaz."""
    r, hata = _kostur({kamera: sapma})
    assert r.red == (kamera,), f"saptiran kamera {kamera} elenmedi: {r.red}"
    assert 0 not in r.red
    assert hata == pytest.approx(0.0, abs=1e-6), f"{sapma} px saptirma sonuca sizdi"
    assert r.neden == "uzlasma"


def test_bagimsiz_eleme_esigi_esik_px_de_devreye_girer():
    """Esik ve uzeri saptirma elenmeli; sonuc sapmasiz kalmali."""
    for sapma in (ESIK_PX, ESIK_PX + 1.0, 20.0, 100.0, 1000.0):
        r, hata = _kostur({3: sapma})
        assert r.red == (3,), f"{sapma} px elenmedi"
        assert hata == pytest.approx(0.0, abs=1e-6)
        assert r.sonuc.goren_kamera == len(r.kabul)


def test_bagimsiz_esik_alti_sapma_elenmez_ve_monoton_olarak_sizar():
    """Esik altindaki sapma aykiri sayilmaz; hatasi buyume sirasinda artmali."""
    hatalar = []
    for sapma in (1.0, 2.0, 4.0, 6.0):
        r, hata = _kostur({3: sapma})
        assert r.kabul == (0, 1, 2, 3), f"{sapma} px esik altinda ama elendi"
        assert r.red == ()
        assert hata > 0.0
        hatalar.append(hata)
    assert all(b > a for a, b in zip(hatalar, hatalar[1:])), hatalar


def test_bagimsiz_aykiri_kamera_iskelet_katmaninda_isaretlenir():
    """Elerme kimligi ve nedeni `ek`te tasinmali (0072 §2), sessizce yutulmamali."""
    from pose3d.iskelet import REFERANS_ISKELET, iskelet_ucgenle
    from pose3d.pose2d import Poz2B

    n = len(REFERANS_ISKELET)
    pts_world = np.array([NOKTA + np.array([0.01 * j, 0.005 * j, 0.0])
                          for j in range(n)])
    pts_cam0 = dunyadan_kamera0(KAMERALAR, pts_world)
    poses = {}
    for i, kam in enumerate(KAMERALAR):
        q = (pts_world @ kam.R.T + kam.t.reshape(3)) @ kam.K.T      # piksel
        xy = q[:, :2] / q[:, 2:3]
        if i == 3:
            xy = xy + 50.0
        poses[i] = Poz2B(REFERANS_ISKELET, xy, np.ones(n), np.ones(n, bool),
                         True, (1280, 720))
    sk = iskelet_ucgenle(KALIB, poses, saglam=True, esik_px=ESIK_PX)
    assert sk.gorunur.all(), "bazi eklemler kestirilemedi"
    assert sk.ek["kullanilan_kameralar"][REFERANS_ISKELET.eklemler[0]] == [0, 1, 2]
    assert sk.ek["red_nedenleri"][REFERANS_ISKELET.eklemler[0]] == "uzlasma"
    np.testing.assert_allclose(sk.noktalar, pts_cam0, atol=1e-6)


# --- (3) iki ve daha fazla kamera birlikte saptiginda --------------------------

@pytest.mark.parametrize("sapma", [15.0, 50.0, 300.0, 1e4])
def test_bagimsiz_iki_kamera_buyuk_sapmada_acikca_reddedilir(sapma):
    """Iki gorus aykiriysa sonuc UYDURULMAZ: None ve neden alani dolu doner."""
    r, hata = _kostur({2: sapma, 3: sapma})
    assert r.sonuc is None and hata is None, f"{sapma} px'te sessiz sonuc: {hata} mm"
    assert r.neden in ("yetersiz_gorus_uzlasmasi", "geometri_veya_artik_gecersiz")
    assert r.kabul == () and r.red == (0, 1, 2, 3)


@pytest.mark.parametrize("sapmalar", [
    {2: 10.0, 3: 10.0},              # iki kamera, her biri esigin ustunde
    {1: 12.0, 2: 12.0, 3: 12.0},     # cogunluk aykiri: tek saglikli kamera eleniyor
    {1: 50.0, 2: 50.0, 3: 50.0},     # kaba sapmada da ayni mekanizma
])
@pytest.mark.xfail(strict=True, reason=(
    "gozlemlenemez: gorusler kendi icinde tutarli saparsa (cogunluk ya da esik "
    "altinda artik) hicbir uzlasma yontemi dogru kumeyi secemez; 0072 sinirliligi"))
def test_bagimsiz_tutarli_sapma_esik_ustundeyken_sessizce_bozmamali(sapmalar):
    """Esigi asan mutabik sapma, konumu sessizce kaydirmamali.

    Saptiran gorusler birbirleriyle tutarli oldugu icin kabul edilen
    cozum onlarin da artigini esik altinda tutar (nokta zayif derinlik
    yonunde kayar). Bu yuzden `neden='uzlasma'` ve `artik < 8 px` gorunur,
    ama konum yer gerceginden onlarca mm sapar. Ya None donmeli ya da hata
    projenin kendi Kapi 2 esigi olan 10 mm'nin altinda kalmali.
    """
    r, hata = _kostur(sapmalar)
    assert r.sonuc is None or hata < 10.0, (
        f"sapma={sapmalar}: kabul={r.kabul} red={r.red} neden={r.neden!r} "
        f"artik={None if r.sonuc is None else round(r.sonuc.artik_px, 2)} px "
        f"hata={hata:.2f} mm")


@pytest.mark.parametrize("sapma", [1.0, 5.0])
def test_bagimsiz_iki_kamera_ayni_yonde_konumu_kaydirir_ama_kabul_saglikli_sanilir(sapma):
    """Iki kamerayi birden kaydirmanin bedeli: hata, sapmayla birlikte buyur."""
    tek, h_tek = _kostur({3: sapma})
    iki, h_iki = _kostur({2: sapma, 3: sapma})
    assert h_tek is not None and h_iki is not None
    assert h_iki > h_tek, f"{sapma} px: iki kamera tek kameradan daha az bozdu"


# --- (4) sinirlar: 3 gorunum, 2 gorunum, 0/1 gozlem ----------------------------

def test_bagimsiz_uc_gorunumde_tek_aykiri_reddedilir():
    """Uc gorunumde aykiri ayrilamaz: en az uc uyum gerekir (0072 §2)."""
    r, hata = _kostur({2: 50.0}, kameralar=(0, 1, 2))
    assert r.sonuc is None and hata is None
    assert r.neden == "yetersiz_gorus_uzlasmasi"


def test_bagimsiz_iki_gorunumde_aykiri_ayrimi_yok_ama_acikca_isaretli():
    """Iki gorunumde aykiri ayrilamaz; bu durum gizlenmemeli (0072 §2)."""
    temiz, hata = _kostur(kameralar=(0, 1))
    assert temiz.kabul == (0, 1) and temiz.neden == "iki_gorus_aykiri_ayrimi_yok"
    assert hata == pytest.approx(0.0, abs=1e-6)

    # Kaba aykirilik yine de yakalanir: iki gorus uyusmuyorsa sonuc yok.
    kaba, h_kaba = _kostur({1: 50.0}, kameralar=(0, 1))
    assert kaba.sonuc is None and h_kaba is None

    # Esik altinda ise sonuc verilir ama neden alani "ayrim yok" der.
    ince, h_ince = _kostur({1: 5.0}, kameralar=(0, 1))
    assert ince.sonuc is not None and h_ince > 0.0
    assert ince.neden == "iki_gorus_aykiri_ayrimi_yok"
    assert ince.kabul == (0, 1)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_bagimsiz_gozlem_sayisi_kabul_kumesini_belirler(n):
    """Saglikli gozlemlerin hepsi kabul edilir; kabul/red boluntusu tamdir."""
    r, hata = _kostur(kameralar=tuple(range(n)))
    assert r.kabul == tuple(range(n))
    assert r.red == ()
    assert set(r.kabul) | set(r.red) == set(range(n))
    assert set(r.kabul).isdisjoint(r.red)
    assert hata == pytest.approx(0.0, abs=1e-6)


def test_bagimsiz_gozlemsiz_ve_tek_gozlemde_cokmeden_none_doner():
    """Hic gozlem yoksa ve tek gozlem varsa sonuc yok; hata firlatilmaz."""
    for kameralar in ((), (2,)):
        r = ucgenle_saglam(KALIB, _gozlemler(kameralar=kameralar or ()))
        assert r.sonuc is None
        assert r.kabul == () and r.red == tuple(sorted(kameralar))
        assert r.neden == "yetersiz_gorus_uzlasmasi"
    # Duz ucgenle ayni girdide acikca hata verir (iki kol farkli sozlesme).
    with pytest.raises(ValueError, match="en az 2 gorus"):
        ucgenle(KALIB, _gozlemler(kameralar=(2,)))


# --- (5) kabul/red, neden, kovaryans, determinizm ------------------------------

def test_bagimsiz_kabul_red_her_zaman_boluntudur():
    rng = np.random.default_rng(1234)
    for _ in range(60):
        n = int(rng.integers(0, 5))
        kameralar = tuple(range(n))
        sapmalar = {i: float(rng.uniform(-60.0, 60.0)) for i in kameralar
                    if rng.random() < 0.4}
        r = ucgenle_saglam(KALIB, _gozlemler(sapmalar=sapmalar, kameralar=kameralar),
                           esik_px=ESIK_PX)
        assert set(r.kabul) | set(r.red) == set(kameralar)
        assert set(r.kabul).isdisjoint(r.red)
        assert list(r.kabul) == sorted(r.kabul) and list(r.red) == sorted(r.red)
        if r.sonuc is None:
            assert r.kabul == () and r.red == tuple(sorted(kameralar))
        else:
            assert r.kabul and r.sonuc.goren_kamera >= 2


def test_bagimsiz_elenen_gorus_kovaryansa_ve_artiga_girmez():
    """Kovaryans ve artik yalniz kabul edilen goruslerden gelmeli (0072 §2)."""
    obs = _gozlemler(sapmalar={3: 50.0})
    r = ucgenle_saglam(KALIB, obs, esik_px=ESIK_PX, sigma_px=1.0)
    beklenen = ucgenle(KALIB, {k: obs[k] for k in r.kabul}, True, 1.0)
    np.testing.assert_allclose(r.sonuc.kovaryans, beklenen.kovaryans)
    assert r.sonuc.artik_px == pytest.approx(beklenen.artik_px)
    assert r.sonuc.artik_px < ESIK_PX


def test_bagimsiz_ayni_girdi_ayni_sonuc():
    """Tohum yok; saglam kol determinist olmali."""
    for sapmalar in ({}, {3: 50.0}, {2: 50.0, 3: 50.0}):
        a = ucgenle_saglam(KALIB, _gozlemler(sapmalar=sapmalar), esik_px=ESIK_PX)
        b = ucgenle_saglam(KALIB, _gozlemler(sapmalar=sapmalar), esik_px=ESIK_PX)
        assert (a.kabul, a.red, a.neden) == (b.kabul, b.red, b.neden)
        assert (a.sonuc is None) == (b.sonuc is None)
        if a.sonuc is not None:
            assert np.array_equal(a.sonuc.nokta, b.sonuc.nokta)


def test_bagimsiz_kamera_anahtari_tipi_iki_kolda_ayni():
    """Iki kol ayni girdi sozlesmesini paylasmali: ayni anahtar, ayni sonuc.

    `ucgenle` numpy tamsayilarini kabul ediyor; `ucgenle_saglam` ise
    `type(k) is not int` ile reddediyor. Cagiran taraf anahtarlari bir
    diziden uretirse saglam kol -- duzeltmesi gereken kol -- coker.
    """
    obs = _gozlemler()
    numpy_anahtarli = {np.int64(k): v for k, v in obs.items()}
    assert ucgenle(KALIB, numpy_anahtarli).gecerli          # duz kol kabul ediyor
    r = ucgenle_saglam(KALIB, numpy_anahtarli, esik_px=ESIK_PX)
    assert r.kabul == (0, 1, 2, 3)


def test_bagimsiz_goren_kamera_aday_gozlemleri_tutar():
    """0072 §2: `Iskelet3B.goren_kamera` ADAY gozlemleri, kabul kumesi ayri tutulur.

    Tanim iskelet duzeyindedir. Tek `Ucgenleme.goren_kamera` ise duz `ucgenle`
    ile ayni anlamda cozume giren kamera sayisidir (kabul sayisi); iki duzey
    burada birlikte sinanir (bulgu 25 Eylul'de iskelet duzeyine tasindi).
    """
    r = ucgenle_saglam(KALIB, _gozlemler(sapmalar={3: 50.0}), esik_px=ESIK_PX)
    assert r.kabul == (0, 1, 2)
    assert r.sonuc.goren_kamera == len(r.kabul)


@pytest.mark.parametrize("kotu", [
    {"esik_px": 0.0}, {"esik_px": -1.0}, {"esik_px": float("nan")},
    {"min_aci_derece": 0.0}, {"min_aci_derece": 90.0}, {"min_aci_derece": -3.0},
])
def test_bagimsiz_gecersiz_esik_reddedilir(kotu):
    with pytest.raises(ValueError):
        ucgenle_saglam(KALIB, _gozlemler(), **kotu)


def test_bagimsiz_gecersiz_kamera_ve_gozlem_reddedilir():
    with pytest.raises(ValueError, match="gecersiz kamera indeksi"):
        ucgenle_saglam(KALIB, {9: np.array([100.0, 100.0])})
    with pytest.raises(ValueError, match="sonlu"):
        ucgenle_saglam(KALIB, {0: np.array([1.0, 2.0]), 1: np.array([np.nan, 2.0])})


# --- (6) hizalama hatti: eleme sonrasi sekil bozulmasi -------------------------

def test_bagimsiz_eleme_sonrasi_hizalama_hatti_sifir_bozulma_verir():
    """Her noktada bir kamera aykiriyken saglam kol sekli korumali.

    Uctan uca: saglam ucgenleme -> Kabsch hizalama. Ham (saglam olmayan) kol
    ayni girdide onlarca mm sekil bozulmasi verir; hizalama bunu gostermeli.
    """
    Xler = [np.array([0.1 * i - 0.2, 0.15 + 0.05 * i, 0.02 * i]) for i in range(5)]
    saglam, ham = [], []
    for j, X in enumerate(Xler):
        obs = _gozlemler(X, sapmalar={j % 4: 50.0})
        r = ucgenle_saglam(KALIB, obs, esik_px=ESIK_PX)
        assert r.red == (j % 4,), f"nokta {j}: elenmedi ({r.red})"
        saglam.append(r.sonuc.nokta)
        ham.append(ucgenle(KALIB, obs).nokta)
    gercek = np.array([_gercek(X) for X in Xler])
    h_saglam = rijit_hizala(np.array(saglam), gercek, olcek_serbest=True)
    h_ham = rijit_hizala(np.array(ham), gercek, olcek_serbest=True)
    assert h_saglam.rms_mm < 1e-3, f"saglam kol artik: {h_saglam.rms_mm} mm"
    assert h_saglam.olcek == pytest.approx(1.0, abs=1e-6)
    assert h_ham.rms_mm > 10.0, "ham kol bozulma gostermedi; kurgu anlamsiz"
