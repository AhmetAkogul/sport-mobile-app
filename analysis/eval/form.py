"""Form karari katmani: 3B iskeletten kusur tespiti.

Uc kusur olculur -- diz valgusu, kalca/omuz hizasi ve govde rotasyonu. Tezin
sordugu soru "kusur var mi" degil, **"kararin ne kadarina guvenilir"**; bu yuzden
her olcum uc sey birden dondurur: olculen aci, karar esigi ve olcumun
belirsizligi. Belirsizlik esigi kesiyorsa karar `BELIRSIZ` olur, uydurma bir
"dogru/kusurlu" uretilmez. A2'deki mansette (form karari dogrulugu vs kamera
acisi) egrinin dustugu yer tam olarak bu bolgedir.

Koordinat sistemi varsayilmaz. `Iskelet3B.noktalar` kamera 0'a goredir ve yer
cekimi yonu bilinmez; bu yuzden olcumler once **iskeletin kendisinden** kurulan
bir govde cercevesine tasinir (`govde_cercevesi`). Boylece ayni durus, kamera
nereye bakarsa baksin ayni sayiyi verir.

Esikler literaturden alinmis **baslangic** degerleridir ve gercek veriyle
kalibre edilecektir; gerekce `docs/kararlar/0010-form-karari-esikleri.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B, IskeletTanimi


class Karar(StrEnum):
    """Bir olcumun sonucu. `BELIRSIZ` bir hata degil, gecerli bir cevaptir."""

    DOGRU = "dogru"
    KUSURLU = "kusurlu"
    BELIRSIZ = "belirsiz"


@dataclass(frozen=True)
class Esik:
    """Karar siniri. `tek_yonlu` ise yalnizca pozitif yon kusur sayilir.

    Diz valgusu tek yonludur: dizin ice cokmesi kusurdur, disa acilmasi (varus)
    ayri bir kusurdur ve bu katmanda **olculmez**, yalnizca isaretli deger
    raporlanir. Kalca egimi ve govde rotasyonu iki yonludur; hangi tarafa
    olursa olsun simetri bozuklugudur.
    """

    deger: float
    tek_yonlu: bool = False


# Baslangic esikleri (derece). Gerekce ve kaynaklar: docs/kararlar/0010.
VARSAYILAN_ESIKLER: dict[str, Esik] = {
    "diz_valgusu_sag": Esik(10.0, tek_yonlu=True),
    "diz_valgusu_sol": Esik(10.0, tek_yonlu=True),
    "kalca_hizasi": Esik(8.0),
    "govde_rotasyonu": Esik(15.0),
}

# Karar, olcumun belirsizliginin kac katini guvenlik payi sayacagi. 2.0 kabaca
# %95 guven araligi demektir; esik bu araligin icinde kaliyorsa karar verilmez.
VARSAYILAN_K = 2.0


@dataclass(frozen=True)
class GovdeCercevesi:
    """Iskeletten kurulan sag-elli ortonormal cerceve.

    `yanal` sag kalcadan sol kalcaya, `yukari` kalca merkezinden boyuna bakar,
    `ileri` ikisinin capraz carpimidir. Kaynak eklemler gorunmuyorsa cerceve
    kurulamaz ve `govde_cercevesi` None doner.
    """

    yanal: np.ndarray
    yukari: np.ndarray
    ileri: np.ndarray
    merkez: np.ndarray


@dataclass(frozen=True)
class Olcum:
    """Tek bir kusurun olcumu.

    `deger` isaretlidir ve derece cinsindendir; hesaplanamadiysa NaN olur.
    `belirsizlik` 1-sigma standart sapmadir; kestirilmediyse NaN'dir ve o zaman
    karar yalin esik karsilastirmasina duser.
    """

    ad: str
    deger: float
    esik: float
    belirsizlik: float
    karar: Karar
    eksik_eklemler: tuple[str, ...] = ()

    @property
    def hesaplandi(self) -> bool:
        return bool(np.isfinite(self.deger))


@dataclass(frozen=True)
class FormRaporu:
    """Bir karenin tum form olcumleri ve birlesik karari."""

    olcumler: dict[str, Olcum] = field(default_factory=dict)

    @property
    def genel_karar(self) -> Karar:
        """Bir kusur bile kesinse KUSURLU; degilse belirsizlik varsa BELIRSIZ."""
        kararlar = {o.karar for o in self.olcumler.values()}
        if Karar.KUSURLU in kararlar:
            return Karar.KUSURLU
        if Karar.BELIRSIZ in kararlar:
            return Karar.BELIRSIZ
        return Karar.DOGRU

    @property
    def kusurlar(self) -> tuple[str, ...]:
        return tuple(ad for ad, o in self.olcumler.items() if o.karar is Karar.KUSURLU)

    def ozet(self) -> dict:
        """JSON'a yazilabilir ozet; rapor uretimi bunu kullanir."""
        return {
            "genel_karar": str(self.genel_karar),
            "kusurlar": list(self.kusurlar),
            "olcumler": {
                ad: {
                    "deger_derece": None if not o.hesaplandi else round(o.deger, 3),
                    "esik_derece": o.esik,
                    "belirsizlik_derece": (
                        None if not np.isfinite(o.belirsizlik) else round(o.belirsizlik, 3)
                    ),
                    "karar": str(o.karar),
                    "eksik_eklemler": list(o.eksik_eklemler),
                }
                for ad, o in self.olcumler.items()
            },
        }


# --- cerceve -----------------------------------------------------------------

_CERCEVE_EKLEMLERI = ("boyun", "sag_kalca", "sol_kalca")


def _birim(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        return None
    return v / n


def _cerceve_kur(P: np.ndarray, tanim: IskeletTanimi) -> GovdeCercevesi | None:
    i = tanim.indeks
    sag, sol = P[i("sag_kalca")], P[i("sol_kalca")]
    merkez = 0.5 * (sag + sol)
    yukari = _birim(P[i("boyun")] - merkez)
    if yukari is None:
        return None
    # Gram-Schmidt: yanal eksen yukaridan arindirilir, yoksa cerceve dik olmaz.
    yanal_ham = sol - sag
    yanal = _birim(yanal_ham - (yanal_ham @ yukari) * yukari)
    if yanal is None:
        return None
    return GovdeCercevesi(yanal=yanal, yukari=yukari,
                          ileri=np.cross(yanal, yukari), merkez=merkez)


def govde_cercevesi(iskelet: Iskelet3B) -> GovdeCercevesi | None:
    """Iskeletten govde cercevesi; kaynak eklemler eksikse None."""
    if not all(iskelet.gorunur[iskelet.tanim.indeks(e)] for e in _CERCEVE_EKLEMLERI):
        return None
    return _cerceve_kur(iskelet.noktalar, iskelet.tanim)


# --- olcumler ----------------------------------------------------------------

_GEREKLI: dict[str, tuple[str, ...]] = {
    "diz_valgusu_sag": ("sag_kalca", "sag_diz", "sag_ayak_bilegi"),
    "diz_valgusu_sol": ("sol_kalca", "sol_diz", "sol_ayak_bilegi"),
    "kalca_hizasi": ("sag_omuz", "sol_omuz"),
    "govde_rotasyonu": ("sag_omuz", "sol_omuz"),
}


def _diz_valgusu(P: np.ndarray, tanim: IskeletTanimi, c: GovdeCercevesi,
                 taraf: str) -> float:
    """Frontal duzlem diz acisinin duzden sapmasi (derece); + ise valgus.

    Literaturdeki FPPA ile ayni tanim: kalca-diz ve ayak bilegi-diz vektorleri
    frontal duzleme yansitilir, aralarindaki aci 180 dereceden ne kadar sapiyorsa
    o kadar bozukluk vardir. Isaret, dizin kalca-ayak bilegi dogrusuna gore
    **orta hatta** mi yoksa disa mi kactigindan gelir.
    """
    i = tanim.indeks
    h, k, a = P[i(f"{taraf}_kalca")], P[i(f"{taraf}_diz")], P[i(f"{taraf}_ayak_bilegi")]

    def yansit(p: np.ndarray) -> np.ndarray:
        d = p - c.merkez
        return np.array([d @ c.yanal, d @ c.yukari])

    h2, k2, a2 = yansit(h), yansit(k), yansit(a)
    u, v = _birim(h2 - k2), _birim(a2 - k2)
    if u is None or v is None:
        return float("nan")
    sapma = 180.0 - float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))

    bacak = a2 - h2
    uzunluk2 = float(bacak @ bacak)
    if uzunluk2 < 1e-12:
        return float("nan")
    t = float((k2 - h2) @ bacak) / uzunluk2
    yanal_kacis = float((k2 - (h2 + t * bacak))[0])
    # `yanal` sagdan sola bakar: sag bacak icin orta hat +yanal, sol icin -yanal.
    medial = 1.0 if taraf == "sag" else -1.0
    return float(np.sign(yanal_kacis * medial) * sapma)


def _kalca_hizasi(P: np.ndarray, tanim: IskeletTanimi, c: GovdeCercevesi) -> float:
    """Omuz ekseninin kalca eksenine gore yan egimi (derece), frontal duzlemde.

    Kalca ekseni cerceveyi kurdugu icin referans `yanal`dir; dolayisiyla bu sayi
    dogrudan omuz-kalca egim farkidir. Tek bacak uzerinde cokme (Trendelenburg)
    ve yana yatma bu olcude gorunur.
    """
    i = tanim.indeks
    omuz = P[i("sol_omuz")] - P[i("sag_omuz")]
    x, y = float(omuz @ c.yanal), float(omuz @ c.yukari)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(y, x)))


def _govde_rotasyonu(P: np.ndarray, tanim: IskeletTanimi, c: GovdeCercevesi) -> float:
    """Omuz ekseninin kalca eksenine gore eksenel donusu (derece).

    Omuz vektoru `yukari` eksenine dik duzleme yansitilir; kalan acisal fark
    govdenin burulmasidir. Squat ve lunge'da agirligin tek tarafa kaymasinin
    en erken isareti budur. Isaret sag el kuralidir: `yukari` ekseni etrafinda
    pozitif donus, omuz cizgisinin sol omuz tarafinin geriye gitmesidir.
    """
    i = tanim.indeks
    omuz = P[i("sol_omuz")] - P[i("sag_omuz")]
    duzlemde = omuz - (omuz @ c.yukari) * c.yukari
    x = float(c.yanal @ duzlemde)
    y = float(np.cross(c.yanal, duzlemde) @ c.yukari)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(y, x)))


def _tum_olcumler(P: np.ndarray, tanim: IskeletTanimi) -> dict[str, float]:
    """Dort olcumu tek noktalar dizisinden hesaplar; cerceve kurulamazsa hepsi NaN."""
    c = _cerceve_kur(P, tanim)
    if c is None:
        return {ad: float("nan") for ad in _GEREKLI}
    return {
        "diz_valgusu_sag": _diz_valgusu(P, tanim, c, "sag"),
        "diz_valgusu_sol": _diz_valgusu(P, tanim, c, "sol"),
        "kalca_hizasi": _kalca_hizasi(P, tanim, c),
        "govde_rotasyonu": _govde_rotasyonu(P, tanim, c),
    }


# --- karar -------------------------------------------------------------------

def _karar_ver(deger: float, esik: Esik, belirsizlik: float, k: float) -> Karar:
    """Esigi belirsizlik bandiyla karsilastirir.

    Band esigi kesiyorsa karar verilmez. Bu, tezin iddiasinin kod karsiligidir:
    olcum hatasi karar sinirindan buyukse o kare hakkinda konusulamaz.
    """
    if not np.isfinite(deger):
        return Karar.BELIRSIZ
    gozlenen = deger if esik.tek_yonlu else abs(deger)
    if not np.isfinite(belirsizlik) or belirsizlik <= 0.0:
        return Karar.KUSURLU if gozlenen > esik.deger else Karar.DOGRU
    pay = k * belirsizlik
    if gozlenen - pay > esik.deger:
        return Karar.KUSURLU
    if gozlenen + pay < esik.deger:
        return Karar.DOGRU
    return Karar.BELIRSIZ


def _belirsizlikler(iskelet: Iskelet3B, konum_belirsizligi_m, n_ornek: int,
                    seed: int) -> dict[str, float]:
    """Eklem konum belirsizligini Monte Carlo ile acisal belirsizlige tasir.

    Analitik turev yerine ornekleme secildi: aci olcumleri arccos ve arctan2
    iceriyor, birinci derece yaklasim esik yakininda yaniltici oluyor. Tohum
    sabit verildigi icin sonuc tekrarlanabilir -- `make reproduce` ayni sayiyi
    uretir.
    """
    sigma = np.asarray(konum_belirsizligi_m, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(len(iskelet.tanim), float(sigma))
    if sigma.shape != (len(iskelet.tanim),):
        raise ValueError("konum_belirsizligi_m skaler ya da eklem sayisi kadar olmali")
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("konum_belirsizligi_m sonlu ve negatif olmayan olmali")

    rng = np.random.default_rng(seed)
    P = iskelet.noktalar
    birikim: dict[str, list[float]] = {ad: [] for ad in _GEREKLI}
    for _ in range(n_ornek):
        bozuk = P + rng.normal(0.0, 1.0, size=P.shape) * sigma[:, None]
        for ad, deger in _tum_olcumler(bozuk, iskelet.tanim).items():
            if np.isfinite(deger):
                birikim[ad].append(deger)
    return {
        ad: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        for ad, v in birikim.items()
    }


def form_degerlendir(
    iskelet: Iskelet3B,
    esikler: dict[str, Esik] | None = None,
    konum_belirsizligi_m: float | np.ndarray | None = None,
    n_ornek: int = 200,
    seed: int = 20260922,
    k: float = VARSAYILAN_K,
) -> FormRaporu:
    """Bir 3B iskeletten form raporu uretir.

    `konum_belirsizligi_m` verilirse (ucgenlemeden gelen 1-sigma eklem konum
    belirsizligi, metre) kararlar belirsizlik bandiyla verilir; verilmezse
    yalin esik karsilastirmasi yapilir ve `belirsizlik` NaN kalir.

    Gorunmeyen eklem **uydurulmaz**: o eklemi gerektiren olcum NaN doner ve
    karari `BELIRSIZ` olur, eksik eklemlerin adi raporda tasinir.
    """
    esikler = {**VARSAYILAN_ESIKLER, **(esikler or {})}
    tanim = iskelet.tanim
    degerler = _tum_olcumler(iskelet.noktalar, tanim)

    if konum_belirsizligi_m is None:
        sapmalar = {ad: float("nan") for ad in _GEREKLI}
    else:
        sapmalar = _belirsizlikler(iskelet, konum_belirsizligi_m, n_ornek, seed)

    cerceve_eksik = tuple(
        e for e in _CERCEVE_EKLEMLERI if not iskelet.gorunur[tanim.indeks(e)]
    )
    olcumler: dict[str, Olcum] = {}
    for ad, gerekli in _GEREKLI.items():
        eksik = cerceve_eksik + tuple(
            e for e in gerekli if not iskelet.gorunur[tanim.indeks(e)]
        )
        deger = float("nan") if eksik else degerler[ad]
        sapma = float("nan") if eksik else sapmalar[ad]
        esik = esikler[ad]
        olcumler[ad] = Olcum(
            ad=ad, deger=deger, esik=esik.deger, belirsizlik=sapma,
            karar=_karar_ver(deger, esik, sapma, k), eksik_eklemler=eksik,
        )
    return FormRaporu(olcumler=olcumler)


# --- sentetik durus ----------------------------------------------------------

# Nominal ayakta durus (metre, y yukari). Diz, kalca-ayak bilegi dogrusunun tam
# ortasinda: boylece verilen valgus acisi geri olculdugunde birebir cikar.
_NOMINAL: dict[str, tuple[float, float, float]] = {
    "boyun": (0.0, 0.55, 0.0),
    "sag_omuz": (-0.18, 0.45, 0.0), "sol_omuz": (0.18, 0.45, 0.0),
    "sag_dirsek": (-0.22, 0.18, 0.05), "sol_dirsek": (0.22, 0.18, 0.05),
    "sag_bilek": (-0.24, -0.08, 0.08), "sol_bilek": (0.24, -0.08, 0.08),
    "sag_kalca": (-0.11, 0.0, 0.0), "sol_kalca": (0.11, 0.0, 0.0),
    "sag_diz": (-0.11, -0.425, 0.0), "sol_diz": (0.11, -0.425, 0.0),
    "sag_ayak_bilegi": (-0.11, -0.85, 0.0), "sol_ayak_bilegi": (0.11, -0.85, 0.0),
}

_UST_GOVDE = ("sag_omuz", "sol_omuz", "sag_dirsek", "sol_dirsek",
              "sag_bilek", "sol_bilek")


def _donme(eksen: np.ndarray, aci_derece: float) -> np.ndarray:
    """Rodrigues donme matrisi."""
    t = np.radians(aci_derece)
    K = np.array([[0.0, -eksen[2], eksen[1]],
                  [eksen[2], 0.0, -eksen[0]],
                  [-eksen[1], eksen[0], 0.0]], dtype=float)
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def sentetik_durus(
    valgus_sag: float = 0.0,
    valgus_sol: float = 0.0,
    kalca_hizasi: float = 0.0,
    govde_rotasyonu: float = 0.0,
    tanim: IskeletTanimi = REFERANS_ISKELET,
    gorunmez: tuple[str, ...] = (),
    donusum: np.ndarray | None = None,
) -> Iskelet3B:
    """Kusur aci degerleri **verilen** sentetik durus; yer gercegi budur.

    Duz bacakli, dik bir govdedir: amaci gercekci bir squat uretmek degil,
    olcum katmanini bilinen bir cevapla sinamaktir. Uretilen iskelette
    `form_degerlendir` girilen acilari geri okumalidir.

    `donusum` verilirse (4x4 ya da 3x3) tum noktalar o donusumden gecirilir;
    olcumlerin kamera durusundan bagimsiz oldugunu sinamak icin kullanilir.
    """
    P = np.array([_NOMINAL[e] for e in tanim.eklemler], dtype=float)
    i = tanim.indeks

    # Valgus: diz, kalca-ayak bilegi dogrusundan yanal olarak kaydirilir. Diz tam
    # ortada oldugu icin sapma acisi 2*atan(2d/L)'dir, yani d = (L/2)*tan(aci/2).
    for taraf, aci in (("sag", valgus_sag), ("sol", valgus_sol)):
        if aci == 0.0:
            continue
        h, a = P[i(f"{taraf}_kalca")], P[i(f"{taraf}_ayak_bilegi")]
        L = float(np.linalg.norm(a - h))
        d = (L / 2.0) * np.tan(np.radians(aci) / 2.0)
        medial = 1.0 if taraf == "sag" else -1.0   # orta hat sag bacak icin +x
        P[i(f"{taraf}_diz")] = P[i(f"{taraf}_diz")] + np.array([medial * d, 0.0, 0.0])

    # Ust govde omuz ortasi etrafinda dondurulur; boyun ve kalca sabit kalir,
    # dolayisiyla govde cercevesi degismez ve acilar birebir geri okunur.
    if kalca_hizasi or govde_rotasyonu:
        R = (_donme(np.array([0.0, 1.0, 0.0]), govde_rotasyonu)
             @ _donme(np.array([0.0, 0.0, 1.0]), kalca_hizasi))
        omuz_orta = 0.5 * (P[i("sag_omuz")] + P[i("sol_omuz")])
        for e in _UST_GOVDE:
            P[i(e)] = omuz_orta + R @ (P[i(e)] - omuz_orta)

    if donusum is not None:
        M = np.asarray(donusum, dtype=float)
        if M.shape == (4, 4):
            P = (P @ M[:3, :3].T) + M[:3, 3]
        elif M.shape == (3, 3):
            P = P @ M.T
        else:
            raise ValueError("donusum 3x3 ya da 4x4 olmali")

    gorunur = np.ones(len(tanim), dtype=bool)
    for e in gorunmez:
        gorunur[i(e)] = False
        P[i(e)] = np.nan
    return Iskelet3B(
        tanim=tanim, noktalar=P, gorunur=gorunur,
        goren_kamera=np.where(gorunur, 4, 0),
        artik_px=np.where(gorunur, 0.0, np.nan),
    )
