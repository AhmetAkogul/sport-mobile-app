"""Coklu gorusten 3B nokta kestirimi (triangulation).

Kalibrasyon "kamera nerede" sorusunu cevaplar; triangulation "nokta nerede"
sorusunu. Duzenegin yer gercegi urettigi yer burasi: `PROJE-PLANI.md` Kapi 2
olcutu, 2 m mesafede bilinen bir nesnede 3B konum hatasinin 10 mm'nin altinda
olmasini istiyor.

Yontem: DLT (Direct Linear Transform). Her kamera bir noktaya iki dogrusal
kisit koyar; N kameradan gelen 2N kisit en kucuk kareler ile cozulur. Iki kamera
yeterlidir ama daha fazlasi hem gurultuyu bastirir hem de bir kamera noktayi
gormedigi (ortulme) durumda hatti ayakta tutar.

Onemli: DLT dogrusal pinhole modeli varsayar. Bozulma (distortion) **once
giderilmeli**, aksi halde hata sessizce sizar. `ucgenle` bunu kendisi yapar;
dogrudan `_dlt` cagiran olursa sorumluluk onundur.

OpenCV'nin `triangulatePoints` fonksiyonu yalnizca **iki** goruse calisir; coklu
kamera duzenegimizin butun kameralarini kullanabilmek icin DLT burada yazildi.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calib.multiview import Kalibrasyon


def projeksiyon_matrisi(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = K [R | t], (3,4)."""
    return np.asarray(K, float) @ np.hstack([np.asarray(R, float).reshape(3, 3),
                                             np.asarray(t, float).reshape(3, 1)])


def kalibrasyondan_projeksiyonlar(kalib: Kalibrasyon) -> list[np.ndarray]:
    """Kalibrasyon sonucundan kamera basina projeksiyon matrisi.

    Rs/Ts kamera 0'a gore oldugu icin cikan 3B noktalar da **kamera 0
    koordinat sisteminde** olur. Olcek board'un fiziksel kare kenarindan gelir,
    yani metre.
    """
    return [projeksiyon_matrisi(K, R, T)
            for K, R, T in zip(kalib.Ks, kalib.Rs, kalib.Ts)]


def bozulma_gider(noktalar: np.ndarray, K: np.ndarray, bozulma: np.ndarray) -> np.ndarray:
    """Piksel noktalarini bozulmadan arindirip yine piksel uzayina dondur."""
    n = np.asarray(noktalar, dtype=np.float64).reshape(-1, 1, 2)
    duz = cv2.undistortPoints(n, np.asarray(K, float), np.asarray(bozulma, float), P=K)
    return duz.reshape(-1, 2)


def _dlt(projeksiyonlar: list[np.ndarray], noktalar: np.ndarray,
         agirliklar: np.ndarray | None = None) -> np.ndarray:
    """Tek nokta icin DLT. noktalar: (M,2), projeksiyonlar: M adet (3,4).

    `agirliklar` (M,): kamera basina goreli agirlik (satirlar normlandiktan sonra
    carpilir); None esit agirlik.
    """
    satirlar = []
    for P, (x, y) in zip(projeksiyonlar, noktalar):
        satirlar.append(x * P[2] - P[0])
        satirlar.append(y * P[2] - P[1])
    A = np.asarray(satirlar, dtype=np.float64)
    # Kosullanma icin olcekle: satir normlari cok farkliysa SVD sayisal olarak bozulur.
    normlar = np.linalg.norm(A, axis=1, keepdims=True)
    normlar[normlar == 0] = 1.0
    A = A / normlar
    if agirliklar is not None:
        A = A * np.repeat(np.asarray(agirliklar, dtype=np.float64), 2)[:, None]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    # SVD'nin sag tekil vektoru birim normludur (|X| = 1), dolayisiyla X[3]
    # sonlu bir nokta icin O(1) buyukluktedir ve esik birimden bagimsizdir
    # (dis inceleme T.2). 1e-12 yalnizca gercekten sonsuzdaki (paralel isin)
    # durumu yakalar; kotu kosullanma ayrica `gecerli` ile elenir.
    if abs(X[3]) < 1e-12:                      # sonsuzdaki nokta -- kesisme yok
        return np.full(3, np.nan)
    return X[:3] / X[3]


@dataclass
class Ucgenleme:
    """Bir noktanin 3B kestirimi ve ne kadar guvenilir oldugu."""

    nokta: np.ndarray            # (3,) kamera 0 koordinatinda, metre
    goren_kamera: int
    artik_px: float              # ortalama yeniden izdusum artigi
    kovaryans: np.ndarray | None = None   # (3,3) m^2; `sigma_px` verilirse dolar

    def __post_init__(self) -> None:
        """Sekil denetimi ve sonsuz artigin NaN'a cevrilmesi (T.3, T.6).

        `artik_px` sonsuz olabiliyordu: kamera duzlemine dusen bir nokta icin
        inf ekleniyor ve ortalamayi da inf yapiyordu. inf bir sayi gibi esikle
        karsilastirilinca "cok kotu olcum" diye okunur; dogrusu "hesaplanamadi",
        yani NaN.
        """
        self.nokta = np.asarray(self.nokta, dtype=np.float64).reshape(-1)
        if self.nokta.shape != (3,):
            raise ValueError(f"nokta (3,) olmali, {self.nokta.shape} geldi")
        if not np.isfinite(self.artik_px):
            self.artik_px = float("nan")
        if self.kovaryans is not None:
            k = np.asarray(self.kovaryans, dtype=np.float64)
            if k.shape != (3, 3):
                raise ValueError(f"kovaryans (3,3) olmali, {k.shape} geldi")
            self.kovaryans = k

    @property
    def gecerli(self) -> bool:
        return bool(np.isfinite(self.nokta).all())

    @property
    def sigma_m(self) -> float:
        """Kovaryansin izotropik karsiligi: sqrt(iz/3), metre.

        Ucgenleme hatasi **anizotropiktir** -- derinlik yonu yanal yonlerden
        belirgin sekilde kotudur. Tek skalara indirgemek o bilgiyi atar; buradaki
        secim, ayni toplam varyansi tasiyan izotropik dagilimin sigmasidir
        (3*sigma^2 = iz). Yone duyarli hesap gerekiyorsa `kovaryans` kullanilmali.
        """
        if self.kovaryans is None:
            return float("nan")
        return float(np.sqrt(np.trace(self.kovaryans) / 3.0))


def _izdusum_jacobiani(P: np.ndarray, X: np.ndarray) -> np.ndarray | None:
    """Piksel izdusumunun 3B noktaya gore turevi, (2,3).

    p = (P*Xh)[:2] / (P*Xh)[2] oldugundan bolum kuralindan:
        dp/dX = (P[:2,:3] - p * P[2,:3]) / w2
    Kamera arkasinda kalan ya da bolenin sifira gittigi nokta icin None.
    """
    w = P @ np.append(np.asarray(X, float), 1.0)
    if abs(w[2]) < 1e-12:
        return None
    p = w[:2] / w[2]
    return (P[:2, :3] - np.outer(p, P[2, :3])) / w[2]


def nokta_kovaryansi(
    projeksiyonlar: list[np.ndarray], X: np.ndarray, sigma_px: float
) -> np.ndarray | None:
    """Birinci derece kovaryans kestirimi: Cov = sigma_px^2 * (J^T J)^-1.

    Gozlem gurultusu kameralar arasi bagimsiz ve her eksende `sigma_px` kabul
    edilir. Bu, tespit gurultusunun **bilindigi** varsayimidir; artiklardan
    kestirmek iki goruste 1 serbestlik derecesi biraktigi icin guvenilmez, o
    yuzden disaridan verilir.

    Birim (dis inceleme T.4): `sigma_px`, `projeksiyonlar`'in olculdugu goruntu
    uzayinda piksel cinsindendir. Projeksiyonlar K iceriyor (P = K[R|t]) ve
    bozulma giderilmis piksel koordinatlariyla calisir; disaridan undistort
    edilmis nokta verilirse gurultu de **o** uzayda ifade edilmelidir (bozulma
    kucuk oldugu surece fark ihmal edilebilir, buyuk bozulmada sigma kenarlarda
    buyur).

    Geometri kotu kosullandiginda (kameralar neredeyse ayni dogrultuda, nokta
    taban cizgisi uzerinde) J^T J tekillesir ve None doner -- buyuk ama uydurma
    bir sayi uretmek yerine "bilinmiyor" demek dogrusu.
    """
    if sigma_px <= 0:
        raise ValueError("sigma_px pozitif olmali")
    bloklar = []
    for P in projeksiyonlar:
        J = _izdusum_jacobiani(P, X)
        if J is None:
            return None
        bloklar.append(J)
    if len(bloklar) < 2:
        return None
    J = np.vstack(bloklar)
    N = J.T @ J
    if not np.isfinite(N).all() or np.linalg.cond(N) > 1e12:
        return None
    return float(sigma_px) ** 2 * np.linalg.inv(N)


def ucgenle(
    kalib: Kalibrasyon,
    gozlemler: dict[int, np.ndarray],
    bozulma_giderildi: bool = False,
    sigma_px: float | None = None,
    agirliklar: dict[int, float] | None = None,
) -> Ucgenleme:
    """Bir 3B noktayi, onu goren kameralarin 2B gozlemlerinden kestir.

    `agirliklar`: {kamera: goreli agirlik} (or. dedektor ham skorunun karesi, 0039);
    olasilik degil, ayni modelin kameralar arasi goreli guveni. None esit.

    `gozlemler`: {kamera_indeksi: (2,) piksel}. En az iki kamera gerekir --
    tek kameradan derinlik cikmaz, bu projenin tum tezi de zaten bu.

    `sigma_px` verilirse sonuca (3,3) kovaryans eklenir. Varsayilan None:
    hesaplanmaz ve `kovaryans` bos kalir -- eski cagiranlar etkilenmez.
    """
    if len(gozlemler) < 2:
        raise ValueError(
            f"triangulation icin en az 2 gorus gerekli, {len(gozlemler)} verildi")
    # Sinir kontrolu basta: aksi halde `kalib.Ks[i]` IndexError firlatir ve
    # `iskelet_ucgenle`'nin ValueError suzgeci onu yakalamaz -- hat coker (T.1).
    n_kamera = len(kalib.Ks)
    if any(not 0 <= i < n_kamera for i in gozlemler):
        raise ValueError(
            f"kamera indeksi gecersiz: {sorted(gozlemler)}, {n_kamera} kamera var")

    indeksler = sorted(gozlemler)
    P_hepsi = kalibrasyondan_projeksiyonlar(kalib)
    Ps, noktalar = [], []
    for i in indeksler:
        p = np.asarray(gozlemler[i], dtype=np.float64).ravel()
        if p.shape != (2,):
            raise ValueError(f"kamera {i} gozlemi (2,) olmali, {p.shape} geldi")
        if not bozulma_giderildi:
            p = bozulma_gider(p, kalib.Ks[i], kalib.bozulmalar[i])[0]
        Ps.append(P_hepsi[i])
        noktalar.append(p)

    w = None if agirliklar is None else np.array([float(agirliklar[i]) for i in indeksler])
    if w is not None and (not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError("agirliklar pozitif sonlu olmali")
    X = _dlt(Ps, np.asarray(noktalar), w)
    if not np.isfinite(X).all():
        return Ucgenleme(nokta=X, goren_kamera=len(indeksler), artik_px=float("nan"))

    Xh = np.append(X, 1.0)
    artiklar = []
    for P, p in zip(Ps, noktalar):
        izd = P @ Xh
        if abs(izd[2]) < 1e-12:
            artiklar.append(np.inf)
            continue
        artiklar.append(float(np.linalg.norm(izd[:2] / izd[2] - p)))
    kov = None if sigma_px is None else nokta_kovaryansi(Ps, X, sigma_px)
    return Ucgenleme(nokta=X, goren_kamera=len(indeksler),
                     artik_px=float(np.mean(artiklar)), kovaryans=kov)


def ucgenle_toplu(
    kalib: Kalibrasyon,
    gozlem_listesi: list[dict[int, np.ndarray]],
    bozulma_giderildi: bool = False,
) -> list[Ucgenleme]:
    """Birden cok nokta icin triangulation. Gormeyen kamera sozlukte bulunmaz."""
    return [ucgenle(kalib, g, bozulma_giderildi) for g in gozlem_listesi]


def mesafe(a: Ucgenleme | np.ndarray, b: Ucgenleme | np.ndarray) -> float:
    """Iki 3B nokta arasi Oklid mesafesi (metre)."""
    pa = a.nokta if isinstance(a, Ucgenleme) else np.asarray(a, float)
    pb = b.nokta if isinstance(b, Ucgenleme) else np.asarray(b, float)
    return float(np.linalg.norm(pa - pb))


def bilinen_mesafe_hatasi(
    kestirimler: list[Ucgenleme],
    gercek_mesafeler: list[tuple[int, int, float]],
) -> dict:
    """Bilinen mesafe duzenegi degerlendirmesi -- juri demosunun temeli.

    `gercek_mesafeler`: (i, j, metre) uclulari; uzerinde olculu isaretler olan
    sert bir cubuk/levhadan gelir ve serit metreyle dogrulanabilir. Sistemin
    olctugu mesafe ile gercek mesafe farki, 3B dogrulugun dogrudan kanitidir.
    """
    if not gercek_mesafeler:
        raise ValueError("en az bir bilinen mesafe gerekli")
    hatalar_mm = []
    satirlar = []
    for i, j, gercek_m in gercek_mesafeler:
        olculen = mesafe(kestirimler[i], kestirimler[j])
        hata_mm = (olculen - gercek_m) * 1000.0
        hatalar_mm.append(abs(hata_mm))
        satirlar.append({"i": i, "j": j, "gercek_mm": gercek_m * 1000.0,
                         "olculen_mm": olculen * 1000.0, "hata_mm": hata_mm})
    h = np.asarray(hatalar_mm)
    return {
        "n": len(h),
        "ortalama_mutlak_hata_mm": float(h.mean()),
        "rms_hata_mm": float(np.sqrt((h ** 2).mean())),
        "en_buyuk_hata_mm": float(h.max()),
        "kapi_2_gecti": bool(h.max() < 10.0),      # PROJE-PLANI.md Kapi 2 olcutu
        "olcumler": satirlar,
    }
