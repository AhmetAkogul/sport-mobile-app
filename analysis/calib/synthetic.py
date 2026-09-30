"""Sentetik coklu kamera sahnesi: kalibrasyon hattini kamera gelmeden dogrula.

Neden bu modul var (KOD-PLANI.md Adim 2):

Gercek kamerayla baslarsaniz kotu bir kalibrasyon sonucunun sebebini bilemezsiniz --
yazilim hatasi mi, fiziksel duzenek mi, tespit gurultusu mu? Burada gercek parametreleri
biz koyuyoruz, dolayisiyla `calibrateMultiview`'in onlari geri bulup bulmadigi
tartisilmaz bir olcuttur. Ayrica gurultuyu kontrollu artirip parametre sacilimini
olcebiliriz; belirsizlik analizinin dogrulugu ancak beklenen cevabi bildiginiz yerde
sinanabilir.

Onemli bulgu: kalibrasyonun kalitesini belirleyen sey kodun dogrulugu degil, **board
poz cesitliligi**. Ayni gurultuyle, dar poz dagiliminda odak uzakligi sacilimi
geniş dagilimdakinin katlarina cikar. Bu, veri toplama protokolunu belirleyen bir
sonuc; `POZ_DAGILIMLARI` iki ucu temsil eder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calib.board import BoardSpec, board_kur


@dataclass(frozen=True)
class Kamera:
    """Gercek (yer gercegi) kamera parametreleri.

    R ve t **dunya -> kamera** donusumudur: x_kam = R @ x_dunya + t
    """

    K: np.ndarray            # (3,3)
    R: np.ndarray            # (3,3)
    t: np.ndarray            # (3,1)
    boyut: tuple[int, int]   # (genislik, yukseklik) piksel

    def __post_init__(self) -> None:
        # Dis inceleme B.5.3: yer gercegi kamerasi bozuksa butun sentetik
        # deney sessizce yanlis olur; kurulumda durulur.
        K, R = np.asarray(self.K, float), np.asarray(self.R, float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"K (3,3), sonlu ve pozitif odakli olmali, {K.shape} geldi")
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-6) \
                or np.linalg.det(R) < 0:
            raise ValueError("R bir donme matrisi olmali (dik, det=+1)")
        if np.asarray(self.t, float).size != 3 or not np.isfinite(self.t).all():
            raise ValueError("t 3 elemanli ve sonlu olmali")
        if len(self.boyut) != 2 or min(self.boyut) <= 0:
            raise ValueError(f"boyut (genislik, yukseklik) pozitif olmali, {self.boyut} geldi")

    @property
    def merkez(self) -> np.ndarray:
        """Kameranin dunya koordinatindaki konumu."""
        return (-self.R.T @ self.t).ravel()


@dataclass(frozen=True)
class PozDagilimi:
    """Board'un sahnede nasil gezdirildigi -- kalibrasyonun kosullanmasi buna bagli."""

    ad: str
    egim_derece: float       # iki eksende tepe egim genligi
    mesafe_m: tuple[float, float]
    yan_kayma_m: float

    def pozlar(self, n_kare: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        cikti = []
        for _ in range(n_kare):
            eksen = rng.normal(size=3)
            eksen /= np.linalg.norm(eksen)
            aci = np.deg2rad(rng.uniform(-self.egim_derece, self.egim_derece))
            R, _ = cv2.Rodrigues((eksen * aci).reshape(3, 1))
            t = np.array([
                [rng.uniform(-self.yan_kayma_m, self.yan_kayma_m)],
                [rng.uniform(-self.yan_kayma_m, self.yan_kayma_m) * 0.5],
                [rng.uniform(*self.mesafe_m) - np.mean(self.mesafe_m)],
            ])
            cikti.append((R, t))
        return cikti


# Iki uc: tembel bir cekim oturumu ve protokole uygun bir oturum.
POZ_DAGILIMLARI = {
    "dar": PozDagilimi("dar", egim_derece=8.0, mesafe_m=(2.4, 2.6), yan_kayma_m=0.05),
    "genis": PozDagilimi("genis", egim_derece=35.0, mesafe_m=(1.5, 3.2), yan_kayma_m=0.45),
}


@dataclass
class Sahne:
    """calibrateMultiview'e hazir sentetik veri + yer gercegi."""

    obj_noktalari: list[np.ndarray]
    img_noktalari: list[list[np.ndarray]]
    mask: np.ndarray
    kameralar: list[Kamera] = field(default_factory=list)
    gurultu_px: float = 0.0

    @property
    def n_kamera(self) -> int:
        return len(self.kameralar)

    @property
    def gorunurluk(self) -> float:
        """Kamera-kare ciftlerinin kaci board'u gordu."""
        return float(self.mask.mean()) if self.mask.size else 0.0


def _bakis_matrisi(merkez: np.ndarray, hedef: np.ndarray) -> np.ndarray:
    """merkez'den hedef'e bakan dunya->kamera rotasyonu."""
    ileri = hedef - merkez
    ileri = ileri / np.linalg.norm(ileri)
    yukari_kaba = np.array([0.0, 1.0, 0.0])
    if abs(float(ileri @ yukari_kaba)) > 0.99:       # tepeden bakis -- kilitlenmeyi onle
        yukari_kaba = np.array([0.0, 0.0, 1.0])
    sag = np.cross(yukari_kaba, ileri)
    sag /= np.linalg.norm(sag)
    yukari = np.cross(ileri, sag)
    return np.stack([sag, yukari, ileri])


def rig_yay(
    acilar_derece: list[float] | tuple[float, ...] = (-40, -14, 14, 40),
    yaricap_m: float = 2.6,
    yukseklik_m: float = 0.5,
    odak_px: float = 900.0,
    boyut: tuple[int, int] = (1280, 720),
) -> list[Kamera]:
    """Merkeze bakan bir yay uzerinde kameralar. Kalici kurulumun basit modeli.

    `yaricap_m` **yatay** (XZ duzlemi) yaricaptir; `yukseklik_m` Y'de eklenir.
    Dolayisiyla kameranin orijine uzakligi sqrt(yaricap^2 + yukseklik^2) olur.
    """
    K = np.array([[odak_px, 0.0, boyut[0] / 2],
                  [0.0, odak_px, boyut[1] / 2],
                  [0.0, 0.0, 1.0]])
    kameralar = []
    for a in acilar_derece:
        r = np.deg2rad(a)
        merkez = np.array([yaricap_m * np.sin(r), yukseklik_m, -yaricap_m * np.cos(r)])
        R = _bakis_matrisi(merkez, np.zeros(3))
        kameralar.append(Kamera(K=K.copy(), R=R, t=(-R @ merkez).reshape(3, 1), boyut=boyut))
    return kameralar


def sahne_uret(
    spec: BoardSpec = BoardSpec(),
    kameralar: list[Kamera] | None = None,
    n_kare: int = 20,
    gurultu_px: float = 0.0,
    dagilim: str | PozDagilimi = "genis",
    seed: int = 0,
) -> Sahne:
    """Board'u sahnede gezdirip her kameraya izdusur.

    Kadraj disina tasan kare o kamera icin "gormedi" sayilir ve maskte 0 olur --
    gercek bir oturumdaki kismi gorunurlugu taklit eder.
    """
    # Dis inceleme B.5.2: bos sahne ve negatif gurultu reddedilir.
    if type(n_kare) is not int or n_kare < 1:
        raise ValueError(f"n_kare pozitif tamsayi olmali, {n_kare!r} geldi")
    if not np.isfinite(gurultu_px) or gurultu_px < 0:
        raise ValueError(f"gurultu_px sonlu ve negatif olmayan olmali, {gurultu_px} geldi")
    kameralar = kameralar or rig_yay()
    dag = POZ_DAGILIMLARI[dagilim] if isinstance(dagilim, str) else dagilim
    model = board_kur(spec).getChessboardCorners().astype(np.float64)
    rng = np.random.default_rng(seed + 9973)

    obj_noktalari: list[np.ndarray] = []
    img_noktalari: list[list[np.ndarray]] = [[] for _ in kameralar]
    mask = np.zeros((len(kameralar), n_kare), dtype=np.uint8)

    for kare, (Rb, tb) in enumerate(dag.pozlar(n_kare, seed)):
        obj_noktalari.append(model.astype(np.float32))
        for ci, kam in enumerate(kameralar):
            R = kam.R @ Rb
            t = kam.R @ tb + kam.t
            # Kamera uzayindaki derinlik: yalnizca onde kalan noktalar gorulur.
            # `cv2.projectPoints` arkadaki noktalari da izdusurur ve sonuc cogu
            # zaman kadrajin **icine** duser (1 m arkadaki nokta tam merkeze).
            # Yalnizca kadraj bakan bir suzgec bunlari "gorulmus" sayar ve
            # sentetik veriye sessizce cop girer. Dis inceleme B.5.1.
            z_cam = (model @ R.T + t.reshape(3))[:, 2]
            p, _ = cv2.projectPoints(model, cv2.Rodrigues(R)[0], t, kam.K, None)
            p = p.reshape(-1, 2)
            if gurultu_px:
                p = p + rng.normal(0.0, gurultu_px, p.shape)
            g, y = kam.boyut
            icinde = (p[:, 0] >= 0) & (p[:, 0] < g) & (p[:, 1] >= 0) & (p[:, 1] < y)
            if icinde.all() and (z_cam > 0).all():
                img_noktalari[ci].append(p.reshape(-1, 1, 2).astype(np.float32))
                mask[ci, kare] = 1
            else:
                # Hiza korunmali: gormeyen kamera icin bos dizi konur, atlanmaz.
                img_noktalari[ci].append(np.empty((0, 1, 2), dtype=np.float32))

    return Sahne(obj_noktalari, img_noktalari, mask, kameralar, gurultu_px)


def dunyadan_kamera0(kameralar: list[Kamera], noktalar: np.ndarray) -> np.ndarray:
    """Dunya koordinatindaki noktalari kamera 0 sistemine tasi.

    Kalibrasyon sonuclari kamera 0 referanslidir (`calib.multiview`), dolayisiyla
    triangulation ciktisi da oradadir. Yer gercegiyle karsilastirmak icin gercek
    noktalari ayni sisteme tasimak gerekir; yoksa sabit bir donusum farki hata
    gibi gorunur.
    """
    k0 = kameralar[0]
    n = np.asarray(noktalar, dtype=np.float64).reshape(-1, 3)
    return (k0.R @ n.T + k0.t).T


def olculu_cubuk(
    uzunluk_m: float = 1.0,
    n_isaret: int = 5,
    merkez: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yon: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Uzerinde olculu isaretler olan sert cubuk -- bilinen mesafe duzenegi.

    `PROJE-PLANI.md` Faz 2: sistemin olctugu mesafe ile gercek mesafe farki 3B
    dogrulugun kanitidir ve serit metreyle dogrulanabilir olmasi juri demosunun
    temelidir. Burada onun sentetik esdegeri uretilir.

    Dondurur: (3B isaret noktalari, (i, j, gercek_mesafe_m) ucluleri)
    """
    if n_isaret < 2:
        raise ValueError("cubukta en az 2 isaret olmali")
    y = np.asarray(yon, dtype=np.float64)
    y = y / np.linalg.norm(y)
    m = np.asarray(merkez, dtype=np.float64)
    adim = uzunluk_m / (n_isaret - 1)
    noktalar = np.array([m + y * (i * adim - uzunluk_m / 2) for i in range(n_isaret)])
    mesafeler = [(i, j, float(abs(j - i) * adim))
                 for i in range(n_isaret) for j in range(i + 1, n_isaret)]
    return noktalar, mesafeler


def nokta_gozlemleri(
    kameralar: list[Kamera],
    noktalar: np.ndarray,
    gurultu_px: float = 0.0,
    seed: int = 0,
) -> list[dict[int, np.ndarray]]:
    """Her 3B noktayi her kameraya izdusur.

    Kadraj disi kalan **ve kamera arkasinda kalan** kamera sozlukte yok.
    Ikincisi sessiz bir tuzakti: arkadaki nokta da izdusurulur ve genelde
    kadrajin icine duser (bkz. B.5.1).
    """
    rng = np.random.default_rng(seed + 31337)
    n = np.asarray(noktalar, dtype=np.float64).reshape(-1, 3)
    cikti: list[dict[int, np.ndarray]] = []
    for X in n:
        gozlem: dict[int, np.ndarray] = {}
        for ci, kam in enumerate(kameralar):
            # Kamera arkasindaki nokta kadraja dusse bile gorulmez (B.5.1).
            if float((kam.R @ X + kam.t.reshape(3))[2]) <= 0.0:
                continue
            p, _ = cv2.projectPoints(X.reshape(1, 3), cv2.Rodrigues(kam.R)[0],
                                     kam.t, kam.K, None)
            p = p.reshape(2)
            if gurultu_px:
                p = p + rng.normal(0.0, gurultu_px, 2)
            g, y = kam.boyut
            if 0 <= p[0] < g and 0 <= p[1] < y:
                gozlem[ci] = p
        cikti.append(gozlem)
    return cikti
