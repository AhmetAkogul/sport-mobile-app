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
            p, _ = cv2.projectPoints(model, cv2.Rodrigues(R)[0], t, kam.K, None)
            p = p.reshape(-1, 2)
            if gurultu_px:
                p = p + rng.normal(0.0, gurultu_px, p.shape)
            g, y = kam.boyut
            icinde = (p[:, 0] >= 0) & (p[:, 0] < g) & (p[:, 1] >= 0) & (p[:, 1] < y)
            if icinde.all():
                img_noktalari[ci].append(p.reshape(-1, 1, 2).astype(np.float32))
                mask[ci, kare] = 1
            else:
                # Hiza korunmali: gormeyen kamera icin bos dizi konur, atlanmaz.
                img_noktalari[ci].append(np.empty((0, 1, 2), dtype=np.float32))

    return Sahne(obj_noktalari, img_noktalari, mask, kameralar, gurultu_px)
