"""2B poz veri sozlesmesi ve koordinat donusumleri.

`HAZIR-TEKNOLOJILER.md` madde 2 ve 3'un kodda karsiligi. Hazir bir poz modeli
(MediaPipe, RTMPose, Roboflow) takildiginda cikti bu sozlesmeye cevrilir;
boylece `pose3d` (duzenek) ve `mono` (telefon) hatlari model degisse bile ayni
seyi karsilastirir.

Iki sessiz hata kaynagi burada kapatiliyor:

1. **Eklem sirasi.** Modeller farkli sayida ve sirada eklem dondurur. Poz,
   eklem tanimini **yaninda tasir**; indeks esitligi hicbir yerde varsayilmaz
   (`pose3d.iskelet.eslestir`).
2. **Goruntu olcegi.** Modeller genelde kucultulmus/letterbox'lanmis goruntuyle
   calisir. Kalibrasyon ise **ozgun piksel koordinatlarini** kullanir. Donusum
   geri alinmazsa hata sessizdir: her sey calisir, sadece sonuclar yanlistir.
   `Donusum.geri` bunu tek yerde yapar ve poz hangi uzayda oldugunu bilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calib.synthetic import Kamera
from pose3d.iskelet import REFERANS_ISKELET, IskeletTanimi


@dataclass(frozen=True)
class Donusum:
    """Olcekleme + oteleme: model uzayi <-> ozgun goruntu uzayi.

    ileri: p_model = olcek * p_ozgun + ofset
    """

    olcek: float
    ofset: np.ndarray             # (2,)

    def ileri(self, noktalar: np.ndarray) -> np.ndarray:
        return np.asarray(noktalar, float).reshape(-1, 2) * self.olcek + self.ofset

    def geri(self, noktalar: np.ndarray) -> np.ndarray:
        """Model uzayindaki noktalari ozgun goruntu koordinatina dondur."""
        if self.olcek == 0:
            raise ValueError("olcek sifir olamaz")
        return (np.asarray(noktalar, float).reshape(-1, 2) - self.ofset) / self.olcek


BIRIM_DONUSUM = Donusum(olcek=1.0, ofset=np.zeros(2))


def letterbox_donusumu(
    ozgun: tuple[int, int],
    hedef: tuple[int, int],
) -> Donusum:
    """En-boy oranini koruyarak sigdirma (letterbox) donusumu.

    Modellerin cogu kare girdi ister ve goruntuyu boyle hazirlar. Donusum
    saklanmazsa eklem noktalari kalibrasyonun uzayina geri dondurulemez.
    """
    og, oy = ozgun
    hg, hy = hedef
    if min(og, oy, hg, hy) <= 0:
        raise ValueError("goruntu boyutlari pozitif olmali")
    olcek = min(hg / og, hy / oy)
    ofset = np.array([(hg - og * olcek) / 2.0, (hy - oy * olcek) / 2.0])
    return Donusum(olcek=olcek, ofset=ofset)


@dataclass
class Poz2B:
    """Tek karede, tek kameradan 2B poz.

    `noktalar` **ozgun goruntu koordinatinda** tutulur -- kalibrasyonun kullandigi
    uzay. Model kucultulmus goruntuyle calistiysa `Donusum.geri` ile cevrilmis
    olmali; `uzay` alani bunu belgeler.
    """

    iskelet: IskeletTanimi
    noktalar: np.ndarray          # (N, 2) piksel
    guven: np.ndarray             # (N,) 0..1
    gorunur: np.ndarray           # (N,) bool -- eksik nokta maskesi
    goruntu_boyutu: tuple[int, int]
    model: str = "bilinmiyor"
    kamera: int | None = None
    kare: int | None = None
    uzay: str = "ozgun"           # "ozgun" | "model"
    ek: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.iskelet)
        self.noktalar = np.asarray(self.noktalar, dtype=np.float64).reshape(-1, 2)
        self.guven = np.asarray(self.guven, dtype=np.float64).ravel()
        self.gorunur = np.asarray(self.gorunur, dtype=bool).ravel()
        for ad, dizi in (("noktalar", self.noktalar), ("guven", self.guven),
                         ("gorunur", self.gorunur)):
            if len(dizi) != n:
                raise ValueError(
                    f"{ad} {len(dizi)} eleman, iskelet '{self.iskelet.ad}' "
                    f"{n} eklem bekliyor -- eklem sirasi esit varsayilamaz")
        if self.uzay not in ("ozgun", "model"):
            raise ValueError("uzay 'ozgun' veya 'model' olmali")

    @property
    def gorunur_oran(self) -> float:
        return float(self.gorunur.mean())

    def ozgun_uzaya(self, donusum: Donusum) -> "Poz2B":
        """Model uzayindaki pozu ozgun goruntu uzayina cevir."""
        if self.uzay == "ozgun":
            return self
        yeni = Poz2B(**{**self.__dict__, "noktalar": donusum.geri(self.noktalar),
                        "uzay": "ozgun"})
        return yeni

    def guven_esikle(self, esik: float) -> "Poz2B":
        """Esigin altindaki eklemleri gorunmez isaretle.

        Nokta silinmez, **maskelenir**: dizi uzunlugu iskeletle ayni kalmali,
        yoksa indeksler kayar.
        """
        yeni_gorunur = self.gorunur & (self.guven >= esik)
        return Poz2B(**{**self.__dict__, "gorunur": yeni_gorunur})


def sentetik_poz(
    kamera: Kamera,
    iskelet_3b: np.ndarray,
    tanim: IskeletTanimi = REFERANS_ISKELET,
    gurultu_px: float = 0.0,
    ortulu: tuple[str, ...] = (),
    seed: int = 0,
    model: str = "sentetik",
    kamera_indeksi: int | None = None,
) -> Poz2B:
    """Bilinen bir 3B iskeleti kameraya izdusurerek 2B poz uret.

    Gercek model takilmadan hattin tamami sinanabilsin diye var. Kadraj disina
    tasan ve `ortulu` listesindeki eklemler gorunmez isaretlenir -- gercek
    ortulmeyi taklit eder.
    """
    rng = np.random.default_rng(seed + 5150)
    n3 = np.asarray(iskelet_3b, dtype=np.float64).reshape(-1, 3)
    if len(n3) != len(tanim):
        raise ValueError(f"3B iskelet {len(n3)} nokta, tanim {len(tanim)} eklem")

    p, _ = cv2.projectPoints(n3, cv2.Rodrigues(kamera.R)[0], kamera.t, kamera.K, None)
    p = p.reshape(-1, 2)
    if gurultu_px:
        p = p + rng.normal(0.0, gurultu_px, p.shape)

    g, y = kamera.boyut
    icinde = (p[:, 0] >= 0) & (p[:, 0] < g) & (p[:, 1] >= 0) & (p[:, 1] < y)
    gorunur = icinde.copy()
    for ad in ortulu:
        gorunur[tanim.indeks(ad)] = False

    # Guven, gercek modellerdeki gibi gorunurlukle iliskili ama ozdes degil.
    guven = np.where(gorunur, rng.uniform(0.7, 1.0, len(tanim)),
                     rng.uniform(0.0, 0.3, len(tanim)))
    return Poz2B(iskelet=tanim, noktalar=p, guven=guven, gorunur=gorunur,
                 goruntu_boyutu=kamera.boyut, model=model,
                 kamera=kamera_indeksi, uzay="ozgun")
