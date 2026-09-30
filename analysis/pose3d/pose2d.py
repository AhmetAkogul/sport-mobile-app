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

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from calib.synthetic import Kamera
from pose3d.iskelet import REFERANS_ISKELET, IskeletTanimi


def _pikseller(noktalar, ad: str = "noktalar") -> np.ndarray:
    """(..., 2) piksel dizisi dondur; sessiz boyut uydurmayi engelle.

    `reshape(-1, 2)` tek basina tehlikelidir: (N, 4) girdiyi (2N, 2) yapar ve
    hata vermez. Son ekseni denetlemek bu sessiz bozulmayi kapatir.
    """
    dizi = np.asarray(noktalar, dtype=np.float64)
    if dizi.ndim == 0 or dizi.shape[-1] != 2:
        raise ValueError(f"{ad} son ekseni 2 olmali, {dizi.shape} geldi")
    return dizi.reshape(-1, 2)


@dataclass(frozen=True)
class Donusum:
    """Olcekleme + oteleme: model uzayi <-> ozgun goruntu uzayi.

    ileri: p_model = olcek * p_ozgun + ofset
    """

    olcek: float
    ofset: np.ndarray             # (2,)

    def __post_init__(self) -> None:
        if not np.isfinite(self.olcek) or self.olcek == 0.0:
            raise ValueError(
                f"olcek sonlu ve sifirdan farkli olmali, {self.olcek} geldi")
        ofset = np.asarray(self.ofset, dtype=np.float64).ravel()
        if ofset.shape != (2,):
            raise ValueError(f"ofset (2,) olmali, {ofset.shape} geldi")
        object.__setattr__(self, "ofset", ofset)

    def ileri(self, noktalar: np.ndarray) -> np.ndarray:
        return _pikseller(noktalar) * self.olcek + self.ofset

    def geri(self, noktalar: np.ndarray) -> np.ndarray:
        """Model uzayindaki noktalari ozgun goruntu koordinatina dondur."""
        return (_pikseller(noktalar) - self.ofset) / self.olcek


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


@dataclass(frozen=True)
class Poz2B:
    """Tek karede, tek kameradan 2B poz.

    `noktalar` **ozgun goruntu koordinatinda** tutulur -- kalibrasyonun kullandigi
    uzay. Model kucultulmus goruntuyle calistiysa `Donusum.geri` ile cevrilmis
    olmali; `uzay` alani bunu belgeler.

    Sinif **frozen**'dir: `uzay` gibi sozlesme alanlari kurulustan sonra
    degistirilemez. Yeni bir poz gerekiyorsa `dataclasses.replace` kullanilir.
    Bu, `uzay="bozuk"` gibi kurulum denetimini atlayan mutasyonlari kapatir.

    Eksik veri politikasi (projenin tek politikasi): **eksik nokta NaN'dir ve
    `gorunur=False`'dur**; 0.0 gibi gecerli gorunen bir degere cevrilmez.
    """

    iskelet: IskeletTanimi
    noktalar: np.ndarray          # (N, 2) piksel; eksik eklem NaN
    guven: np.ndarray             # (N,) 0..1
    gorunur: np.ndarray           # (N,) bool -- eksik nokta maskesi
    tespit: bool                  # karede hic poz bulundu mu (0 nokta != "hepsi eksik")
    goruntu_boyutu: tuple[int, int]
    model: str = "bilinmiyor"
    kamera: int | None = None
    kare: int | None = None
    uzay: str = "ozgun"           # "ozgun" | "model"
    ek: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Diziler normalize edilirken **yeni** dizi uretilir: kisinin elinde
        # tuttugu `poz.noktalar` referansi replace() sonrasi baska bir diziyi
        # gosterir. Referans kimligine guvenilmemeli, dizi kopyalanmali.
        n = len(self.iskelet)
        object.__setattr__(self, "noktalar", _pikseller(self.noktalar))
        object.__setattr__(self, "guven",
                           np.asarray(self.guven, dtype=np.float64).ravel())
        object.__setattr__(self, "gorunur",
                           np.asarray(self.gorunur, dtype=bool).ravel())
        for ad, dizi in (("noktalar", self.noktalar), ("guven", self.guven),
                         ("gorunur", self.gorunur)):
            if len(dizi) != n:
                raise ValueError(
                    f"{ad} {len(dizi)} eleman, iskelet '{self.iskelet.ad}' "
                    f"{n} eklem bekliyor -- eklem sirasi esit varsayilamaz")
        if self.uzay not in ("ozgun", "model"):
            raise ValueError("uzay 'ozgun' veya 'model' olmali")
        if type(self.tespit) is not bool:
            raise ValueError(f"tespit bool olmali, {type(self.tespit).__name__} geldi")
        if not self.tespit:
            if self.gorunur.any():
                raise ValueError(
                    "tespit=False iken gorunur eklem olamaz: karede poz bulunamadiysa "
                    "butun eklemler gorunmez olmali")
            if np.isfinite(self.noktalar).any():
                raise ValueError(
                    "tespit=False iken koordinat tasinamaz: tespit yoksa nokta da "
                    "yoktur, 0.0 gibi gecerli gorunen bir deger yazilmaz (NaN beklenir)")

    @property
    def gorunur_oran(self) -> float:
        return float(self.gorunur.mean())

    def ozgun_uzaya(self, donusum: Donusum) -> "Poz2B":
        """Model uzayindaki pozu ozgun goruntu uzayina cevir."""
        if self.uzay == "ozgun":
            return self
        return replace(self, noktalar=donusum.geri(self.noktalar), uzay="ozgun",
                       ek=dict(self.ek))

    def guven_esikle(self, esik: float) -> "Poz2B":
        """Esigin altindaki eklemleri gorunmez isaretle.

        Nokta silinmez, **maskelenir**: dizi uzunlugu iskeletle ayni kalmali,
        yoksa indeksler kayar.
        """
        yeni_gorunur = self.gorunur & (self.guven >= esik)
        return replace(self, gorunur=yeni_gorunur, ek=dict(self.ek))


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
    tasan, **kamera arkasinda kalan** ve `ortulu` listesindeki eklemler gorunmez
    isaretlenir -- gercek ortulmeyi taklit eder.
    """
    rng = np.random.default_rng(seed + 5150)
    n3 = np.asarray(iskelet_3b, dtype=np.float64).reshape(-1, 3)
    if len(n3) != len(tanim):
        raise ValueError(f"3B iskelet {len(n3)} nokta, tanim {len(tanim)} eklem")
    if gurultu_px < 0:
        raise ValueError(f"gurultu_px negatif olamaz: {gurultu_px}")

    p, _ = cv2.projectPoints(n3, cv2.Rodrigues(kamera.R)[0], kamera.t, kamera.K, None)
    p = p.reshape(-1, 2)
    if gurultu_px:
        p = p + rng.normal(0.0, gurultu_px, p.shape)

    # Kamera arkasindaki nokta da "projekte edilir" ve sonuc genelde kadrajin
    # ICINE duser (1 m arkadaki nokta tam merkeze gelir). Yalnizca kadraj
    # filtresi kullanilirsa bu cop veri "goruldu" sayilir. Z>0 sart.
    p_cam = (np.asarray(kamera.R, float) @ n3.T + np.asarray(kamera.t, float).reshape(3, 1)).T
    ileride = p_cam[:, 2] > 0

    g, y = kamera.boyut
    icinde = (p[:, 0] >= 0) & (p[:, 0] < g) & (p[:, 1] >= 0) & (p[:, 1] < y)
    gorunur = icinde & ileride
    for ad in ortulu:
        gorunur[tanim.indeks(ad)] = False
    # Kamera arkasindaki projeksiyon anlamsizdir (isaret degisir): koordinat da
    # yazilmaz.
    p = np.where(ileride[:, None], p, np.nan)
    # "Tespit" = kisi kadrajda: en az bir eklem kameranin onunde ve kadraj icinde.
    # Hepsi ortulu olsa da kisi kadrajdadir (tespit var, gorunur eklem yok).
    # Kisi kadraj disindaysa koordinat da tasinmaz: "tespit yoksa nokta yok".
    tespit = bool((icinde & ileride).any())
    if not tespit:
        p = np.full_like(p, np.nan)

    # Guven, gercek modellerdeki gibi gorunurlukle iliskili ama ozdes degil.
    guven = np.where(gorunur, rng.uniform(0.7, 1.0, len(tanim)),
                     rng.uniform(0.0, 0.3, len(tanim)))
    return Poz2B(iskelet=tanim, noktalar=p, guven=guven, gorunur=gorunur,
                 tespit=tespit, goruntu_boyutu=kamera.boyut, model=model,
                 kamera=kamera_indeksi, uzay="ozgun")
