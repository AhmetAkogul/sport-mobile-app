"""Iskelet tanimi ve coklu gorusten 3B iskelet kestirimi.

Iki modul arasindaki sozlesme burada: `pose3d` (duzenek hatti) ve `mono`
(telefon hatti) **ayni eklem tanimini** kullanmak zorunda, yoksa karsilastirma
anlamsizlasir. HAZIR-TEKNOLOJILER.md madde 2 bunu zaten sart kosuyor:
"MediaPipe ve baska modellerin nokta sayisi/sirasi dogrudan esit kabul edilmez."

Bu yuzden eklem listesi **isimle** tasinir, indeksle degil. Model degistiginde
`eslestir` cagrilir; sessizce yanlis eklemi karsilastirmak yerine eksik eklem
acikca isaretlenir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calib.multiview import Kalibrasyon
from pose3d.triangulate import Ucgenleme, ucgenle


@dataclass(frozen=True)
class IskeletTanimi:
    """Eklem isimleri ve aralarindaki baglantilar.

    Isimler sozlesmenin kendisidir. Iki modul ayni `ad`i tasiyan tanimi
    kullaniyorsa karsilastirilabilir; tasimiyorsa once `eslestir` gerekir.
    """

    ad: str
    eklemler: tuple[str, ...]
    baglantilar: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(set(self.eklemler)) != len(self.eklemler):
            raise ValueError("eklem isimleri benzersiz olmali")
        bilinmeyen = {e for cift in self.baglantilar for e in cift} - set(self.eklemler)
        if bilinmeyen:
            raise ValueError(f"baglantida bilinmeyen eklem: {sorted(bilinmeyen)}")
        gorulen: set[frozenset[str]] = set()
        for a, b in self.baglantilar:
            if a == b:
                raise ValueError(f"baglanti eklemin kendisine olamaz: {a}")
            anahtar = frozenset((a, b))
            if anahtar in gorulen:
                raise ValueError(f"baglanti tekrarli: {a}-{b}")
            gorulen.add(anahtar)

    def __len__(self) -> int:
        return len(self.eklemler)

    def indeks(self, eklem: str) -> int:
        try:
            return self.eklemler.index(eklem)
        except ValueError as e:
            raise KeyError(f"'{eklem}' bu iskelette yok: {self.ad}") from e


# Projenin referans iskeleti. Egzersiz secimi (squat, sinav, lunge) diz valgusu,
# kalca/bel hizasi ve govde rotasyonu olcecegi icin kalca-diz-ayak bilegi ve
# omuz-kalca zinciri sart. Yuz noktalari yok: KVKK acisindan gereksiz veri.
REFERANS_ISKELET = IskeletTanimi(
    ad="bitirme-13",
    eklemler=(
        "boyun",
        "sag_omuz", "sag_dirsek", "sag_bilek",
        "sol_omuz", "sol_dirsek", "sol_bilek",
        "sag_kalca", "sag_diz", "sag_ayak_bilegi",
        "sol_kalca", "sol_diz", "sol_ayak_bilegi",
    ),
    baglantilar=(
        ("boyun", "sag_omuz"), ("sag_omuz", "sag_dirsek"), ("sag_dirsek", "sag_bilek"),
        ("boyun", "sol_omuz"), ("sol_omuz", "sol_dirsek"), ("sol_dirsek", "sol_bilek"),
        ("sag_omuz", "sag_kalca"), ("sag_kalca", "sag_diz"), ("sag_diz", "sag_ayak_bilegi"),
        ("sol_omuz", "sol_kalca"), ("sol_kalca", "sol_diz"), ("sol_diz", "sol_ayak_bilegi"),
        ("sag_kalca", "sol_kalca"), ("sag_omuz", "sol_omuz"),
    ),
)


@dataclass
class Iskelet3B:
    """3B eklem konumlari. Gorulemeyen eklem NaN ve `gorunur=False`.

    Iki alan farkli sorulari cevaplar ve karistirilmamalidir:

    - `goren_kamera[j]` -- `j` eklemini kac kamera **gordu** (triangulation
      denenmeden once). Ucgenleme sonradan basarisiz olabilir (kotu kosullanma),
      o zaman goren yuksek ama `gorunur[j]=False` olur.
    - `gorunur[j]` -- `j` eklemi icin **guvenilir** bir 3B nokta uretildi mi.
    """

    tanim: IskeletTanimi
    noktalar: np.ndarray          # (N, 3) metre, kamera 0 koordinatinda
    gorunur: np.ndarray           # (N,) bool
    goren_kamera: np.ndarray      # (N,) int -- eklem basina kac kamera gordu
    artik_px: np.ndarray          # (N,) yeniden izdusum artigi
    ek: dict = field(default_factory=dict)   # hat/uretici kaynakli ek bilgi

    def __post_init__(self) -> None:
        n = len(self.tanim)
        noktalar = np.asarray(self.noktalar, dtype=np.float64)
        if noktalar.shape != (n, 3):
            raise ValueError(
                f"noktalar {noktalar.shape}, ({n}, 3) bekleniyor -- 2B bir dizi "
                "3B iskelet yerine gecmez")
        gorunur = np.asarray(self.gorunur)
        if gorunur.shape != (n,) or gorunur.dtype != bool:
            raise ValueError(
                f"gorunur {gorunur.shape}/{gorunur.dtype}, ({n},) bool bekleniyor")
        for ad, dizi in (("goren_kamera", self.goren_kamera), ("artik_px", self.artik_px)):
            if np.asarray(dizi).shape != (n,):
                raise ValueError(f"{ad} {np.asarray(dizi).shape}, ({n},) bekleniyor")
        sonlu = np.isfinite(noktalar).all(axis=1)
        eksik = gorunur & ~sonlu
        if eksik.any():
            raise ValueError(
                "gorunur ama sonlu olmayan eklem: "
                f"{[self.tanim.eklemler[i] for i in np.flatnonzero(eksik)]}")
        if not isinstance(self.ek, dict):
            raise ValueError(f"ek sozluk olmali, {type(self.ek).__name__} geldi")
        fazla = ~gorunur & sonlu
        if fazla.any():
            raise ValueError(
                "gorunmez eklem deger tasiyor (NaN beklenir): "
                f"{[self.tanim.eklemler[i] for i in np.flatnonzero(fazla)]}")
        object.__setattr__(self, "noktalar", noktalar)

    def al(self, eklem: str) -> np.ndarray:
        return self.noktalar[self.tanim.indeks(eklem)]

    def uzunluk(self, a: str, b: str) -> float:
        """Iki eklem arasi mesafe (metre). Biri gorunmuyorsa NaN."""
        ia, ib = self.tanim.indeks(a), self.tanim.indeks(b)
        if not (self.gorunur[ia] and self.gorunur[ib]):
            return float("nan")
        return float(np.linalg.norm(self.noktalar[ia] - self.noktalar[ib]))

    @property
    def gorunur_oran(self) -> float:
        return float(self.gorunur.mean())


def eslestir(
    kaynak: IskeletTanimi,
    hedef: IskeletTanimi,
    esleme: dict[str, str] | None = None,
) -> np.ndarray:
    """hedef eklemleri icin kaynak indeksleri; eslesmeyen eklem -1.

    `esleme` verilmezse isim esitligi kullanilir. Donen dizi ile yeniden
    siralama yapilir; -1 olan eklemler **eksik** sayilir, uydurulmaz.
    """
    esleme = esleme or {}
    # {isim: indeks} bir kez kurulur: `eklemler.index()` her eklem icin O(n) tarama.
    kaynak_indeks = {ad: i for i, ad in enumerate(kaynak.eklemler)}
    cikti = np.full(len(hedef), -1, dtype=int)
    for i, ad in enumerate(hedef.eklemler):
        kaynak_ad = esleme.get(ad, ad)
        if kaynak_ad in kaynak_indeks:
            cikti[i] = kaynak_indeks[kaynak_ad]
    return cikti


def iskelet_ucgenle(
    kalib: Kalibrasyon,
    gozlemler: dict[int, "object"],
    tanim: IskeletTanimi = REFERANS_ISKELET,
    min_gorus: int = 2,
    bozulma_giderildi: bool = False,
) -> Iskelet3B:
    """Kamera basina 2B pozdan 3B iskelet.

    `gozlemler`: {kamera_indeksi: Poz2B}. Her eklem **bagimsiz** olarak, yalnizca
    o eklemi goren kameralardan ucgenlenir; bir kamera bir eklemi ortulme
    yuzunden gormuyorsa digerleri isi surdurur. Iki gorusun altina dusen eklem
    uydurulmaz, gorunmez isaretlenir.

    `ucgenle`'nin `ValueError`'u **yutulmaz**: buraya gelmeden once en az iki
    gorus sarti zaten denetlendi, dolayisiyla oradan gelen hata artik "bu eklem
    olculemedi" degil, gercek bir sozlesme/kalibrasyon sorunudur ve sessizce
    yutulursa kaynagi gizlenir (dis inceleme I.2).
    """
    n = len(tanim)
    noktalar = np.full((n, 3), np.nan)
    gorunur = np.zeros(n, dtype=bool)
    goren = np.zeros(n, dtype=int)
    artik = np.full(n, np.nan)

    # Her kameranin pozu referans iskelete tasinir; indeks esitligi varsayilmaz.
    yerlesim: dict[int, tuple[object, np.ndarray]] = {}
    for kamera, poz in gozlemler.items():
        yerlesim[kamera] = (poz, eslestir(poz.iskelet, tanim))

    for j in range(n):
        eklem_gozlem: dict[int, np.ndarray] = {}
        for kamera, (poz, harita) in yerlesim.items():
            k = harita[j]
            if k < 0 or not poz.gorunur[k]:
                continue
            eklem_gozlem[kamera] = poz.noktalar[k]
        goren[j] = len(eklem_gozlem)
        if len(eklem_gozlem) < max(2, min_gorus):
            continue
        sonuc: Ucgenleme = ucgenle(kalib, eklem_gozlem, bozulma_giderildi)
        if not sonuc.gecerli:
            continue
        noktalar[j] = sonuc.nokta
        artik[j] = sonuc.artik_px
        gorunur[j] = True

    return Iskelet3B(tanim=tanim, noktalar=noktalar, gorunur=gorunur,
                     goren_kamera=goren, artik_px=artik)
