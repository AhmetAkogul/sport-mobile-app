"""Tek kamerada cok kisi: herkesi bul, kareler arasi kimlik ver, kisi basina iskelet.

Kestirici (RTMW) her karede kimliksiz TAM_VUCUT pozlari dondurur; kimlik
supervision'in ByteTrack'i ile verilir (hazir arac, yeniden yazilmadi; 0036).
Her kimligin pozlari kendi zaman serisinde (`KisiGecmisi`) birikir; hareket
tanima ve hareket basina dogru/yanlis bu seriden beslenir.

ByteTrack yalniz kutu hareketine bakar: kisiler kesisip uzun sure ortulurse
kimlik degisebilir. Bu sinir olculur (`scripts/deney_coklu_kisi_panoptic.py`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pose3d.pose2d import Poz2B


def poz_kutusu(poz: Poz2B, pay: float = 0.1) -> np.ndarray | None:
    """Pozun kutusu: `ek["kutu"]` varsa o, yoksa gorunur noktalarin kutusu (+pay)."""
    if "kutu" in poz.ek:
        return np.asarray(poz.ek["kutu"], float)
    P = poz.noktalar[poz.gorunur]
    if len(P) < 2:
        return None
    lo, hi = P.min(axis=0), P.max(axis=0)
    d = (hi - lo) * pay
    return np.concatenate([lo - d, hi + d])


def poz_guveni(poz: Poz2B) -> float:
    """Takipci icin tespit guveni: gorunur govde + bas eklemlerinin orani (0..1).

    Tam vucutta eller sayilmaz: el bulunamadiginda oran 21/65 = 0,32'ye iner ve
    ByteTrack'in yeni iz esiginin (0,25 + 0,1) altinda kalip kisi hic takip
    edilmez (Panoptic'te MediaPipe ile 0 eslesme boyle cikti).
    """
    from pose3d.tam_vucut import GRUPLAR, TAM_VUCUT
    g = poz.gorunur
    if poz.iskelet.ad == TAM_VUCUT.ad:
        g = g[list(GRUPLAR["bas"] + GRUPLAR["govde"])]
    return float(g.mean()) if len(g) else 0.0


class ByteTrackSarmal:
    """supervision.ByteTrack -> (kutular, guven, sira) -> (kimlikler, sira)."""

    def __init__(self, *, kare_hizi: float = 30.0, kayip_s: float = 2.0):
        import warnings

        import supervision as sv
        self._sv = sv
        with warnings.catch_warnings():
            # 0.28'de kullanimdan kalkacak diye isaretli; surum 0.30.5'e sabit (0036).
            warnings.simplefilter("ignore", FutureWarning)
            # Dikkat: supervision tamponu `frame_rate / 30 * lost_track_buffer`
            # kare olarak olcekler, yani tampon "30 fps karesi" birimindedir.
            # Saniyeyi dogrudan kare sayisi sanmak 15 fps'te sureyi yariya indirir.
            self._t = sv.ByteTrack(frame_rate=int(round(kare_hizi)),
                                   lost_track_buffer=int(round(kayip_s * 30)),
                                   minimum_consecutive_frames=1)

    def __call__(self, kutular, guven, sira):
        d = self._sv.Detections(xyxy=kutular, confidence=guven,
                                class_id=np.zeros(len(sira), int), data={"sira": sira})
        r = self._t.update_with_detections(d)
        # Bos karede supervision `data`yi dusurur; takipci yine de cagrilir ki
        # kayip sayaclari ilerlesin.
        return r.tracker_id, r.data.get("sira", np.zeros(0, int))


class KisiTakip:
    """Kimliksiz poz listesi -> [(kimlik, poz)]; kimlik 1'den baslar.

    `tracker(kutular (n,4), guven (n,), sira (n,)) -> (kimlikler, sira)`;
    verilmezse supervision ByteTrack (`ByteTrackSarmal`).
    """

    def __init__(self, *, kare_hizi: float = 30.0, kayip_s: float = 2.0, tracker=None):
        self._tracker = tracker or ByteTrackSarmal(kare_hizi=kare_hizi, kayip_s=kayip_s)

    def guncelle(self, pozlar: list[Poz2B]) -> list[tuple[int, Poz2B]]:
        kutular, secilen = [], []
        for i, p in enumerate(pozlar):
            k = poz_kutusu(p)
            if k is not None and np.isfinite(k).all():
                kutular.append(k)
                secilen.append(i)
        n = len(secilen)
        kimlik, sira = self._tracker(np.asarray(kutular, float).reshape(n, 4),
                                     np.array([poz_guveni(pozlar[i]) for i in secilen], float),
                                     np.asarray(secilen, int))
        return [(int(t), pozlar[int(s)]) for t, s in zip(kimlik, sira)]


@dataclass
class KisiGecmisi:
    """Bir kimligin son `uzunluk` karedeki zaman damgali pozlari."""

    kimlik: int
    uzunluk: int = 300
    zaman: deque = field(default_factory=deque)
    pozlar: deque = field(default_factory=deque)

    def ekle(self, t: float, poz: Poz2B) -> None:
        self.zaman.append(float(t))
        self.pozlar.append(poz)
        while len(self.zaman) > self.uzunluk:
            self.zaman.popleft()
            self.pozlar.popleft()

    def dizi(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(T,) zaman, (T, N, 2) nokta (eksik NaN), (T, N) gorunurluk."""
        if not self.pozlar:
            return np.zeros(0), np.zeros((0, 0, 2)), np.zeros((0, 0), bool)
        return (np.array(self.zaman), np.stack([p.noktalar for p in self.pozlar]),
                np.stack([p.gorunur for p in self.pozlar]))


class CokKisiHatti:
    """Takip + kisi basina gecmis; kare basina bir `adim` cagrisi.

    `unut_s` saniyedir gorulmeyen kimligin gecmisi silinir.
    """

    def __init__(self, takip: KisiTakip, *, uzunluk: int = 300, unut_s: float = 5.0):
        self.takip = takip
        self.uzunluk = uzunluk
        self.unut_s = unut_s
        self.kisiler: dict[int, KisiGecmisi] = {}
        self._son: dict[int, float] = {}

    def adim(self, t: float, pozlar: list[Poz2B]) -> list[tuple[int, Poz2B]]:
        eslesen = self.takip.guncelle(pozlar)
        for kimlik, poz in eslesen:
            g = self.kisiler.setdefault(kimlik, KisiGecmisi(kimlik, self.uzunluk))
            g.ekle(t, poz)
            self._son[kimlik] = t
        for kimlik in [k for k, s in self._son.items() if t - s > self.unut_s]:
            del self._son[kimlik]
            del self.kisiler[kimlik]
        return eslesen


# Kimlik renkleri (BGR); kimlik % len ile dolanir.
KISI_RENKLERI = ((80, 200, 255), (120, 255, 120), (255, 160, 80), (200, 120, 255),
                 (80, 255, 255), (255, 255, 120), (160, 160, 255), (120, 200, 160))


def kisi_rengi(kimlik: int) -> tuple[int, int, int]:
    return KISI_RENKLERI[(kimlik - 1) % len(KISI_RENKLERI)]
