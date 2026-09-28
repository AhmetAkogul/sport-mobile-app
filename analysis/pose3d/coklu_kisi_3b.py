"""Coklu kamerada cok kisi (3B): kameralar arasi kisi eslestirme, ucgenleme, 3B kimlik.

Adimlar (0039):

1. Her kamerada kisi basina tam vucut 2B poz (RTMW, `mono.rtmw_model`).
2. **Kameralar arasi eslestirme** (`kisileri_esle`): iki kameradaki iki poz ayni
   kisiyse eklemleri birbirinin epipolar dogrusuna yakin duser. Maliyet,
   ortak gorunen govde eklemlerinin simetrik epipolar uzakliginin medyanidir
   (piksel). Butun ciftler maliyete gore siralanir, birlesim-bul ile kumelenir;
   bir kumede her kameradan en fazla bir poz olur ve kumenin butun ciftleri
   esigin altinda kalmalidir. Bu, cok gorunumlu poz kestirimindeki yerlesik
   yontemin (epipolar yakinlik + kumeleme; Dong ve ark. 2019, MVPose) yalin
   bir surumudur; yeni yontem degildir.
3. Kume basina `iskelet_ucgenle` (saglam mod) -> TAM_VUCUT 3B (eller, ayaklar dahil).
4. **3B kimlik** (`Takip3B`): kareler arasi pelvis mesafesiyle Macar eslemesi.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from calib.multiview import Kalibrasyon
from pose3d.iskelet import Iskelet3B, iskelet_ucgenle
from pose3d.pose2d import Poz2B
from pose3d.tam_vucut import GRUPLAR, TAM_VUCUT
from pose3d.triangulate import bozulma_gider

GOVDE = list(GRUPLAR["bas"] + GRUPLAR["govde"])


def _capraz(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float).ravel()
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def temel_matris(kalib: Kalibrasyon, i: int, j: int) -> np.ndarray:
    """F_ij: kamera i'deki (bozulmasi giderilmis) x_i icin l_j = F_ij x_i.

    Rs/Ts kamera 0'a gore: X_k = R_k X_0 + T_k. Goreli: R = R_j R_i^T,
    t = T_j - R T_i; E = [t]x R; F = K_j^-T E K_i^-1.
    """
    R = kalib.Rs[j] @ kalib.Rs[i].T
    t = np.asarray(kalib.Ts[j], float).reshape(3) - R @ np.asarray(kalib.Ts[i], float).reshape(3)
    E = _capraz(t) @ R
    return np.linalg.inv(kalib.Ks[j]).T @ E @ np.linalg.inv(kalib.Ks[i])


def _duzelt(kalib, k, P):
    ok = np.isfinite(P).all(axis=1)
    out = np.full(P.shape, np.nan)
    if ok.any():
        out[ok] = bozulma_gider(P[ok], kalib.Ks[k], kalib.bozulmalar[k])
    return out


def epipolar_maliyet(F: np.ndarray, a: np.ndarray, b: np.ndarray, en_az: int = 4) -> float:
    """Ortak gorunen noktalarin simetrik epipolar uzakliginin medyani (px).

    `a`, `b`: (N, 2) bozulmasi giderilmis piksel; eksik NaN. Ortak nokta
    `en_az`'dan azsa inf.
    """
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if ok.sum() < en_az:
        return float("inf")
    xa = np.c_[a[ok], np.ones(ok.sum())]
    xb = np.c_[b[ok], np.ones(ok.sum())]
    lb = xa @ F.T                                  # b'deki dogrular
    la = xb @ F                                    # a'daki dogrular
    db = np.abs(np.sum(lb * xb, 1)) / np.linalg.norm(lb[:, :2], axis=1)
    da = np.abs(np.sum(la * xa, 1)) / np.linalg.norm(la[:, :2], axis=1)
    return float(np.median((da + db) / 2))


def kisileri_esle(kalib: Kalibrasyon, kameralar: dict[int, list[Poz2B]],
                  esik_px: float = 15.0) -> list[dict[int, int]]:
    """{kamera: [poz, ...]} -> kumeler [{kamera: poz sirasi}], her kume >= 2 kamera.

    Maliyet `esik_px`'in altindaki ciftler artan maliyetle birlestirilir; iki
    kume ancak ortak kamerasi yoksa ve aralarindaki butun ciftler esigin
    altindaysa birlesir (tek bir iyi cift iki farkli kisiyi baglayamaz).
    """
    duz = {k: [_duzelt(kalib, k, p.noktalar[GOVDE]) for p in ps] for k, ps in kameralar.items()}
    kam = sorted(duz)
    maliyet: dict[tuple, float] = {}
    for x, i in enumerate(kam):
        for j in kam[x + 1:]:
            F = temel_matris(kalib, i, j)
            for a, pa in enumerate(duz[i]):
                for b, pb in enumerate(duz[j]):
                    c = epipolar_maliyet(F, pa, pb)
                    if c < esik_px:
                        maliyet[((i, a), (j, b))] = maliyet[((j, b), (i, a))] = c
    kume = {(k, a): frozenset({(k, a)}) for k in kam for a in range(len(duz[k]))}
    for (u, v), _ in sorted(maliyet.items(), key=lambda kv: kv[1]):
        ku, kv = kume[u], kume[v]
        if ku == kv:
            continue
        if {k for k, _ in ku} & {k for k, _ in kv}:
            continue
        if all((p, q) in maliyet for p in ku for q in kv):
            birlesik = ku | kv
            for e in birlesik:
                kume[e] = birlesik
    return [dict(sorted(s)) for s in set(kume.values()) if len(s) >= 2]


def kisileri_ucgenle(kalib: Kalibrasyon, kameralar: dict[int, list[Poz2B]],
                     kumeler: list[dict[int, int]], *, saglam: bool = True,
                     esik_px: float = 8.0) -> list[Iskelet3B]:
    """Kume basina TAM_VUCUT 3B iskelet (kamera 0 cercevesi, kalibrasyon birimi)."""
    return [iskelet_ucgenle(kalib, {k: kameralar[k][a] for k, a in kume.items()}, TAM_VUCUT,
                            saglam=saglam, esik_px=esik_px)
            for kume in kumeler]


def yakin_kumeleri_birlestir(kalib: Kalibrasyon, kameralar: dict[int, list[Poz2B]],
                             kumeler: list[dict[int, int]], iskeletler: list[Iskelet3B], *,
                             esik_m: float = 0.2, **ucgen) -> tuple[list, list]:
    """Ayni kisinin iki kumeye bolunmesini onarir.

    Epipolar kumeleme "butun ciftler esigin altinda" ister; tek bir kamerada
    kotu kestirilmis poz bir kisiyi iki kumeye bolebilir (Panoptic'te ayni
    kisiden pelvisleri 2 cm arayla iki iskelet goruldu). Ortak kamerasi olmayan
    ve ortak govde eklemlerinin medyan uzakligi `esik_m`'den kucuk iki kume
    birlestirilip yeniden ucgenlenir.
    """
    kumeler, iskeletler = list(kumeler), list(iskeletler)
    degisti = True
    while degisti:
        degisti = False
        for a in range(len(kumeler)):
            for b in range(a + 1, len(kumeler)):
                if set(kumeler[a]) & set(kumeler[b]):
                    continue
                d = np.linalg.norm(iskeletler[a].noktalar[GOVDE] - iskeletler[b].noktalar[GOVDE],
                                   axis=1)
                if np.isfinite(d).sum() >= 3 and np.nanmedian(d) < esik_m:
                    kume = {**kumeler[a], **kumeler[b]}
                    isk = kisileri_ucgenle(kalib, kameralar, [kume], **ucgen)[0]
                    kumeler = [k for i, k in enumerate(kumeler) if i not in (a, b)] + [kume]
                    iskeletler = [s for i, s in enumerate(iskeletler) if i not in (a, b)] + [isk]
                    degisti = True
                    break
            if degisti:
                break
    return kumeler, iskeletler


def pelvis(isk: Iskelet3B) -> np.ndarray:
    """Kalca ortasi; kalcalar yoksa gorunur govde eklemlerinin merkezi (takip kopmasin:
    saglam ucgenleme kalcayi atinca Panoptic'te kimlik degisimi 39'a cikiyordu)."""
    i, j = TAM_VUCUT.indeks("sol_kalca"), TAM_VUCUT.indeks("sag_kalca")
    P = isk.noktalar[[i, j]]
    P = P[np.isfinite(P).all(1)]
    if len(P):
        return P.mean(0)
    G = isk.noktalar[GOVDE]
    G = G[np.isfinite(G).all(1)]
    return G.mean(0) if len(G) else np.full(3, np.nan)


class Takip3B:
    """3B iskeletlere kareler arasi kimlik: pelvis mesafesiyle Macar eslemesi.

    `kapi_m`: bu mesafeden uzak eslesme yapilmaz; `kayip_kare` kare gorulmeyen
    kimlik biter. Kimlik 1'den baslar.
    """

    def __init__(self, kapi_m: float = 0.5, kayip_kare: int = 30):
        self.kapi_m, self.kayip_kare = kapi_m, kayip_kare
        self._iz: dict[int, tuple[np.ndarray, int]] = {}      # kimlik -> (pelvis, son kare)
        self._sonraki, self._kare = 1, 0

    def guncelle(self, iskeletler: list[Iskelet3B]) -> list[tuple[int, Iskelet3B]]:
        self._kare += 1
        self._iz = {k: v for k, v in self._iz.items() if self._kare - v[1] <= self.kayip_kare}
        P = [pelvis(s) for s in iskeletler]
        kim = list(self._iz)
        C = np.full((len(P), len(kim)), 1e9)
        for a, p in enumerate(P):
            for b, k in enumerate(kim):
                if np.isfinite(p).all():
                    d = float(np.linalg.norm(p - self._iz[k][0]))
                    if d <= self.kapi_m:
                        C[a, b] = d
        atama = {}
        if len(P) and len(kim):
            for a, b in zip(*linear_sum_assignment(C)):
                if C[a, b] < 1e9:
                    atama[a] = kim[b]
        out = []
        for a, s in enumerate(iskeletler):
            if a not in atama:
                atama[a] = self._sonraki
                self._sonraki += 1
            if np.isfinite(P[a]).all():
                self._iz[atama[a]] = (P[a], self._kare)
            out.append((atama[a], s))
        return out
