"""Deterministik gorus uzlasmasi; ham skor olasilik kabul edilmez.

Aday secimi ve aykiri ayrimi yalniz geometriyle yapilir. Istege bagli
`agirliklar` yalniz kabul edilen gorusler icindeki son DLT'yi agirliklandirir
(ayni modelin kameralar arasi goreli guveni; Panoptic'te medyan 21,1 -> ~19 mm, 0039).

Iki goruste aykiri kamerayi ayirt edemeyiz. Uc ve fazla goruste en az uc
uyumlu gorus gerekir. Piksel/ray esikleri gelistirme ayaridir, fiziksel kabul degil.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from pose3d.triangulate import (
    Ucgenleme, bozulma_gider, kalibrasyondan_projeksiyonlar, ucgenle,
)


@dataclass(frozen=True)
class SaglamSonuc:
    sonuc: Ucgenleme | None
    kabul: tuple[int, ...]
    red: tuple[int, ...]
    neden: str


def ucgenle_saglam(kalib, gozlemler, *, esik_px=8.0, min_aci_derece=1.0,
                   sigma_px=None, bozulma_giderildi=False, agirliklar=None):
    """Cift adaylar -> tum goruslerde uzlasma -> inlier DLT -> yeniden denetim."""
    if not np.isfinite(esik_px) or esik_px <= 0:
        raise ValueError("esik_px pozitif sonlu olmali")
    if not np.isfinite(min_aci_derece) or not 0 < min_aci_derece < 90:
        raise ValueError("min_aci_derece 0..90 arasinda olmali")
    # `ucgenle` ile ayni girdi sozlesmesi: numpy tamsayilari da kamera indeksidir
    # (bagimsiz dogrulama, 25 Eylul); bool tamsayi sayilmaz.
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
           for k in gozlemler):
        raise ValueError("gecersiz kamera indeksi")
    gozlemler = {int(k): v for k, v in gozlemler.items()}
    keys = tuple(sorted(gozlemler))
    if any(not 0 <= k < len(kalib.Ks) for k in keys):
        raise ValueError("gecersiz kamera indeksi")
    pts = {}
    for k, p in gozlemler.items():
        p = np.asarray(p, float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("gozlem (2,) sonlu olmali")
        pts[k] = p if bozulma_giderildi else bozulma_gider(
            p, kalib.Ks[k], kalib.bozulmalar[k])[0]
    Ps = kalibrasyondan_projeksiyonlar(kalib)
    centers = {k: -np.asarray(kalib.Rs[k]).T @ np.asarray(kalib.Ts[k]).reshape(3)
               for k in keys}

    def inliers(X):
        iyi, hatalar = [], []
        for k in keys:
            q = Ps[k] @ np.append(X, 1)
            if q[2] <= 0:
                continue
            err = float(np.linalg.norm(q[:2] / q[2] - pts[k]))
            if err <= esik_px:
                iyi.append(k)
                hatalar.append(err)
        return tuple(iyi), float(np.mean(hatalar)) if hatalar else np.inf

    adaylar = []
    gereken = 2 if len(keys) == 2 else 3
    for pair in combinations(keys, 2):
        result = ucgenle(kalib, {k: pts[k] for k in pair}, True)
        if not result.gecerli:
            continue
        iyi, err = inliers(result.nokta)
        if len(iyi) >= gereken:
            adaylar.append((-len(iyi), err, iyi))
    if not adaylar:
        return SaglamSonuc(None, (), keys, "yetersiz_gorus_uzlasmasi")
    for _, _, selected in sorted(adaylar):
        w = None if agirliklar is None else {k: agirliklar[k] for k in selected}
        result = ucgenle(kalib, {k: pts[k] for k in selected}, True, sigma_px, w)
        if not result.gecerli:
            continue
        iyi, _ = inliers(result.nokta)
        if not set(selected) <= set(iyi):
            continue
        rays = [result.nokta - centers[k] for k in selected]
        rays = [r / np.linalg.norm(r) for r in rays if np.linalg.norm(r) > 1e-12]
        # Paralel VE antiparalel isinlar derinlik icin kotu kosulludur.
        acilar = [np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))
                  for a, b in combinations(rays, 2)]
        if not acilar or max(acilar) < min_aci_derece:
            continue
        return SaglamSonuc(result, selected, tuple(k for k in keys if k not in selected),
                           "iki_gorus_aykiri_ayrimi_yok" if len(keys) == 2 else "uzlasma")
    return SaglamSonuc(None, (), keys, "geometri_veya_artik_gecersiz")
