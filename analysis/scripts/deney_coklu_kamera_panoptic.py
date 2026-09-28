"""0039: coklu kamerada cok kisi (3B) -- Panoptic 160224_haggling1, 4 HD kamera, 3 kisi.

1. `--tespit`: her kamerada RTMW (kisi basina TAM_VUCUT 2B); sonuc
   `out/coklu_kamera/tespit_<ilk>_<n>_<adim>.npz` onbellegine yazilir.
2. Degerlendirme: kameralar arasi epipolar eslestirme -> saglam ucgenleme ->
   3B kimlik; Panoptic'in kimlikli 3B iskeletine (kamera 0 cercevesi, metre)
   karsi govde 17 eklem MPJPE (mm), tespit orani (MPJPE < 250 mm), fazla kisi,
   3B kimlik degisimi. Rijit hizalama yok: mutlak konum.

    PYTHONPATH=. <mediapipe-env>/bin/python scripts/deney_coklu_kamera_panoptic.py --tespit
    .venv/bin/python scripts/deney_coklu_kamera_panoptic.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from pose3d.coklu_kisi_3b import (Takip3B, kisileri_esle, kisileri_ucgenle,  # noqa: E402
                                  yakin_kumeleri_birlestir)
from pose3d.pose2d import Poz2B  # noqa: E402
from pose3d.tam_vucut import GRUPLAR, TAM_VUCUT  # noqa: E402
from veri.panoptic import (COCO19_TAM_VUCUT, dunyadan_kamera0, kalibrasyon_oku,  # noqa: E402
                           kimlikli_iskeletler_oku)

KOK = Path("data/dis/cmu_panoptic/160224_haggling1")
KAMERALAR = ["00_00", "00_01", "00_02", "00_03"]
GOVDE = list(GRUPLAR["bas"] + GRUPLAR["govde"])
ELLER = list(GRUPLAR["sol_el"] + GRUPLAR["sag_el"])
ESLESME_MM = 250.0


def _onbellek(a):
    return Path("out/coklu_kamera") / f"tespit_{a.baslangic}_{a.kare}_{a.adim}.npz"


def tespit(a):
    import cv2

    from mono.rtmw_model import RTMWEstimator
    kareler = [a.baslangic + i * a.adim for i in range(a.kare)]
    kayit = {}
    with RTMWEstimator("../bitirme-capture/data/models/rtmpose/detector.onnx",
                       "data/dis/rtmw_x_l_384/acik/end2end.onnx") as m:
        for ci, ad in enumerate(KAMERALAR):
            cap = cv2.VideoCapture(str(KOK / "hdVideos" / f"hd_{ad}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, kareler[0])
            onceki = kareler[0] - 1
            for k in kareler:
                while onceki < k - 1:
                    cap.grab()
                    onceki += 1
                ok, im = cap.read()
                onceki = k
                if not ok:
                    break
                for pi, p in enumerate(m.kisiler(im)):
                    kayit[f"{k}_{ci}_{pi}"] = np.c_[p.noktalar, p.gorunur, p.guven]
            print(ad, "bitti", flush=True)
    _onbellek(a).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_onbellek(a), **kayit)


def _kameralar_kare(d, k, boyut=(1920, 1080)):
    out = {ci: [] for ci in range(len(KAMERALAR))}
    for ci in range(len(KAMERALAR)):
        pi = 0
        while f"{k}_{ci}_{pi}" in d:
            v = d[f"{k}_{ci}_{pi}"]
            out[ci].append(Poz2B(TAM_VUCUT, v[:, :2], v[:, 3], v[:, 2] > 0.5, True, boyut,
                                 model="rtmw"))
            pi += 1
    return out


def degerlendir(a):
    d = np.load(_onbellek(a))
    pk = kalibrasyon_oku(KOK / "calibration_160224_haggling1.json", KAMERALAR)
    kareler = [a.baslangic + i * a.adim for i in range(a.kare)]
    etiket = kimlikli_iskeletler_oku(KOK / "hdPose3d_stage1_coco19.tar", kareler)
    takip = Takip3B(kapi_m=0.5, kayip_kare=int(2 * 30 / a.adim))
    gt_n = eslesen = fazla = 0
    hatalar, el_orani, kam_sayisi = [], [], []
    iz = defaultdict(list)
    for k in kareler:
        kameralar = _kameralar_kare(d, k)
        kumeler = kisileri_esle(pk.kalib, kameralar, esik_px=a.esik_px)
        ucgen = {"saglam": a.artik_px > 0, "esik_px": a.artik_px}
        isk = kisileri_ucgenle(pk.kalib, kameralar, kumeler, **ucgen)
        if a.birlestir:
            kumeler, isk = yakin_kumeleri_birlestir(pk.kalib, kameralar, kumeler, isk, **ucgen)
        isk = [s for s in isk if s.gorunur[GOVDE].sum() >= 8]
        kam_sayisi += [len(c) for c in kumeler]
        kimlikli = takip.guncelle(isk)
        gts = []
        for kid, j in etiket.get(k, []):
            g = j[list(COCO19_TAM_VUCUT)]
            ok = g[:, 3] >= 0.1
            if ok.sum() >= 8:
                P = dunyadan_kamera0(g[:, :3], pk)
                P[~ok] = np.nan
                gts.append((kid, P))
        gt_n += len(gts)
        if not gts or not kimlikli:
            fazla += len(kimlikli)
            continue
        C = np.full((len(gts), len(kimlikli)), 1e9)
        for i, (_, P) in enumerate(gts):
            for j, (_, s) in enumerate(kimlikli):
                e = np.linalg.norm(s.noktalar[GOVDE] - P, axis=1) * 1000
                if np.isfinite(e).sum() >= 6:
                    C[i, j] = np.nanmean(e)
        es = 0
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < ESLESME_MM:
                es += 1
                kid, P = gts[i]
                tid, s = kimlikli[j]
                iz[kid].append(tid)
                e = np.linalg.norm(s.noktalar[GOVDE] - P, axis=1) * 1000
                hatalar += e[np.isfinite(e)].tolist()
                el_orani.append(float(s.gorunur[ELLER].mean()))
        eslesen += es
        fazla += len(kimlikli) - es
    h = np.array(hatalar)
    degisim = sum(sum(1 for x, y in zip(v[:-1], v[1:]) if x != y) for v in iz.values())
    sonuc = {
        "kareler": {"ilk": kareler[0], "n": len(kareler), "adim": a.adim},
        "kamera": KAMERALAR, "epipolar_esik_px": a.esik_px, "artik_esik_px": a.artik_px,
        "birlestir": a.birlestir,
        "etiketli_kisi_kare": gt_n, "tespit_orani": round(eslesen / max(gt_n, 1), 3),
        "fazla_kisi_kare_basina": round(fazla / len(kareler), 3),
        "govde_mpjpe_mm": {"medyan": round(float(np.median(h)), 1),
                           "ortalama": round(float(np.mean(h)), 1),
                           "p90": round(float(np.percentile(h, 90)), 1)} if len(h) else None,
        "pck_100mm": round(float(np.mean(h < 100)), 3) if len(h) else None,
        "kume_kamera_sayisi": {str(k): v for k, v in Counter(kam_sayisi).items()},
        "el_eklemi_ucgenlenen_oran": round(float(np.mean(el_orani)), 3) if el_orani else None,
        "kimlik_degisimi_3b": degisim,
        "kimlik_safligi_3b": {str(k): round(Counter(v).most_common(1)[0][1] / len(v), 3)
                              for k, v in iz.items()},
    }
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))
    Path("out/coklu_kamera/sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tespit", action="store_true", help="2B tespiti calistir (model ortami)")
    p.add_argument("--baslangic", type=int, default=3000)
    p.add_argument("--kare", type=int, default=300)
    p.add_argument("--adim", type=int, default=2)
    p.add_argument("--esik-px", type=float, default=15.0, help="epipolar eslestirme esigi")
    p.add_argument("--artik-px", type=float, default=40.0,
                   help="saglam ucgenleme artik esigi; 0 saglam modu kapatir")
    p.add_argument("--birlestir", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    if a.tespit:
        tespit(a)
    else:
        degerlendir(a)


if __name__ == "__main__":
    main()
