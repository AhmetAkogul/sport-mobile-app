"""0036: tek kamerada cok kisi -- tespit, eklem hatasi ve kimlik surekliligi.

CMU Panoptic 160224_haggling1 (3 kisi ayni anda, kimlikli 3B etiket). Etiket
her HD kameraya izdusurulur; her karede tahminler etiketle Macar eslemesiyle
eslestirilir (eklem hatasi / etiket kutusu kosegeni < ESLESME).

Olculen:
- tespit orani: goruntude >= 8 eklemi gorunen etiketli kisilerin eslesen payi
- fazla tespit: eslesmeyen tahmin sayisi / kare
- govde eklem hatasi (px ve etiket kutusu kosegenine oran)
- kimlik degisimi: bir etiketli kisinin eslestigi takip kimliginin degismesi
- kimlik safligi: eslesen karelerde en sik takip kimliginin payi

    PYTHONPATH=. <mediapipe-env>/bin/python scripts/deney_coklu_kisi_panoptic.py \\
        --kamera 00_00 --baslangic 3000 --kare 600 --adim 2 [--kestirici rtmw|mediapipe]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mono.coklu_kisi import KisiTakip
from pose3d.tam_vucut import GRUPLAR
from veri.panoptic import COCO19_TAM_VUCUT, izdusur, kamera_oku, kimlikli_iskeletler_oku

KOK = Path("data/dis/cmu_panoptic/160224_haggling1")
ESLESME = 0.15            # eklem hatasi / etiket kutusu kosegeni
EN_AZ_EKLEM = 8
GOVDE = list(GRUPLAR["bas"]) + list(GRUPLAR["govde"])


def etiket_2b(j19: np.ndarray, kamera: dict) -> np.ndarray:
    """(19,4) metre -> TAM_VUCUT govde sirasinda (17,2) piksel; gorunmeyen NaN."""
    w, h = kamera["resolution"]
    px = izdusur(j19[:, :3], kamera)[list(COCO19_TAM_VUCUT)]
    gecerli = (j19[list(COCO19_TAM_VUCUT), 3] >= 0.1) & (px[:, 0] >= 0) & (px[:, 0] < w) \
        & (px[:, 1] >= 0) & (px[:, 1] < h)
    px[~gecerli] = np.nan
    return px


def kestirici_kur(ad: str):
    if ad == "rtmw":
        from mono.rtmw_model import RTMWEstimator
        m = RTMWEstimator("../bitirme-capture/data/models/rtmpose/detector.onnx",
                          "data/dis/rtmw_x_l_384/acik/end2end.onnx")
        return m, m.kisiler
    from mono.mediapipe_model import MediaPipeEstimator
    m = MediaPipeEstimator("../bitirme-capture/data/models/pose_landmarker_full-float16-v1.task",
                           kisi_sayisi=6)
    return m, m.kisiler


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kamera", default="00_00")
    p.add_argument("--baslangic", type=int, default=3000)
    p.add_argument("--kare", type=int, default=600, help="islenen kare sayisi")
    p.add_argument("--adim", type=int, default=2, help="kac karede bir (30 fps / adim)")
    p.add_argument("--kestirici", default="rtmw", choices=("rtmw", "mediapipe"))
    p.add_argument("--kayip-s", type=float, default=2.0,
                   help="takipcinin kayip kisiyi bekledigi sure (s)")
    p.add_argument("--cikti", type=Path)
    a = p.parse_args(argv)

    kamera = kamera_oku(KOK / "calibration_160224_haggling1.json", a.kamera)
    kareler = [a.baslangic + i * a.adim for i in range(a.kare)]
    etiket = kimlikli_iskeletler_oku(KOK / "hdPose3d_stage1_coco19.tar", kareler)
    cap = cv2.VideoCapture(str(KOK / "hdVideos" / f"hd_{a.kamera}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, kareler[0])
    model, kestir = kestirici_kur(a.kestirici)
    takip = KisiTakip(kare_hizi=30.0 / a.adim, kayip_s=a.kayip_s)

    gt_sayi = eslesen = fazla = islenen = 0
    hatalar, oranlar = [], []
    iz: dict[int, list[int]] = defaultdict(list)
    iz_kare: dict[int, list[int]] = defaultdict(list)
    etiket_kare: dict[int, list[int]] = defaultdict(list)
    onceki = kareler[0] - 1
    for k in kareler:
        while onceki < k - 1:           # atlanan kareler: grab, seek her karede pahali
            cap.grab()
            onceki += 1
        ok, im = cap.read()
        onceki = k
        if not ok:
            break
        islenen += 1
        tahmin = takip.guncelle(kestir(im))
        gts = []
        for kid, j in etiket.get(k, []):
            px = etiket_2b(j, kamera)
            if np.isfinite(px).all(axis=1).sum() >= EN_AZ_EKLEM:
                lo, hi = np.nanmin(px, 0), np.nanmax(px, 0)
                gts.append((kid, px, float(np.linalg.norm(hi - lo))))
                etiket_kare[kid].append(k)
        gt_sayi += len(gts)
        if not gts or not tahmin:
            fazla += len(tahmin)
            continue
        C = np.full((len(gts), len(tahmin)), 1e9)
        for i, (_, px, kosegen) in enumerate(gts):
            for j, (_, poz) in enumerate(tahmin):
                d = np.linalg.norm(poz.noktalar[GOVDE] - px, axis=1)
                if np.isfinite(d).sum() >= EN_AZ_EKLEM // 2:
                    C[i, j] = np.nanmean(d) / kosegen
        eslesti = 0
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < ESLESME:
                eslesti += 1
                kid, px, kosegen = gts[i]
                tid, poz = tahmin[j]
                iz[kid].append(tid)
                iz_kare[kid].append(k)
                d = np.linalg.norm(poz.noktalar[GOVDE] - px, axis=1)
                hatalar += d[np.isfinite(d)].tolist()
                oranlar += (d[np.isfinite(d)] / kosegen).tolist()
        eslesen += eslesti
        fazla += len(tahmin) - eslesti
    model.close()

    degisim = sum(sum(1 for x, y in zip(v[:-1], v[1:]) if x != y) for v in iz.values())
    saflik = {kid: Counter(v).most_common(1)[0][1] / len(v) for kid, v in iz.items() if v}
    sonuc = {
        "kestirici": a.kestirici, "kamera": a.kamera,
        "kareler": {"ilk": kareler[0], "islenen": islenen, "adim": a.adim},
        "etiketli_kisi_kare": gt_sayi, "tespit_orani": eslesen / max(gt_sayi, 1),
        "fazla_tespit_kare_basina": fazla / max(islenen, 1),
        "govde_hata_px_medyan": float(np.median(hatalar)) if hatalar else None,
        "govde_hata_px_p90": float(np.percentile(hatalar, 90)) if hatalar else None,
        "govde_hata_kosegen_medyan": float(np.median(oranlar)) if oranlar else None,
        "kimlik_degisimi": degisim,
        "kimlik_safligi": {str(k): round(v, 3) for k, v in saflik.items()},
        "takip_kimlikleri": {str(k): sorted(set(v)) for k, v in iz.items()},
        # kimligin degistigi kare ve oncesindeki eslesmeme boslugu (kare)
        "degisim_yerleri": {str(kid): [
            {"kare": iz_kare[kid][i + 1], "bosluk_kare": iz_kare[kid][i + 1] - iz_kare[kid][i],
             "etiket_gorunur_bosluk": sum(1 for x in etiket_kare[kid]
                                         if iz_kare[kid][i] < x < iz_kare[kid][i + 1])}
            for i in range(len(v) - 1) if v[i] != v[i + 1]] for kid, v in iz.items()},
    }
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))
    if a.cikti:
        a.cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
