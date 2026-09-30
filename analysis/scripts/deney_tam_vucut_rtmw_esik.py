"""0035: RTMW skoru kadraj disi / karartilmis eklemi ayiriyor mu?

REHAB24-6 Camera17 videolarinin 90. karesi. Iki sinama:
1. govde: kalca hizasindan asagisi karartilir, bacak/ayak skorlari olculur;
2. el: iki elin kutusu karartilir, parmak skorlari olculur.
Kutu ayni kalir (dedektor atlanir), yalniz goruntu degisir.

    PYTHONPATH=. <rtmpose-env>/bin/python scripts/deney_tam_vucut_rtmw_esik.py
"""
import glob

import cv2
import numpy as np

from mono.rtmw_model import RTMWEstimator
from pose3d.tam_vucut import GRUPLAR

DEDEKTOR = "../bitirme-capture/data/models/rtmpose/detector.onnx"
RTMW = "data/dis/rtmw_x_l_384/acik/end2end.onnx"
EL = list(GRUPLAR["sol_el"]) + list(GRUPLAR["sag_el"])


def _alan(p):
    x1, y1, x2, y2 = p.ek["kutu"]
    return (x2 - x1) * (y2 - y1)


def main():
    vids = sorted(glob.glob("data/dis/rehab24_6/videos/Ex*/*Camera17*.mp4"))[:16]
    govde_ic, govde_dis, el_ic, el_dis = [], [], [], []
    with RTMWEstimator(DEDEKTOR, RTMW, threshold=0) as m:
        for v in vids:
            cap = cv2.VideoCapture(v)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
            ok, im = cap.read()
            kisiler = m.kisiler(im) if ok else []
            if not kisiler:
                continue
            ana = max(kisiler, key=_alan)
            kutu = [ana.ek["kutu"]]
            h = np.array(ana.ek["ham_skor"])
            el_ic += h[EL].tolist()
            # 1. kalcadan asagisi kadraj disi
            kes = int(np.nanmean(ana.noktalar[[11, 12], 1]))
            im2 = im.copy()
            im2[kes:] = 0
            h2 = np.array(m.kisiler(im2, kutular=kutu)[0].ek["ham_skor"])
            govde_dis += h2[ana.noktalar[:, 1] > kes + 40].tolist()
            govde_ic += h2[ana.noktalar[:, 1] < kes - 40].tolist()
            # 2. eller karartilir
            im3 = im.copy()
            for g in ("sol_el", "sag_el"):
                P = ana.noktalar[list(GRUPLAR[g])]
                x0, y0 = np.nanmin(P, 0) - 25
                x1, y1 = np.nanmax(P, 0) + 25
                im3[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)] = 0
            el_dis += np.array(m.kisiler(im3, kutular=kutu)[0].ek["ham_skor"])[EL].tolist()
    for ad, ic, dis in (("govde", govde_ic, govde_dis), ("el", el_ic, el_dis)):
        ic, dis = np.array(ic), np.array(dis)
        print(f"{ad}: gorunur n={len(ic)} medyan {np.median(ic):.2f} | "
              f"karartilmis n={len(dis)} medyan {np.median(dis):.2f}")
        for t in (2.0, 3.0, 3.5, 4.0):
            print(f"  esik {t}: gorunur kabul {(ic >= t).mean():.3f}  "
                  f"karartilmis kabul {(dis >= t).mean():.3f}")


if __name__ == "__main__":
    main()
