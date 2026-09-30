"""0035: MediaPipe (poz + el kirpintisi) ile RTMW parmak noktalarinin uyumu.

Parmaklar icin yer gercegi yok; iki bagimsiz modelin uyumu tutarlilik kanitidir,
dogruluk kaniti degildir. Iki ortam gerektigi icin iki adim:

    PYTHONPATH=. <mediapipe-env>/bin/python scripts/deney_tam_vucut_uyum.py mp <dizin>
    PYTHONPATH=. <rtmpose-env>/bin/python scripts/deney_tam_vucut_uyum.py rtmw <dizin>
"""
import glob
import sys

import cv2
import numpy as np

from pose3d.tam_vucut import GRUPLAR

KARE = 75


def _videolar():
    return sorted(glob.glob("data/dis/rehab24_6/videos/Ex*/*.mp4"))[::7][:40]


def _kare(v):
    cap = cv2.VideoCapture(v)
    cap.set(cv2.CAP_PROP_POS_FRAMES, KARE)
    ok, im = cap.read()
    return im if ok else None


def mp_adimi(dizin):
    from mono.mediapipe_model import MediaPipeEstimator
    out = {}
    with MediaPipeEstimator("../bitirme-capture/data/models/pose_landmarker_full-float16-v1.task",
                            el_modeli="data/dis/mediapipe_hand/hand_landmarker.task") as m:
        for v in _videolar():
            im = _kare(v)
            if im is not None:
                _, tam = m.tam_vucut(im)
                out[v] = np.concatenate([tam.noktalar, tam.gorunur[:, None]], 1)
    np.save(f"{dizin}/uyum_mp.npy", out, allow_pickle=True)
    print(len(out), "kare")


def rtmw_adimi(dizin):
    from mono.rtmw_model import RTMWEstimator
    mp = np.load(f"{dizin}/uyum_mp.npy", allow_pickle=True).item()
    bulunan, toplam, hata = 0, 0, []
    g = list(GRUPLAR["govde"])
    with RTMWEstimator("../bitirme-capture/data/models/rtmpose/detector.onnx",
                       "data/dis/rtmw_x_l_384/acik/end2end.onnx", threshold=0) as m:
        for v, arr in mp.items():
            kisiler = m.kisiler(_kare(v))
            if not kisiler:
                continue
            P, gor = arr[:, :2], arr[:, 2] > 0
            r = min(kisiler, key=lambda k: np.nanmedian(np.linalg.norm(k.noktalar[g] - P[g], axis=1)))
            for el in ("sol_el", "sag_el"):
                ix = list(GRUPLAR[el])
                toplam += 1
                if gor[ix].all():
                    bulunan += 1
                    avuc = np.linalg.norm(P[ix[0]] - P[ix[9]])
                    hata.append(np.linalg.norm(r.noktalar[ix] - P[ix], axis=1) / avuc)
    h = np.array(hata)
    print(f"MediaPipe el buldu: {bulunan}/{toplam}")
    print(f"RTMW-MP parmak farki / avuc boyu: medyan {np.median(h):.2f}, p90 "
          f"{np.percentile(h, 90):.2f}; el basina medyani >0.5 olan {(np.median(h, 1) > 0.5).sum()}"
          f"/{len(h)}")


if __name__ == "__main__":
    {"mp": mp_adimi, "rtmw": rtmw_adimi}[sys.argv[1]](sys.argv[2])
