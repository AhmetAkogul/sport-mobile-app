"""Fitness-AQA squat (etiketli) videolarinda MediaPipe poz tespiti; model ortaminda kosulur.

Her video tek tekrardir (medyan 96 kare, 30 fps). Kare basina REFERANS
sirasinda 2B nokta/guven/gorunurluk ve `pose_world` 3B'si -- hem gorunur
eklemlerle sinirli (`dunya`) hem maskesiz (`dunya_tam`, gorunmez eklem
tahmini dahil) -- `out/fitness_aqa_tespit/mediapipe/<anahtar>.npz` dosyasina
yazilir. Yalnizca "dizler ice" etiketi olan 1.623 video islenir; biten
dosya atlanir (yarida kalirsa kaldigi yerden surer).
Degerlendirme ayri betikte (`deney_fitness_aqa_valgus.py`).

    PYTHONPATH=. <mediapipe-env>/bin/python scripts/fitness_aqa_tespit.py \\
        --model <pose_landmarker_full.task>
    PYTHONPATH=. <rtmpose-env>/bin/python scripts/fitness_aqa_tespit.py --backend rtmw \\
        --model <end2end.onnx> --detector <detector.onnx> --bolum val

`--bolum` (train/val/test) yalniz o resmi bolumu isler; resmi test bolumu son
olcume saklanir. RTMPose/RTMW'de dunya (3B) alanlari yoktur (NaN).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mono.backend import kestirici_olustur  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, eslestir  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
SQUAT = KOK / "data/dis/fitness_aqa/Squat/Labeled_Dataset"
ETIKET = SQUAT / "Labels/error_knees_inward.json"
VIDEOLAR = SQUAT / "videos/videos"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=("mediapipe", "rtmpose", "rtmw"), default="mediapipe")
    p.add_argument("--detector", help="RTMPose/RTMW kisi dedektoru (YOLOX onnx)")
    p.add_argument("--bolum", choices=("train", "val", "test"))
    p.add_argument("--ilk", type=int, help="yalnizca ilk N video (deneme icin)")
    a = p.parse_args()

    anahtarlar = sorted(json.loads(ETIKET.read_text()))
    if a.bolum:
        secili = set(json.loads((SQUAT / f"Splits/{a.bolum}_keys.json").read_text()))
        anahtarlar = [k for k in anahtarlar if k in secili]
    anahtarlar = anahtarlar[:a.ilk]
    cikti = KOK / "out/fitness_aqa_tespit" / a.backend
    cikti.mkdir(parents=True, exist_ok=True)
    n = len(REFERANS_ISKELET)
    if a.backend == "mediapipe":
        baglam = kestirici_olustur("mediapipe", model=a.model)
    else:
        from mono.rtmpose_model import RTMPoseEstimator
        ayar = {"rtmpose": ("rtmpose-m", (192, 256)), "rtmw": ("rtmw-x", (288, 384))}
        ad, giris = ayar[a.backend]
        baglam = RTMPoseEstimator(a.detector, a.model, coklu_kisi="en_buyuk", ad=ad,
                                  giris_boyutu=giris)
    with baglam as model:
        for sira, anahtar in enumerate(anahtarlar, 1):
            hedef = cikti / f"{anahtar}.npz"
            if hedef.exists():
                continue
            kayit: dict[str, list] = {k: [] for k in (
                "noktalar", "guven", "gorunur", "tespit", "dunya", "dunya_gorunur",
                "dunya_tam", "dunya_guven")}
            cap = cv2.VideoCapture(str(VIDEOLAR / f"{anahtar}.mp4"))
            fps, boyut = cap.get(cv2.CAP_PROP_FPS), None
            try:
                while True:
                    ok, im = cap.read()
                    if not ok:
                        break
                    boyut = (im.shape[1], im.shape[0])
                    poz = replace(model(im), kamera=0, kare=len(kayit["tespit"]))
                    harita = eslestir(poz.iskelet, REFERANS_ISKELET)
                    P, g, v = np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool)
                    for j, k in enumerate(harita):
                        if k >= 0:
                            P[j], g[j], v[j] = poz.noktalar[k], poz.guven[k], poz.gorunur[k]
                    kayit["noktalar"].append(P)
                    kayit["guven"].append(g)
                    kayit["gorunur"].append(v)
                    kayit["tespit"].append(bool(poz.tespit))
                    for ad, anah, bos in (
                            ("dunya", "world_points_m", np.full((n, 3), np.nan)),
                            ("dunya_gorunur", "world_visible", np.zeros(n, bool)),
                            ("dunya_tam", "world_points_all_m", np.full((n, 3), np.nan)),
                            ("dunya_guven", "world_confidence", np.zeros(n))):
                        kayit[ad].append(np.asarray(poz.ek.get(anah, bos)))
            finally:
                cap.release()
            if boyut is None:
                print(f"{anahtar}: video okunamadi", flush=True)
                continue
            np.savez_compressed(hedef, model=str(model.model_id), boyut=np.array(boyut),
                                fps=float(fps), **{k: np.asarray(v) for k, v in kayit.items()})
            if sira % 50 == 0:
                print(f"{sira}/{len(anahtarlar)}", flush=True)


if __name__ == "__main__":
    main()
