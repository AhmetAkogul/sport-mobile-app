"""0037: REHAB24-6 videolarinin tamaminda tam vucut tespiti (hareket tanima girdisi).

Butun kareler (tekrar disi dahil: "hareket yok" sinifi icin) `--adim` karede bir
islenir. Kare basina TAM_VUCUT 2B (65 nokta; el modeli verilmez, bu hareketlerde
gerekmiyor) ve MediaPipe `pose_world` (referans 13 eklem + 4 ayak noktasi)
`out/hareket_tespit/mediapipe/<video>-<c17|c18>.npz` dosyasina yazilir.
Var olan dosya atlanir (kesilip yeniden baslatilabilir).

    PYTHONPATH=. <mediapipe-env>/bin/python scripts/hareket_tespit.py \\
        --model <pose_landmarker_full.task> [--egzersiz 1 2 3] [--adim 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mono.mediapipe_model import MediaPipeEstimator  # noqa: E402
from pose3d.tam_vucut import TAM_VUCUT  # noqa: E402

KOK = Path("data/dis/rehab24_6/videos")
CIKTI = Path("out/hareket_tespit/mediapipe")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--egzersiz", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--adim", type=int, default=3, help="kac karede bir (30 fps dizini)")
    a = p.parse_args()
    CIKTI.mkdir(parents=True, exist_ok=True)
    n = len(TAM_VUCUT)
    with MediaPipeEstimator(a.model) as model:
        for ex in a.egzersiz:
            for v in sorted((KOK / f"Ex{ex}").glob("*.mp4")):
                video = v.name.split("-")[0]
                kamera = "c17" if "Camera17" in v.name else "c18"
                hedef = CIKTI / f"{video}-{kamera}.npz"
                if hedef.exists():
                    continue
                cap = cv2.VideoCapture(str(v))
                kareler, P, G, D, A, tespit = [], [], [], [], [], []
                boyut = (0, 0)
                i = -1
                while True:
                    i += 1
                    if i % a.adim:
                        if not cap.grab():
                            break
                        continue
                    ok, im = cap.read()
                    if not ok:
                        break
                    poz, tam = model.tam_vucut(im)
                    kareler.append(i)
                    P.append(tam.noktalar)
                    G.append(tam.gorunur)
                    tespit.append(tam.tespit)
                    D.append(poz.ek.get("world_points_all_m", np.full((13, 3), np.nan)))
                    A.append(poz.ek.get("ayak_dunya_m", np.full((4, 3), np.nan)))
                    boyut = tam.goruntu_boyutu
                cap.release()
                tmp = hedef.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp, model=model.model_id, egzersiz=ex, boyut=np.array(boyut),
                    kare=np.array(kareler), noktalar=np.array(P, float).reshape(-1, n, 2),
                    gorunur=np.array(G, bool).reshape(-1, n), tespit=np.array(tespit, bool),
                    dunya_tam=np.array(D, float).reshape(-1, 13, 3),
                    ayak_dunya=np.array(A, float).reshape(-1, 4, 3))
                tmp.replace(hedef)
                print(f"{hedef.name}: {len(kareler)} kare, tespit %{100 * np.mean(tespit):.0f}",
                      flush=True)


if __name__ == "__main__":
    main()
