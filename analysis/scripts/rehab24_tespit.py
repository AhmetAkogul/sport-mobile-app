"""REHAB24-6 videolarinda poz tespiti (Katman 2 girdisi); model ortaminda kosulur.

Yalnizca squat (Ex6) tekrarlarinin kareleri islenir. Kare basina REFERANS
sirasinda 2B nokta/guven/gorunurluk ve (MediaPipe'ta) `pose_world` 3B'si
`out/rehab24_tespit/<backend>/<video>-<kamera>.npz` dosyasina yazilir.
Degerlendirme ayri betikte (`deney_rehab24_tek_gorus.py`, tespit dosyasi varsa otomatik).

    PYTHONPATH=. <mediapipe-env>/bin/python scripts/rehab24_tespit.py \\
        --backend mediapipe --model <pose_landmarker_full.task>
    PYTHONPATH=. <rtmpose-env>/bin/python scripts/rehab24_tespit.py \\
        --backend rtmpose --model <pose.onnx> --detector <detector.onnx>

Video adi veri setinde `<video>` ve `c17` / `c18` iceren dosyadir; kare
numarasi 30 fps dizinidir (Segmentation.txt).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mono.backend import kestirici_olustur  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, eslestir  # noqa: E402
from veri.rehab24 import tekrarlar_oku  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data/dis/rehab24_6"
# Veri setindeki dosya adlari: Ex6/PM_008-Camera17-30fps.mp4, ...-Camera18-30fps-transposed.mp4
_KAMERA_ADI = {"c17": "Camera17", "c18": "Camera18"}


def video_bul(video: str, kamera: str) -> Path:
    adaylar = sorted(p for p in (VERI / "videos").rglob("*")
                     if p.suffix.lower() in (".mp4", ".avi", ".mov")
                     and video in p.name and _KAMERA_ADI[kamera] in p.name)
    if len(adaylar) != 1:
        raise FileNotFoundError(f"{video}-{kamera}: {len(adaylar)} aday ({adaylar[:3]})")
    return adaylar[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("mediapipe", "rtmpose"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--detector")
    p.add_argument("--video", help="yalnizca bu video (deneme icin)")
    a = p.parse_args()

    tekrarlar = [t for t in tekrarlar_oku(VERI / "Segmentation.csv") if not t.mocap_hatali]
    videolar = sorted({t.video for t in tekrarlar if a.video in (None, t.video)})
    cikti = KOK / "out/rehab24_tespit" / a.backend
    cikti.mkdir(parents=True, exist_ok=True)
    n = len(REFERANS_ISKELET)
    # RTMPose dedektoru birden fazla kisi bulursa en buyuk kutu (yer gercegine bakmaz).
    coklu = "en_buyuk" if a.backend == "rtmpose" else "hata"
    with kestirici_olustur(a.backend, model=a.model, detector=a.detector,
                           coklu_kisi=coklu) as model:
        for video in videolar:
            kareler = sorted({k for t in tekrarlar if t.video == video
                              for k in range(t.ilk, t.son + 1)})
            istenen = set(kareler)
            for kamera in ("c17", "c18"):
                hedef = cikti / f"{video}-{kamera}.npz"
                if hedef.exists():
                    continue
                kayit: dict[str, list] = {k: [] for k in (
                    "kare", "noktalar", "guven", "gorunur", "tespit", "dunya", "dunya_gorunur",
                    "dunya_tam", "dunya_guven")}
                cap = cv2.VideoCapture(str(video_bul(video, kamera)))
                i, son, boyut = 0, max(kareler), None
                try:
                    while i <= son:
                        ok, im = cap.read()
                        if not ok:
                            break
                        if i in istenen:
                            boyut = (im.shape[1], im.shape[0])
                            poz = replace(model(im), kamera=0 if kamera == "c17" else 1, kare=i)
                            harita = eslestir(poz.iskelet, REFERANS_ISKELET)
                            P, g, v = np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool)
                            for j, k in enumerate(harita):
                                if k >= 0:
                                    P[j], g[j], v[j] = poz.noktalar[k], poz.guven[k], poz.gorunur[k]
                            kayit["kare"].append(i)
                            kayit["noktalar"].append(P)
                            kayit["guven"].append(g)
                            kayit["gorunur"].append(v)
                            kayit["tespit"].append(bool(poz.tespit))
                            kayit["dunya"].append(np.asarray(
                                poz.ek.get("world_points_m", np.full((n, 3), np.nan)), float))
                            kayit["dunya_gorunur"].append(np.asarray(
                                poz.ek.get("world_visible", np.zeros(n, bool)), bool))
                            # Maskelenmemis dunya (gorunmez eklem tahmini dahil) ve guveni.
                            kayit["dunya_tam"].append(np.asarray(
                                poz.ek.get("world_points_all_m", np.full((n, 3), np.nan)), float))
                            kayit["dunya_guven"].append(np.asarray(
                                poz.ek.get("world_confidence", np.zeros(n)), float))
                        i += 1
                finally:
                    cap.release()
                np.savez_compressed(hedef, model=str(model.model_id), boyut=np.array(boyut),
                                    **{k: np.asarray(v) for k, v in kayit.items()})
                print(f"{video}-{kamera}: {len(kayit['kare'])}/{len(kareler)} kare", flush=True)


if __name__ == "__main__":
    main()
