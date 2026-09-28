"""RTMW (COCO-WholeBody, 133 nokta) -> kisi basina TAM_VUCUT Poz2B.

`RTMPoseEstimator` tek kisi sozlesmesindedir ve 133 noktanin yalniz ilk 17'sini
alir. Bu sinif karedeki **her kisiyi** dondurur: govde, bas, ayaklar ve parmak
eklemleri (yuz noktalari haric, bkz. `pose3d.tam_vucut`). Kimlik tasimaz;
kareler arasi kimlik `mono.coklu_kisi.KisiTakip` ile verilir.

Skor SIMCC kaynaklidir, olasilik degildir (0031); esik modele ozgudur.
"""

from __future__ import annotations

from importlib.metadata import version
import math
from pathlib import Path

import numpy as np

from capture.alignment import file_sha256
from pose3d.tam_vucut import coco_wholebody_poz

# Olculdu (0035): esik 3,5'te gorunur ellerin %98'i kabul, karartilmis ellerin
# %56'si de "gorunur" -- skor el varligini guvenilir ayirmiyor.
RTMW_GUVEN_ESIGI = 3.5


class RTMWEstimator:
    def __init__(self, detector_path, pose_path, *, threshold=RTMW_GUVEN_ESIGI,
                 ad="rtmw-x", giris_boyutu=(288, 384)):
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Guven esigi sonlu ve negatif olmayan bir sayi olmali.")
        for path in (detector_path, pose_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        from rtmlib import YOLOX, RTMPose
        self.model_id = (f"rtmlib-{version('rtmlib')}/{ad}/pose:{file_sha256(pose_path)}"
                         f"/detector:{file_sha256(detector_path)}")
        self.threshold = threshold
        self._closed = False
        self._detector = YOLOX(str(detector_path), model_input_size=(640, 640),
                               backend="onnxruntime", device="cpu")
        self._pose = RTMPose(str(pose_path), model_input_size=tuple(giris_boyutu),
                             to_openpose=False, backend="onnxruntime", device="cpu")

    def kisiler(self, image_bgr, kutular=None):
        """Karedeki her kisi icin TAM_VUCUT Poz2B; `ek["kutu"]` = [x1, y1, x2, y2].

        `kutular` verilirse dedektor atlanir (takipcinin tahmin ettigi kutularla
        her karede dedektor calistirmamak icin).
        """
        if self._closed:
            raise RuntimeError("Model kapali.")
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0):
            raise ValueError("Girdi bos olmayan BGR uint8 goruntu olmali.")
        height, width = image_bgr.shape[:2]
        boxes = self._detector(image_bgr) if kutular is None else kutular
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        points, scores = self._pose(image_bgr, bboxes=boxes)
        if len(points) != len(boxes):
            raise ValueError("Kutu ve poz sayisi uyusmuyor.")
        return [coco_wholebody_poz(p, s, (width, height), model=self.model_id,
                                   esik=self.threshold, ek={"kutu": b.tolist()})
                for p, s, b in zip(points, scores, boxes)]

    def close(self):
        self._closed = True
        self._pose = self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
