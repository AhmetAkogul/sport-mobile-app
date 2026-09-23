"""RTMPose/COCO-17 -> mevcut bitirme-13 Poz2B arayüzü.

**Esik uyarisi (B2 karsilastirmasi).** RTMPose'un skoru SIMCC dagilimindan
gelen ham bir skordur; kalibre edilmis olasilik **degildir**. Tipik degerlerde
0.3 "dusuk guven" demek degil, 0.7 "cok yuksek" demektir. Ayni esigi MediaPipe
ile paylasmak, iki modeli ayni olcekte olcuyormus gibi gosterir ve B2 tablosunu
yaniltir. Bu yuzden:

- her adaptor kendi varsayilan esigini tasir (`SIMCC_GUVEN_ESIGI`),
- karar kaydinda/esikte hangi esigin kullanildigi `ek["guven_esigi"]` ile
  pozun yaninda saklanir,
- iki modelin skorlari karsilastirilirken esik degil **ham skor** esas alinir
  (`ek["raw_scores"]`).
"""

from importlib.metadata import version
import math
from pathlib import Path

import numpy as np

from capture.alignment import file_sha256
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B

# COCO-17'nin yüz dışındaki eklemleri; referans sözleşmesi değişmez.
COCO_MAP = {"sol_omuz": 5, "sag_omuz": 6, "sol_dirsek": 7, "sag_dirsek": 8,
            "sol_bilek": 9, "sag_bilek": 10, "sol_kalca": 11, "sag_kalca": 12,
            "sol_diz": 13, "sag_diz": 14, "sol_ayak_bilegi": 15, "sag_ayak_bilegi": 16}

# Referans iskeletin ihtiyac duydugu COCO indeksleri (5..16). Yuz noktalari
# (0..4) saklanmaz: hicbir olcumde kullanilmiyor, KVKK acisindan da gereksiz.
KULLANILAN_COCO = tuple(sorted(set(COCO_MAP.values())))

# SIMCC skoru icin varsayilan esik. MediaPipe'in 0.5'i ile ayni sayi **degildir**:
# ayni esik iki modelde farkli anlam tasir, bkz. modul docstring'i.
SIMCC_GUVEN_ESIGI = 0.3


def coco_to_pose(points, scores, size, *, model, threshold=SIMCC_GUVEN_ESIGI):
    points, scores = np.asarray(points, float), np.asarray(scores, float)
    if points.shape != (17, 2) or scores.shape != (17,):
        raise ValueError("COCO-17 için (17,2) nokta ve (17,) skor gerekli.")
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError("Model koordinat ve skorları sonlu olmalı.")
    n = len(REFERANS_ISKELET)
    xy, confidence = np.zeros((n, 2)), np.zeros(n)
    for i, name in enumerate(REFERANS_ISKELET.eklemler):
        indices = [5, 6] if name == "boyun" else [COCO_MAP[name]]
        xy[i] = points[indices].mean(axis=0)
        confidence[i] = np.clip(scores[indices].min(), 0, 1)
    return Poz2B(REFERANS_ISKELET, xy, confidence, confidence >= threshold, True, size,
                 model=model,
                 ek={"kaynak_iskelet": "coco-17", "turetilmis": ["boyun"],
                     "guven_esigi": float(threshold),
                     # Ham skorlar esik kalibrasyonu icin saklanir; yalnizca
                     # kullanilan eklemler (yuz skorlari disarida).
                     "raw_scores": [float(scores[i]) for i in KULLANILAN_COCO],
                     "raw_scores_joints": list(KULLANILAN_COCO),
                     "score_semantics": "simcc_not_calibrated_probability"})


class RTMPoseEstimator:
    def __init__(self, detector_path, pose_path, *, threshold=SIMCC_GUVEN_ESIGI):
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Güven eşiği 0..1 aralığında olmalı.")
        for path in (detector_path, pose_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        from rtmlib import YOLOX, RTMPose
        self.model_id = (f"rtmlib-{version('rtmlib')}/rtmpose-m/pose:{file_sha256(pose_path)}"
                         f"/detector:{file_sha256(detector_path)}")
        self.threshold = threshold
        self._closed = False
        self._detector = YOLOX(str(detector_path), model_input_size=(640, 640),
                               backend="onnxruntime", device="cpu")
        self._pose = RTMPose(str(pose_path), model_input_size=(192, 256),
                             to_openpose=False, backend="onnxruntime", device="cpu")

    def __call__(self, image_bgr):
        if self._closed:
            raise RuntimeError("Model kapalı.")
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0):
            raise ValueError("Girdi boş olmayan BGR uint8 görüntü olmalı.")
        height, width = image_bgr.shape[:2]
        boxes = self._detector(image_bgr)
        if len(boxes) == 0:
            # Kisi yoksa koordinat da yok: NaN -> JSON'da null (tek politika).
            n = len(REFERANS_ISKELET)
            return Poz2B(REFERANS_ISKELET, np.full((n, 2), np.nan), np.zeros(n),
                         np.zeros(n, bool), False, (width, height), model=self.model_id)
        if len(boxes) != 1:
            raise ValueError("Birden fazla kişi bulundu; kişi seçimi/izlemesi gerekli.")
        points, scores = self._pose(image_bgr, bboxes=boxes)
        if len(points) != 1 or len(scores) != 1:
            raise ValueError("Tek kişi için beklenmeyen model çıktısı.")
        return coco_to_pose(points[0], scores[0], (width, height), model=self.model_id, threshold=self.threshold)

    def close(self):
        if self._closed:                      # idempotent: ikinci cagri zararsiz
            return
        self._closed = True
        self._pose = None
        self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
