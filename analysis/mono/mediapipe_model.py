"""MediaPipe Tasks modelini mevcut Poz2B sözleşmesine bağlar."""

import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from pose3d.adaptorler import mediapipe_poz2b
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B


def result_to_pose(result, size, *, model, threshold=0.5):
    """Tespit yoksa görünmez poz döner; eski karenin pozunu taşımaz."""
    if not result.pose_landmarks:
        n = len(REFERANS_ISKELET)
        return Poz2B(REFERANS_ISKELET, np.zeros((n, 2)), np.zeros(n), np.zeros(n, bool),
                     size, model=model, ek={"tespit": False})
    if len(result.pose_landmarks) != 1:
        raise ValueError("Tek kişi sözleşmesinde birden fazla poz döndü.")
    pose = mediapipe_poz2b(result.pose_landmarks[0], size, guven_esigi=threshold, model=model)
    pose.ek["tespit"] = True
    return pose


class MediaPipeEstimator:
    """BGR uint8 -> özgün piksellerde Poz2B; her çağrı bağımsız IMAGE modu.

    Durumsuz mod, aynı kareyi farklı video/kameralar arasında izleme geçmişi
    taşımadan işler. VIDEO takip optimizasyonu bu ilk baseline'a dahil değildir.
    """

    def __init__(self, model_path, *, threshold=0.5):
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Güven eşiği 0..1 aralığında olmalı.")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        import mediapipe as mp
        if sys.platform == "darwin" and mp.__version__ in {"1.0.0", "1.0.1"}:
            raise RuntimeError("Bu Mac için MediaPipe 0.10.35 kullanın; 1.0.x yerel süreç çökmesine yol açabilir.")
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model_id = f"mediapipe-{mp.__version__}/pose_landmarker/sha256:{self.sha256}"
        self.threshold = threshold
        self._mp = mp
        self._closed = False
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path.resolve()), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            output_segmentation_masks=False)
        self._detector = PoseLandmarker.create_from_options(options)

    def __call__(self, image_bgr):
        if self._closed:
            raise RuntimeError("Model kapalı.")
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0):
            raise ValueError("Girdi boş olmayan BGR uint8 görüntü olmalı.")
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        pose = result_to_pose(result, (width, height), model=self.model_id, threshold=self.threshold)
        pose.ek.update(model_sha256=self.sha256, running_mode="IMAGE", device="CPU")
        return pose

    def close(self):
        if not self._closed:
            self._closed = True
            self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
