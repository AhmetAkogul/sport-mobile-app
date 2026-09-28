"""MediaPipe Tasks modelini mevcut Poz2B sözleşmesine bağlar."""

from dataclasses import replace
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from capture.alignment import file_sha256
from pose3d.adaptorler import mediapipe_poz2b, mediapipe_dunya_noktalari
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B


def _sayisal_surum(metin: str) -> tuple[int, ...]:
    """'1.0.0.post1' -> (1, 0, 0); sayisal olmayan ilk parcada durur."""
    parcalar: list[int] = []
    for parca in str(metin).split("."):
        if not parca.isdigit():
            break
        parcalar.append(int(parca))
    return tuple(parcalar)


def result_to_pose(result, size, *, model, threshold=0.5):
    """Tespit yoksa görünmez poz döner; eski karenin pozunu taşımaz.

    "Tespit var mı" bilgisi artık `ek["tespit"]` sözlüğünde değil, `Poz2B.tespit`
    alanında taşınır: yeni bir adaptör yazarken unutulması sessiz bir hataya
    dönüşemez, çünkü alan zorunludur.
    """
    if not result.pose_landmarks:
        # Tespit yoksa koordinat da yoktur: NaN yazilir, 0.0 (goruntunun sol ust
        # kosesi) gibi gecerli gorunen bir deger yazilmaz (tek eksik veri politikasi).
        n = len(REFERANS_ISKELET)
        return Poz2B(REFERANS_ISKELET, np.full((n, 2), np.nan), np.zeros(n),
                     np.zeros(n, bool), False, size, model=model)
    if len(result.pose_landmarks) != 1:
        raise ValueError("Tek kişi sözleşmesinde birden fazla poz döndü.")
    return mediapipe_poz2b(result.pose_landmarks[0], size, guven_esigi=threshold,
                           model=model, tespit=True)


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
        surum = _sayisal_surum(mp.__version__)
        # Koruma "1.0.0 ve 1.0.1" ile sinirli degil: sikayet edilen davranis 1.0
        # hattinin tamami icin gecerli (0030). '1.0.2' cikarsa koruma kalkmaz;
        # 1.1+ dogrulanmadigi icin engellenmez, denenip karar kaydina yazilmali.
        if sys.platform == "darwin" and surum[:2] == (1, 0):
            raise RuntimeError(
                f"Bu Mac için MediaPipe 0.10.35 kullanın; {mp.__version__} yerel süreç "
                "çökmesine yol açabilir (docs/kararlar/0030).")
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        # capture.alignment'taki tek sha256 uygulamasi kullanilir (tum dosyayi
        # belleğe alan ikinci bir kopya yok).
        self.sha256 = file_sha256(path)
        # Varyant (lite/full/heavy) model kimliginde gorunur; aksi halde üç ay
        # sonra "hangi MediaPipe" sorusu karar kaydina bakmadan cevaplanamaz.
        self.model_id = (f"mediapipe-{mp.__version__}/{path.stem}"
                         f"/sha256:{self.sha256}")
        self.threshold = threshold
        self._mp = mp
        self._closed = False
        # CPU delegate bilincli: donanim bagimsiz ve iki backend ayni kaynak
        # kosullarinda olculsun (GPU delegate olsaydi B2 tablosu donanima bagli
        # olurdu). GPU yolu denenirse karar kaydinda ayrica belirtilmeli.
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
        world = getattr(result, "pose_world_landmarks", [])
        extra = {}
        if pose.tespit and len(world) == 1:
            xyz, conf, visible = mediapipe_dunya_noktalari(world[0], guven_esigi=self.threshold)
            # Maskelenmemis kopya: modelin gorunmez (or. profilde arkadaki bacak)
            # eklem icin de verdigi tahmin. Kullanan taraf guveni kendisi tartar;
            # `world_points_m` sozlesmesi (yalniz gorunur) degismez.
            extra = {"world_points_all_m": xyz.tolist(), "world_confidence": conf.tolist()}
            visible = visible & pose.gorunur
            xyz = xyz.copy()
            xyz[~visible] = np.nan
            extra |= {"world_points_m": xyz.tolist(), "world_visible": visible.tolist()}
            # Yuz gorunurlugu (burun, gozler): Poz2B yuzu tasimaz; sirti donuk kisiyi
            # (onden bakisla ayni govde acisi) ayirmak icin kullanilir.
            yuz = [result.pose_landmarks[0][i] for i in (0, 2, 5)]
            extra["yuz_guveni"] = float(max(min(getattr(p, "visibility", 0.0) or 0.0,
                                                getattr(p, "presence", 1.0) or 0.0) for p in yuz))
            # Ayak noktalari (REFERANS_ISKELET tasimaz): sag topuk, sag ayak ucu, sol
            # topuk, sol ayak ucu -- MediaPipe 30, 32, 29, 31. Lunge'da "diz ayak
            # ucunu geciyor" icin (docs/deney/2026-09-28-ec3d-lunge.md).
            ayak = (30, 32, 29, 31)
            lm2, lm3 = result.pose_landmarks[0], world[0]
            extra["ayak_px"] = [[lm2[i].x * width, lm2[i].y * height] for i in ayak]
            extra["ayak_dunya_m"] = [[lm3[i].x, lm3[i].y, lm3[i].z] for i in ayak]
            extra["ayak_guveni"] = [float(min(getattr(lm2[i], "visibility", 0.0) or 0.0,
                                              getattr(lm2[i], "presence", 1.0) or 0.0))
                                    for i in ayak]
        # `replace`: Poz2B frozen; `ek.update` yerine yeni bir kopya uretilir.
        return replace(pose, ek={**pose.ek, **extra, "model_sha256": self.sha256,
                                 "running_mode": "IMAGE", "device": "CPU"})

    def close(self):
        if not self._closed:
            self._closed = True
            self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
