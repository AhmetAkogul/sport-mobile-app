"""mono.backend: iki CLI'nin ortak model secimi.

Esik **modele gore** farklidir; ayni sayiyi iki modele uygulamak B2
karsilastirmasini yaniltir (dis inceleme V.3, docs/kararlar/0031).
"""
import numpy as np
import pytest

from mono.backend import MEDIAPIPE_GUVEN_ESIGI, kestirici_olustur
from mono.rtmpose_model import SIMCC_GUVEN_ESIGI, coco_to_pose


def test_bilinmeyen_backend_reddedilir():
    with pytest.raises(ValueError, match="bilinmeyen backend"):
        kestirici_olustur("yolo", model="m.task")


def test_rtmpose_dedektorsuz_reddedilir():
    with pytest.raises(ValueError, match="detector"):
        kestirici_olustur("rtmpose", model="pose.onnx")


def test_varsayilan_esikler_modelden_gelir():
    assert MEDIAPIPE_GUVEN_ESIGI == 0.5
    assert SIMCC_GUVEN_ESIGI < MEDIAPIPE_GUVEN_ESIGI
    # SIMCC 0.4 skoru MediaPipe'in 0.5 esigiyle gorunmez olurdu.
    pose = coco_to_pose(np.zeros((17, 2)), np.full(17, 0.4), (640, 480), model="test")
    assert pose.gorunur.all()
