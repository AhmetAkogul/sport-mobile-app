"""Poz modeli secimi: iki CLI de ayni fabrikayi kullanir.

B2 karsilastirmasinin kabul olcutu "ayni arayuz, ayni test goruntuleri" idi ama
`run_video.py`/`run_phone.py` MediaPipe'a gomuluydu; RTMPose ile ayni isi yapan
bir giris noktasi yoktu (dis inceleme V.3/U.8).

Esik **modele gore** farklidir ve varsayilani buradan gelir: MediaPipe
`visibility/presence` olasiligi icin 0.5, RTMPose SIMCC skoru icin 0.3. Ayni
sayiyi iki modele uygulamak karsilastirmayi yaniltir (`docs/kararlar/0031`).
"""
from __future__ import annotations

MEDIAPIPE_GUVEN_ESIGI = 0.5


def kestirici_olustur(backend: str, *, model: str, detector: str | None = None,
                      threshold: float | None = None, coklu_kisi: str = "hata",
                      el_modeli=None):
    """Backend adindan estimator uret (context manager olarak kullanilir).

    Donen nesnenin `model_id` alani rapora yazilir; hangi surum/varyant ile
    olculdugu boylece kayitta kalir.
    """
    if backend == "mediapipe":
        from mono.mediapipe_model import MediaPipeEstimator
        return MediaPipeEstimator(model, threshold=(
            MEDIAPIPE_GUVEN_ESIGI if threshold is None else threshold), el_modeli=el_modeli)
    if backend == "rtmpose":
        if not detector:
            raise ValueError("rtmpose icin dedektor modeli (--detector) gerekli")
        from mono.rtmpose_model import SIMCC_GUVEN_ESIGI, RTMPoseEstimator
        return RTMPoseEstimator(detector, model, threshold=(
            SIMCC_GUVEN_ESIGI if threshold is None else threshold), coklu_kisi=coklu_kisi)
    raise ValueError(f"bilinmeyen backend: {backend!r} (mediapipe | rtmpose)")
