"""mono.run_video testi: sahte kaynak ve sahte model ile JSONL sozlesmesi.

Sinananlar: eksik eklem JSON'da null olur (0.0 degil), `tespit` alani satirda ve
ozette tutarli, `frame_limit`/`source_end` ayrimi dogru, `physical_validation`
daima False.
"""
import json

import numpy as np
import pytest

from mono import run_video
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B

BOYUT = (320, 240)


class SahteKaynak:
    def __init__(self, source, **ayar):
        self.kalan = 3

    def isOpened(self):
        return True

    def grab(self):
        if self.kalan <= 0:
            return False
        self.kalan -= 1
        return True

    def retrieve(self):
        return True, np.zeros((BOYUT[1], BOYUT[0], 3), np.uint8)

    def release(self):
        pass


class SahteModel:
    model_id = "sahte/mediapipe-9.9/pose_landmarker_full/sha256:abc"

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def __call__(self, image):
        n = len(REFERANS_ISKELET)
        noktalar = np.array([[50.0 + 4.0 * i, 40.0 + 2.0 * i] for i in range(n)])
        gorunur = np.ones(n, dtype=bool)
        noktalar[1] = np.nan
        gorunur[1] = False
        return Poz2B(REFERANS_ISKELET, noktalar, np.ones(n), gorunur, True, BOYUT,
                     model=self.model_id)


class SahteTespitsiz(SahteModel):
    """Hic poz bulamayan model: tespit False, butun eklemler gorunmez."""

    def __call__(self, image):
        n = len(REFERANS_ISKELET)
        return Poz2B(REFERANS_ISKELET, np.full((n, 2), np.nan), np.zeros(n),
                     np.zeros(n, bool), False, BOYUT, model=self.model_id)


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    video = tmp_path / "klip.mp4"
    video.write_bytes(b"sahte video")
    monkeypatch.setattr(run_video, "ProcessCapture", SahteKaynak)
    monkeypatch.setattr(run_video, "kestirici_olustur", lambda *a, **k: SahteModel())
    return video, tmp_path / "cikti"


def test_ozet_ve_jsonl_tutarli(ortam):
    video, cikti = ortam
    rapor = run_video.run_video(video, "model.task", cikti, max_frames=10)
    assert rapor["status"] == "completed"
    assert rapor["stop_reason"] == "source_end_or_read_failure"
    assert rapor["frames"] == 3 and rapor["detected_frames"] == 3
    assert rapor["physical_validation"] is False
    assert rapor["model"] == SahteModel.model_id
    satirlar = [json.loads(s) for s in (cikti / "poses.jsonl").read_text().splitlines()]
    assert len(satirlar) == 3
    for i, satir in enumerate(satirlar):
        assert satir["frame_index"] == i
        assert satir["tespit"] is True
        assert satir["points_px"][1] == [None, None]     # NaN -> null
        assert satir["visible"][1] is False
        assert satir["joints"] == list(REFERANS_ISKELET.eklemler)
    assert json.loads((cikti / "summary.json").read_text())["frames"] == 3


def test_tespitsiz_kare_detected_frames_i_artirmaz(ortam, monkeypatch):
    video, cikti = ortam
    monkeypatch.setattr(run_video, "kestirici_olustur", lambda *a, **k: SahteTespitsiz())
    rapor = run_video.run_video(video, "model.task", cikti, max_frames=10)
    assert rapor["frames"] == 3 and rapor["detected_frames"] == 0
    satir = json.loads((cikti / "poses.jsonl").read_text().splitlines()[0])
    assert satir["tespit"] is False
    assert all(p == [None, None] for p in satir["points_px"])


def test_frame_limit(ortam, monkeypatch):
    video, cikti = ortam

    class Sinirsiz(SahteKaynak):
        def __init__(self, source, **ayar):
            self.kalan = 10 ** 6

    monkeypatch.setattr(run_video, "ProcessCapture", Sinirsiz)
    rapor = run_video.run_video(video, "model.task", cikti, max_frames=4)
    assert rapor["stop_reason"] == "frame_limit" and rapor["frames"] == 4


def test_cikti_dizini_yeni_olmali(ortam):
    video, cikti = ortam
    run_video.run_video(video, "model.task", cikti, max_frames=1)
    with pytest.raises(FileExistsError):
        run_video.run_video(video, "model.task", cikti, max_frames=1)


def test_max_frames_dogrulanir(ortam):
    video, cikti = ortam
    with pytest.raises(ValueError, match="max_frames"):
        run_video.run_video(video, "model.task", cikti, max_frames=-1)
