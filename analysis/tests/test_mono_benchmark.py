"""mono.benchmark: 0031'deki gecikme/bellek sayilarini ureten arac.

Gercek model yerine sahte kestirici; olculen sey olcum duzenegi: kare okuma,
isinma cagrilari, olcum sayisi, rapor alanlari ve az ornek uyarisi.
"""
import json

import cv2
import numpy as np
import pytest

import mono.mediapipe_model as mp_modul
from mono import benchmark
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B


class SahteKestirici:
    model_id = "sahte/benchmark"

    def __init__(self, *a, **k):
        self.cagri = 0

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def __call__(self, image):
        self.cagri += 1
        n = len(REFERANS_ISKELET)
        return Poz2B(REFERANS_ISKELET, np.zeros((n, 2)), np.ones(n), np.ones(n, bool),
                     True, (image.shape[1], image.shape[0]), model=self.model_id)


@pytest.fixture
def video(tmp_path):
    yol = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for i in range(5):
        w.write(np.full((32, 32, 3), 20 * i, np.uint8))
    w.release()
    return yol


def test_rapor_olcum_sayisini_ve_kapsami_tasir(tmp_path, video, monkeypatch, capsys):
    monkeypatch.setattr(mp_modul, "MediaPipeEstimator", SahteKestirici)
    hedef = tmp_path / "rapor.json"
    rapor = benchmark.benchmark("mediapipe", video, hedef, model="m.task", frames=5, repeats=2)
    assert rapor["measured_calls"] == 10 and rapor["warmup_calls"] == 2
    assert rapor["detected_calls"] == 10
    assert rapor["initialization_scope"] == "model_load_only_imports_excluded"
    assert rapor["latency_ms_median"] >= 0.0
    assert json.loads(hedef.read_text())["model"] == "sahte/benchmark"
    # 10 cagri < 100: az ornek uyarisi basilir
    assert "en az 100" in capsys.readouterr().err


def test_videoda_az_kare_varsa_uyarir_ama_calisir(tmp_path, video, monkeypatch, capsys):
    monkeypatch.setattr(mp_modul, "MediaPipeEstimator", SahteKestirici)
    rapor = benchmark.benchmark("mediapipe", video, tmp_path / "r.json", model="m", frames=50,
                                repeats=1)
    assert rapor["measured_calls"] == 5
    assert "50 karenin 5" in capsys.readouterr().err


def test_var_olan_cikti_ve_bilinmeyen_backend_reddedilir(tmp_path, video):
    (tmp_path / "var.json").write_text("{}")
    with pytest.raises(FileExistsError):
        benchmark.benchmark("mediapipe", video, tmp_path / "var.json", model="m")
    with pytest.raises(ValueError, match="Bilinmeyen"):
        benchmark.benchmark("openpose", video, tmp_path / "yeni.json", model="m")
