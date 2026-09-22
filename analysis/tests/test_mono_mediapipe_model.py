from types import SimpleNamespace

import numpy as np
import pytest

from mono.mediapipe_model import MediaPipeEstimator, result_to_pose


def landmarks():
    return [SimpleNamespace(x=0.25, y=0.75, z=0, visibility=0.9, presence=0.8) for _ in range(33)]


def test_missing_pose_is_explicitly_invisible():
    pose = result_to_pose(SimpleNamespace(pose_landmarks=[]), (640, 480), model="test")
    assert pose.noktalar.shape == (13, 2)
    assert not pose.gorunur.any()
    assert not pose.ek['tespit']


def test_real_adapter_preserves_pixel_space_and_model():
    pose = result_to_pose(SimpleNamespace(pose_landmarks=[landmarks()]), (640, 480), model="test")
    assert np.allclose(pose.noktalar, [160, 360])
    assert pose.uzay == 'ozgun' and pose.model == 'test'
    assert pose.gorunur.all()


def test_multiple_poses_not_silently_selected():
    with pytest.raises(ValueError):
        result_to_pose(SimpleNamespace(pose_landmarks=[landmarks(), landmarks()]), (640, 480), model="test")


@pytest.mark.parametrize('threshold', [-1, 2, float('nan'), True])
def test_bad_threshold_fails_before_model_import(threshold):
    with pytest.raises(ValueError):
        MediaPipeEstimator('missing.task', threshold=threshold)


def test_bgr_converted_to_rgb_and_close_idempotent():
    estimator = MediaPipeEstimator.__new__(MediaPipeEstimator)
    estimator._closed = False
    estimator.threshold = 0.5
    estimator.model_id = 'test'
    estimator.sha256 = 'abc'
    images = []
    closed = []
    estimator._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB='rgb'), Image=lambda **kw: kw['data'])
    def detect(rgb):
        images.append(rgb.copy())
        return SimpleNamespace(pose_landmarks=[])
    estimator._detector = SimpleNamespace(detect=detect, close=lambda: closed.append(True))
    image = np.full((10, 20, 3), [10, 20, 30], np.uint8)
    pose = estimator(image)
    assert images[0][0, 0].tolist() == [30, 20, 10]
    assert image[0, 0].tolist() == [10, 20, 30]
    assert pose.goruntu_boyutu == (20, 10)
    estimator.close()
    estimator.close()
    assert closed == [True]
    with pytest.raises(RuntimeError):
        estimator(image)


def test_known_macos_crash_version_rejected_before_native_creation(tmp_path, monkeypatch):
    import sys
    import mono.mediapipe_model as module
    model = tmp_path / 'test.task'
    model.write_bytes(b'test')
    monkeypatch.setitem(sys.modules, 'mediapipe', SimpleNamespace(__version__='1.0.1'))
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    with pytest.raises(RuntimeError, match='0.10.35'):
        MediaPipeEstimator(model)
