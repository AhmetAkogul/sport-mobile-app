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
    assert pose.tespit is False      # tespit artik sozlesme alani (X.1)


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


def test_gorunmez_eklemin_dunya_tahmini_ayri_anahtarda_korunur():
    # Profilde arkadaki diz: 2B ve dunya gorunurlugu dusuk. `world_points_m`
    # sozlesmesi onu NaN yapar; `world_points_all_m` modelin tahminini saklar.
    from pose3d.iskelet import REFERANS_ISKELET

    def isaretler(z):
        lm = [SimpleNamespace(x=0.25, y=0.75, z=z, visibility=0.9, presence=0.9) for _ in range(33)]
        lm[26] = SimpleNamespace(x=0.3, y=0.7, z=z, visibility=0.1, presence=0.9)  # sag diz
        return lm

    estimator = MediaPipeEstimator.__new__(MediaPipeEstimator)
    estimator._closed, estimator.threshold = False, 0.5
    estimator.model_id, estimator.sha256 = 'test', 'abc'
    estimator._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB='rgb'), Image=lambda **kw: kw['data'])
    estimator._detector = SimpleNamespace(
        detect=lambda rgb: SimpleNamespace(pose_landmarks=[isaretler(0.0)],
                                           pose_world_landmarks=[isaretler(0.2)]),
        close=lambda: None)
    ek = estimator(np.zeros((10, 20, 3), np.uint8)).ek
    diz = REFERANS_ISKELET.indeks("sag_diz")
    gorunur, tam = np.array(ek["world_points_m"]), np.array(ek["world_points_all_m"])
    assert np.isnan(gorunur[diz]).all() and not ek["world_visible"][diz]
    np.testing.assert_allclose(tam[diz], [0.3, 0.7, 0.2])
    assert ek["world_confidence"][diz] == pytest.approx(0.1)
    ayni = np.array(ek["world_visible"])
    np.testing.assert_array_equal(gorunur[ayni], tam[ayni])
