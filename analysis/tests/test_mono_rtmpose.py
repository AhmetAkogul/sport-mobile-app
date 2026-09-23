import numpy as np
import pytest

from mono.rtmpose_model import RTMPoseEstimator, coco_to_pose, COCO_MAP
from pose3d.iskelet import REFERANS_ISKELET


def test_named_mapping_and_neck_confidence():
    points = np.arange(34).reshape(17,2)
    scores = np.ones(17)*0.9
    scores[5] = 0.2
    pose = coco_to_pose(points,scores,(640,480),model='test')
    for name,index in COCO_MAP.items():
        assert np.array_equal(pose.noktalar[REFERANS_ISKELET.indeks(name)],points[index])
    neck = REFERANS_ISKELET.indeks('boyun')
    assert np.array_equal(pose.noktalar[neck],points[[5,6]].mean(axis=0))
    assert pose.guven[neck] == 0.2 and not pose.gorunur[neck]
    assert pose.uzay == 'ozgun'


@pytest.mark.parametrize('points,scores', [
    (np.zeros((16,2)),np.ones(17)), (np.zeros((17,2)),np.ones(16)),
    (np.full((17,2),np.nan),np.ones(17)), (np.zeros((17,2)),np.full(17,np.inf)),
])
def test_invalid_model_output_rejected(points,scores):
    with pytest.raises(ValueError):
        coco_to_pose(points,scores,(640,480),model='test')


def test_scores_are_not_claimed_as_probabilities():
    scores=np.ones(17)*1.2
    pose=coco_to_pose(np.zeros((17,2)),scores,(640,480),model='test')
    assert (pose.guven == 1).all()
    assert pose.ek['score_semantics'] == 'simcc_not_calibrated_probability'
    # Yalnizca kullanilan eklemlerin ham skorlari saklanir (yuz skorlari degil).
    assert pose.ek['raw_scores_joints'] == list(range(5, 17))
    assert pose.ek['raw_scores'] == [scores[i] for i in pose.ek['raw_scores_joints']]


def test_simcc_esigi_mediapipe_esiginden_ayridir():
    """SIMCC skoru olasilik degil: ayni esik iki modelde farkli anlam tasir."""
    from mono.rtmpose_model import SIMCC_GUVEN_ESIGI
    assert SIMCC_GUVEN_ESIGI < 0.5
    scores = np.full(17, 0.4)
    pose = coco_to_pose(np.zeros((17, 2)), scores, (640, 480), model='test')
    assert pose.gorunur.all(), "0.4 SIMCC'de dusuk guven degil; MediaPipe esigiyle saklanmamali"
    assert pose.ek['guven_esigi'] == SIMCC_GUVEN_ESIGI


def estimator(boxes):
    model=RTMPoseEstimator.__new__(RTMPoseEstimator)
    model._closed=False
    model.model_id='test'
    model.threshold=0.5
    model._detector=lambda _: boxes
    return model


def test_no_person_does_not_run_pose_on_full_image():
    model=estimator([])
    model._pose=lambda *a,**kw: pytest.fail('Kişi yokken poz modeli çağrıldı')
    pose=model(np.zeros((100,100,3),np.uint8))
    assert not pose.gorunur.any() and pose.tespit is False


def test_multiple_people_need_explicit_selection():
    model=estimator([[0,0,50,50],[50,50,100,100]])
    with pytest.raises(ValueError,match='Birden fazla'):
        model(np.zeros((100,100,3),np.uint8))


def test_single_person_and_close():
    model=estimator([[0,0,50,50]])
    model._pose=lambda *a,**kw: (np.ones((1,17,2))*25,np.ones((1,17))*.9)
    pose=model(np.zeros((100,100,3),np.uint8))
    assert pose.gorunur.all() and pose.goruntu_boyutu == (100,100)
    model.close()
    model.close()
    with pytest.raises(RuntimeError):
        model(np.zeros((100,100,3),np.uint8))


@pytest.mark.parametrize('threshold',[-1,2,float('nan'),True])
def test_invalid_threshold_before_optional_import(threshold):
    with pytest.raises(ValueError):
        RTMPoseEstimator('missing','missing',threshold=threshold)
