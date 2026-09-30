import json

import numpy as np
import pytest

from capture.reader import iter_batches
from capture.record import record_session
from capture.session import CameraConfig, SessionConfig


class Source:
    def isOpened(self):
        return True
    def grab(self):
        return True
    def retrieve(self):
        return True, np.full((8, 12, 3), (10, 20, 30), np.uint8)
    def release(self):
        pass


@pytest.fixture
def recorded(tmp_path):
    config = SessionConfig("synthetic", "sabit", (
        CameraConfig("left", 0, "auto"), CameraConfig("right", 1, "auto")))
    path = tmp_path / "session"
    record_session(config, path, max_frames=2, capture_factory=lambda _: Source())
    return path


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_roundtrip_preserves_identity_pixels_and_times(recorded):
    batches = list(iter_batches(recorded))
    assert len(batches) == 2
    for i, batch in enumerate(batches):
        assert [f.camera_id for f in batch] == ["left", "right"]
        for frame in batch:
            assert frame.batch_id == i
            assert frame.session_path == recorded.resolve()
            assert frame.image_bgr[0, 0].tolist() == [10, 20, 30]
            assert frame.grab_started_ns <= frame.grab_finished_ns
            assert frame.timestamp_semantics == "host_monotonic_grab_interval_not_exposure_time"


def test_uses_committed_groups_when_index_missing(recorded):
    (recorded / "frames.jsonl").unlink()
    assert len(list(iter_batches(recorded))) == 2


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version=99),
    lambda d: d.update(status="recording"),
    lambda d: d.update(completed_batches=3),
    lambda d: d.update(timestamp_semantics="exposure"),
])
def test_bad_session_metadata(recorded, change):
    edit(recorded / "session.json", change)
    with pytest.raises(ValueError):
        list(iter_batches(recorded))


@pytest.mark.parametrize("change", [
    lambda d: d.update(batch_id=5),
    lambda d: d["frames"].pop(),
    lambda d: d["frames"][0].update(path="../../outside.png"),
    lambda d: d["frames"][0].update(camera_id="other"),
    lambda d: d["frames"][0].update(shape=[1, 1, 3]),
    lambda d: d["frames"][0].update(grab_finished_ns=-1),
])
def test_bad_frame_metadata(recorded, change):
    edit(recorded / "batches/000000/frames.json", change)
    with pytest.raises(ValueError):
        list(iter_batches(recorded))


def test_missing_image_never_yields_partial_batch(recorded):
    (recorded / "batches/000000/right.png").unlink()
    with pytest.raises(ValueError, match="okunamadı"):
        next(iter_batches(recorded))


def test_backward_time_rejected(recorded):
    edit(recorded / "batches/000001/frames.json",
         lambda d: d["frames"][0].update(grab_started_ns=0, grab_finished_ns=1))
    with pytest.raises(ValueError, match="geriye"):
        list(iter_batches(recorded))
