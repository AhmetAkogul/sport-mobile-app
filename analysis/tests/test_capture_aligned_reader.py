import json

import cv2
import numpy as np
import pytest

from capture.alignment import VideoTimeline, align_timelines, file_sha256, scan_video
from capture.aligned_reader import open_aligned_batches


@pytest.fixture
def plan_path(tmp_path):
    timelines = []
    for key, fps, count in [("a", 10, 3), ("b", 20, 5)]:
        path = tmp_path / f"{key}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
        assert writer.isOpened()
        try:
            for index in range(count):
                writer.write(np.full((24, 32, 3), 20 + index * 30, np.uint8))
        finally:
            writer.release()
        timelines.append(VideoTimeline(key, str(path), tuple(i * 1000 / fps for i in range(count)), file_sha256(path)))
    plan = align_timelines(timelines, reference_camera="a", offsets_ms={"a": 0, "b": 0}, tolerance_ms=1)
    output = tmp_path / "plan.json"
    output.write_text(json.dumps(plan))
    return output


def test_real_pixels_and_source_identity_match(plan_path):
    with open_aligned_batches(plan_path) as batches:
        result = list(batches)
    assert len(result) == 3
    for group_id, frames in enumerate(result):
        assert [f.camera_id for f in frames] == ["a", "b"]
        assert [f.source_frame_index for f in frames] == [group_id, group_id * 2]
        for frame in frames:
            assert frame.image_bgr.mean() == pytest.approx(20 + frame.source_frame_index * 30, abs=3)
            assert frame.aligned_time_ms == pytest.approx(group_id * 100)
            assert frame.source_path.is_absolute()


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version=1),
    lambda d: d["sources"][0].update(sha256=None),
    lambda d: d["groups"][0]["frames"].pop(),
    lambda d: d["groups"][1]["frames"][0].update(source_frame_index=0),
    lambda d: d["groups"][0]["frames"][0].update(source_frame_index=999),
    lambda d: d["groups"][0]["frames"][0].update(aligned_time_ms=50),
    lambda d: d["groups"][0].update(spread_ms=20),
    lambda d: d.update(physical_synchronization_verified=True),
])
def test_bad_plan_rejected_before_opening_source(plan_path, change):
    plan = json.loads(plan_path.read_text())
    change(plan)
    plan_path.write_text(json.dumps(plan))
    def forbidden(_):
        pytest.fail("Plan doğrulanmadan kaynak açıldı")
    with pytest.raises(ValueError):
        with open_aligned_batches(plan_path, capture_factory=forbidden):
            pass


def test_changed_source_rejected_before_decode(plan_path):
    video = plan_path.parent / "a.avi"
    with video.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="değişmiş"):
        with open_aligned_batches(plan_path):
            pass


class FakeSource:
    def __init__(self, path):
        self.released = False
        self.index = -1
        self.fps = 10 if path.endswith("a.avi") else 20
        self.bad_time = False
        self.stop = False
    def isOpened(self):
        return True
    def grab(self):
        self.index += 1
        return not self.stop
    def retrieve(self):
        return True, np.zeros((24, 32, 3), np.uint8)
    def get(self, prop):
        return 999 if self.bad_time else self.index * 1000 / self.fps
    def release(self):
        self.released = True


def test_early_exit_closes_all_sources(plan_path):
    sources = []
    def factory(path):
        source = FakeSource(path)
        sources.append(source)
        return source
    with open_aligned_batches(plan_path, capture_factory=factory) as batches:
        assert len(next(batches)) == 2
    assert all(s.released for s in sources)
    with pytest.raises(RuntimeError, match="with"):
        next(batches)


@pytest.mark.parametrize("failure", ["time", "eof"])
def test_no_partial_group_on_source_failure(plan_path, failure):
    sources = []
    def factory(path):
        source = FakeSource(path)
        if path.endswith("b.avi"):
            source.bad_time = failure == "time"
            source.stop = failure == "eof"
        sources.append(source)
        return source
    with pytest.raises(ValueError):
        with open_aligned_batches(plan_path, capture_factory=factory) as batches:
            next(batches)
    assert all(s.released for s in sources)


def test_later_open_failure_closes_earlier_source(plan_path):
    sources = []
    def factory(path):
        if path.endswith("b.avi"):
            raise RuntimeError("open failed")
        source = FakeSource(path)
        sources.append(source)
        return source
    with pytest.raises(RuntimeError):
        with open_aligned_batches(plan_path, capture_factory=factory):
            pass
    assert sources[0].released


def test_scan_plan_reader_roundtrip(plan_path):
    timelines = [scan_video(key, plan_path.parent / f"{key}.avi") for key in ("a", "b")]
    plan = align_timelines(timelines, reference_camera="a", offsets_ms={"a": 0, "b": 0}, tolerance_ms=1)
    plan_path.write_text(json.dumps(plan))
    with open_aligned_batches(plan_path) as batches:
        assert len(list(batches)) == 3


def test_change_between_groups_is_detected(plan_path):
    with open_aligned_batches(plan_path, capture_factory=FakeSource) as batches:
        next(batches)
        with (plan_path.parent / "a.avi").open("ab") as file:
            file.write(b"changed")
        with pytest.raises(ValueError, match="okuma sırasında"):
            next(batches)


def test_empty_plan_opens_no_decoders(plan_path):
    plan = json.loads(plan_path.read_text())
    plan["groups"] = []
    plan_path.write_text(json.dumps(plan))
    def forbidden(_):
        pytest.fail("Boş plan için kaynak açıldı")
    with open_aligned_batches(plan_path, capture_factory=forbidden) as batches:
        assert list(batches) == []
