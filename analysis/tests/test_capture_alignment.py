import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from capture.alignment import VideoTimeline, align_timelines, scan_video


def timeline(key, times):
    return VideoTimeline(key, f"{key}.avi", tuple(times))


def align(a, b, tolerance=1, offsets=None):
    return align_timelines([timeline("a", a), timeline("b", b)], reference_camera="a",
                           offsets_ms=offsets or {"a": 0, "b": 0}, tolerance_ms=tolerance)


def test_different_fps_matches_time_not_frame_number():
    result = align([0, 100, 200], [0, 50, 100, 150, 200])
    assert [g["frames"][1]["source_frame_index"] for g in result["groups"]] == [0, 2, 4]
    assert result["unused_frame_indices"]["b"] == [1, 3]
    assert not result["physical_synchronization_verified"]


def test_explicit_offset():
    result = align([0, 100], [50, 150], offsets={"a": 50, "b": 0})
    assert len(result["groups"]) == 2
    assert result["groups"][0]["frames"][0]["source_time_ms"] == 0
    assert result["groups"][0]["frames"][0]["aligned_time_ms"] == 50


def test_no_frame_reuse():
    result = align([0, 25, 50], [0, 50], tolerance=30)
    ids = [g["frames"][1]["source_frame_index"] for g in result["groups"]]
    assert ids == [0, 1]
    assert result["unmatched_reference_frames"] == [{"reference_frame_index": 2, "reason": "source_exhausted"}]


def test_tolerance_rejection_does_not_consume_candidate():
    result = align([0, 100], [100, 200], tolerance=1)
    assert result["groups"][0]["frames"][1]["source_frame_index"] == 0
    assert result["unmatched_reference_frames"][0]["reason"] == "outside_tolerance"


def test_tolerance_is_entire_group_spread():
    result = align_timelines([timeline("a", [10]), timeline("b", [5]), timeline("c", [15])],
                            reference_camera="a", offsets_ms={"a": 0, "b": 0, "c": 0}, tolerance_ms=6)
    assert result["groups"] == []


def test_tie_prefers_earlier_frame():
    result = align([50], [0, 100], tolerance=50)
    assert result["groups"][0]["frames"][1]["source_frame_index"] == 0


@pytest.mark.parametrize("times", [[], [0, 0], [1, 0], [-1, 0], [0, float("nan")], [0, float("inf")]])
def test_bad_timeline(times):
    with pytest.raises(ValueError):
        timeline("a", times)


@pytest.mark.parametrize("offsets", [{"a": 0}, {"a": 0, "b": float("nan")}, {"a": 0, "b": 0, "c": 0}])
def test_offset_required_for_every_camera(offsets):
    with pytest.raises(ValueError):
        align_timelines([timeline("a", [0]), timeline("b", [0])], reference_camera="a",
                        offsets_ms=offsets, tolerance_ms=1)


@pytest.mark.parametrize("tolerance", [-1, True, float("nan")])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        align([0], [0], tolerance=tolerance)


class FakeVideo:
    def __init__(self, times):
        self.times = times
        self.index = -1
        self.released = False
    def isOpened(self):
        return True
    def grab(self):
        self.index += 1
        return self.index < len(self.times)
    def retrieve(self):
        return True, np.zeros((8, 8, 3), np.uint8)
    def get(self, key):
        assert key == cv2.CAP_PROP_POS_MSEC
        return self.times[self.index]
    def release(self):
        self.released = True


@pytest.mark.parametrize("times", [[0, 0, 0], [10, 5], [0], [0, float("nan")]])
def test_scan_rejects_unusable_times_without_fps_fallback(tmp_path, times):
    path = tmp_path / "fake.avi"
    path.touch()
    handle = FakeVideo(times)
    with pytest.raises(ValueError):
        scan_video("a", path, capture_factory=lambda _: handle)
    assert handle.released


def test_scan_limit_never_returns_partial_timeline(tmp_path):
    path = tmp_path / "fake.avi"
    path.touch()
    handle = FakeVideo([0, 100, 200])
    with pytest.raises(ValueError, match="sınır"):
        scan_video("a", path, max_frames=2, capture_factory=lambda _: handle)
    assert handle.released


def test_real_videos_cli(tmp_path):
    videos = []
    for key, fps, count in [("a", 10, 3), ("b", 20, 5)]:
        path = tmp_path / f"{key}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
        assert writer.isOpened()
        try:
            for index in range(count):
                writer.write(np.full((24, 32, 3), index * 20, np.uint8))
        finally:
            writer.release()
        videos.append({"camera_id": key, "path": str(path)})
    config = {"videos": videos, "reference_camera": "a", "offsets_ms": {"a": 0, "b": 0}, "tolerance_ms": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "alignment.json"
    command = [sys.executable, "-m", "capture.alignment", "--config", str(config_path), "--output", str(output)]
    subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    result = json.loads(output.read_text())
    assert len(result["groups"]) == 3
    assert [g["frames"][1]["source_frame_index"] for g in result["groups"]] == [0, 2, 4]
    before = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert repeated.returncode != 0
    assert output.read_bytes() == before
