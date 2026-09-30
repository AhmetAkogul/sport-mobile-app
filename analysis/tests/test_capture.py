"""Kayıt sözleşmesi ve gerçek OpenCV video okuma kabul testleri."""
import json

import cv2
import numpy as np
import pytest

from capture.record import record_session
from capture.session import CameraConfig, SessionConfig


def config():
    return SessionConfig("synthetic", "sentetik sabit", (
        CameraConfig("left", 0, "varsayılan"), CameraConfig("right", 1, "varsayılan"),
    ))


class FakeCapture:
    def __init__(self, count=3, opened=True, decode=True):
        self.remaining = count
        self.opened = opened
        self.decode = decode
        self.released = False

    def isOpened(self):
        return self.opened

    def grab(self):
        self.remaining -= 1
        return self.remaining >= 0

    def retrieve(self):
        return self.decode, np.full((24, 32, 3), 100, dtype=np.uint8)

    def release(self):
        self.released = True


def run(tmp_path, handles, count=3):
    return record_session(config(), tmp_path / "session", max_frames=count,
                          capture_factory=lambda source: handles[source])


def test_equal_counts_timestamps_and_metadata(tmp_path):
    handles = [FakeCapture(), FakeCapture()]
    result = run(tmp_path, handles)
    assert result["status"] == "completed"
    assert result["completed_batches"] == 3
    rows = [json.loads(line) for line in (tmp_path / "session/frames.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    for batch in rows:
        assert len(batch["frames"]) == 2
        for frame in batch["frames"]:
            assert frame["grab_started_ns"] <= frame["grab_finished_ns"]
            assert cv2.imread(str(tmp_path / "session" / frame["path"])).shape == (24, 32, 3)
    assert all(h.released for h in handles)
    assert not result["hardware_synchronized"]


def test_short_source_discards_incomplete_group(tmp_path):
    result = run(tmp_path, [FakeCapture(3), FakeCapture(1)])
    assert result["status"] == "source_stopped"
    assert result["stopped_camera"] == "right"
    assert result["completed_batches"] == 1
    assert len(list((tmp_path / "session/batches").glob("*/*.png"))) == 2


@pytest.mark.parametrize("handles", [
    [FakeCapture(), FakeCapture(opened=False)],
    [FakeCapture(), FakeCapture(decode=False)],
])
def test_failure_releases_resources_and_reports_state(tmp_path, handles):
    with pytest.raises(RuntimeError):
        run(tmp_path, handles)
    assert all(h.released for h in handles)
    state = json.loads((tmp_path / "session/session.json").read_text())
    assert state["status"] == "failed"
    assert state["completed_batches"] == 0


def test_write_failure_rolls_back_group(tmp_path, monkeypatch):
    original = cv2.imwrite
    def fail_second(path, frame):
        return False if path.endswith("right.png") else original(path, frame)
    monkeypatch.setattr(cv2, "imwrite", fail_second)
    handles = [FakeCapture(), FakeCapture()]
    with pytest.raises(OSError):
        run(tmp_path, handles)
    assert not list((tmp_path / "session/batches").iterdir())
    assert not (tmp_path / "session/.pending").exists()
    assert all(h.released for h in handles)


def test_never_overwrites_existing_session(tmp_path):
    run(tmp_path, [FakeCapture(), FakeCapture()])
    with pytest.raises(FileExistsError):
        run(tmp_path, [FakeCapture(), FakeCapture()])


@pytest.mark.parametrize("kwargs", [
    {"participant_code": ""}, {"lighting": " "}, {"cameras": ()},
    {"cameras": (CameraConfig("same", 0, "auto"), CameraConfig("same", 1, "auto"))},
    {"cameras": (CameraConfig("a", 0, "auto"), CameraConfig("b", 0, "auto"))},
])
def test_metadata_required(kwargs):
    values = dict(participant_code="synthetic", lighting="sabit", cameras=config().cameras)
    values.update(kwargs)
    with pytest.raises(ValueError):
        SessionConfig(**values)


@pytest.mark.parametrize("camera_id,source,settings", [
    ("../escape", 0, "auto"), ("a", -1, "auto"), ("a", True, "auto"),
    ("a", "", "auto"), ("a", 0, ""),
])
def test_invalid_camera_config(camera_id, source, settings):
    with pytest.raises(ValueError):
        CameraConfig(camera_id, source, settings)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_limit_does_not_start_recording(tmp_path, count):
    with pytest.raises(ValueError):
        run(tmp_path, [], count)
    assert not (tmp_path / "session").exists()


def test_missing_config_does_not_start_recording(tmp_path):
    with pytest.raises(ValueError):
        record_session(None, tmp_path / "session", max_frames=1)
    assert not (tmp_path / "session").exists()


def test_json_config_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config().to_dict()))
    assert SessionConfig.from_json(path) == config()


def test_actual_opencv_video_sources(tmp_path):
    cameras = []
    for camera_id in ("left", "right"):
        path = tmp_path / f"{camera_id}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        assert writer.isOpened(), "Sentetik video için MJPG kodlayıcısı gerekli"
        try:
            for value in (30, 100, 200):
                writer.write(np.full((24, 32, 3), value, np.uint8))
        finally:
            writer.release()
        cameras.append(CameraConfig(camera_id, str(path), "MJPG 32x24 10 fps"))
    session = SessionConfig("synthetic", "sabit", tuple(cameras))
    result = record_session(session, tmp_path / "session", max_frames=3)
    assert result["completed_batches"] == 3
    assert result["status"] == "completed"
    for camera in cameras:
        for index, expected in enumerate((30, 100, 200)):
            image = cv2.imread(str(tmp_path / f"session/batches/{index:06d}/{camera.camera_id}.png"))
            assert image.mean() == pytest.approx(expected, abs=3)


def test_all_grabs_precede_decoding(tmp_path):
    events = []
    class OrderedCapture(FakeCapture):
        def grab(self):
            events.append("grab")
            return super().grab()
        def retrieve(self):
            events.append("retrieve")
            return super().retrieve()
    run(tmp_path, [OrderedCapture(), OrderedCapture()], count=1)
    assert events == ["grab", "grab", "retrieve", "retrieve"]


def test_interrupt_preserves_completed_groups(tmp_path):
    class InterruptedCapture(FakeCapture):
        def grab(self):
            if self.remaining == 2:
                raise KeyboardInterrupt()
            return super().grab()
    handles = [InterruptedCapture(), FakeCapture()]
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, handles)
    state = json.loads((tmp_path / "session/session.json").read_text())
    assert state["status"] == "interrupted"
    assert state["completed_batches"] == 1
    assert len(list((tmp_path / "session/batches").glob("*/*.png"))) == 2
    assert all(h.released for h in handles)


def test_camera_names_cannot_collide_on_case_insensitive_filesystem():
    with pytest.raises(ValueError):
        SessionConfig("synthetic", "sabit", (
            CameraConfig("Left", 0, "auto"), CameraConfig("left", 1, "auto")))


@pytest.mark.parametrize("image", [np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8),
                                    np.zeros((8, 8, 3), np.float32)])
def test_unsupported_image_is_not_published(tmp_path, image):
    class InvalidImage(FakeCapture):
        def retrieve(self):
            return True, image
    handles = [FakeCapture(), InvalidImage()]
    with pytest.raises(ValueError, match="BGR"):
        run(tmp_path, handles)
    assert not list((tmp_path / "session/batches").iterdir())
    assert all(h.released for h in handles)
