import json

import cv2
import numpy as np
import pytest

from capture.settings import configure_source, PROPERTIES
from capture.session import CameraConfig, SessionConfig
from capture.record import record_session
from capture.source import ProcessCapture


class Device:
    def __init__(self, source=0):
        self.values = {PROPERTIES["width"]: 32, PROPERTIES["height"]: 24, PROPERTIES["fps"]: 30}
        self.events = []
        self.released = False
    def isOpened(self):
        return True
    def set(self, key, value):
        self.events.append("set")
        self.values[key] = value
        return True
    def get(self, key):
        self.events.append("get")
        return self.values[key]
    def grab(self):
        return True
    def retrieve(self):
        return True, np.zeros((24, 32, 3), np.uint8)
    def release(self):
        self.released = True


def camera(**kwargs):
    values = dict(camera_id="a", source=0, settings="test", width=32, height=24, fps=30)
    values.update(kwargs)
    return CameraConfig(**values)


def test_apply_then_read_back():
    device = Device()
    result = configure_source(device, camera())
    assert device.events == ["set"] * 3 + ["get"] * 3
    assert result["issues"] == []
    assert result["reported"] == {"width": 32, "height": 24, "fps": 30}
    assert all(result["set_accepted"].values())


def test_file_only_validates_and_never_sets():
    device = Device()
    result = configure_source(device, camera(source="file.avi"))
    assert device.events == ["get"] * 3
    assert result["mode"] == "validate_only"
    assert result["issues"] == []


@pytest.mark.parametrize("value", [0, float("nan"), float("inf")])
def test_unknown_driver_values_report_null(value):
    device = Device()
    device.values[PROPERTIES["fps"]] = value
    report = configure_source(device, camera(source="file.avi"))
    assert report["reported"]["fps"] is None
    assert report["issues"]
    json.dumps(report, allow_nan=False)


def test_rejected_setting_stops_before_frames(tmp_path):
    class Reject(Device):
        def set(self, key, value):
            return False
    device = Reject()
    cfg = SessionConfig("synthetic", "sabit", (camera(),))
    with pytest.raises(ValueError, match="kabul etmedi"):
        record_session(cfg, tmp_path / "session", max_frames=1, capture_factory=lambda _: device)
    metadata = json.loads((tmp_path / "session/session.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["completed_batches"] == 0
    assert metadata["camera_settings"]["a"]["issues"]
    assert device.released


def test_later_setting_changing_resolution_is_detected():
    class ChangesWidth(Device):
        def set(self, key, value):
            result = super().set(key, value)
            if key == PROPERTIES["fps"]:
                self.values[PROPERTIES["width"]] = 640
            return result
    result = configure_source(ChangesWidth(), camera())
    assert any("width" in issue for issue in result["issues"])


def test_actual_image_checked_even_if_driver_claims_match(tmp_path):
    cfg = SessionConfig("synthetic", "sabit", (camera(width=640),))
    with pytest.raises(ValueError, match="Gerçek kare boyutu"):
        record_session(cfg, tmp_path / "session", max_frames=1, capture_factory=Device)
    assert not list((tmp_path / "session/batches").iterdir())


@pytest.mark.parametrize("fps,accepted", [(29.97, True), (15, False)])
def test_fps_tolerance(fps, accepted):
    device = Device()
    device.values[PROPERTIES["fps"]] = fps
    report = configure_source(device, camera(source="file.avi"))
    assert (not report["issues"]) is accepted


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"height": -1}, {"width": 1.2}, {"height": True},
    {"fps": True}, {"fps": 0}, {"fps": float("nan")}, {"fps": float("inf")},
])
def test_bad_setting_rejected(kwargs):
    with pytest.raises(ValueError):
        camera(**kwargs)


def test_process_get_set_roundtrip():
    source = ProcessCapture(0, factory=Device)
    try:
        report = configure_source(source, camera())
        assert report["issues"] == []
        assert report["reported"]["fps"] == 30
    finally:
        source.release()


def test_real_video_report_and_dimensions(tmp_path):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    try:
        writer.write(np.zeros((24, 32, 3), np.uint8))
    finally:
        writer.release()
    cfg = SessionConfig("synthetic", "sabit", (camera(source=str(path), fps=10),))
    result = record_session(cfg, tmp_path / "session", max_frames=1)
    assert result["status"] == "completed"
    report = result["camera_settings"]["a"]
    assert report["mode"] == "validate_only"
    assert report["reported"] == {"width": 32, "height": 24, "fps": 10}
