"""Gerçek alt süreçlerle kilitlenme ve temiz kapanma kabul testleri."""
import json
import multiprocessing as mp
import time

import numpy as np
import pytest

from capture.source import ProcessCapture
from capture.record import record_session
from capture.session import CameraConfig, SessionConfig


class FakeDevice:
    def __init__(self, mode):
        self.mode = mode
        self.count = 0
        if mode == "open":
            time.sleep(30)
    def isOpened(self):
        return self.mode != "unopened"
    def grab(self):
        self.count += 1
        if self.mode == "grab" or self.mode == "second" and self.count == 2:
            time.sleep(30)
        if self.mode == "error":
            raise ValueError("driver failure")
        if self.mode == "exit":
            import os
            os._exit(3)
        return True
    def retrieve(self):
        if self.mode == "retrieve":
            time.sleep(30)
        return True, np.full((24, 32, 3), 7, np.uint8)
    def get(self, property_id):
        if self.mode == "get":
            time.sleep(30)
        return 0.0
    def set(self, property_id, value):
        if self.mode == "set":
            time.sleep(30)
        return False
    def release(self):
        if self.mode == "close":
            time.sleep(30)


def device(mode):
    return ProcessCapture(mode, timeout_s=0.3, open_timeout_s=5, factory=FakeDevice)


def test_normal_roundtrip_and_idempotent_close():
    source = device("normal")
    try:
        assert source.isOpened()
        assert source.grab()
        ok, frame = source.retrieve()
        assert ok and frame.shape == (24, 32, 3) and frame.mean() == 7
    finally:
        source.release()
    source.release()
    assert not source.isOpened()
    assert not source._process.is_alive()
    assert not source._thread.is_alive()


@pytest.mark.parametrize("operation", ["grab", "retrieve", "close"])
def test_hung_operation_is_bounded_and_worker_exits(operation):
    source = device(operation)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match=operation):
        {"grab": source.grab, "retrieve": source.retrieve, "close": source.release}[operation]()
    assert time.monotonic() - started < 3
    assert not source._process.is_alive()
    assert not source._thread.is_alive()
    source.release()


def test_hung_open_leaves_no_child():
    before = {p.pid for p in mp.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="open"):
        ProcessCapture("open", timeout_s=0.3, open_timeout_s=1, factory=FakeDevice)
    assert time.monotonic() - started < 4
    assert {p.pid for p in mp.active_children()} == before


@pytest.mark.parametrize("mode", ["error", "exit"])
def test_worker_error_and_crash_reach_parent(mode):
    source = device(mode)
    with pytest.raises(RuntimeError):
        source.grab()
    assert not source._process.is_alive()


def test_open_failure_closes_worker():
    source = device("unopened")
    assert not source.isOpened()
    assert not source._process.is_alive()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        ProcessCapture("normal", timeout_s=timeout, factory=FakeDevice)


def test_timeout_preserves_complete_groups_and_failed_metadata(tmp_path):
    handles = []
    def factory(mode):
        handle = device(mode)
        handles.append(handle)
        return handle
    config = SessionConfig("synthetic", "sabit", (
        CameraConfig("a", "normal", "auto"), CameraConfig("b", "second", "auto")))
    with pytest.raises(TimeoutError):
        record_session(config, tmp_path / "session", max_frames=3, capture_factory=factory)
    meta = json.loads((tmp_path / "session/session.json").read_text())
    assert meta["status"] == "failed"
    assert meta["completed_batches"] == 1
    assert "TimeoutError" in meta["error"]
    assert len(list((tmp_path / "session/batches").glob("*/*.png"))) == 2
    assert all(not h._process.is_alive() for h in handles)


def test_release_timeout_marks_session_failed(tmp_path):
    cfg = SessionConfig("synthetic", "sabit", (CameraConfig("a", "close", "auto"),))
    result = record_session(cfg, tmp_path / "session", max_frames=1, capture_factory=device)
    assert result["status"] == "failed"
    assert result["completed_batches"] == 1
    assert "TimeoutError" in result["cleanup_errors"][0]


def test_cli_uses_process_isolation_with_real_video(tmp_path):
    import subprocess
    import sys
    import cv2
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    try:
        for value in (30, 100):
            writer.write(np.full((24, 32, 3), value, np.uint8))
    finally:
        writer.release()
    cfg = SessionConfig("synthetic", "sabit", (CameraConfig("a", str(path), "MJPG"),))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg.to_dict()))
    result = subprocess.run([
        sys.executable, "-m", "capture.record", "--config", str(cfg_path),
        "--output", str(tmp_path / "session"), "--max-frames", "2",
        "--source-timeout", "2", "--open-timeout", "10",
    ], capture_output=True, text=True, timeout=20, check=True)
    metadata = json.loads(result.stdout)
    assert metadata["status"] == "completed"
    assert metadata["source_isolation"] == "process"
    assert metadata["source_timeout_s"] == 2
    assert metadata["completed_batches"] == 2


@pytest.mark.parametrize("operation", ["get", "set"])
def test_hung_property_call_is_bounded(operation):
    source = device(operation)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        if operation == "get":
            source.get(3)
        else:
            source.set(3, 640)
    assert time.monotonic() - started < 3
    assert not source._process.is_alive()
    assert not source._thread.is_alive()
