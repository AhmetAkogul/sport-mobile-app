"""Disk hataları sonlandırmada asıl kayıt hatasını gizlememeli."""
import errno
import json

import numpy as np
import pytest

import capture.record as module
from capture.session import CameraConfig, SessionConfig


class Source:
    def __init__(self):
        self.released = False
    def isOpened(self):
        return True
    def grab(self):
        return True
    def retrieve(self):
        return True, np.zeros((8, 8, 3), np.uint8)
    def release(self):
        self.released = True


def setup_record(tmp_path):
    handles = [Source(), Source()]
    config = SessionConfig("synthetic", "sabit", (
        CameraConfig("a", 0, "auto"), CameraConfig("b", 1, "auto")))
    return handles, lambda: module.record_session(config, tmp_path / "session", max_frames=2,
                                                capture_factory=lambda key: handles[key])


def test_disk_full_preserves_primary_error_and_previous_batch(tmp_path, monkeypatch):
    handles, run = setup_record(tmp_path)
    original_write = module.cv2.imwrite
    original_json = module._write_json
    calls = 0
    full = False
    def write_image(path, image):
        nonlocal calls, full
        calls += 1
        if calls == 4:
            full = True
            raise OSError(errno.ENOSPC, "image disk full")
        return original_write(path, image)
    def write_json(path, value):
        if full:
            raise OSError(errno.ENOSPC, "metadata disk full")
        return original_json(path, value)
    monkeypatch.setattr(module.cv2, "imwrite", write_image)
    monkeypatch.setattr(module, "_write_json", write_json)
    with pytest.raises(OSError, match="image disk full") as error:
        run()
    assert any("metadata disk full" in note for note in error.value.__notes__)
    assert all(h.released for h in handles)
    assert len(list((tmp_path / "session/batches").glob("*/*.png"))) == 2
    assert not (tmp_path / "session/.pending").exists()
    # Disk doluyken son durumu yazabildiğimizi iddia etmiyoruz.
    assert json.loads((tmp_path / "session/session.json").read_text())["status"] == "recording"


def test_cleanup_failure_does_not_hide_write_error(tmp_path, monkeypatch):
    handles, run = setup_record(tmp_path)
    def fail_write(*args):
        raise OSError(errno.ENOSPC, "primary write failure")
    def fail_remove(*args):
        raise PermissionError("cleanup denied")
    monkeypatch.setattr(module.cv2, "imwrite", fail_write)
    monkeypatch.setattr(module.shutil, "rmtree", fail_remove)
    with pytest.raises(OSError, match="primary write failure") as error:
        run()
    assert any("cleanup denied" in note for note in error.value.__notes__)
    assert all(h.released for h in handles)
    state = json.loads((tmp_path / "session/session.json").read_text())
    assert state["status"] == "failed"
    assert state["cleanup_errors"]


def test_final_metadata_failure_is_not_success(tmp_path, monkeypatch):
    handles, run = setup_record(tmp_path)
    original = module._write_json
    def fail_final(path, value):
        if path.name == "session.json" and value["status"] == "completed":
            raise OSError(errno.ENOSPC, "final metadata full")
        return original(path, value)
    monkeypatch.setattr(module, "_write_json", fail_final)
    with pytest.raises(OSError, match="final metadata full"):
        run()
    assert all(h.released for h in handles)
    assert len(list((tmp_path / "session/batches").glob("*/*.png"))) == 4
