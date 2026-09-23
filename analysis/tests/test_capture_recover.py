import json

import numpy as np
import pytest

from capture.reader import iter_batches
from capture.record import record_session
from capture.recover import recover_session
from capture.session import CameraConfig, SessionConfig


class Source:
    def isOpened(self):
        return True
    def grab(self):
        return True
    def retrieve(self):
        return True, np.full((8, 12, 3), 42, np.uint8)
    def release(self):
        pass


@pytest.fixture
def crashed(tmp_path):
    source = tmp_path / "source"
    cfg = SessionConfig("synthetic", "sabit", (
        CameraConfig("a", 0, "auto"), CameraConfig("b", 1, "auto")))
    record_session(cfg, source, max_frames=2, capture_factory=lambda _: Source())
    metadata = json.loads((source / "session.json").read_text())
    metadata.update(status="recording", completed_batches=0)
    metadata.pop("finished_at_utc")
    (source / "session.json").write_text(json.dumps(metadata))
    (source / "frames.jsonl").write_text('{"partial":')
    (source / ".pending").mkdir()
    (source / ".pending/incomplete.png").write_bytes(b"partial")
    return source


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_recovers_complete_groups_without_changing_original(crashed, tmp_path):
    original = snapshot(crashed)
    target = tmp_path / "recovered"
    result = recover_session(crashed, target)
    assert snapshot(crashed) == original
    assert result["status"] == "recovered"
    assert result["completed_batches"] == 2
    assert result["recovery"]["pending_ignored"]
    assert "finished_at_utc" not in result
    assert not (target / ".pending").exists()
    assert (target / "recovery-original-session.json").read_bytes() == original["session.json"]
    assert len((target / "frames.jsonl").read_text().splitlines()) == 2
    groups = list(iter_batches(target))
    assert len(groups) == 2
    assert all(f.image_bgr.mean() == 42 for group in groups for f in group)


@pytest.mark.parametrize("corruption", ["image", "gap", "manifest", "symlink", "version"])
def test_corrupt_committed_group_is_not_silently_discarded(crashed, tmp_path, corruption):
    if corruption == "image":
        (crashed / "batches/000001/a.png").write_bytes(b"bad")
    elif corruption == "gap":
        (crashed / "batches/000001").rename(crashed / "batches/000002")
    elif corruption == "manifest":
        (crashed / "batches/000001/frames.json").write_text("invalid")
    elif corruption == "symlink":
        path = crashed / "batches/000001/a.png"
        path.unlink()
        path.symlink_to(crashed / "batches/000000/a.png")
    else:
        path = crashed / "session.json"
        meta = json.loads(path.read_text())
        meta["schema_version"] = 99
        path.write_text(json.dumps(meta))
    before = snapshot(crashed)
    with pytest.raises(ValueError):
        recover_session(crashed, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()
    assert snapshot(crashed) == before
    assert not list(tmp_path.glob(".capture-recovery-*"))


def test_existing_output_is_preserved(crashed, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        recover_session(crashed, target)
    assert (target / "keep").read_text() == "unchanged"


def test_nested_output_rejected(crashed):
    with pytest.raises(ValueError):
        recover_session(crashed, crashed / "recovered")
    assert not (crashed / "recovered").exists()


def test_source_change_during_copy_rejected(crashed, tmp_path, monkeypatch):
    import capture.recover as module
    original = module.shutil.copytree
    def copy_and_change(*args, **kwargs):
        result = original(*args, **kwargs)
        path = crashed / "session.json"
        data = json.loads(path.read_text())
        data["completed_batches"] += 1
        path.write_text(json.dumps(data))
        return result
    monkeypatch.setattr(module.shutil, "copytree", copy_and_change)
    with pytest.raises(RuntimeError, match="değişti"):
        recover_session(crashed, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()


def test_cli_recovers_and_produces_readable_output(crashed, tmp_path):
    import subprocess
    import sys
    target = tmp_path / "cli-recovered"
    result = subprocess.run([
        sys.executable, "-m", "capture.recover", "--source", str(crashed),
        "--output", str(target),
    ], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["completed_batches"] == 2
    assert len(list(iter_batches(target))) == 2


def test_symlink_hatasi_kurtarma_baglamiyla_raporlanir(crashed, tmp_path):
    """Dis inceleme C.5.1: okuyucunun hatasi 'kurtarma' baglamiyla gelir."""
    path = crashed / "batches/000001/a.png"
    path.unlink()
    path.symlink_to(crashed / "batches/000000/a.png")
    with pytest.raises(ValueError, match="Kurtarma"):
        recover_session(crashed, tmp_path / "recovered")
