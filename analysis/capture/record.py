"""N kaynaktan eşit sayıda kare grubu kaydı; donanım senkronu sağlamaz."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from capture.session import SessionConfig
from capture.source import ProcessCapture
from capture.settings import configure_source


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def record_session(config, output_dir, *, max_frames, capture_factory=None, source_timeout_s=5.0, open_timeout_s=15.0):
    """Yeni dizine PNG grupları, session.json ve frames.jsonl yazar.

    Tüm kaynaklar önce grab, sonra retrieve edilir. Zamanlar ana bilgisayarın
    monotonic saatidir; sensör pozlama zamanı veya gerçek senkron kayması değildir.
    Tam gruplar atomik dizin taşımasıyla yayımlanır. frames.jsonl indeksidir;
    çökme sonrası asıl kayıt batches/*/frames.json içindeki grup meta verisidir.
    Herhangi bir kaynak durursa tamamlanmamış grup atılır, kayıt durur.
    """
    if not isinstance(config, SessionConfig):
        raise ValueError("Doğrulanmış SessionConfig gerekli.")
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames pozitif tamsayı olmalı.")
    for value in (source_timeout_s, open_timeout_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Zaman aşımı sonlu pozitif saniye olmalı.")
    factory = capture_factory or (lambda source: ProcessCapture(
        source, timeout_s=source_timeout_s, open_timeout_s=open_timeout_s))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    batches = output / "batches"
    try:
        batches.mkdir()
    except BaseException:
        # Yarıda kalan dizin sonraki denemeyi `exist_ok=False` yüzünden kilitler;
        # kullanıcı elle silmek zorunda kalmasın (dış inceleme C.3.1).
        shutil.rmtree(output, ignore_errors=True)
        raise
    metadata = {
        "schema_version": 1, "config": config.to_dict(), "started_at_utc": _utc(),
        "status": "recording", "completed_batches": 0,
        "timestamp_semantics": "host_monotonic_grab_interval_not_exposure_time",
        "hardware_synchronized": False,
        "camera_settings": {},
        "source_isolation": "process" if capture_factory is None else "custom_factory",
        "source_timeout_s": source_timeout_s if capture_factory is None else None,
        "open_timeout_s": open_timeout_s if capture_factory is None else None,
    }
    _write_json(output / "session.json", metadata)
    handles = []
    pending = output / ".pending"
    primary_error = None
    try:
        for camera in config.cameras:
            handle = factory(camera.source)
            handles.append(handle)
            if not handle.isOpened():
                raise RuntimeError(f"Kaynak açılamadı: {camera.camera_id}")
            settings_report = configure_source(handle, camera)
            metadata["camera_settings"][camera.camera_id] = settings_report
            # Kamera basina yazim bilincli: acilis sirasinda surec cokerse diskte
            # hangi kameranin hangi ayarla acildigi kalir (dis inceleme C.3.2).
            _write_json(output / "session.json", metadata)
            if settings_report["issues"]:
                raise ValueError(f"Kamera ayarları doğrulanamadı: {camera.camera_id}: "
                                 + "; ".join(settings_report["issues"]))
        with (output / "frames.jsonl").open("x", encoding="utf-8") as index:
            for batch_id in range(max_frames):
                timings = []
                stopped = None
                for camera, handle in zip(config.cameras, handles):
                    before = time.monotonic_ns()
                    ok = handle.grab()
                    after = time.monotonic_ns()
                    if not ok:
                        stopped = camera.camera_id
                        break
                    timings.append((before, after))
                if stopped is not None:
                    metadata.update(status="source_stopped", stopped_camera=stopped)
                    break
                frames = []
                for camera, handle in zip(config.cameras, handles):
                    ok, frame = handle.retrieve()
                    if not ok or frame is None or frame.size == 0:
                        raise RuntimeError(f"Kare çözülemedi: {camera.camera_id}")
                    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                            or frame.ndim != 3 or frame.shape[2] != 3):
                        raise ValueError(f"Kare BGR uint8 olmalı: {camera.camera_id}")
                    if ((camera.width is not None and frame.shape[1] != camera.width)
                            or (camera.height is not None and frame.shape[0] != camera.height)):
                        raise ValueError(f"Gerçek kare boyutu istenen ayarla uyuşmuyor: {camera.camera_id}")
                    frames.append(frame)
                pending.mkdir()
                rows = []
                for camera, frame, (before, after) in zip(config.cameras, frames, timings):
                    filename = f"{camera.camera_id}.png"
                    if not cv2.imwrite(str(pending / filename), frame):
                        raise OSError(f"Kare yazılamadı: {camera.camera_id}")
                    rows.append({
                        "camera_id": camera.camera_id,
                        "path": f"batches/{batch_id:06d}/{filename}",
                        "grab_started_ns": before, "grab_finished_ns": after,
                        "shape": list(frame.shape),
                    })
                batch = {"batch_id": batch_id, "frames": rows}
                _write_json(pending / "frames.json", batch)
                pending.rename(batches / f"{batch_id:06d}")
                metadata["completed_batches"] += 1
                index.write(json.dumps(batch, ensure_ascii=False) + "\n")
                index.flush()
            else:
                metadata["status"] = "completed"
    except BaseException as exc:
        primary_error = exc
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for handle in handles:
            try:
                handle.release()
            except Exception as exc:
                metadata.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
        if pending.exists():
            try:
                shutil.rmtree(pending)
            except Exception as exc:
                metadata.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
                if primary_error is not None:
                    primary_error.add_note(f"Yarım grup temizlenemedi: {exc}")
        if metadata.get("cleanup_errors") and metadata["status"] in {"completed", "source_stopped"}:
            metadata["status"] = "failed"
        metadata["finished_at_utc"] = _utc()
        try:
            _write_json(output / "session.json", metadata)
        except Exception as exc:
            if primary_error is None:
                raise
            primary_error.add_note(f"Oturumun son durumu diske yazılamadı: {exc}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Çok kaynaklı kare kaydı")
    parser.add_argument("--config", required=True, help="Oturum JSON dosyası")
    parser.add_argument("--output", required=True, help="Yeni kayıt dizini; data/ altında tutun")
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--source-timeout", type=float, default=5.0, help="Kaynak çağrısı sınırı (saniye)")
    parser.add_argument("--open-timeout", type=float, default=15.0, help="Kaynak açma sınırı (saniye)")
    args = parser.parse_args()
    result = record_session(SessionConfig.from_json(args.config), args.output, max_frames=args.max_frames,
                            source_timeout_s=args.source_timeout, open_timeout_s=args.open_timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
