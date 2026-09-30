"""Kamera açmadan N sentetik video ile tekrar üretilebilir kayıt kabul koşusu."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from capture.reader import iter_batches
from capture.record import record_session, _write_json
from capture.session import CameraConfig, SessionConfig


def validate_capture(output_dir, *, cameras=4, frames=300, width=320, height=240):
    """Yeni çıktı dizininde kaynakları, kaydı ve doğrulama sonucunu saklar."""
    for name, value in (("cameras", cameras), ("frames", frames), ("width", width), ("height", height)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} pozitif tamsayı olmalı.")
    if width % 2 or height % 2 or min(width, height) < 16:
        raise ValueError("MJPG testi için boyutlar en az 16 ve çift olmalı.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    configs = []
    for camera in range(cameras):
        path = inputs / f"camera-{camera}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("MJPG yazıcı açılamadı.")
        try:
            for index in range(frames):
                image = np.empty((height, width, 3), np.uint8)
                image[:] = ((index * 7) % 256, (camera * 43) % 256, (index * 13) % 256)
                cv2.putText(image, f"{camera}:{index}", (8, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)
                writer.write(image)
        finally:
            writer.release()
        configs.append(CameraConfig(f"camera-{camera}", str(path.resolve()), "Sentetik MJPG 30 fps",
                                    width=width, height=height, fps=30))
    started = time.perf_counter()
    result = record_session(SessionConfig("synthetic", "sentetik", tuple(configs)),
                            root / "session", max_frames=frames)
    record_seconds = time.perf_counter() - started
    references = []
    verified = 0
    try:
        for config in configs:
            source = cv2.VideoCapture(config.source)
            references.append(source)
            if not source.isOpened():
                raise RuntimeError("Karşılaştırma videosu açılamadı.")
        for group in iter_batches(root / "session"):
            if len(group) != cameras:
                raise AssertionError("Kamera sayısı uyuşmuyor.")
            for config, reference, frame in zip(configs, references, group):
                ok, expected = reference.read()
                if not ok or frame.camera_id != config.camera_id or not np.array_equal(expected, frame.image_bgr):
                    raise AssertionError("Kayıt kaynak videonun çözülen görüntüsüyle uyuşmuyor.")
                verified += 1
    finally:
        for reference in references:
            reference.release()
    if result["status"] != "completed" or verified != cameras * frames:
        raise AssertionError("Kayıt tamamlanmadı veya doğrulanan kare sayısı yanlış.")
    report = {"status": "passed", "cameras": cameras, "groups": frames, "verified_images": verified,
              "width": width, "height": height, "source_fps": 30,
              "record_wall_seconds": record_seconds,
              "recorded_bytes": sum(p.stat().st_size for p in (root / "session").rglob("*") if p.is_file()),
              "opencv_version": cv2.__version__,
              "physical_validation": False, "real_time_capture_claim": False,
              "description": "Yerel videodan çevrimdışı kayıt; piksel eşitliği doğrulandı."}
    _write_json(root / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Sentetik çok kaynaklı kayıt doğrulaması")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(validate_capture(args.output, cameras=args.cameras, frames=args.frames),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
