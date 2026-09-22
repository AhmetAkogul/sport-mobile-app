"""Yerel videodan gerçek modelle Poz2B JSONL üretir; 3B doğruluk iddia etmez."""

import argparse
import json
from pathlib import Path
import time

from capture.alignment import file_sha256
from capture.record import _write_json
from capture.source import ProcessCapture
from mono.mediapipe_model import MediaPipeEstimator


def run_video(video_path, model_path, output_dir, *, max_frames=300):
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames pozitif tamsayı olmalı.")
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "input": str(path), "input_sha256": file_sha256(path),
              "frames": 0, "detected_frames": 0, "physical_validation": False,
              "running_mode": "IMAGE", "frame_limit": max_frames}
    _write_json(output / "summary.json", report)
    handle = None
    started = time.perf_counter()
    try:
        with MediaPipeEstimator(model_path) as estimator:
            report["model"] = estimator.model_id
            handle = ProcessCapture(str(path))
            if not handle.isOpened():
                raise ValueError("Video açılamadı.")
            with (output / "poses.jsonl").open("x", encoding="utf-8") as file:
                for index in range(max_frames):
                    if not handle.grab():
                        report["stop_reason"] = "source_end_or_read_failure"
                        break
                    ok, image = handle.retrieve()
                    if not ok:
                        raise ValueError("Video karesi çözülemedi.")
                    pose = estimator(image)
                    pose.kare = index
                    row = {"frame_index": index, "skeleton": pose.iskelet.ad,
                           "joints": list(pose.iskelet.eklemler), "points_px": pose.noktalar.tolist(),
                           "confidence": pose.guven.tolist(), "visible": pose.gorunur.tolist(),
                           "image_size": pose.goruntu_boyutu, "coordinate_space": pose.uzay,
                           "model": pose.model, "extra": pose.ek}
                    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    file.flush()
                    report["frames"] += 1
                    report["detected_frames"] += int(pose.ek["tespit"])
                else:
                    report["stop_reason"] = "frame_limit"
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if handle is not None:
                handle.release()
        except Exception as exc:
            report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            _write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="MediaPipe ile videodan Poz2B üret")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run_video(args.video, args.model, args.output, max_frames=args.max_frames),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
