"""İzole süreçte aynı video için model gecikmesi/bellek ölçümü; doğruluk testi değil."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np

from capture.alignment import file_sha256


# Medyan/p95 icin anlamli bir ornek: 100 cagri. Daha azinda olcum gurultulu
# olur ve p95 tek bir aykiri degere bagli kalir (dis inceleme B.2).
ONERILEN_OLCUM = 100


def benchmark(backend, video, output, *, model, detector=None, frames=ONERILEN_OLCUM, repeats=3):
    if type(frames) is not int or frames <= 0 or type(repeats) is not int or repeats <= 0:
        raise ValueError("Kare ve tekrar sayıları pozitif tamsayı olmalı.")
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    # Lazy import'lar ölçümün dışında tutulur: aksi halde ilk benchmark'ta
    # import süresi initialization_s'e dahil olur, sonrakilerde olmaz (B.3).
    if backend == 'mediapipe':
        from mono.mediapipe_model import MediaPipeEstimator
    elif backend == 'rtmpose':
        from mono.rtmpose_model import RTMPoseEstimator
    else:
        raise ValueError("Bilinmeyen model.")
    images = []
    source = cv2.VideoCapture(str(video))
    try:
        if not source.isOpened():
            raise ValueError("Video açılamadı.")
        for _ in range(frames):
            ok, image = source.read()
            if not ok:
                break
            images.append(image)
    finally:
        source.release()
    if not images:
        raise ValueError(f"Video'da hiç kare okunamadı: {video}")
    if len(images) < frames:
        print(f"uyari: istenen {frames} karenin {len(images)} tanesi okunabildi "
              f"({len(images) * repeats} cagri olculecek)", file=sys.stderr)
    with ExitStack() as stack:
        start = time.perf_counter()
        if backend == 'mediapipe':
            estimator = stack.enter_context(MediaPipeEstimator(model))
        else:
            estimator = stack.enter_context(RTMPoseEstimator(detector, model))
        initialization_s = time.perf_counter() - start
        for image in images[:2]:
            estimator(image)
        times, detected, visible = [], 0, []
        for _ in range(repeats):
            for image in images:
                start = time.perf_counter_ns()
                pose = estimator(image)
                times.append((time.perf_counter_ns() - start) / 1e6)
                detected += int(pose.tespit)
                visible.append(int(pose.gorunur.sum()))
        if len(times) < ONERILEN_OLCUM:
            print(f"uyari: {len(times)} cagri olculdu -- medyan/p95 icin en az "
                  f"{ONERILEN_OLCUM} onerilir (kare ve tekrar sayisini artirin)",
                  file=sys.stderr)
        report = {
            'backend': backend, 'model': estimator.model_id,
            'input_sha256': file_sha256(video), 'image_size': list(images[0].shape),
            'source_frames': frames, 'repeats': repeats, 'measured_calls': len(times), 'warmup_calls': min(2,frames),
            'detected_calls': detected, 'visible_joint_counts': visible,
            'latency_ms_median': float(np.median(times)), 'latency_ms_p95': float(np.percentile(times,95)),
            # initialization = yalnizca model kurulumu; import'lar oncesinde yapilir.
            'initialization_seconds': initialization_s,
            'initialization_scope': 'model_load_only_imports_excluded',
            # ru_maxrss: Darwin'de bayt, Linux'ta KB. Carpan bu farki kapatir.
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024),
            'resource_scope': 'whole_process_including_loaded_frames_and_imports',
            'timing_scope': 'full_image_person_detection_and_pose_inference_excludes_video_decode',
            'accuracy': None, 'accuracy_reason': 'No labeled ground truth; repeated still-image video only.',
            'opencv_version': cv2.__version__, 'physical_validation': False,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x',encoding='utf-8') as file:
        json.dump(report,file,ensure_ascii=False,indent=2,allow_nan=False)
    # Terminale ozet basılır: `visible_joint_counts` kare başına uzun bir liste
    # olduğu için ve `model` zaten `model_id` ile aynı bilgiyi taşıdığı için
    # atlanır -- ikisi de JSON dosyasında durur.
    print(json.dumps({k: v for k,v in report.items() if k not in ('visible_joint_counts','model')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('mediapipe','rtmpose'),required=True)
    parser.add_argument('--video',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--detector')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    benchmark(args.backend,args.video,args.output,model=args.model,detector=args.detector)


if __name__ == '__main__':
    main()
