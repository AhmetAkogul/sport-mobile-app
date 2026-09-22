"""Gerçek poz modeli ile telefon hattını videoda uçtan uca çalıştırır."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from capture.alignment import file_sha256
from mono.mediapipe_model import MediaPipeEstimator
from mono.phone import KareKaydi, hatti_kostur

def load_lengths(path):
    """JSON listesi: [{"a":"sag_kalca","b":"sag_diz","metre":0.42}]."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("lengths JSON liste olmalı")
    result = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"a", "b", "metre"}:
            raise ValueError("her uzunluk a, b ve metre alanlarını taşımalı")
        result[(item["a"], item["b"])] = float(item["metre"])
    return result

def run(video, model, output, K, lengths, max_frames=300):
    video = Path(video).resolve(); output = Path(output)
    if not video.is_file() or output.exists(): raise ValueError("video mevcut, çıktı yeni olmalı")
    output.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video));
    if not cap.isOpened(): raise ValueError("video açılamadı")
    rows=[]; frames=[]
    try:
        with MediaPipeEstimator(model) as estimator:
            for i in range(max_frames):
                ok, image = cap.read()
                if not ok: break
                frames.append(KareKaydi(i, image))
                if len(frames) >= max_frames: break
            sonuc = hatti_kostur(frames, estimator, np.asarray(K, float), lengths)
    finally: cap.release()
    for i, (poz, skel) in enumerate(zip(sonuc.pozlar, sonuc.iskeletler)):
        rows.append({"frame_index": i, "points_px": poz.noktalar.tolist(),
                     "confidence": poz.guven.tolist(), "visible": poz.gorunur.tolist(),
                     "points_3d_m": np.nan_to_num(skel.noktalar, nan=0).tolist(),
                     "visible_3d": skel.gorunur.tolist(), "model": poz.model,
                     "coordinate_space": poz.uzay})
    (output/'results.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    report={"status":"completed", "input":str(video), "input_sha256":file_sha256(video),
            "frames":len(rows), "model":sorted({p.model for p in sonuc.pozlar}),
            "summary":sonuc.ozet(), "physical_validation":False}
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--video',required=True); p.add_argument('--model',required=True); p.add_argument('--output',required=True); p.add_argument('--intrinsics',required=True); p.add_argument('--lengths',required=True); p.add_argument('--max-frames',type=int,default=300); a=p.parse_args()
    print(json.dumps(run(a.video,a.model,a.output,json.loads(Path(a.intrinsics).read_text()),load_lengths(a.lengths),a.max_frames),ensure_ascii=False,indent=2))
if __name__ == '__main__': main()
