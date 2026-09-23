"""Gerçek poz modeli ile telefon hattını videoda uçtan uca çalıştırır.

Video okuma **`capture.source.ProcessCapture`** ile yapılır: `run_video.py` ile
aynı disiplin (süreç izolasyonu + zaman aşımı). Düz `cv2.VideoCapture` bozuk bir
kodek veya sürücü kilidinde ana süreci de kilitler (dış inceleme U.1/X.2).

Eksik veri politikası: gorunmeyen eklem **NaN**'dır ve yazarken JSON `null`
olur. `np.nan_to_num(..., nan=0)` gibi bir dönüşüm yok — 0.0 geçerli bir
koordinattır, "yok" bilgisini saklamaz (dış inceleme U.3).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from capture.alignment import file_sha256
from capture.record import _write_json
from capture.source import ProcessCapture
from mono.backend import kestirici_olustur
from mono.phone import KareKaydi, hatti_kostur, json_uyumlu
from pose3d.iskelet import REFERANS_ISKELET

_KEMIKLER = {frozenset(c) for c in REFERANS_ISKELET.baglantilar}


def load_lengths(path):
    """JSON listesi: [{"a":"sag_kalca","b":"sag_diz","metre":0.42}].

    Kemik adları referans iskelette bulunmalı ve uzunluk sonlu/pozitif olmalı;
    yazım hatası sessizce "öncüsüz kemik" olarak dolaşmaz (dış inceleme U.5).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("lengths JSON boş olmayan liste olmalı")
    result = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"a", "b", "metre"}:
            raise ValueError("her uzunluk a, b ve metre alanlarını taşımalı")
        a, b = item["a"], item["b"]
        for ad in (a, b):
            if ad not in REFERANS_ISKELET.eklemler:
                raise ValueError(f"bilinmeyen eklem: {ad}")
        if a == b:
            raise ValueError(f"kemik iki ucu aynı olamaz: {a}")
        if frozenset((a, b)) not in _KEMIKLER:
            raise ValueError(f"iskelette tanımlı bir kemik değil: {a}-{b}")
        if (a, b) in result or (b, a) in result:
            raise ValueError(f"kemik iki kez verildi: {a}-{b}")
        metre = float(item["metre"])
        if not math.isfinite(metre) or metre <= 0:
            raise ValueError(f"uzunluk sonlu ve pozitif olmalı: {a}-{b}={metre}")
        result[(a, b)] = metre
    return result


def run(video, model, output, K, lengths, max_frames=300, *, backend="mediapipe",
        detector=None, threshold=None):
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames pozitif tamsayı olmalı.")
    video = Path(video).resolve()
    output = Path(output)
    if not video.is_file():
        raise ValueError("video mevcut olmalı")
    if output.exists():
        raise ValueError("çıktı yeni olmalı")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"K (3, 3) sonlu olmalı, {K.shape} geldi")
    output.mkdir(parents=True, exist_ok=False)
    rapor = {"status": "running", "input": str(video), "input_sha256": file_sha256(video),
             "frames": 0, "physical_validation": False, "frame_limit": max_frames,
             "backend": backend,
             "coordinate_space": "telefon kamerasi (ozgun piksel -> metre)",
             "tespit_politikasi": "eksik eklem NaN, JSON'da null"}
    _write_json(output / "summary.json", rapor)

    handle = None
    basladi = time.perf_counter()
    try:
        with kestirici_olustur(backend, model=model, detector=detector,
                               threshold=threshold) as estimator:
            rapor["model"] = estimator.model_id
            # Deneyin aynen tekrarlanabilmesi icin girdiler ozette saklanir:
            # modele ozgu esik, kamera matrisi ve kemik oncu uzunluklari.
            rapor["threshold"] = float(estimator.threshold)
            rapor["intrinsics_K"] = K.tolist()
            rapor["lengths_m"] = [{"a": a, "b": b, "metre": float(m)}
                                  for (a, b), m in sorted(lengths.items())]
            handle = ProcessCapture(str(video))
            if not handle.isOpened():
                raise ValueError("Video açılamadı.")

            def kareler():
                """Kareleri tembel üret: tüm video belleğe alınmaz."""
                for i in range(max_frames):
                    if not handle.grab():
                        rapor["stop_reason"] = "source_end_or_read_failure"
                        return
                    ok, image = handle.retrieve()
                    if not ok:
                        raise ValueError("Video karesi çözülemedi.")
                    yield KareKaydi(i, image, kamera_id="telefon")

            sonuc = hatti_kostur(kareler(), estimator, K, lengths)
            rapor["stop_reason"] = rapor.get("stop_reason", "frame_limit")

            rows = []
            for i, (poz, skel) in enumerate(zip(sonuc.pozlar, sonuc.iskeletler)):
                rows.append({
                    "frame_index": i,
                    "tespit": bool(poz.tespit),
                    "points_px": json_uyumlu(poz.noktalar),
                    "confidence": json_uyumlu(poz.guven),
                    "visible": poz.gorunur.tolist(),
                    "points_3d_m": json_uyumlu(skel.noktalar),
                    "visible_3d": skel.gorunur.tolist(),
                    "artik_px": json_uyumlu(skel.artik_px),
                    # Atlanan kemikler + derinlik sacilimi: hata analizinde
                    # "bu eklem neden dustu / derinligi ne kadar guvenilir".
                    "iskelet_ek": json_uyumlu(skel.ek),
                    "model": poz.model,
                    "coordinate_space": poz.uzay,
                })
            (output / "results.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                        for r in rows), encoding="utf-8")
            rapor["frames"] = len(rows)
            rapor["detected_frames"] = sum(1 for r in rows if r["tespit"])
            rapor["model"] = sorted({p.model for p in sonuc.pozlar})
            rapor["summary"] = sonuc.ozet()
            rapor["status"] = "completed"
    except BaseException as exc:
        rapor.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if handle is not None:
                handle.release()
        finally:
            rapor["elapsed_seconds"] = time.perf_counter() - basladi
            _write_json(output / "summary.json", rapor)
    return rapor


def main():
    p = argparse.ArgumentParser(description="Telefon hattını videoda uçtan uca koştur")
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--backend", choices=("mediapipe", "rtmpose"), default="mediapipe")
    p.add_argument("--detector", default=None, help="rtmpose için YOLOX dedektör modeli")
    p.add_argument("--threshold", type=float, default=None)
    a = p.parse_args()
    print(json.dumps(run(a.video, a.model, a.output,
                         json.loads(Path(a.intrinsics).read_text(encoding="utf-8")),
                         load_lengths(a.lengths), a.max_frames,
                         backend=a.backend, detector=a.detector, threshold=a.threshold),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
