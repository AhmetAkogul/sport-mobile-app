"""Video dosyalarını açık zaman ofsetleriyle eşler; fiziksel senkron ölçmez."""

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import json
import hashlib
import math
from pathlib import Path

import cv2

from capture.source import ProcessCapture


def _number(value, name, *, nonnegative=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError(f"{name} sonlu {'negatif olmayan ' if nonnegative else ''}sayı olmalı.")


def file_sha256(path):
    """Video kimliğini içerikten hesaplar; dosya tarama sırasında değişmemeli."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoTimeline:
    camera_id: str
    path: str
    timestamps_ms: tuple[float, ...]
    sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("Kamera kimliği boş olamaz.")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Video yolu gerekli.")
        object.__setattr__(self, "timestamps_ms", tuple(self.timestamps_ms))
        if not self.timestamps_ms:
            raise ValueError("Video zaman çizelgesi boş olamaz.")
        previous = -1.0
        for value in self.timestamps_ms:
            _number(value, "Video zamanı", nonnegative=True)
            if value <= previous:
                raise ValueError("Video zamanları kesin artmalı; tekrar/geriye gidiş var.")
            previous = value


def scan_video(camera_id, path, *, max_frames=100000, source_timeout_s=5.0,
               open_timeout_s=15.0, capture_factory=None):
    """OpenCV POS_MSEC okur; eksik zamanda FPS'den zaman uydurmaz.

    Yalnızca yerel dosya. En az iki kare gerekir: tek sıfır değerin desteklenmeyen
    zaman özelliğinden ayrılması mümkün değildir. Tarama sınırında kesmez, hata verir.
    """
    if type(max_frames) is not int or max_frames < 2:
        raise ValueError("max_frames en az 2 olmalı.")
    video_path = Path(path).resolve()
    if not video_path.is_file():
        raise ValueError("Yerel video dosyası gerekli.")
    fingerprint = file_sha256(video_path)
    handle = (capture_factory(str(video_path)) if capture_factory else ProcessCapture(
        str(video_path), timeout_s=source_timeout_s, open_timeout_s=open_timeout_s))
    timestamps = []
    try:
        if not handle.isOpened():
            raise ValueError("Video açılamadı.")
        while handle.grab():
            if len(timestamps) >= max_frames:
                raise ValueError("Video tarama sınırını aştı; kısmi çizelge üretilmedi.")
            ok, frame = handle.retrieve()
            if not ok or frame is None or frame.size == 0:
                raise ValueError("Video karesi çözülemedi.")
            value = float(handle.get(cv2.CAP_PROP_POS_MSEC))
            _number(value, "POS_MSEC", nonnegative=True)
            if timestamps and value <= timestamps[-1]:
                raise ValueError("POS_MSEC desteklenmiyor veya kesin artmıyor.")
            timestamps.append(value)
    finally:
        handle.release()
    if len(timestamps) < 2:
        raise ValueError("Zaman desteğini doğrulamak için en az iki kare gerekli.")
    if file_sha256(video_path) != fingerprint:
        raise ValueError("Video tarama sırasında değişti.")
    return VideoTimeline(camera_id, str(video_path), tuple(timestamps), fingerprint)


def align_timelines(timelines, *, reference_camera, offsets_ms, tolerance_ms):
    """Referans sırasıyla en yakın kullanılmamış kareleri seçer.

    Ortak zaman = dosya zamanı + kullanıcının açıkça verdiği ofset.
    Her kamera için ofset zorunlu (bilerek sıfır verilebilir). Bir kare tekrar
    kullanılmaz. Tam gruptaki en erken/en geç zaman farkı toleransı aşamaz.
    Açgözlü eşleştirmedir; küresel en fazla eşleşme garantisi vermez.
    """
    timelines = tuple(timelines)
    if len(timelines) < 2 or not all(isinstance(t, VideoTimeline) for t in timelines):
        raise ValueError("En az iki VideoTimeline gerekli.")
    ids = [t.camera_id for t in timelines]
    if len(set(ids)) != len(ids) or reference_camera not in ids:
        raise ValueError("Kamera kimlikleri benzersiz ve referans mevcut olmalı.")
    if not isinstance(offsets_ms, dict) or set(offsets_ms) != set(ids):
        raise ValueError("Her kamera için açık zaman ofseti gerekli.")
    _number(tolerance_ms, "Tolerans", nonnegative=True)
    adjusted = {}
    for timeline in timelines:
        offset = offsets_ms[timeline.camera_id]
        _number(offset, "Ofset")
        values = tuple(t + offset for t in timeline.timestamps_ms)
        if not all(math.isfinite(t) for t in values):
            raise ValueError("Ofsetli zaman sayısal aralığı aşıyor.")
        adjusted[timeline.camera_id] = values
    next_index = {key: 0 for key in ids}
    used = {key: set() for key in ids}
    groups, unmatched = [], []
    originals = {t.camera_id: t for t in timelines}
    for ref_index, reference_time in enumerate(adjusted[reference_camera]):
        selected = {reference_camera: ref_index}
        for key in ids:
            if key == reference_camera:
                continue
            values = adjusted[key]
            pos = bisect_left(values, reference_time, lo=next_index[key])
            candidates = [i for i in (pos - 1, pos) if next_index[key] <= i < len(values)]
            if not candidates:
                break
            selected[key] = min(candidates, key=lambda i: (abs(values[i] - reference_time), i))
        if len(selected) != len(ids):
            unmatched.append({"reference_frame_index": ref_index, "reason": "source_exhausted"})
            continue
        times = [adjusted[key][selected[key]] for key in ids]
        spread = max(times) - min(times)
        if spread > tolerance_ms + 1e-9:
            unmatched.append({"reference_frame_index": ref_index, "reason": "outside_tolerance"})
            continue
        rows = []
        for key in ids:
            index = selected[key]
            next_index[key] = index + 1
            used[key].add(index)
            rows.append({"camera_id": key, "source_frame_index": index,
                         "source_time_ms": originals[key].timestamps_ms[index],
                         "aligned_time_ms": adjusted[key][index]})
        groups.append({"group_id": len(groups), "spread_ms": spread, "frames": rows})
    return {
        "schema_version": 2, "timestamp_semantics": "opencv_pos_msec_plus_explicit_offset",
        "physical_synchronization_verified": False,
        "method": "reference_greedy_nearest_without_reuse",
        "reference_camera": reference_camera, "offsets_ms": dict(offsets_ms),
        "tolerance_ms": tolerance_ms,
        "sources": [{"camera_id": t.camera_id, "path": t.path,
                     "frame_count": len(t.timestamps_ms), "sha256": t.sha256} for t in timelines],
        "groups": groups, "unmatched_reference_frames": unmatched,
        "unused_frame_indices": {key: [i for i in range(len(adjusted[key])) if i not in used[key]] for key in ids},
    }


def main():
    parser = argparse.ArgumentParser(description="Yerel videolar için zaman eşleştirme planı")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=100000)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    timelines = [scan_video(item["camera_id"], item["path"], max_frames=args.max_frames)
                 for item in config["videos"]]
    result = align_timelines(timelines, reference_camera=config["reference_camera"],
                             offsets_ms=config["offsets_ms"], tolerance_ms=config["tolerance_ms"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Eşleşen grup: {len(result['groups'])}; eşleşmeyen referans karesi: {len(result['unmatched_reference_frames'])}")
    return 0 if result["groups"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
