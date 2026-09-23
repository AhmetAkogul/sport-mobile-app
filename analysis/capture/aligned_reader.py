"""İçerik kimliği doğrulanmış video planından tam görüntü grupları okur."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from capture.alignment import _number, file_sha256
from capture.source import ProcessCapture


@dataclass(frozen=True)
class AlignedFrame:
    camera_id: str
    source_path: Path
    group_id: int
    source_frame_index: int
    source_time_ms: float
    aligned_time_ms: float
    image_bgr: np.ndarray
    timestamp_semantics: str = "opencv_pos_msec_plus_explicit_offset"


def _validate(plan):
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 2:
        raise ValueError("İçerik kimlikli plan sürümü 2 gerekli; planı yeniden üretin.")
    if plan.get("timestamp_semantics") != "opencv_pos_msec_plus_explicit_offset":
        raise ValueError("Planın zaman anlamı desteklenmiyor.")
    if plan.get("physical_synchronization_verified") is not False:
        raise ValueError("Bu okuyucu fiziksel senkron doğrulaması kabul etmez.")
    sources = plan["sources"]
    ids = [s["camera_id"] for s in sources]
    if (len(ids) < 2 or not all(isinstance(key, str) and key for key in ids)
            or len(set(ids)) != len(ids) or plan["reference_camera"] not in ids):
        raise ValueError("Plan kamera kimlikleri geçersiz.")
    if set(plan["offsets_ms"]) != set(ids):
        raise ValueError("Ofsetler kamera kimlikleriyle uyuşmuyor.")
    for offset in plan["offsets_ms"].values():
        _number(offset, "Ofset")
    _number(plan["tolerance_ms"], "Tolerans", nonnegative=True)
    counts = {}
    for source in sources:
        if (not isinstance(source.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
            raise ValueError("Kaynak içerik kimliği gerekli; planı yeniden üretin.")
        path = Path(source["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Kaynak mutlak yerel dosya yolu olmalı.")
        count = source["frame_count"]
        if type(count) is not int or count < 2:
            raise ValueError("Kaynak kare sayısı geçersiz.")
        counts[source["camera_id"]] = count
    previous = {key: (-1, -1.0) for key in ids}
    for group_id, group in enumerate(plan["groups"]):
        if type(group["group_id"]) is not int or group["group_id"] != group_id:
            raise ValueError("Grup sırası geçersiz.")
        if [row["camera_id"] for row in group["frames"]] != ids:
            raise ValueError("Gruptaki kamera kimlikleri/sırası geçersiz.")
        times = []
        for row in group["frames"]:
            key, index = row["camera_id"], row["source_frame_index"]
            if type(index) is not int or not previous[key][0] < index < counts[key]:
                raise ValueError("Kare indeksi geçersiz veya tekrar kullanılmış.")
            source_time, aligned_time = row["source_time_ms"], row["aligned_time_ms"]
            _number(source_time, "Kaynak zamanı", nonnegative=True)
            _number(aligned_time, "Ortak zaman")
            if source_time <= previous[key][1]:
                raise ValueError("Kaynak zamanları artmalı.")
            if not math.isclose(source_time + plan["offsets_ms"][key], aligned_time, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Ofset ile ortak zaman uyuşmuyor.")
            previous[key] = index, source_time
            times.append(aligned_time)
        spread = max(times) - min(times)
        _number(group["spread_ms"], "Grup zaman farkı", nonnegative=True)
        if (spread > plan["tolerance_ms"] + 1e-9 or
                not math.isclose(spread, group["spread_ms"], rel_tol=0, abs_tol=1e-6)):
            raise ValueError("Grup zaman farkı/toleransı geçersiz.")
    return sources


@contextmanager
def open_aligned_batches(plan_path, *, source_timeout_s=5.0, open_timeout_s=15.0,
                         capture_factory=None):
    """with bloğunda tam AlignedFrame demetleri sunar; erken çıkışta da kapatır.

    Rastgele seek yerine baştan sıralı çözme kullanır. Her seçili karede POS_MSEC
    plana karşı 0.001 ms toleransla sınanır. Tüm videolar açılmadan SHA-256 kontrol
    edilir; kullanım sırasında dosyalar değiştirilmemelidir.
    """
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    sources = _validate(plan)
    signatures = {}
    for source in sources:
        signatures[source["camera_id"]] = Path(source["path"]).stat()
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Kaynak video değişmiş: {source['camera_id']}")
    handles, paths = {}, {}
    active = True
    with ExitStack() as stack:
        if plan["groups"]:
            for source in sources:
                key = source["camera_id"]
                path = Path(source["path"])
                paths[key] = path
                handle = (capture_factory(str(path)) if capture_factory else ProcessCapture(
                    str(path), timeout_s=source_timeout_s, open_timeout_s=open_timeout_s))
                stack.callback(handle.release)
                if not handle.isOpened():
                    raise ValueError(f"Video açılamadı: {key}")
                handles[key] = handle
        positions = {key: -1 for key in handles}

        def batches():
            for group in plan["groups"]:
                if not active:
                    raise RuntimeError("Okuyucu with bloğu dışında kullanılamaz.")
                frames = []
                for row in group["frames"]:
                    key = row["camera_id"]
                    path, handle = paths[key], handles[key]
                    now, before = path.stat(), signatures[key]
                    if (now.st_size, now.st_mtime_ns, now.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
                        raise ValueError(f"Video okuma sırasında değişti: {key}")
                    # `image` her satirda sifirlanir: dongu calismazsa onceki
                    # kameranin karesi bu kameraya yazilmasin (dis inceleme C.9.1).
                    # `_validate` kare indekslerinin artmasini zaten sart kosuyor;
                    # bu, o dolayli guvenceyi yerel ve acik hale getirir.
                    image = None
                    while positions[key] < row["source_frame_index"]:
                        if not handle.grab():
                            raise ValueError(f"Video planlanan kareden önce bitti: {key}")
                        ok, image = handle.retrieve()
                        if not ok or image is None:
                            raise ValueError(f"Video karesi çözülemedi: {key}")
                        positions[key] += 1
                    if image is None:
                        raise ValueError(
                            f"Plan ayni kareyi ikinci kez istiyor veya geri gidiyor: {key}")
                    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
                        raise ValueError("Görüntü BGR uint8 olmalı.")
                    actual_time = float(handle.get(cv2.CAP_PROP_POS_MSEC))
                    if not math.isfinite(actual_time) or not math.isclose(
                            actual_time, row["source_time_ms"], rel_tol=0, abs_tol=0.001):
                        raise ValueError(f"Kare zamanı planla uyuşmuyor: {key}")
                    frames.append(AlignedFrame(key, path, group["group_id"], positions[key],
                                               actual_time, row["aligned_time_ms"], image))
                yield tuple(frames)
        try:
            yield batches()
        finally:
            active = False
