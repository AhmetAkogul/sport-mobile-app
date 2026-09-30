"""Kaydedilmiş tam grupları poz/ekipman modellerine veri kaybetmeden aktarır."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from capture.session import CameraConfig, SessionConfig


@dataclass(frozen=True)
class RecordedFrame:
    session_path: Path
    batch_id: int
    camera_id: str
    image_bgr: np.ndarray
    grab_started_ns: int
    grab_finished_ns: int
    timestamp_semantics: str


def iter_batches(session_dir):
    """Her adımda tüm kameraların RecordedFrame demetini verir.

    Görüntü özgün boyutta BGR uint8'dir. RGB dönüşümü ve model ön işlemesi
    adaptöre aittir. JSONL indeks yerine atomik yayımlanan grup kayıtları okunur.
    Bu okuyucu tamamlanmış/durdurulmuş oturum içindir; canlı izleme yapmaz.
    """
    root = Path(session_dir).resolve()
    metadata = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Desteklenmeyen oturum sürümü.")
    if metadata.get("status") not in {"completed", "source_stopped", "failed", "interrupted", "recovered"}:
        raise ValueError("Oturum henüz kapatılmamış; canlı okuma desteklenmiyor.")
    semantics = metadata.get("timestamp_semantics")
    if semantics != "host_monotonic_grab_interval_not_exposure_time":
        raise ValueError("Bilinmeyen zaman damgası anlamı.")
    cfg = dict(metadata["config"])
    cfg["cameras"] = tuple(CameraConfig(**item) for item in cfg["cameras"])
    config = SessionConfig(**cfg)
    expected_ids = [c.camera_id for c in config.cameras]
    batch_root = root / "batches"
    if batch_root.is_symlink():
        raise ValueError("Grup kökü bağlantı olamaz.")
    groups = sorted(batch_root.iterdir())
    count = metadata.get("completed_batches")
    if type(count) is not int or count < 0 or len(groups) != count:
        raise ValueError("Tamamlanmış grup sayısı diskle uyuşmuyor.")
    previous_times = {}
    for batch_id, group in enumerate(groups):
        if group.name != f"{batch_id:06d}" or not group.is_dir() or group.is_symlink():
            raise ValueError("Grup sırası veya dizini geçersiz.")
        manifest = group / "frames.json"
        if manifest.is_symlink():
            raise ValueError("Grup meta verisi bağlantı olamaz.")
        batch = json.loads(manifest.read_text(encoding="utf-8"))
        if type(batch.get("batch_id")) is not int or batch["batch_id"] != batch_id:
            raise ValueError("Grup kimliği dizinle uyuşmuyor.")
        rows = batch["frames"]
        if [row["camera_id"] for row in rows] != expected_ids:
            raise ValueError("Kamera kimlikleri veya sırası yapılandırmayla uyuşmuyor.")
        frames = []
        for row in rows:
            camera_id = row["camera_id"]
            expected_path = f"batches/{batch_id:06d}/{camera_id}.png"
            if row["path"] != expected_path:
                raise ValueError("Kare yolu kayıt sözleşmesine uymuyor.")
            path = root / expected_path
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Kare yolu oturum dışına çıkamaz.")
            before, after = row["grab_started_ns"], row["grab_finished_ns"]
            if (type(before) is not int or type(after) is not int or before < 0
                    or after < before or before < previous_times.get(camera_id, 0)):
                raise ValueError("Kare zaman damgaları geçersiz veya geriye gidiyor.")
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Kare okunamadı: {expected_path}")
            if (list(image.shape) != row["shape"] or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3):
                raise ValueError("Kare boyutu/türü beklenen BGR uint8 sözleşmesine uymuyor.")
            previous_times[camera_id] = after
            frames.append(RecordedFrame(root, batch_id, camera_id, image, before, after, semantics))
        yield tuple(frames)
