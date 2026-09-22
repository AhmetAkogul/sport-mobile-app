"""Kayıt başlamadan doğrulanan oturum ve kamera yapılandırması."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} boş olamaz.")


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: int | str
    settings: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    def __post_init__(self):
        if not isinstance(self.camera_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.camera_id):
            raise ValueError("camera_id yalnızca ASCII harf, rakam, _ ve - içerebilir.")
        if type(self.source) not in (int, str):
            raise ValueError("source kamera indeksi veya video dosyası yolu olmalı.")
        if isinstance(self.source, int) and self.source < 0:
            raise ValueError("Kamera indeksi negatif olamaz.")
        if isinstance(self.source, str):
            _text(self.source, "source")
        _text(self.settings, "Kamera ayarları")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} pozitif tamsayı olmalı.")
        if self.fps is not None and (type(self.fps) not in (int, float)
                or not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("fps sonlu pozitif sayı olmalı.")


@dataclass(frozen=True)
class SessionConfig:
    participant_code: str
    lighting: str
    cameras: tuple[CameraConfig, ...]
    notes: str = ""

    def __post_init__(self):
        _text(self.participant_code, "Katılımcı kodu (insansız deneyde synthetic)")
        _text(self.lighting, "Işık koşulu")
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if not self.cameras or not all(isinstance(c, CameraConfig) for c in self.cameras):
            raise ValueError("En az bir CameraConfig gerekli.")
        ids = [c.camera_id.casefold() for c in self.cameras]
        sources = [c.source for c in self.cameras]
        if len(set(ids)) != len(ids) or len(set(sources)) != len(sources):
            raise ValueError("Kamera kimlikleri ve kaynakları benzersiz olmalı.")
        if not isinstance(self.notes, str):
            raise ValueError("notes metin olmalı.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["cameras"] = tuple(CameraConfig(**item) for item in data["cameras"])
        return cls(**data)
