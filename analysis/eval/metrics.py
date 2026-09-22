"""Piksel koordinatlarındaki eşleşmiş gözlemlerin hata metrikleri."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReprojectionError:
    """Nokta başına Öklid uzaklığı özeti; tüm hatalar piksel cinsindedir."""

    count: int
    excluded_count: int
    mean_px: float
    rms_px: float
    max_px: float


def reprojection_error(observed, projected, *, mask=None) -> ReprojectionError:
    """Aynı sıradaki 2B noktaları karşılaştırır: şekil (..., 2).

    RMS = sqrt(mean(dx² + dy²)); koordinat başına RMS değildir.
    Maske, nokta eksenleriyle aynı şekilli boolean dizidir; True dahil eder.
    Eksikler otomatik elenmez: yalnızca maskelenmiş noktalarda NaN/Inf olabilir.
    Kamera/kare/köşe kimliklerinin eşleştirilmesi çağıranın sorumluluğudur.
    """
    observed = np.asarray(observed, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    if observed.shape != projected.shape:
        raise ValueError("Gözlem ve izdüşüm şekilleri aynı olmalı.")
    if observed.ndim < 2 or observed.shape[-1] != 2:
        raise ValueError("Nokta dizilerinin şekli (..., 2) olmalı.")

    point_shape = observed.shape[:-1]
    if mask is None:
        valid = np.ones(point_shape, dtype=bool)
    else:
        valid = np.asarray(mask)
        if valid.dtype != np.bool_ or valid.shape != point_shape:
            raise ValueError("Maske boolean olmalı ve nokta eksenleriyle eşleşmeli.")

    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("En az bir geçerli nokta gerekli.")
    actual, predicted = observed[valid], projected[valid]
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Maskeye dahil noktalar sonlu olmalı.")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = actual - predicted
        distances = np.hypot(delta[:, 0], delta[:, 1])
    if not np.isfinite(distances).all():
        raise ValueError("Hesaplanan uzaklıklar sayısal aralığı aşıyor.")
    maximum = float(distances.max())
    # Ölçekleme, büyük ama sonlu uzaklıkların karesinde taşmayı engeller.
    scaled = distances / maximum if maximum else distances
    return ReprojectionError(
        count=count,
        excluded_count=int(valid.size - count),
        mean_px=float(scaled.mean() * maximum),
        rms_px=float(np.sqrt(np.mean(scaled**2)) * maximum),
        max_px=maximum,
    )
