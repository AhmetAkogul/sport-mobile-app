"""İstenen ayar ile sürücünün bildirdiği değerleri ayrı tutar."""

import math

import cv2


PROPERTIES = {
    "width": cv2.CAP_PROP_FRAME_WIDTH,
    "height": cv2.CAP_PROP_FRAME_HEIGHT,
    "fps": cv2.CAP_PROP_FPS,
}


def configure_source(handle, camera):
    """Kamera indeksinde uygular; dosya/URL kaynağında yalnızca beklentiyi sınar.

    FPS sürücü beyanıdır, ölçülmüş gerçek kare hızı değildir. İstenen alanlar
    doğrulanamazsa issues doludur. Tüm set çağrılarından SONRA get yapılır;
    FPS ayarı çözünürlüğü değiştirebilen sürücüler böylece yakalanır.
    """
    requested = {name: getattr(camera, name) for name in PROPERTIES}
    report = {"requested": requested, "reported": {}, "set_accepted": {},
              "mode": "apply" if type(camera.source) is int else "validate_only",
              "issues": [], "fps_semantics": "driver_reported_not_measured"}
    if report["mode"] == "apply":
        for name, property_id in PROPERTIES.items():
            if requested[name] is None:
                continue
            # hasattr yeterli degil: `set` bir property/ozel nesne olabilir ve
            # cagri aninda TypeError verir. Denemek tek guvenilir denetim.
            try:
                accepted = bool(handle.set(property_id, requested[name]))
            except (AttributeError, TypeError):
                accepted = False
            report["set_accepted"][name] = accepted
            if not accepted:
                report["issues"].append(f"{name}: sürücü ayarı kabul etmedi")
    for name, property_id in PROPERTIES.items():
        try:
            raw = handle.get(property_id)
        except (AttributeError, TypeError):
            raw = None
        value = float(raw) if raw is not None else None
        if value is not None and (not math.isfinite(value) or value <= 0):
            value = None
        report["reported"][name] = value
        expected = requested[name]
        if expected is not None:
            matches = value is not None and (
                math.isclose(value, expected, rel_tol=0.01, abs_tol=0.1) if name == "fps"
                else value == expected)
            if not matches:
                report["issues"].append(f"{name}: istenen={expected}, bildirilen={value}")
    return report
