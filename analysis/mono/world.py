"""MediaPipe'in hazir 3B tabani. Mutlak telefon derinligi degildir."""
import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B


def dunya_iskeleti(poz):
    n = len(REFERANS_ISKELET)
    points = np.asarray(poz.ek.get("world_points_m", np.full((n, 3), np.nan)), float)
    visible = np.asarray(poz.ek.get("world_visible", np.zeros(n, bool)), bool)
    visible = visible & poz.gorunur
    points = points.copy()
    points[~visible] = np.nan
    return Iskelet3B(REFERANS_ISKELET, points, visible, visible.astype(int),
                     np.full(n, np.nan), cerceve="mediapipe_kalca_merkezi",
                     kaynak=poz.model, ek={"mutlak_konum": False,
                                           "yontem": "mediapipe_world"})
