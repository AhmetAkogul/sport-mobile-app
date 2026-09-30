"""mono.demo uctan uca: sahte model + gercek kucuk video.

Sinanan: iki karar kolu yaziliyor, belirsizligi bilen kol yalin esikten daha
az kesin konusuyor, isaretli video uretiliyor ve ozet kaynagini tasiyor.
"""
import json

import cv2
import numpy as np
import pytest

from mono import demo, run_phone
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B

BOYUT = (320, 240)


class SahteModel:
    model_id = "sahte/demo"
    threshold = 0.5

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def __call__(self, image):
        # Onden duran bir insan: omuzlar ustte, kalca ortada, ayaklar altta.
        xy = {"boyun": (160, 50), "sag_omuz": (135, 55), "sol_omuz": (185, 55),
              "sag_dirsek": (125, 90), "sol_dirsek": (195, 90),
              "sag_bilek": (120, 120), "sol_bilek": (200, 120),
              "sag_kalca": (148, 120), "sol_kalca": (172, 120),
              "sag_diz": (146, 170), "sol_diz": (174, 170),
              "sag_ayak_bilegi": (146, 220), "sol_ayak_bilegi": (174, 220)}
        n = len(REFERANS_ISKELET)
        pts = np.array([xy[e] for e in REFERANS_ISKELET.eklemler], float)
        return Poz2B(REFERANS_ISKELET, pts, np.ones(n), np.ones(n, bool), True, BOYUT,
                     model=self.model_id)


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, BOYUT)
    for _ in range(3):
        w.write(np.full((BOYUT[1], BOYUT[0], 3), 40, np.uint8))
    w.release()
    monkeypatch.setattr(run_phone, "kestirici_olustur", lambda *a, **k: SahteModel())
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    uzunluk = {tuple(c): 0.3 for c in REFERANS_ISKELET.baglantilar}
    return video, K, uzunluk, tmp_path / "demo"


def test_demo_iki_karar_kolunu_ve_videoyu_uretir(ortam):
    video, K, uzunluk, cikti = ortam
    ozet = demo.run(video, "m.task", cikti, K, uzunluk, max_frames=3)
    assert ozet["status"] == "completed" and ozet["frames"] == 3
    assert sum(ozet["yalin_esik_kararlari"].values()) == 3
    assert sum(ozet["belirsizligi_bilen_kararlari"].values()) == 3
    assert ozet["konum_belirsizligi_m"] == demo.VARSAYILAN_KONUM_BELIRSIZLIGI_M
    assert "derinlik-duyarliligi" in ozet["belirsizlik_kaynagi"]
    satirlar = [json.loads(s) for s in (cikti / "form.jsonl").read_text().splitlines()]
    assert {"yalin_esik", "belirsizligi_bilen"} <= set(satirlar[0])
    cap = cv2.VideoCapture(str(cikti / "overlay.avi"))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 3


def test_76_mm_belirsizlikle_telefon_kesin_konusmaz(ortam):
    """Tezin mesaji: tek gorus hatasi kadar belirsizlikle karar verilemez."""
    video, K, uzunluk, cikti = ortam
    ozet = demo.run(video, "m.task", cikti, K, uzunluk, max_frames=3)
    assert ozet["belirsizligi_bilen_kararlari"] == {"belirsiz": 3}
    assert "belirsiz" not in ozet["yalin_esik_kararlari"]


@pytest.mark.parametrize("sigma", [0.0, -0.1, float("nan")])
def test_gecersiz_belirsizlik_reddedilir(ortam, sigma):
    video, K, uzunluk, cikti = ortam
    with pytest.raises(ValueError, match="konum_belirsizligi_m"):
        demo.run(video, "m.task", cikti, K, uzunluk, konum_belirsizligi_m=sigma)
