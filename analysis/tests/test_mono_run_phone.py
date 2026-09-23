"""mono.run_phone uctan uca testi: sahte kaynak ve sahte model ile.

Gercek video/model gerekmez: `ProcessCapture` ve `MediaPipeEstimator` monkeypatch
edilir; boylece surec izolasyonlu okuma sozlesmesi, JSONL yazimi ve **eksik veri
politikasi** (NaN -> JSON null, asla 0.0) sinanir. Dis inceleme U.1-U.6.
"""
import json

import numpy as np
import pytest

from mono import run_phone
from mono.phone import json_uyumlu
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B

BOYUT = (320, 240)


class SahteKaynak:
    """`ProcessCapture` yerine: kare sayisina kadar sahte goruntu uretir."""

    def __init__(self, source, **ayar):
        self.kalan = 2                    # iki kare sonra kaynak biter
        self.kapandi = False

    def isOpened(self):
        return True

    def grab(self):
        if self.kalan <= 0:
            return False
        self.kalan -= 1
        return True

    def retrieve(self):
        return True, np.zeros((BOYUT[1], BOYUT[0], 3), np.uint8)

    def release(self):
        self.kapandi = True


class SahteModel:
    """`MediaPipeEstimator` yerine: `tamam` kadar nokta, `eksik` eklem NaN."""

    model_id = "sahte/model-1"
    threshold = 0.3

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def __call__(self, image):
        n = len(REFERANS_ISKELET)
        noktalar = np.array([[100.0 + 5.0 * i, 80.0 + 3.0 * i] for i in range(n)])
        gorunur = np.ones(n, dtype=bool)
        noktalar[0] = np.nan          # eksik eklem: NaN, 0.0 DEGIL
        gorunur[0] = False
        return Poz2B(REFERANS_ISKELET, noktalar, np.ones(n), gorunur, True, BOYUT,
                     model=self.model_id)


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    video = tmp_path / "telefon.mp4"
    video.write_bytes(b"sahte video icerigi")     # kaynak okunmuyor, kimligi yeter
    monkeypatch.setattr(run_phone, "ProcessCapture", SahteKaynak)
    monkeypatch.setattr(run_phone, "kestirici_olustur", lambda *a, **k: SahteModel())
    K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    uzunluklar = {("sag_kalca", "sag_diz"): 0.42, ("sag_diz", "sag_ayak_bilegi"): 0.43}
    return video, K, uzunluklar, tmp_path / "cikti"


def test_jsonl_eksik_eklemi_null_yazar(ortam):
    video, K, uzunluklar, cikti = ortam
    rapor = run_phone.run(video, "model.task", cikti, K, uzunluklar, max_frames=5)
    assert rapor["status"] == "completed"
    assert rapor["frames"] == 2 and rapor["detected_frames"] == 2
    assert rapor["stop_reason"] == "source_end_or_read_failure"
    assert rapor["physical_validation"] is False
    satirlar = [json.loads(s) for s in (cikti / "results.jsonl").read_text().splitlines()]
    assert len(satirlar) == 2
    for satir in satirlar:
        assert satir["points_px"][0] == [None, None]          # NaN -> null
        assert satir["points_3d_m"][0] == [None, None, None]  # 0.0 degil
        assert satir["visible"][0] is False
        assert satir["tespit"] is True
        assert satir["coordinate_space"] == "ozgun"
        # gorunen eklemler sonlu olmali
        assert all(v is not None for v in satir["points_px"][1])
        # tek gorus iskeletinin ek bilgisi de yaziliyor
        assert set(satir["iskelet_ek"]) == {"atlanan_kemikler", "derinlik_sacilimi_m"}
    assert rapor["summary"]["n_kare"] == 2


def test_cikti_dizini_yeni_olmali(ortam):
    video, K, uzunluklar, cikti = ortam
    run_phone.run(video, "model.task", cikti, K, uzunluklar, max_frames=1)
    with pytest.raises(ValueError, match="yeni olmal"):
        run_phone.run(video, "model.task", cikti, K, uzunluklar, max_frames=1)


def test_gecersiz_k_reddedilir(ortam):
    video, _, uzunluklar, cikti = ortam
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        run_phone.run(video, "model.task", cikti, np.zeros((2, 2)), uzunluklar)


def test_video_yoksa_hata(ortam):
    _, K, uzunluklar, cikti = ortam
    with pytest.raises(ValueError, match="mevcut"):
        run_phone.run("yok.mp4", "model.task", cikti, K, uzunluklar)


def test_max_frames_dogrulanir(ortam):
    video, K, uzunluklar, cikti = ortam
    with pytest.raises(ValueError, match="max_frames"):
        run_phone.run(video, "model.task", cikti, K, uzunluklar, max_frames=0)


def test_load_lengths_dogrular(tmp_path):
    iyi = tmp_path / "iyi.json"
    iyi.write_text(json.dumps([{"a": "sag_kalca", "b": "sag_diz", "metre": 0.42}]))
    assert run_phone.load_lengths(iyi) == {("sag_kalca", "sag_diz"): 0.42}

    for bozuk, mesaj in (
        ([{"a": "sag_kalca", "b": "yok_boyle_bir_eklem", "metre": 0.4}], "bilinmeyen eklem"),
        ([{"a": "sag_kalca", "b": "sag_kalca", "metre": 0.4}], "aynı olamaz"),
        # iki eklem de gecerli ama aralarinda kemik yok: hicbir 3B eklem uretmezdi
        ([{"a": "sag_bilek", "b": "sol_ayak_bilegi", "metre": 0.9}], "tanımlı bir kemik değil"),
        ([{"a": "sag_kalca", "b": "sag_diz", "metre": 0.42},
          {"a": "sag_diz", "b": "sag_kalca", "metre": 0.40}], "iki kez"),
        ([{"a": "sag_kalca", "b": "sag_diz", "metre": 0.0}], "pozitif"),
        ([{"a": "sag_kalca", "b": "sag_diz"}], "taşımalı"),
        ([], "liste"),
    ):
        yol = tmp_path / "bozuk.json"
        yol.write_text(json.dumps(bozuk))
        with pytest.raises(ValueError, match=mesaj):
            run_phone.load_lengths(yol)


def test_json_uyumlu_nan_ve_numpy_cevirir():
    deger = json_uyumlu({"a": np.float64(1.5), "b": np.array([np.nan, 2.0]),
                         "c": (np.int64(3), float("inf"))})
    assert deger == {"a": 1.5, "b": [None, 2.0], "c": [3, None]}
    assert json.dumps(deger, allow_nan=False)     # NaN sizmaz


def test_ozet_deneyi_tekrarlamaya_yetecek_girdileri_saklar(ortam):
    """Esik, K ve oncu uzunluklar ozette: deney sonradan aynen tekrarlanabilir."""
    video, K, uzunluklar, cikti = ortam
    run_phone.run(video, "model.task", cikti, K, uzunluklar, max_frames=5)
    ozet = json.loads((cikti / "summary.json").read_text())
    assert ozet["threshold"] == 0.3
    assert ozet["intrinsics_K"] == K.tolist()
    assert {(u["a"], u["b"]): u["metre"] for u in ozet["lengths_m"]} == uzunluklar
