"""capture.validate: kamerasiz kayit kabul kosusu (22 Eylul dort kaynak dogrulamasi).

Kucuk boyutla uctan uca kosulur: sentetik videolar -> kayit -> geri okuma ->
her goruntunun dogrulanmasi.
"""
import json

import pytest

from capture.validate import validate_capture


def test_kucuk_kosu_basarili_ve_rapor_yazilir(tmp_path):
    rapor = validate_capture(tmp_path / "v", cameras=2, frames=4, width=32, height=32)
    assert rapor["status"] == "passed"
    assert rapor["verified_images"] == 2 * 4
    assert json.loads((tmp_path / "v" / "validation.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("ayar,mesaj", [
    ({"cameras": 0}, "pozitif"),
    ({"frames": 1.5}, "pozitif"),
    ({"width": 33}, "çift"),
    ({"height": 8}, "en az 16"),
])
def test_gecersiz_ayar_reddedilir(tmp_path, ayar, mesaj):
    with pytest.raises(ValueError, match=mesaj):
        validate_capture(tmp_path / "v", **{"cameras": 2, "frames": 2, "width": 32,
                                            "height": 32, **ayar})


def test_var_olan_dizine_yazmaz(tmp_path):
    (tmp_path / "v").mkdir()
    with pytest.raises(FileExistsError):
        validate_capture(tmp_path / "v", cameras=1, frames=1, width=32, height=32)
