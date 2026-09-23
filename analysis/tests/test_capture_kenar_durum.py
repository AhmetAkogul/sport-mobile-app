"""capture/ kenar durum testleri: dış inceleme C.1.1, C.2.1, C.2.2, C.6.2.

Bu maddelerin ortak özelliği **sessiz** davranış olması: bozuk JSON, pickle
edilemeyen factory, release() içinde hata, `set`i çağrılamayan handle. Hepsi
"çalışıyor gibi görünen" hatalar olduğu için test edilmeden güvenilmez.
"""
import json
from pathlib import Path

import pytest

from capture.session import CameraConfig, SessionConfig
from capture.settings import configure_source
from capture.source import ProcessCapture


# --- C.1.1: from_json yapısal doğrulama ---

def _kamera():
    return dict(camera_id="a", source="video.avi", settings="test",
                width=32, height=24, fps=30)


def _oturum():
    return dict(participant_code="synthetic", lighting="studio",
                cameras=[_kamera()], notes="")


def test_from_json_turu_birebir_okur(tmp_path):
    yol = tmp_path / "session.json"
    yol.write_text(json.dumps(_oturum()))
    config = SessionConfig.from_json(yol)
    assert config.cameras[0] == CameraConfig(**_kamera())
    sozluk = config.to_dict()
    assert sozluk["participant_code"] == "synthetic"
    assert list(sozluk["cameras"])[0] == _kamera()   # to_dict tuple döndürür


def test_from_json_notes_istege_baglidir(tmp_path):
    """`notes` varsayilanli bir alan: elle yazilmis JSON'da olmayabilir."""
    veri = _oturum()
    veri.pop("notes", None)
    yol = tmp_path / "session.json"
    yol.write_text(json.dumps(veri))
    assert SessionConfig.from_json(yol).notes == ""


@pytest.mark.parametrize("icerik,mesaj", [
    ("[]", "nesne olmalı"),
    ("{", "çözülemedi"),
    (json.dumps({"participant_code": "s", "lighting": "l"}), "eksik alan"),
    (json.dumps({**_oturum(), "ekstra": 1}), "bilinmeyen alan"),
    (json.dumps({**_oturum(), "cameras": {"a": 1}}), "liste olmalı"),
    (json.dumps({**_oturum(), "cameras": ["a"]}), "nesne olmalı"),
])
def test_from_json_bozuk_yapiyi_acik_mesajla_reddeder(tmp_path, icerik, mesaj):
    yol = tmp_path / "bozuk.json"
    yol.write_text(icerik)
    with pytest.raises(ValueError, match=mesaj):
        SessionConfig.from_json(yol)


# --- C.2.1 / C.2.2: factory sözleşmesi ve release ---

class DosyaliKaynak:
    """release() çağrısını bir dosyaya yazar; ilk çağrıda hata verir.

    Sayaç dosyada tutulur çünkü handle alt süreçte yaşar: ana süreç nesneyi
    göremez, ama dosyayı okuyabilir.
    """

    def __init__(self, source):
        self.yol = Path(source)

    def isOpened(self):
        return True

    def grab(self):
        return False

    def retrieve(self):
        return False, None

    def release(self):
        with self.yol.open("a", encoding="utf-8") as dosya:
            dosya.write("release\n")
        raise RuntimeError("release hatası (test)")


def _dosyali_ac(source):
    """Modül seviyesinde: spawn factory'yi pickle eder."""
    return DosyaliKaynak(source)


class BosKaynak:
    def isOpened(self):
        return True

    def release(self):
        pass


def _bos_ac(source):
    return BosKaynak()


def test_pickle_edilemeyen_factory_erken_reddedilir():
    with pytest.raises(ValueError, match="pickle"):
        ProcessCapture("kaynak", factory=lambda source: None)


def test_pickle_edilebilir_factory_kabul_edilir():
    handle = ProcessCapture("kaynak", factory=_bos_ac)
    try:
        assert handle.isOpened()
    finally:
        handle.release()


def test_release_hatasi_cift_release_uretmez(tmp_path):
    """release() hata verse de handle **bir kez** kapatılır (C.2.1)."""
    sayac = tmp_path / "release.log"
    handle = ProcessCapture(str(sayac), factory=_dosyali_ac)
    with pytest.raises(Exception):
        handle.release()
    assert sayac.read_text().count("release") == 1
    handle.release()                     # idempotent: ikinci çağrı zararsız
    assert sayac.read_text().count("release") == 1


# --- C.6.2: hasattr yeterli değil ---

class TuzakliAyarlar:
    """`set`i çağrılamayan nesne: hasattr True der, çağrı TypeError verir."""

    set = "bu bir metin"

    def get(self, property_id):
        return 30.0


def test_cagrilamayan_set_hata_yerine_issue_uretir():
    camera = CameraConfig(camera_id="a", source=0, settings="test",
                          width=32, height=24, fps=30)
    report = configure_source(TuzakliAyarlar(), camera)
    assert report["mode"] == "apply"
    assert not any(report["set_accepted"].values())
    assert any("kabul etmedi" in issue for issue in report["issues"])


def test_kaynaklar_buyuk_kucuk_harf_duyarsiz_benzersiz():
    """Dis inceleme C.1.2: macOS'ta `Video.avi` ile `video.avi` ayni dosyadir."""
    iki = [_kamera(), {**_kamera(), "camera_id": "b", "source": "VIDEO.avi"}]
    with pytest.raises(ValueError, match="benzersiz"):
        SessionConfig(participant_code="s", lighting="l",
                      cameras=tuple(CameraConfig(**k) for k in iki))
