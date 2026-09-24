"""scripts/veri_indir.py: katalog butunlugu ve erisim kurali (ag yok)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import veri_indir  # noqa: E402


def test_katalog_eksiksiz():
    assert veri_indir.katalog_denetle() == []


def test_bozuk_katalog_yakalanir():
    bozuk = {"x": {"url": "ftp://a", "lisans": "", "erisim": "gizli", "rol": "R", "not": "n"}}
    h = veri_indir.katalog_denetle(bozuk)
    assert any("lisans" in s for s in h) and any("erisim" in s for s in h)
    assert any("http" in s for s in h)


@pytest.mark.parametrize("ad", ["opencap_labvalidation", "flex", "ec3d", "ui_prmd"])
def test_hesap_basvuru_ve_klasor_adresleri_indirilmez(ad, tmp_path):
    """Hesap/basvuru isteyen ya da klasor/sayfa adresi olanlar otomatik indirilmez."""
    with pytest.raises(ValueError):
        veri_indir.indir(ad, hedef_kok=tmp_path)
    assert not any(tmp_path.iterdir())


def test_dosya_adi_zenodo_ve_hf_adreslerinden():
    assert veri_indir._dosya_adi(
        "https://zenodo.org/api/records/7672767/files/w12_rgb.mp4/content") == "w12_rgb.mp4"
    assert veri_indir._dosya_adi(
        "https://huggingface.co/x/y/resolve/main/model.safetensors") == "model.safetensors"


def test_eski_tek_dosyali_kayit_donusturulur_ve_yeniden_indirilmez(tmp_path, monkeypatch):
    import json
    dizin = tmp_path / "aistpp_kamera"
    dizin.mkdir()
    (dizin / "cameras.zip").write_bytes(b"x")
    (dizin / "KAYNAK.json").write_text(json.dumps({
        "ad": "aistpp_kamera", "url": veri_indir.KATALOG["aistpp_kamera"]["url"],
        "lisans": "l", "rol": "R", "dosya": "cameras.zip", "sha256": "0" * 64, "bayt": 1,
        "indirme_utc": "2026-09-23T00:00:00+00:00"}))
    monkeypatch.setattr(veri_indir, "_curl", lambda *a: pytest.fail("yeniden indirildi"))
    veri_indir.indir("aistpp_kamera", hedef_kok=tmp_path)
    kayit = json.loads((dizin / "KAYNAK.json").read_text())
    assert "cameras.zip" in kayit["dosyalar"] and "dosya" not in kayit
