"""scripts/sayi_kilidi.py: rapordaki sayilarin sessizce degismesini yakalar.

`make reproduce` ~90 s surdugu icin testte kosulmaz; burada yalnizca
karsilastirma mantigi sahte ciktilarla sinanir.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sayi_kilidi  # noqa: E402


def test_duzlestir_ic_ice_ve_listeleri_acar():
    d = sayi_kilidi.duzlestir({"ozet": {"mm": 14.1, "ad": "metin"}, "l": [1, None, True]})
    assert d == {"ozet.mm": 14.1, "l[0]": 1.0, "l[1]": None, "l[2]": 1.0}


def test_ayni_sayi_fark_uretmez_kucuk_sayisal_gurultu_de():
    b = {"a.json": {"x": 14.1127}}
    assert sayi_kilidi.karsilastir(b, {"a.json": {"x": 14.1127 * (1 + 1e-9)}}) == []


@pytest.mark.parametrize("gozlenen,parca", [
    ({"a.json": {"x": 15.3}}, "kilit 14.1 -> simdi 15.3"),
    ({"a.json": {"x": None}}, "simdi None"),            # olculdu -> olculemedi
    ({"a.json": {"x": 14.1, "y": 1.0}}, "yeni alan"),
    ({"a.json": {}}, "alan kayboldu"),
    ({}, "ciktida yok"),
])
def test_her_degisiklik_turu_yakalanir(gozlenen, parca):
    farklar = sayi_kilidi.karsilastir({"a.json": {"x": 14.1}}, gozlenen)
    assert any(parca in f for f in farklar), farklar


def test_main_sapmada_bir_dondurur(tmp_path, monkeypatch):
    monkeypatch.setattr(sayi_kilidi, "DENEY_CIKTILARI", ("a.json",))
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text(json.dumps({"ozet": {"mm": 14.1}}))
    kilit = tmp_path / "kilit.json"
    assert sayi_kilidi.main(["--out", str(out), "--kilit", str(kilit), "--guncelle"]) == 0
    assert sayi_kilidi.main(["--out", str(out), "--kilit", str(kilit)]) == 0
    (out / "a.json").write_text(json.dumps({"ozet": {"mm": 15.3}}))
    assert sayi_kilidi.main(["--out", str(out), "--kilit", str(kilit)]) == 1


def test_kayitli_kilit_butun_deney_ciktilarini_kapsar():
    kilit = json.loads(sayi_kilidi.KILIT.read_text(encoding="utf-8"))
    assert set(kilit) == set(sayi_kilidi.DENEY_CIKTILARI)
    # mansetteki sayilar kilitte
    assert kilit["senkron_kaymasi.json"]["ozet.serbest_30fps_1ms_p95_mm"] == pytest.approx(14.1127, abs=1e-4)
    assert kilit["form_belirsizligi.json"]["ozet.duzenek_karar_cozunurlugu_derece"] == pytest.approx(1.781)


def test_guncelle_once_degisen_ile_yeni_alani_ayirir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sayi_kilidi, "DENEY_CIKTILARI", ("a.json",))
    out = tmp_path / "out"
    out.mkdir()
    kilit = tmp_path / "kilit.json"
    (out / "a.json").write_text(json.dumps({"x": 1.0}))
    sayi_kilidi.main(["--out", str(out), "--kilit", str(kilit), "--guncelle"])
    (out / "a.json").write_text(json.dumps({"x": 2.0, "y": 3.0}))
    capsys.readouterr()
    sayi_kilidi.main(["--out", str(out), "--kilit", str(kilit), "--guncelle"])
    cikti = capsys.readouterr().out
    assert "1 yeni alan, 1 degisen/kaybolan" in cikti
    assert "kilit 1.0 -> simdi 2.0" in cikti
