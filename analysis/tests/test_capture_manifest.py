"""capture.manifest: kayit manifesti kurallari (0070 §3, 0071 §5)."""
import copy
import json
from pathlib import Path

import pytest

from capture.manifest import bolme_ve_cikarilanlar, denetle, main

ORNEK = json.loads((Path(__file__).resolve().parent.parent
                    / "docs/veri/ornek-manifest.json").read_text(encoding="utf-8"))


def _m():
    return copy.deepcopy(ORNEK)


def test_ornek_manifest_gecerli():
    assert denetle(_m()) == []


@pytest.mark.parametrize("bozma,parca", [
    (lambda m: m["oturumlar"][0].update(kisi="YOK"), "bilinmeyen kisi"),
    (lambda m: m["oturumlar"][0].update(kalibrasyon="YOK"), "bilinmeyen kalibrasyon"),
    (lambda m: m["oturumlar"][0]["kosul"].pop("isik"), "kosul.isik eksik"),
    (lambda m: m["oturumlar"][0]["referans_ucgenleme"].append("tel"), "telefon referans"),
    (lambda m: m["oturumlar"][0].update(cihazlar=[c for c in m["oturumlar"][0]["cihazlar"]
                                                  if c["rol"] != "telefon"]), "telefon kaydi yok"),
    (lambda m: m["oturumlar"][0]["cihazlar"][0].update(video_sha256="abc"), "SHA-256"),
    (lambda m: m["oturumlar"][0]["cihazlar"][0].update(fps=0), "fps"),
    (lambda m: m["oturumlar"][0]["tekrarlar"][1].update(baslangic_ms=2000), "cakisiyor"),
    (lambda m: m["oturumlar"][0].update(kemik_uzunlugu={"a": 1}), "kaynagi"),
    (lambda m: m["kisiler"].append({"kod": "P99"}), "hic oturumu yok"),
    (lambda m: m["oturumlar"].append(copy.deepcopy(m["oturumlar"][0])), "2 kez"),
    (lambda m: m.update(teste_ayrilan_kosullar={"hava": ["yagmur"]}), "bilinmeyen kosul"),
])
def test_kural_ihlali_yakalanir(bozma, parca):
    m = _m()
    bozma(m)
    assert any(parca in h for h in denetle(m)), denetle(m)


def test_bolme_kisi_butunlugu_ve_tutulan_kosul_cikarilir():
    b = bolme_ve_cikarilanlar(_m())
    assert (len(b["test"]), len(b["dogrulama"]), len(b["egitim"])) == (2, 2, 2)
    assert b["genelleme_kabulu_mumkun"]
    # Egitim/dogrulama kisilerinin "los" isikli oturumlari cikarilir ve sayilir.
    assert b["n_cikarilan"] == 4
    assert all(c["bolum"] != "test" and c["kosul"] == "isik" for c in b["cikarilan_oturumlar"])


def test_cli_gecersizde_bir_gecerlide_kilit(tmp_path, capsys):
    yol = tmp_path / "m.json"
    yol.write_text(json.dumps(_m()))
    assert main([str(yol)]) == 0
    assert len(json.loads(capsys.readouterr().out)["kilit_sha256"]) == 64
    bozuk = _m()
    bozuk["oturumlar"][0]["referans_ucgenleme"].append("tel")
    yol.write_text(json.dumps(bozuk))
    assert main([str(yol)]) == 1


def _etiket(**k):
    e = {"oturum": "P01-S1", "tekrar": 0, "kusur": "diz_valgusu_sag", "kaynak": "uzman",
         "karar": "kusurlu", "etiketleyen": "U1"}
    return {**e, **k}


def test_etiketler_manifeste_bagli_ve_kaynaklar_ayri():
    from capture.manifest import etiketleri_denetle
    m = _m()
    ikisi = [_etiket(), _etiket(kaynak="geometrik", karar="dogru", etiketleyen="kod-0010")]
    assert etiketleri_denetle(ikisi, m) == []         # iki kaynak birlikte durabilir
    for bozuk, parca in [
        (_etiket(oturum="YOK"), "manifestte olmayan"),
        (_etiket(kusur="ayak_acisi"), "bilinmeyen kusur"),
        (_etiket(kaynak="model"), "kaynak"),
        (_etiket(karar="belirsiz"), "gerekce"),
        (_etiket(etiketleyen=""), "etiketleyen"),
    ]:
        assert any(parca in h for h in etiketleri_denetle([bozuk], m)), parca
    assert any("tekrarli" in h for h in etiketleri_denetle([_etiket(), _etiket()], m))
