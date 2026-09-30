"""Olay semasi, panel durumu ve SSE servisi (0042)."""

import json
import threading
import urllib.request
from datetime import datetime, timezone

import pytest

from mono.olay import OLAY_SURUMU, PanelDurumu, olay_dogrula, olay_yap
from mono.olay_servisi import OlayKaynagi, sunucu

Z = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)


def _olay(tur, kisi=1, istasyon="squat-1", **ek):
    b = {"tur": tur, "kimlik": kisi, "t": 1.0, "metin": f"{tur} metni", **ek}
    return olay_yap(b, yuzey="salon", istasyon=istasyon, kamera="0", model_surumu="m:1", zaman=Z)


def test_olay_semasi_ve_dogrulama():
    o = _olay("tekrar", hareket="squat", tekrar=2, karar="yanlis", p_yanlis=0.7123,
              olculer={"diz_onde": 1.0})
    olay_dogrula(o)
    assert o["surum"] == OLAY_SURUMU and o["kaynak"]["istasyon"] == "squat-1"
    assert o["p_yanlis"] == 0.712 and "olculer" not in o          # ayrinti olaya girmez
    assert o["zaman"].startswith("2026-09-29T02:00:00.000")
    with pytest.raises(ValueError):
        _olay("bilinmeyen")
    with pytest.raises(ValueError):
        olay_yap({"tur": "tekrar", "kimlik": 1, "t": 0, "metin": ""}, yuzey="tv",
                 istasyon="a", kamera="0", model_surumu="m")
    with pytest.raises(ValueError):
        olay_dogrula({k: v for k, v in o.items() if k != "kisi"})


def test_problemli_uyarisi_iki_duzeltte_bir_kez():
    d = PanelDurumu()
    assert d.ekle(_olay("hareket", hareket="squat")) is None
    assert d.ekle(_olay("tekrar", hareket="squat", tekrar=1, karar="yanlis")) is None
    assert d.ekle(_olay("ipucu", olcu="diz_onde")) is None
    u = d.ekle(_olay("tekrar", hareket="squat", tekrar=2, karar="yanlis"))
    assert u and u["tur"] == "uyari" and "2'i duzeltilecek" in u["metin"]
    assert d.ekle(_olay("tekrar", hareket="squat", tekrar=3, karar="yanlis")) is None  # yinelenmez
    assert d.ekle(_olay("tekrar", kisi=1, istasyon="lunge-2", karar="yanlis")) is None  # baska kisi
    ozet = d.ozet()
    assert ozet[0]["problemli"] and ozet[0]["duzelt"] == 3 and ozet[0]["istasyon"] == "squat-1"


def test_kaynak_yarim_satiri_bekler_bozuk_satiri_atlar(tmp_path):
    yol = tmp_path / "o.jsonl"
    yol.write_text(json.dumps(_olay("hareket", hareket="squat")) + "\nbozuk\n{\"yarim\":")
    k = OlayKaynagi(yol)
    assert k.oku() == 1 and k.bozuk == 1
    with open(yol, "a") as f:
        f.write(" 1}\n")                       # yarim satir tamamlandi ama semaya uymuyor
    assert k.oku() == 0 and k.bozuk == 2


def test_servis_durum_ve_sse(tmp_path):
    yol = tmp_path / "o.jsonl"
    yol.write_text("")
    s, kaynak, dur = sunucu(yol, 0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    taban = f"http://127.0.0.1:{s.server_address[1]}"
    try:
        assert "Antrenör" in urllib.request.urlopen(taban + "/", timeout=5).read().decode()
        with open(yol, "a") as f:
            for o in (_olay("hareket", hareket="squat"),
                      _olay("tekrar", hareket="squat", tekrar=1, karar="yanlis"),
                      _olay("tekrar", hareket="squat", tekrar=2, karar="yanlis")):
                f.write(json.dumps(o) + "\n")
        kaynak.oku()
        durum = json.loads(urllib.request.urlopen(taban + "/durum", timeout=5).read())
        assert durum["kisiler"][0]["problemli"] is True
        r = urllib.request.urlopen(taban + "/olaylar", timeout=5)
        turler = []
        while len(turler) < 4:
            satir = r.readline().decode()
            if satir.startswith("data: "):
                turler.append(json.loads(satir[6:])["tur"])
        assert turler == ["hareket", "tekrar", "tekrar", "uyari"]
    finally:
        dur.set()
        s.shutdown()
        s.server_close()
