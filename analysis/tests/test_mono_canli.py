"""mono.canli: kamerasiz, sentetik iskeletle canli degerlendirme mantigi."""
from __future__ import annotations

import numpy as np
import pytest

from eval.form import form_degerlendir
from mono import canli
from pose3d.iskelet import REFERANS_ISKELET as R


def _iskelet(fleksiyon=0.0, ice=0.0, yaw=0.0):
    """MediaPipe dunya cercevesi (X sag, Y asagi, Z ileri), metre.

    `fleksiyon`: diz bukulmesi (derece); `ice`: dizlerin orta hatta kaymasi (m);
    `yaw`: govdenin kameraya gore donusu (0 onden, 90 profil).
    """
    t = np.radians(fleksiyon / 2)
    P = {}
    for taraf, x in (("sag", -0.1), ("sol", 0.1)):
        ayak = np.array([x, 0.9, 0.0])
        diz = ayak + 0.45 * np.array([0, -np.cos(t), -np.sin(t)])
        diz[0] -= np.sign(x) * ice
        kalca = diz + 0.45 * np.array([0, -np.cos(t), np.sin(t)])
        kalca[0] = x
        P[f"{taraf}_ayak_bilegi"], P[f"{taraf}_diz"], P[f"{taraf}_kalca"] = ayak, diz, kalca
        P[f"{taraf}_omuz"] = kalca + [np.sign(x) * 0.08, -0.5, 0]
        P[f"{taraf}_dirsek"] = P[f"{taraf}_omuz"] + [0, 0.3, 0]
        P[f"{taraf}_bilek"] = P[f"{taraf}_dirsek"] + [0, 0.25, 0]
    P["boyun"] = (P["sag_omuz"] + P["sol_omuz"]) / 2
    M = np.array([P[e] for e in R.eklemler])
    merkez = (M[R.indeks("sag_kalca")] + M[R.indeks("sol_kalca")]) / 2
    a = np.radians(yaw)
    Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return (M - merkez) @ Ry.T


def _ek(P):
    return {"world_points_all_m": P.tolist()}


def test_kisi_yoksa_belirsiz():
    assert canli.dunya_iskeleti({}) is None
    kare, bilgi = canli.kare_hazirla(None)
    assert kare["fleksiyon_derece"] is None
    assert {o["karar"] for o in kare["olcumler"].values()} == {"belirsiz"}
    assert bilgi["aci"] is None


@pytest.mark.parametrize("yaw", [0.0, 30.0, 75.0, 90.0])
def test_govde_acisi_donusu_izler(yaw):
    assert canli.govde_acisi(_iskelet(yaw=yaw)) == pytest.approx(yaw, abs=1e-6)


def test_onden_kayma_cikarilir():
    isk = canli.dunya_iskeleti(_ek(_iskelet(fleksiyon=80, ice=0.05)))
    ham = form_degerlendir(isk).olcumler
    kare, bilgi = canli.kare_hazirla(isk)
    assert bilgi["aci"] == pytest.approx(0.0, abs=1e-6)
    assert bilgi["kayma"] == pytest.approx(canli.KAYMA_B0)
    for ad, v in zip(canli.VALGUS, bilgi["valgus"]):
        assert v == pytest.approx(ham[ad].deger - canli.KAYMA_B0)
    assert not bilgi["olculemez"]


def test_profilde_olculemez_ve_karar_yok():
    isk = canli.dunya_iskeleti(_ek(_iskelet(fleksiyon=80, ice=0.1, yaw=80)))
    kare, bilgi = canli.kare_hazirla(isk)
    assert bilgi["olculemez"]
    assert {o["karar"] for o in kare["olcumler"].values()} == {"belirsiz"}


def _squat(d, ice_dipte=0.0, yaw=0.0, t0=0.0, fps=30):
    """Ayakta -> 100 derece -> ayakta, 3 s; bir tekrarin son durumunu dondurur."""
    acilar = np.concatenate([np.linspace(0, 100, 45), np.linspace(100, 0, 45), np.zeros(15)])
    durum = None
    for i, f in enumerate(acilar):
        ice = ice_dipte * f / 100
        durum = d.ekle(t0 + i / fps, _ek(_iskelet(fleksiyon=f, ice=ice, yaw=yaw)))
    return durum


def test_duz_dizli_squat_dogru_ice_kacan_kusurlu():
    d = canli.CanliDegerlendirici()
    s = _squat(d)["son_tekrar"]
    assert s is not None and s["tamamlandi"] and s["karar"] == "dogru"
    assert s["medyan_aci"] == pytest.approx(0.0, abs=1e-6) and not s["olculemez"]
    s2 = _squat(d, ice_dipte=0.12, t0=10.0)["son_tekrar"]
    assert s2["tekrar"] == 2 and s2["karar"] == "kusurlu"


def test_yandan_squat_olculemez_isaretlenir():
    s = _squat(canli.CanliDegerlendirici(), ice_dipte=0.12, yaw=85)["son_tekrar"]
    assert s is not None and s["olculemez"] and s["karar"] != "kusurlu"


def test_zaman_geri_giderse_cokmez():
    d = canli.CanliDegerlendirici()
    d.ekle(1.0, _ek(_iskelet()))
    d.ekle(1.0, _ek(_iskelet()))        # ayni damga: kesin artana cekilir
    assert d.son_zaman > 1.0


@pytest.mark.parametrize("bacak,yuz,aci,kip", [
    (False, 0.9, 0.0, "kadraj"), (True, 0.9, None, "kadraj"),
    (True, 0.1, 10.0, "sirt"), (True, 0.1, 80.0, "yandan"),   # yandan yuz gorunmeyebilir
    (True, 0.9, 10.0, "onden"), (True, 0.9, 45.0, "ara"), (True, None, 10.0, "onden"),
])
def test_rehber_kipleri(bacak, yuz, aci, kip):
    assert canli.rehber(bacak, yuz, aci)[0] == kip


def test_sagital_olculer_ayakta_ve_cokmus():
    ayakta = canli.sagital_olculer(canli.dunya_iskeleti(_ek(_iskelet())))
    assert ayakta["derinlik"] == pytest.approx(0.0, abs=1e-6)
    assert ayakta["govde_egimi"] == pytest.approx(0.0, abs=1e-6)
    assert ayakta["diz_onde"] == pytest.approx(0.0, abs=1e-6)
    cok = canli.sagital_olculer(canli.dunya_iskeleti(_ek(_iskelet(fleksiyon=100))))
    assert cok["derinlik"] == pytest.approx(100.0, abs=1e-6)
    assert cok["diz_onde"] == pytest.approx(np.sin(np.radians(50)), abs=1e-6)


def test_sirti_donuk_ve_kadraj_disi_valgus_karari_yok():
    ek = {**_ek(_iskelet(fleksiyon=80, ice=0.1)), "yuz_guveni": 0.05}
    kare, bilgi = canli.kare_hazirla(canli.dunya_iskeleti(ek), True, ek["yuz_guveni"])
    assert bilgi["kip"] == "sirt" and bilgi["olculemez"]
    assert {o["karar"] for o in kare["olcumler"].values()} == {"belirsiz"}
    _, bilgi = canli.kare_hazirla(canli.dunya_iskeleti(_ek(_iskelet())), False, 0.9)
    assert bilgi["kip"] == "kadraj" and bilgi["olculemez"] and bilgi["sagital"] is None


def test_yandan_tekrar_ozetinde_sagital_var_onden_yok():
    s = _squat(canli.CanliDegerlendirici(), yaw=85)["son_tekrar"]
    assert s["sagital"]["derinlik"] == pytest.approx(100.0, abs=1.0)
    s = _squat(canli.CanliDegerlendirici())["son_tekrar"]
    assert s["sagital"] is None


def test_kaynak_ac_akis_ve_video_ayrimi(tmp_path):
    import argparse

    from mono.canli import kaynak_ac
    a = argparse.Namespace(video=None, akis="rtsp://k:p@127.0.0.1:1/yok", kamera=0)
    cap, dosya, ad = kaynak_ac(a)
    assert dosya is False and ad == "127.0.0.1:1/yok"      # parola adda yok
    cap.release()
