"""Mimari regresyonlar: birim, aykiri gorus, belirsizlik, zaman ve veri sizintisi."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calib.multiview import Kalibrasyon
from capture.alignment import file_sha256
from eval.egzersiz import SQUAT, kare_degerlendir, duzenek_degerlendir
from eval.form import form_degerlendir, sentetik_durus
from eval.karsilastirma import karsilastir, konum_hatasi_mm
from eval.tekrar import SquatTakip
from mono.world import dunya_iskeleti
from mono.mediapipe_model import MediaPipeEstimator
from pose3d.iskelet import REFERANS_ISKELET, iskelet_ucgenle
from pose3d.pose2d import Poz2B
from pose3d.saglam import ucgenle_saglam
from pose3d.triangulate import ucgenle
from uncertainty.eklem import eklem_kovaryanslari
from veri.ec3d import kareyi_cevir


def rig(n=4):
    K = np.array([[800., 0, 640], [0, 800, 480], [0, 0, 1]])
    return Kalibrasyon(0., [K.copy() for _ in range(n)], [np.zeros(5) for _ in range(n)],
                      [np.eye(3) for _ in range(n)], [np.array([-i*.5, 0., 0.]) for i in range(n)])


def observe(kal, point):
    out = {}
    for i in range(len(kal.Ks)):
        q = kal.Ks[i] @ (point+kal.Ts[i])
        out[i] = q[:2]/q[2]
    return out


def test_ec3d_birim_boyut_hatasi_sessiz_olamaz():
    s = kareyi_cevir(np.ones((3, 25))*.2)
    assert s.birim == "normalize" and not s.goren_kamera.any()
    with pytest.raises(ValueError, match="metre"):
        s.uzunluk("sag_kalca", "sag_diz")
    with pytest.raises(ValueError, match="metre"):
        form_degerlendir(s, konum_belirsizligi_m=.076)
    with pytest.raises(ValueError, match="metre"):
        konum_hatasi_mm(s, s)
    form_degerlendir(s)  # boyutsuz aci motoru halen kullanilabilir


def test_yanlis_kamera_elenir_gercek_nokta_korunur():
    k, point = rig(), np.array([.2, .1, 4.])
    obs = observe(k, point)
    obs[3] += [140, 110]
    assert np.linalg.norm(ucgenle(k, obs).nokta-point) > .1
    r = ucgenle_saglam(k, obs, esik_px=2, sigma_px=1)
    assert r.kabul == (0, 1, 2) and r.red == (3,)
    np.testing.assert_allclose(r.sonuc.nokta, point, atol=1e-8)
    np.testing.assert_allclose(r.sonuc.kovaryans, ucgenle(k, {i:obs[i] for i in r.kabul}, sigma_px=1).kovaryans)


def test_iki_iyi_bir_kotu_gorus_kesin_referans_olmaz():
    k = rig(3)
    obs = observe(k, np.array([0., 0., 3.]))
    obs[2] += [100, 100]
    assert ucgenle_saglam(k, obs).sonuc is None


def test_paralel_isin_ve_kamera_arkasi_reddedilir():
    k = rig(2)
    assert ucgenle_saglam(k, observe(k, np.array([0., 0., 10000.]))).sonuc is None
    assert ucgenle_saglam(k, observe(k, np.array([0., 0., -3.]))).sonuc is None


def test_saglam_iskelet_kovaryans_karar_baglantisi():
    k = rig()
    s = sentetik_durus()
    pts = s.noktalar + [0, 0, 4]
    poses = {}
    for i in range(4):
        xy = np.array([observe(k, p)[i] for p in pts])
        if i == 3:
            xy += [100, 100]
        poses[i] = Poz2B(REFERANS_ISKELET, xy, np.ones(13), np.ones(13, bool), True, (1280,960))
    sk, report = duzenek_degerlendir(k, poses, sigma_px=1, esik_px=2)
    assert sk.gorunur.all() and sk.kovaryans.shape == (13,3,3)
    np.testing.assert_allclose(sk.noktalar, pts, atol=1e-8)
    np.testing.assert_allclose(sk.kovaryans, eklem_kovaryanslari(k, sk, poses))
    assert all(o["karar"] == "belirsiz" for o in report["olcumler"].values())
    assert sk.belirsizlik_durumu == "kosullu"
    raw = iskelet_ucgenle(k, poses)
    assert np.linalg.norm(raw.noktalar-pts) > .1


def test_belirsizlik_yok_veya_dogrulanmamis_kesin_konusmaz():
    sk = sentetik_durus(valgus_sag=20)
    assert all(x["karar"] == "belirsiz" for x in kare_degerlendir(sk)["olcumler"].values())
    cov = np.tile(np.eye(3)*1e-8, (13,1,1))
    conditional = replace(sk, kovaryans=cov, belirsizlik_durumu="kosullu", belirsizlik_kaynagi="test")
    assert kare_degerlendir(conditional)["olcumler"]["diz_valgusu_sag"]["karar"] == "belirsiz"
    assert kare_degerlendir(conditional, kosullu_izin=True)["olcumler"]["diz_valgusu_sag"]["karar"] == "kusurlu"


def frame(angle, karar="dogru"):
    return {"profil": SQUAT.surum, "fleksiyon_derece": angle,
            "olcumler": {f:{"karar":karar} for f in SQUAT.kapsam}}


def exercise(decisions=None):
    angles = [0, 30, 45, 70, 80, 70, 50, 30, 0]
    tracker = SquatTakip()
    for i, a in enumerate(angles):
        tracker.ekle(i*.1, frame(a, (decisions or {}).get(i, "dogru")))
    return tracker.bitir()


def test_tekrar_evreleri_sure_ve_tek_kare_sicramasi():
    r = exercise()[0]
    assert r["tamamlandi"] and r["karar"] == "dogru"
    assert exercise({4:"kusurlu"})[0]["karar"] == "dogru"
    r = exercise(dict.fromkeys(range(2,7),"kusurlu"))[0]
    assert r["karar"] == "kusurlu" and len(r["kusurlar"]) == 4


def test_eksik_kare_kapsamayi_dusurur():
    r = exercise(dict.fromkeys(range(2,7),"belirsiz"))[0]
    assert r["karar"] == "belirsiz" and max(r["kapsama"].values()) < .8


def test_bosluk_yarim_tekrar_ve_monoton_zaman():
    t = SquatTakip()
    t.ekle(0, frame(0))
    t.ekle(.1, frame(40))
    out = t.ekle(1, frame(70))
    assert out["sonuc"]["karar"] == "belirsiz" and not out["sonuc"]["tamamlandi"]
    with pytest.raises(ValueError, match="artmali"):
        t.ekle(1, frame(60))
    t = SquatTakip()
    t.ekle(0, frame(0))
    t.ekle(.1, frame(40))
    assert t.bitir()[0]["neden"] == "kayit_sonu"


def test_world_mutlak_kamera_derinligi_gibi_karsilastirilmaz():
    p = SimpleNamespace(ek={"world_points_m":np.ones((13,3)), "world_visible":np.ones(13,bool)},
                        gorunur=np.ones(13,bool), model="mp")
    s = dunya_iskeleti(p)
    assert s.cerceve == "mediapipe_kalca_merkezi"
    with pytest.raises(ValueError, match="cerceve"):
        konum_hatasi_mm(s,sentetik_durus())


def test_hazir_world_ayni_model_cagrisindan_ve_2b_maskesiyle_gelir():
    estimator = MediaPipeEstimator.__new__(MediaPipeEstimator)
    estimator._closed, estimator.threshold = False, .5
    estimator.model_id, estimator.sha256 = "mp/test", "abc"
    image = [SimpleNamespace(x=.5, y=.5, z=0., visibility=1., presence=1.) for _ in range(33)]
    image[28].visibility = .1  # sag ayak bilegi
    world = [SimpleNamespace(x=.1, y=.2, z=.3, visibility=1., presence=1.) for _ in range(33)]
    calls = []

    def detect(x):
        calls.append(x)
        return SimpleNamespace(pose_landmarks=[image], pose_world_landmarks=[world])
    estimator._detector = SimpleNamespace(detect=detect)
    estimator._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB="rgb"), Image=lambda **k:k["data"])
    s = dunya_iskeleti(estimator(np.zeros((20,20,3),np.uint8)))
    assert len(calls) == 1 and s.birim == "metre"
    idx = REFERANS_ISKELET.indeks("sag_ayak_bilegi")
    assert not s.gorunur[idx] and np.isnan(s.noktalar[idx]).all()


def test_kilitli_manifest_hata_kapsama_ve_eksik_tahmin(tmp_path):
    p = tmp_path / "poses.jsonl"
    sk = sentetik_durus()
    p.write_text(json.dumps({"frame_index":0, "points_3d_m":sk.noktalar.tolist(),
        "visible_3d":[True]*13,"birim_3d":"metre","cerceve_3d":"kamera0","kaynak_3d":"test"})+"\n")
    spec = {"path":p.name,"sha256":file_sha256(p)}
    manifest = {"split":"test","participants":{"p1":"test"},"methods":["a","b"],
                "label_definition":"valgus-v1","locked_before_evaluation":True,"faults":["valgus"],
                "samples":[{"id":"x","person":"p1","session":"s1","frame_index":0,
                    "reference":spec,"reference_labels":{"valgus":"dogru"},
                    "predictions":{"a":spec},"prediction_labels":{"a":{"valgus":"dogru"}}}]}
    r = karsilastir(manifest,kok=tmp_path)
    assert r["methods"]["b"]["form"]["valgus"]["sayim"]["U"] == 1
    assert r["methods"]["a"]["mpjpe_mm"] is None  # ortak kume bos; sifir hata degil
    manifest["samples"][0]["predictions"]["b"] = spec
    r = karsilastir(manifest,kok=tmp_path)
    assert r["methods"]["a"]["mpjpe_mm"] == 0
    manifest["participants"]["p1"] = "train"
    with pytest.raises(ValueError,match="sizinti"):
        karsilastir(manifest,kok=tmp_path)
    manifest["participants"]["p1"] = "test"
    manifest["samples"][0]["reference_labels"] = {}
    r = karsilastir(manifest,kok=tmp_path)
    assert r["methods"]["a"]["form"]["valgus"]["sayim"]["R"] == 1
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="hash"):
        karsilastir(manifest,kok=tmp_path)
