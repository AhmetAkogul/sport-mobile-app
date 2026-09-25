"""REHAB24-6: gercek harekette tek gorus form karari, bakis yonune gore.

Tezin manset sorusunun (form karari dogrulugu vs kamera acisi) ilk gercek
hareket verisiyle olcumu. Ayni squat tekrari iki senkron kameradan gorulur;
kamera 17'ye gore "front" olan tekrar kamera 18'den "profile"dir.

**Katman 1 -- kahin 2B (bu betik).** Telefonun 2B'si, mocap iskeletinin o
kameraya izdusumudur (veri setinin kendi 2B'si): dedektor hatasi yok. Tek
gorus 3B `mono.phone.tek_gorus_3b` ile kurulur; kemik uzunlugu onculeri kisi
basina mocap medyanidir ("anatomi bir kez olculmus"). Olculen hata yalnizca
**tek gorus geometrisinin** hatasidir; gercek dedektorle (Katman 2, video)
hata bundan buyuk olacaktir -- telefon hatasinin alt siniri.

Referans karar, ayni kuralin mocap iskeletine uygulanmasidir: soru "telefon
referansla ayni karari veriyor mu" (0070 iddia 2).

Tekrar karari (referans ve telefon icin ayni, profil `eval.egzersiz.SQUAT`):
valgus (sag ya da sol) en az `kusur_suresi_s` kesintisiz esigi asarsa KUSURLU;
yoksa karelerin en az `min_kapsama`'sinda kesin karar varsa DOGRU; yoksa
BELIRSIZ.

    python scripts/deney_rehab24_tek_gorus.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.egzersiz import SQUAT  # noqa: E402
from eval.form import Karar, form_degerlendir  # noqa: E402
from eval.protokol import Oge, Sayim, kisi_agirlikli, kisi_bootstrap  # noqa: E402
from mono.phone import tek_gorus_3b  # noqa: E402
from pose3d.hizalama import rijit_hizala  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B  # noqa: E402
from pose3d.pose2d import Poz2B  # noqa: E402
from veri.rehab24 import (  # noqa: E402
    EKLEMLER, _TURETILMIS, kamera_kestir, kareyi_cevir, tekrar_kareleri, tekrarlar_oku,
)

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data/dis/rehab24_6"
FPS = 30.0
KAMERALAR = {"c17": (1920, 1080), "c18": (1080, 1920)}
# Segmentation.txt: kamera 17 yonu -> kamera 18 yonu (kameralar dik).
C18_YONU = {"front": "profile", "half-profile": "half-profile", "profile": "front"}
VALGUS = ("diz_valgusu_sag", "diz_valgusu_sol")


def _referans_2b(x26: np.ndarray) -> np.ndarray:
    """(26, 2) -> REFERANS sirasinda (13, 2); boyun iki omuz ortalamasi."""
    P = np.full((len(REFERANS_ISKELET), 2), np.nan)
    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        idx = _TURETILMIS.get(ad, (EKLEMLER.get(ad),))
        P[i] = x26[list(idx)].mean(axis=0)
    return P


def poz_kur(x26: np.ndarray, boyut: tuple[int, int], kamera: int, kare: int) -> Poz2B:
    """Kahin 2B: kadraj disindaki eklem gorunmez (gercek dedektor de goremez)."""
    P = _referans_2b(x26)
    w, h = boyut
    icinde = (np.isfinite(P).all(axis=1) & (P[:, 0] >= 0) & (P[:, 0] < w)
              & (P[:, 1] >= 0) & (P[:, 1] < h))
    P[~icinde] = np.nan
    return Poz2B(iskelet=REFERANS_ISKELET, noktalar=P, guven=icinde.astype(float),
                 gorunur=icinde, tespit=bool(icinde.any()), goruntu_boyutu=boyut,
                 model="rehab24_kahin_2b", kamera=kamera, kare=kare)


def kisi_onculeri(tekrarlar, kareler_al) -> dict[str, dict]:
    """Kisi basina kemik uzunlugu medyani (metre), butun squat karelerinden."""
    uzunluk: dict = defaultdict(lambda: defaultdict(list))
    for t in tekrarlar:
        for q in kareler_al(t)[::5]:
            isk = kareyi_cevir(q)
            for a, b in REFERANS_ISKELET.baglantilar:
                u = isk.uzunluk(a, b)
                if np.isfinite(u) and u > 0:
                    uzunluk[t.kisi][(a, b)].append(u)
    return {k: {bag: float(np.median(v)) for bag, v in d.items()} for k, d in uzunluk.items()}


def tekrar_karari(kare_kararlari: list[tuple[Karar, Karar]]) -> str:
    """Kare basina (sag, sol) valgus kararlarindan tekrar karari (profil SQUAT)."""
    gerekli = int(np.ceil(SQUAT.kusur_suresi_s * FPS))
    for taraf in (0, 1):
        seri = 0
        for kk in kare_kararlari:
            seri = seri + 1 if kk[taraf] is Karar.KUSURLU else 0
            if seri >= gerekli:
                return "kusurlu"
    kesin = np.mean([all(k is not Karar.BELIRSIZ for k in kk) for kk in kare_kararlari])
    return "dogru" if kesin >= SQUAT.min_kapsama else "belirsiz"


def _valgus_kararlari(isk) -> tuple[tuple[Karar, Karar], tuple[float, float]]:
    r = form_degerlendir(isk)
    return (tuple(r.olcumler[a].karar for a in VALGUS),
            tuple(float(r.olcumler[a].deger) for a in VALGUS))


def _ozet(v) -> dict | None:
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "medyan": round(float(np.median(v)), 2),
            "p95": round(float(np.percentile(v, 95)), 2)}


def _temiz(o):
    """NaN -> None (JSON null), ic ice."""
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _temiz(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_temiz(v) for v in o]
    return o


def _tespit_yukle(backend: str, video: str, cam: str) -> dict | None:
    yol = KOK / f"out/rehab24_tespit/{backend}/{video}-{cam}.npz"
    if not yol.exists():
        return None
    d = np.load(yol)
    return {"indeks": {int(k): i for i, k in enumerate(d["kare"])},
            **{k: d[k] for k in ("noktalar", "guven", "gorunur", "tespit", "dunya",
                                 "dunya_gorunur", "boyut")}}


def _tespit_pozu(d: dict, i: int, ci: int, kare: int) -> Poz2B:
    P = d["noktalar"][i].copy()
    g = d["gorunur"][i].astype(bool) & np.isfinite(P).all(axis=1)
    P[~g] = np.nan
    return Poz2B(iskelet=REFERANS_ISKELET, noktalar=P, guven=np.where(g, d["guven"][i], 0.0),
                 gorunur=g, tespit=bool(d["tespit"][i]),
                 goruntu_boyutu=tuple(int(x) for x in d["boyut"]),
                 model="rehab24_tespit", kamera=ci, kare=kare)


def _dunya_iskeleti(d: dict, i: int) -> Iskelet3B:
    n = len(REFERANS_ISKELET)
    g = d["dunya_gorunur"][i].astype(bool) & d["gorunur"][i].astype(bool)
    P = np.asarray(d["dunya"][i], float).copy()
    g &= np.isfinite(P).all(axis=1)
    P[~g] = np.nan
    return Iskelet3B(REFERANS_ISKELET, P, g, g.astype(int), np.full(n, np.nan),
                     cerceve="mediapipe_kalca_merkezi", kaynak="mediapipe_world")


def main() -> None:
    tekrarlar = [t for t in tekrarlar_oku(VERI / "Segmentation.csv") if not t.mocap_hatali]
    onbellek: dict[str, np.ndarray] = {}

    def iki_b(video: str, cam: str) -> np.ndarray:
        anahtar = f"{video}-{cam}"
        if anahtar not in onbellek:
            onbellek[anahtar] = np.load(VERI / f"2d_joints/Ex6/{anahtar}-30fps.npy")
        return onbellek[anahtar]

    # Kameralar butun kayitlarda sabit (DLT: rms 0,000 px, 25 Eylul); tek bir
    # video uzerinden kamera basina bir kez kestirilir.
    ornek = tekrarlar[0].video
    X_ornek = np.load(VERI / f"3d_joints/Ex6/{ornek}-30fps.npy")[::10, :, :3]
    kameralar = {c: kamera_kestir(X_ornek.reshape(-1, 3), iki_b(ornek, c)[::10].reshape(-1, 2))
                 for c in KAMERALAR}
    onculer = kisi_onculeri(tekrarlar, lambda t: tekrar_kareleri(VERI, t))

    # Yontemler: kahin 2B her zaman; tespit dosyasi olan her backend icin
    # geometri tabani, MediaPipe icin ayrica pose_world tabani.
    backendler = sorted(p.name for p in (KOK / "out/rehab24_tespit").glob("*") if p.is_dir())
    yontem_adlari = ["geometri_kahin"] + [f"geometri_{b}" for b in backendler] + (
        ["mediapipe_world"] if "mediapipe" in backendler else [])

    satirlar = []
    for t in tekrarlar:
        kareler = tekrar_kareleri(VERI, t)
        ref_iskelet = [kareyi_cevir(q) for q in kareler]
        ref = [_valgus_kararlari(r) for r in ref_iskelet]
        ref_karar = tekrar_karari([r[0] for r in ref])
        for ci, (cam, boyut) in enumerate(KAMERALAR.items()):
            kam = kameralar[cam]
            x = iki_b(t.video, cam)[t.ilk:t.son + 1]
            tespitler = {b: _tespit_yukle(b, t.video, cam) for b in backendler}
            for yontem in yontem_adlari:
                b = yontem.split("_", 1)[1] if yontem.startswith("geometri_") else "mediapipe"
                d = tespitler.get(b)
                if yontem != "geometri_kahin" and d is None:
                    continue
                tel, hata, sekil, gorunur_oran, valgus_fark = [], [], [], [], []
                for j, q in enumerate(kareler):
                    kare = t.ilk + j
                    if yontem == "geometri_kahin":
                        isk = tek_gorus_3b(poz_kur(x[j], boyut, ci, kare), kam.K, onculer[t.kisi])
                    elif kare not in d["indeks"]:
                        tel.append((Karar.BELIRSIZ, Karar.BELIRSIZ))
                        gorunur_oran.append(0.0)
                        continue
                    elif yontem == "mediapipe_world":
                        isk = _dunya_iskeleti(d, d["indeks"][kare])
                    else:
                        isk = tek_gorus_3b(_tespit_pozu(d, d["indeks"][kare], ci, kare),
                                           kam.K, onculer[t.kisi])
                    kk, deg = _valgus_kararlari(isk)
                    tel.append(kk)
                    valgus_fark.extend(abs(a - b_) for a, b_ in zip(deg, ref[j][1])
                                       if np.isfinite(a) and np.isfinite(b_))
                    P_ref = (kam.R @ ref_iskelet[j].noktalar.T + kam.t[:, None]).T
                    m = isk.gorunur & np.isfinite(P_ref).all(axis=1)
                    gorunur_oran.append(float(m.mean()))
                    if not m.any():
                        continue
                    # Birincil: mutlak hata, kamera cercevesinde (donusum kalibrasyondan).
                    # MediaPipe world mutlak konum vermez: yalnizca ikincil sekil hatasi.
                    if yontem != "mediapipe_world":
                        hata.append(float(np.mean(np.linalg.norm(
                            isk.noktalar[m] - P_ref[m], axis=1)) * 1000))
                    # Ikincil (0070: tani, basari olcutu degil): kare basina rijit hizali sekil.
                    if m.sum() >= 3:
                        h = rijit_hizala(isk.noktalar[m], P_ref[m])
                        sekil.append(float(np.mean(np.linalg.norm(
                            h.uygula(isk.noktalar[m]) - P_ref[m], axis=1)) * 1000))
                satirlar.append({
                    "yontem": yontem, "kisi": t.kisi, "video": t.video, "tekrar": t.tekrar_no,
                    "kamera": cam, "yon": t.yon if cam == "c17" else C18_YONU[t.yon],
                    "fizyoterapist_dogru": t.dogru, "referans": ref_karar,
                    "telefon": tekrar_karari(tel),
                    "mpjpe_mm": float(np.mean(hata)) if hata else float("nan"),
                    "sekil_mm_hizali": float(np.mean(sekil)) if sekil else float("nan"),
                    "valgus_mutlak_fark_derece": float(np.median(valgus_fark)) if valgus_fark
                    else float("nan"),
                    "gorunur_eklem_orani": float(np.mean(gorunur_oran)),
                })

    yontemler: dict = {}
    for yontem in yontem_adlari:
        yonler = {}
        for yon in ("front", "half-profile", "profile"):
            s = [r for r in satirlar if r["yontem"] == yontem and r["yon"] == yon]
            if not s:
                continue
            og = [Oge(r["kisi"], f'{r["video"]}-{r["kamera"]}', r["referans"], r["telefon"])
                  for r in s]
            yonler[yon] = {
                "n_tekrar": len(s), "n_kisi": len({r["kisi"] for r in s}),
                "protokol_0070": Sayim.ciftlerden((o.referans, o.telefon) for o in og).oranlar(),
                "kisi_agirlikli": kisi_agirlikli(og),
                "kisi_bootstrap": kisi_bootstrap(og),
                "mpjpe_mm": _ozet([r["mpjpe_mm"] for r in s]),
                "sekil_mm_hizali": _ozet([r["sekil_mm_hizali"] for r in s]),
                "valgus_mutlak_fark_derece": _ozet([r["valgus_mutlak_fark_derece"] for r in s]),
                "gorunur_eklem_orani": round(float(np.mean(
                    [r["gorunur_eklem_orani"] for r in s])), 4),
                "referans_kusurlu": sum(r["referans"] == "kusurlu" for r in s),
            }
        yontemler[yontem] = yonler

    sonuc = {
        "deney": "rehab24_tek_gorus",
        "katmanlar": {"geometri_kahin": "kahin 2B (mocap izdusumu) + kemik onculu geometri",
                      "geometri_<backend>": "gercek dedektor 2B + kemik onculu geometri",
                      "mediapipe_world": "MediaPipe pose_world (ogrenilmis 3B, mutlak konum yok)"},
        "kaynak": "REHAB24-6 Ex6; 3d_joints + 2d_joints 30 fps; videolar (tespit)",
        "kameralar": {c: {"fx": round(float(k.K[0, 0]), 1), "fy": round(float(k.K[1, 1]), 1),
                          "rms_px": round(k.rms_px, 6),
                          "merkez_m": np.round(-k.R.T @ k.t, 3).tolist()}
                      for c, k in kameralar.items()},
        "tekrar_kurali": {"kusur_suresi_s": SQUAT.kusur_suresi_s,
                          "min_kapsama": SQUAT.min_kapsama, "fps": FPS},
        "onculer": "kisi basina mocap kemik uzunlugu medyani",
        "yontemler": yontemler,
        "not": ("9 kisi < 10: on kanit. Referans karar ayni kuralin mocap'e uygulanmasi; "
                "0014 valgus tanimi EC3D'de secildi. sekil_mm_hizali kare basina rijit "
                "hizalamadir: 0070'e gore yalnizca ikincil tani."),
    }
    (KOK / "out").mkdir(exist_ok=True)
    (KOK / "out/rehab24_tek_gorus.json").write_text(
        json.dumps(_temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (KOK / "out/rehab24_tek_gorus_satirlar.json").write_text(
        json.dumps(_temiz(satirlar), ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8")
    for yontem, yonler in yontemler.items():
        for yon, y in yonler.items():
            o = y["protokol_0070"]
            print(f"{yontem:18s} {yon:12s}", {k: o[k] for k in ("N", "D", "C", "A", "K", "Y")},
                  "mpjpe", (y["mpjpe_mm"] or {}).get("medyan"),
                  "sekil", (y["sekil_mm_hizali"] or {}).get("medyan"),
                  "valgus_fark", (y["valgus_mutlak_fark_derece"] or {}).get("medyan"))


if __name__ == "__main__":
    main()
