"""Telefon sagital form olculerini (diz onde, derinlik, govde egimi) olcebiliyor mu?

REHAB24'te uzmanin "yanlis" kararini en cok sagital olculer acikladi
(`2026-09-28-tanim-fizyoterapist.md`): diz onde AUC 0,894, derinlik 0,792,
govde egimi 0,759. Valgus profilde olculemiyor; sagital olculer tam da orada
en iyi gorunmeli.

A. REHAB24 (9 kisi): telefon (MediaPipe world maskesiz) ile mocap'in ayni
   tekrardaki olcusu -- yon basina medyan mutlak fark ve Spearman; telefon
   olcusunun fizyoterapist etiketini ayirmasi (AUC).
B. Fitness-AQA (egitim + dogrulama; resmi test son olcume saklanir): telefon
   "diz onde" olcusu bagimsiz "dizler one" etiketini ayiriyor mu? Govde
   acisina gore.

Olculer `deney_rehab24_tanim.tekrar_olculeri` ile, telefon ve mocap icin ayni
kod. MediaPipe world'de dikey = -Y (kamera eksenli, Y asagi; kamera yatay
varsayimi).

    python scripts/deney_sagital.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

KOK = Path(__file__).resolve().parent.parent


def _yukle(ad: str, dosya: str):
    spec = importlib.util.spec_from_file_location(ad, KOK / "scripts" / dosya)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dt = _yukle("dt", "deney_rehab24_tanim.py")
tg = dt.tg
OLCULER = ("diz_onde", "derinlik", "govde_egimi", "valgus")
MP_YUKARI = np.array([0.0, -1.0, 0.0])
SQUAT = KOK / "data/dis/fitness_aqa/Squat/Labeled_Dataset"


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return None
    rx, ry = np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def rehab24() -> dict:
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv") if not t.mocap_hatali]
    gerekli = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))
    yukari: dict = {}
    satir = []
    for t in tekrarlar:
        kareler = tg.tekrar_kareleri(tg.VERI, t)
        ref = [tg.kareyi_cevir(q) for q in kareler]
        if t.video not in yukari:
            yukari[t.video] = dt.dikey_kestir(np.array([s.noktalar for s in ref[:10]]))
        o_ref = dt.tekrar_olculeri(ref, yukari[t.video], gerekli)
        for cam in tg.KAMERALAR:
            d = tg._tespit_yukle("mediapipe", t.video, cam)
            if d is None or "dunya_tam" not in d:
                continue
            isk = [tg._dunya_tam_iskeleti(d, d["indeks"][t.ilk + j])
                   for j in range(len(kareler)) if t.ilk + j in d["indeks"]]
            if len(isk) < gerekli:
                continue
            o_tel = dt.tekrar_olculeri(isk, MP_YUKARI, gerekli)
            satir.append({"kisi": t.kisi, "dogru": t.dogru,
                          "yon": t.yon if cam == "c17" else tg.C18_YONU[t.yon],
                          **{f"ref_{m}": o_ref[m] for m in OLCULER},
                          **{f"tel_{m}": o_tel[m] for m in OLCULER}})
    out = {}
    for yon in ("front", "half-profile", "profile"):
        s = [x for x in satir if x["yon"] == yon]
        out[yon] = {}
        for m in OLCULER:
            r = np.array([x[f"ref_{m}"] for x in s], float)
            p = np.array([x[f"tel_{m}"] for x in s], float)
            ok = np.isfinite(r) & np.isfinite(p)
            a_tel = dt.auc([x[f"tel_{m}"] for x in s if not x["dogru"]],
                           [x[f"tel_{m}"] for x in s if x["dogru"]])
            a_ref = dt.auc([x[f"ref_{m}"] for x in s if not x["dogru"]],
                           [x[f"ref_{m}"] for x in s if x["dogru"]])
            out[yon][m] = {"n": int(ok.sum()),
                           "medyan_mutlak_fark": round(float(np.median(np.abs(p[ok] - r[ok]))), 3)
                           if ok.any() else None,
                           "spearman": (round(v, 3) if (v := spearman(p, r)) is not None else None),
                           "auc_fizyo_telefon": round(a_tel, 3) if a_tel is not None else None,
                           "auc_fizyo_mocap": round(a_ref, 3) if a_ref is not None else None}
    return out


def fitness_aqa() -> dict:
    etiket = json.loads((SQUAT / "Labels/error_knees_forward.json").read_text())
    test = set(json.loads((SQUAT / "Splits/test_keys.json").read_text()))
    satirlar = json.loads((KOK / "out/fitness_aqa_valgus_satirlar.json").read_text())
    aci = {s["anahtar"]: s["govde_acisi_derece"] for s in satirlar}
    gerekli = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))
    satir = []
    for k, ar in sorted(etiket.items()):
        yol = KOK / f"out/fitness_aqa_tespit/mediapipe/{k}.npz"
        if k in test or not yol.exists():
            continue
        d = dict(np.load(yol))
        isk = [tg._dunya_tam_iskeleti(d, i) for i in range(len(d["tespit"]))]
        isk = [s for s in isk if s.gorunur.all()]
        if len(isk) < gerekli:
            continue
        o = dt.tekrar_olculeri(isk, MP_YUKARI, gerekli)
        satir.append({"kusurlu": bool(ar), "aci": aci.get(k), **o})
    out = {"n": len(satir), "n_kusurlu": sum(s["kusurlu"] for s in satir), "dilim": {}}
    for ad, lo, hi in (("tum", 0, 91), ("0-30", 0, 30), ("30-60", 30, 60), ("60-90", 60, 91)):
        s = [x for x in satir if x["aci"] is not None and lo <= x["aci"] < hi]
        out["dilim"][ad] = {"n": len(s), **{m: (round(v, 3) if (v := dt.auc(
            [x[m] for x in s if x["kusurlu"]], [x[m] for x in s if not x["kusurlu"]]))
            is not None else None) for m in OLCULER}}
    return out


def main() -> None:
    sonuc = {"deney": "sagital", "rehab24": rehab24(), "fitness_aqa_dizler_one": fitness_aqa()}
    (KOK / "out/sagital.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    for yon, v in sonuc["rehab24"].items():
        print("REHAB24", yon)
        for m, o in v.items():
            print(f"   {m:12s}", o)
    f = sonuc["fitness_aqa_dizler_one"]
    print(f"Fitness-AQA dizler one (egitim+dogrulama) n={f['n']} kusurlu={f['n_kusurlu']}")
    for ad, o in f["dilim"].items():
        print(f"   {ad:6s}", o)


if __name__ == "__main__":
    main()
