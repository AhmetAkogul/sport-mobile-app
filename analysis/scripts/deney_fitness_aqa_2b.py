"""Fitness-AQA dogrulama bolumu: dedektor gucu "dizler ice" ayrimini iyilestiriyor mu?

EC3D'de gercek dedektorun (OpenPose) 2B'sinden KASR "dizler ice"yi ayirdi
(AUC 0,94-0,95); Fitness-AQA'da MediaPipe 2B'sinden ayni olcu 0,59 kaldi.
Hipotez: sorun anahtar nokta kalitesi. Sinama: ayni videolarda daha guclu
dedektor (RTMPose-m, RTMW-x) KASR'in AUC'sini yukseltiyor mu?

Yalniz resmi **dogrulama** bolumu (243 video); resmi test bolumu son olcume
saklanir. Olculer (tekrar basina, 0,2 s kesintisiz tutulan en kotu deger):

- kasr: -(diz arasi / ayak bilegi arasi); buyuk = dizler ice
- kasr_goreli: -(KASR / tekrarin ilk 10 karesinin KASR medyani); kisinin
  durus genisligini sadelestirir
- valgus_3b: yalniz MediaPipe (dunya, maskesiz), 0014

    python scripts/deney_fitness_aqa_2b.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
SQUAT = KOK / "data/dis/fitness_aqa/Squat/Labeled_Dataset"
_spec = importlib.util.spec_from_file_location("fa", KOK / "scripts/deney_fitness_aqa_valgus.py")
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)
tg = fa.tg
I = tg.REFERANS_ISKELET.indeks  # noqa: E741
GEREKLI = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))
BACKENDLER = ("mediapipe", "rtmpose", "rtmw")


def surekli(v: np.ndarray) -> float:
    en = -np.inf
    for i in range(len(v) - GEREKLI + 1):
        w = v[i:i + GEREKLI]
        if np.isfinite(w).all():
            en = max(en, float(w.min()))
    return en


def kasr_dizisi(noktalar: np.ndarray, gorunur: np.ndarray) -> np.ndarray:
    """(T, 13, 2) -> (T,) diz arasi / ayak bilegi arasi; eksik ya da dar -> NaN."""
    j = [I("sag_diz"), I("sol_diz"), I("sag_ayak_bilegi"), I("sol_ayak_bilegi")]
    ok = gorunur[:, j].all(axis=1)
    diz = np.linalg.norm(noktalar[:, j[0]] - noktalar[:, j[1]], axis=1)
    ayak = np.linalg.norm(noktalar[:, j[2]] - noktalar[:, j[3]], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (ayak > 1), diz / ayak, np.nan)


def skorlar(d: dict, backend: str) -> dict:
    k = kasr_dizisi(np.asarray(d["noktalar"], float), np.asarray(d["gorunur"], bool))
    bas = k[:10][np.isfinite(k[:10])]
    s = {"kasr": surekli(-k),
         "kasr_goreli": surekli(-k / np.median(bas)) if bas.size else -np.inf}
    if backend == "mediapipe" and "dunya_tam" in d:
        deg = np.array([tg._valgus_kararlari(tg._dunya_tam_iskeleti(d, i))[1]
                        for i in range(len(d["tespit"]))], float)
        s["valgus_3b"] = max(surekli(deg[:, 0]), surekli(deg[:, 1]))
    s["tespit_orani"] = float(np.mean(d["tespit"]))
    s["kasr_kapsama"] = float(np.isfinite(k).mean())
    return s


def main() -> None:
    etiket = json.loads((SQUAT / "Labels/error_knees_inward.json").read_text())
    val = sorted(set(json.loads((SQUAT / "Splits/val_keys.json").read_text())) & set(etiket))
    sonuc = {"deney": "fitness_aqa_2b_dogrulama", "bolum": "val", "n_video": len(val),
             "n_kusurlu": sum(bool(etiket[k]) for k in val), "backendler": {}}
    for b in BACKENDLER:
        klasor = KOK / "out/fitness_aqa_tespit" / b
        mevcut = [k for k in val if (klasor / f"{k}.npz").exists()]
        if len(mevcut) < len(val):
            print(f"{b}: {len(mevcut)}/{len(val)} video; eksik, atlandi")
            continue
        satir = []
        for k in val:
            d = dict(np.load(klasor / f"{k}.npz"))
            satir.append({"kusurlu": bool(etiket[k]), **skorlar(d, b)})
        poz = [s for s in satir if s["kusurlu"]]
        neg = [s for s in satir if not s["kusurlu"]]
        olcu = {}
        for m in ("kasr", "kasr_goreli", "valgus_3b"):
            if m in satir[0]:
                a = fa.auc([s[m] for s in poz], [s[m] for s in neg])
                olcu[m] = round(a, 3) if a is not None else None
        sonuc["backendler"][b] = {
            "auc": olcu,
            "tespit_orani": round(float(np.mean([s["tespit_orani"] for s in satir])), 3),
            "kasr_kapsama": round(float(np.mean([s["kasr_kapsama"] for s in satir])), 3)}
        print(b, sonuc["backendler"][b])
    (KOK / "out/fitness_aqa_2b.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
