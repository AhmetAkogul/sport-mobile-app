"""0037: hareket tanima -- REHAB24-6'da kisi-disarida, EC3D'de dis sinama.

Girdi: `scripts/hareket_tespit.py` ciktisi (MediaPipe tam vucut, 10 kare/s).
Etiket: tekrar icindeki kare o hareketin, tekrar disi "yok" (0). 2 s pencere,
0,5 s kaydirma; pencerenin >= %60'i tek harekette ise o hareket, <= %10'u
tekrar icindeyse "yok", arasi (gecis) atlanir.

Dogrulama: her seferinde bir kisi disarida (egitimde o kisinin hicbir videosu
yok), iki kamera birlikte. Tekrar duzeyi: merkezi tekrarin icinde kalan
pencerelerin cogunluk oyu.

Dis sinama (EC3D): baska veri seti, baska dedektor (OpenPose BODY_25), dort
kamera; squat ve lunge REHAB24 siniflariyla ayni; plank ve pick-up egitimde
olmayan hareketler (dogru cevap "bilinen hareket degil").

    .venv/bin/python scripts/deney_hareket_tanima.py [--model-cikti out/hareket_modeli.joblib]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.hareket import (HAREKETLER, aynala, kare_ozellikleri, pencere_ozellikleri,  # noqa: E402
                          pencereler, siniflandirici)
from veri.ec3d import BODY25_TAM_VUCUT  # noqa: E402

TESPIT = Path("out/hareket_tespit/mediapipe")
SEGMENT = Path("data/dis/rehab24_6/Segmentation.csv")
EC3D = Path("data/dis/ec3d/data.pickle")
HIZ = 10.0                 # tespit 30 fps'ten 3 karede bir


def segmentler():
    """video -> (kisi, egzersiz, [(ilk, son, tekrar_no, yon, dogru)])."""
    out = {}
    with SEGMENT.open() as f:
        for r in csv.DictReader(f, delimiter=";"):
            v = out.setdefault(r["video_id"], [r["person_id"], int(r["exercise_id"]), []])
            v[2].append((int(r["first_frame"]), int(r["last_frame"]), int(r["repetition_number"]),
                         r["cam17_orientation"], int(r["correctness"])))
    return out


def rehab24_pencereleri():
    seg = segmentler()
    X, y, kisi, video, kamera, tekrar = [], [], [], [], [], []
    for dosya in sorted(TESPIT.glob("*.npz")):
        ad, kam = dosya.stem.split("-")
        if ad not in seg:
            continue
        kisi_id, ex, reps = seg[ad]
        d = np.load(dosya)
        kare, P, G = d["kare"], d["noktalar"], d["gorunur"]
        etiket = np.zeros(len(kare), int)
        rep_no = np.full(len(kare), -1)
        for ilk, son, no, _, _ in reps:
            ic = (kare >= ilk) & (kare <= son)
            etiket[ic] = ex
            rep_no[ic] = no
        for aynali in (False, True):
            if aynali:
                P2, G2 = aynala(P, G, float(d["boyut"][0]))
            else:
                P2, G2 = P, G
            for b, s in pencereler(len(kare), HIZ):
                pay = np.mean(etiket[b:s] == ex)
                if pay >= 0.6:
                    lab = ex
                elif pay <= 0.1:
                    lab = 0
                else:
                    continue
                X.append(pencere_ozellikleri(kare_ozellikleri(P2[b:s], G2[b:s])))
                y.append(lab)
                kisi.append(kisi_id)
                video.append(ad)
                kamera.append(kam + ("-ayna" if aynali else ""))
                tekrar.append(rep_no[(b + s) // 2])
    return (np.array(X), np.array(y), np.array(kisi), np.array(video), np.array(kamera),
            np.array(tekrar), seg)


def makro_f1(y, p, siniflar):
    f = []
    for c in siniflar:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def ec3d_sinama(model):
    """EC3D tekrarlarini 2 s pencerelerle siniflandir; pencere oylarinin dagilimi."""
    with EC3D.open("rb") as f:
        kareler = pickle.load(f)["frames"]
    beklenen = {"SQUAT": 6, "Lunges": 5, "Plank": None, "Pick-up": None}
    sonuc = {}
    for ex, grup in kareler.items():
        oylar = Counter()
        for etiketler in grup.values():
            for ks in etiketler.values():
                adlar = sorted(ks)
                kameralar = sorted({c for a in adlar for c in ks[a].get("2D_op", {})})
                for cam in kameralar:
                    P = np.full((len(adlar), 65, 2), np.nan)
                    for i, a in enumerate(adlar):
                        op = ks[a].get("2D_op", {}).get(cam, {})
                        for j_op, j_tv in enumerate(BODY25_TAM_VUCUT):
                            if j_tv >= 0 and j_op in op:
                                P[i, j_tv] = np.asarray(op[j_op], float)[:2]
                    P = P[::3]                                    # 30 -> 10 kare/s
                    G = np.isfinite(P).all(-1) & (P != 0).any(-1)
                    ws = pencereler(len(P), HIZ) or [(0, len(P))]
                    Xw = np.array([pencere_ozellikleri(kare_ozellikleri(P[b:s], G[b:s]))
                                   for b, s in ws])
                    oylar.update(model.predict(Xw).tolist())
        toplam = sum(oylar.values())
        dagilim = {HAREKETLER[int(k)]: round(v / toplam, 3) for k, v in oylar.most_common()}
        dogru = beklenen[ex]
        sonuc[ex] = {"pencere": toplam, "dagilim": dagilim,
                     "dogru_oran": (round(oylar[dogru] / toplam, 3) if dogru else None)}
    return sonuc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-cikti", type=Path, default=Path("out/hareket_modeli.joblib"))
    p.add_argument("--cikti", type=Path, default=Path("out/hareket_tanima.json"))
    a = p.parse_args()
    X, y, kisi, video, kamera, tekrar, seg = rehab24_pencereleri()
    siniflar = sorted(set(y.tolist()))
    print(f"{len(X)} pencere (aynali dahil), {len(set(kisi))} kisi, siniflar {Counter(y.tolist())}")

    tahmin = np.full(len(y), -1)
    for k in sorted(set(kisi)):
        test = kisi == k
        m = siniflandirici().fit(X[~test], y[~test])
        tahmin[test] = m.predict(X[test])
    orj = ~np.char.endswith(kamera.astype(str), "-ayna")           # degerlendirme aynasiz
    yo, po = y[orj], tahmin[orj]
    karisiklik = {HAREKETLER[c]: {HAREKETLER[d]: int(np.sum((yo == c) & (po == d)))
                                  for d in siniflar} for c in siniflar}
    # tekrar duzeyi: (video, kamera, tekrar) cogunluk oyu
    tekrar_oy = defaultdict(list)
    for i in np.flatnonzero(orj & (tekrar >= 0) & (y > 0)):
        tekrar_oy[(video[i], kamera[i], int(tekrar[i]))].append((int(y[i]), int(tahmin[i])))
    yon = {(v, r[2]): r[3] for v, (_, _, reps) in seg.items() for r in reps}
    tk = defaultdict(lambda: [0, 0])
    for (v, kam, no), oy in tekrar_oy.items():
        gercek = oy[0][0]
        kazanan = Counter(t for _, t in oy).most_common(1)[0][0]
        # c18'in yonu c17'ye gore bilinmiyor; yon kirilimi yalniz c17 icin
        anahtar = f"{kam}:{yon.get((v, no), '?')}" if kam == "c17" else kam
        for a_ in ("hepsi", anahtar):
            tk[a_][0] += int(kazanan == gercek)
            tk[a_][1] += 1
    sonuc = {
        "pencere": {"n": int(orj.sum()), "dogruluk": round(float(np.mean(yo == po)), 3),
                    "makro_f1": round(makro_f1(yo, po, siniflar), 3),
                    "sinif_duyarliligi": {HAREKETLER[c]: round(float(np.mean(po[yo == c] == c)), 3)
                                          for c in siniflar},
                    "karisiklik": karisiklik},
        "tekrar": {a_: {"dogru": v[0], "toplam": v[1], "oran": round(v[0] / v[1], 3)}
                   for a_, v in sorted(tk.items())},
    }
    model = siniflandirici().fit(X, y)
    sonuc["ec3d"] = ec3d_sinama(model)
    import joblib
    a.model_cikti.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, a.model_cikti)
    a.cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in sonuc["pencere"].items() if k != "karisiklik"},
                     ensure_ascii=False))
    print("tekrar:", {k: v["oran"] for k, v in sonuc["tekrar"].items()})
    print("ec3d:", json.dumps(sonuc["ec3d"], ensure_ascii=False))


if __name__ == "__main__":
    main()
