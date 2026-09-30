"""REHAB24-6 lunge (Ex5): telefon "diz ayak ucunu geciyor"u olcebiliyor mu?

Mocap'te diz onde fizyoterapist kararini kisi ici 0,88-1,00 AUC ile ayirdi
(`2026-09-28-ec3d-lunge.md`). Burada ayni olcu telefondan (MediaPipe, gercek
video) ve ayni fonksiyonla (`deney_rehab24_lunge_tanim.kare_olculeri`)
hesaplanir: telefon iskeleti mocap'in 26 eklem duzenine yerlestirilir.

Kollar:
- telefon_3b: MediaPipe dunya (maskesiz govde + ayak), dikey = -Y (kamera yatay)
- telefon_2b: goruntu duzlemi (piksel), dikey = -y
- mocap: referans

Yon basina: telefon-mocap Spearman, fizyoterapist AUC (telefon ve mocap).

    python scripts/deney_rehab24_lunge_telefon.py
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


rl = _yukle("rl", "deney_rehab24_lunge_tanim.py")
tg = _yukle("tg", "deney_rehab24_tek_gorus.py")
sg = _yukle("sg", "deney_sagital.py")
I = tg.REFERANS_ISKELET.indeks  # noqa: E741
OLCULER = ("diz_onde", "derin")
# REFERANS eklemi -> REHAB24 26 eklem indeksi; ayak (sag topuk, sag uc, sol topuk, sol uc)
YERLESIM = {"sag_kalca": 21, "sag_diz": 22, "sag_ayak_bilegi": 23,
            "sol_kalca": 16, "sol_diz": 17, "sol_ayak_bilegi": 18, "boyun": 3}
AYAK_UCU = {1: 24, 3: 19}          # ayak dizisindeki sira -> REHAB24 ToeBase


def telefon_iskeleti(govde: np.ndarray, ayak: np.ndarray) -> np.ndarray:
    """(13, d) REFERANS + (4, d) ayak -> (26, d) REHAB24 duzeni; bos eklem NaN."""
    d = govde.shape[1]
    X = np.full((26, d), np.nan)
    for ad, j in YERLESIM.items():
        X[j] = govde[I(ad)]
    X[0] = (govde[I("sag_kalca")] + govde[I("sol_kalca")]) / 2
    for i, j in AYAK_UCU.items():
        X[j] = ayak[i]
    return X


def main() -> None:
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv", egzersiz=rl.LUNGE)
                 if not t.mocap_hatali]
    yukari_mocap: dict = {}
    satirlar = []
    for t in tekrarlar:
        try:
            X = np.asarray(tg.tekrar_kareleri(tg.VERI, t, egzersiz=rl.LUNGE), float)[:, :, :3]
        except ValueError:
            continue
        if t.video not in yukari_mocap:
            yukari_mocap[t.video] = rl.dikey(X[:10])
        ref = [rl.kare_olculeri(x, yukari_mocap[t.video]) for x in X]
        for cam in tg.KAMERALAR:
            yol = KOK / f"out/rehab24_tespit/mediapipe/{t.video}-{cam}.npz"
            if not yol.exists():
                continue
            ek = dict(np.load(yol))
            if "ayak_dunya" not in ek:
                continue
            indeks = {int(k): i for i, k in enumerate(ek["kare"])}
            k3, k2 = [], []
            for j in range(len(X)):
                i = indeks.get(t.ilk + j)
                if i is None:
                    k3.append({m: np.nan for m in rl.OLCULER})
                    k2.append({m: np.nan for m in rl.OLCULER})
                    continue
                k3.append(rl.kare_olculeri(telefon_iskeleti(
                    np.asarray(ek["dunya_tam"][i], float), ek["ayak_dunya"][i]),
                    np.array([0.0, -1.0, 0.0])))
                k2.append(rl.kare_olculeri(telefon_iskeleti(
                    np.asarray(ek["noktalar"][i], float), ek["ayak_px"][i]),
                    np.array([0.0, -1.0])))
            s = {"kisi": t.kisi, "dogru": t.dogru,
                 "yon": t.yon if cam == "c17" else tg.C18_YONU[t.yon]}
            for m in OLCULER:
                s[f"mocap_{m}"] = rl.tekrar_skoru(ref, m)
                s[f"tel3b_{m}"] = rl.tekrar_skoru(k3, m)
                s[f"tel2b_{m}"] = rl.tekrar_skoru(k2, m)
            satirlar.append(s)

    sonuc = {"deney": "rehab24_lunge_telefon", "n": len(satirlar), "yonler": {}}
    for yon in ("front", "half-profile", "profile"):
        s = [x for x in satirlar if x["yon"] == yon]
        v = {"n": len(s), "n_yanlis": sum(not x["dogru"] for x in s)}
        for m in OLCULER:
            for kol in ("mocap", "tel3b", "tel2b"):
                a = rl.el.auc([x[f"{kol}_{m}"] for x in s if not x["dogru"]],
                              [x[f"{kol}_{m}"] for x in s if x["dogru"]])
                v[f"auc_{kol}_{m}"] = round(a, 3) if a is not None else None
            for kol in ("tel3b", "tel2b"):
                r = sg.spearman([x[f"{kol}_{m}"] for x in s], [x[f"mocap_{m}"] for x in s])
                v[f"spearman_{kol}_{m}"] = round(r, 3) if r is not None else None
        sonuc["yonler"][yon] = v
    (KOK / "out/rehab24_lunge_telefon.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    for yon, v in sonuc["yonler"].items():
        print(f"== {yon} n={v['n']} yanlis={v['n_yanlis']}")
        for m in OLCULER:
            print(f"   {m:9s} AUC mocap {v[f'auc_mocap_{m}']}  tel3b {v[f'auc_tel3b_{m}']}  "
                  f"tel2b {v[f'auc_tel2b_{m}']} | Spearman 3b {v[f'spearman_tel3b_{m}']}  "
                  f"2b {v[f'spearman_tel2b_{m}']}")


if __name__ == "__main__":
    main()
