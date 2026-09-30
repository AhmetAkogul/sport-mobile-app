"""EC3D lunge: hangi olcu uzman etiketini ayiriyor? (once tanim, sonra telefon)

Squat'ta ogrenilen sira (`docs/deney/2026-09-28-tanim-fizyoterapist.md`): telefona
gecmeden once olcunun kendisi 3B yer gercegiyle sinanir.

Etiketler yazarlarin kodundan (`utils.py`, Zhao ve ark., ACCV 2022):
1 dogru, 4 yeterince inmiyor, 6 diz ayak ucunu geciyor. 4 kisi.

Olculer (kare basina; tekrar skoru 0,2 s kesintisiz tutulan en kotu deger):
- diz_onde: ondeki bacakta dizin ayak basparmaginin onunde kalan mesafesi, ayak
  yonunde (topuk -> basparmak, yatay) ve kaval boyuna oranla. Kusur: buyuk.
- derinlik: ondeki dizin 0,2 s kesintisiz tutulan en buyuk fleksiyonu (derece).
  Kusur: kucuk; skor = -derinlik.
Ondeki bacak: ayak bilegi govdenin ileri yonunde daha onde olan.

3B: BODY_25, z yukari (veri/ec3d.py). 2B: OpenPose BODY_25, kamera basina;
dikey = goruntu -y, "ileri" = ayak yonunun goruntudeki yatay bileseni.

    python scripts/deney_ec3d_lunge.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from veri.ec3d import tekrarlar  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
KAMERALAR = ("6_1", "6_2", "6_3", "6_4")
GEREKLI = 6
# BODY_25: sag (kalca, diz, ayak bilegi, basparmak, topuk) = 9,10,11,22,24; sol = 12,13,14,19,21
BACAK = {"sag": (9, 10, 11, 22, 24), "sol": (12, 13, 14, 19, 21)}
ETIKET = {1: "dogru", 4: "yeterince_inmiyor", 6: "diz_ayak_ucunu_geciyor"}


def surekli_en_buyuk(v: np.ndarray, g: int = GEREKLI) -> float:
    en = -np.inf
    for i in range(len(v) - g + 1):
        w = v[i:i + g]
        if np.isfinite(w).all():
            en = max(en, float(w.min()))
    return en


def _fleksiyon(h, k, a) -> float:
    u, v = h - k, a - k
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(180 - np.degrees(np.arccos(np.clip(c, -1, 1))))


def kare_olculeri(X: np.ndarray, yukari: np.ndarray) -> tuple[float, float]:
    """X: (25, d) BODY_25 (d = 2 ya da 3). (diz_onde, derinlik) ondeki bacak icin."""
    def yatay(v):
        v = v - (v @ yukari) * yukari
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    ileriler = {}
    for ad, (h, k, a, t, tp) in BACAK.items():
        if not np.isfinite(X[[h, k, a, t, tp]]).all():
            return np.nan, np.nan
        ileriler[ad] = yatay(X[t] - X[tp])
    if any(v is None for v in ileriler.values()):
        return np.nan, np.nan
    ileri = yatay(ileriler["sag"] + ileriler["sol"])
    if ileri is None:
        return np.nan, np.nan
    on = max(BACAK, key=lambda ad: X[BACAK[ad][2]] @ ileri)
    h, k, a, t, tp = BACAK[on]
    kaval = np.linalg.norm(X[k] - X[a])
    diz_onde = float((X[k] - X[t]) @ ileriler[on] / kaval) if kaval > 1e-9 else np.nan
    return diz_onde, _fleksiyon(X[h], X[k], X[a])


def auc(pozitif, negatif) -> float | None:
    p = np.asarray([x for x in pozitif if not np.isnan(x)], float)
    n = np.asarray([x for x in negatif if not np.isnan(x)], float)
    if not p.size or not n.size:
        return None
    return float(((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])).mean())


def main() -> None:
    reps = tekrarlar(KOK / "data/dis/ec3d/data_3D.pickle", "Lunges")
    with open(KOK / "data/dis/ec3d/data_3D.pickle", "rb") as f:
        etiket = pickle.load(f)["labels"]
    with open(KOK / "data/dis/ec3d/data.pickle", "rb") as f:
        kareler2b = pickle.load(f)["frames"]["Lunges"]
    kare_no: dict = {}
    for eg, kisi, lab, tek, kare in etiket:
        if eg == "Lunges":
            kare_no.setdefault((str(kisi), int(lab), int(tek)), []).append(int(kare))

    satirlar = []
    for (_, kisi, lab, tek), poz in reps.items():
        X3 = np.asarray(poz, float).transpose(0, 2, 1)           # (T, 25, 3)
        o3 = np.array([kare_olculeri(x, np.array([0.0, 0.0, 1.0])) for x in X3])
        s = {"kisi": kisi, "etiket": lab, "tekrar": tek,
             "diz_onde_3b": surekli_en_buyuk(o3[:, 0]),
             # en derin kesintisiz bukulme; sig = onun eksi isaretlisi (kusur: sig)
             "sig_3b": -surekli_en_buyuk(o3[:, 1])}
        nolar = sorted(kare_no[(kisi, lab, tek)])
        grup = kareler2b.get(kisi, {}).get(lab, {})
        for cam in KAMERALAR:
            o2 = []
            for n in nolar:
                op = grup.get(f"frame_{n:06d}", {}).get("2D_op", {}).get(cam, {})
                X = np.full((25, 2), np.nan)
                for j, v in op.items():
                    X[int(j)] = v
                o2.append(kare_olculeri(X, np.array([0.0, -1.0])))
            o2 = np.array(o2, float)
            s[f"diz_onde_{cam}"] = surekli_en_buyuk(o2[:, 0])
            s[f"sig_{cam}"] = -surekli_en_buyuk(o2[:, 1])
            s[f"kapsama_{cam}"] = float(np.isfinite(o2[:, 0]).mean())
        satirlar.append(s)

    dogru = [s for s in satirlar if s["etiket"] == 1]
    sonuc = {"deney": "ec3d_lunge", "n": {ETIKET[k]: sum(s["etiket"] == k for s in satirlar)
                                           for k in ETIKET},
             "n_kisi": len({s["kisi"] for s in satirlar}), "auc": {}}
    for ad, lab, olcu in (("diz_ayak_ucunu_geciyor", 6, "diz_onde"),
                          ("yeterince_inmiyor", 4, "sig")):
        kusur = [s for s in satirlar if s["etiket"] == lab]
        v = {"3b": auc([s[f"{olcu}_3b"] for s in kusur], [s[f"{olcu}_3b"] for s in dogru])}
        for cam in KAMERALAR:
            v[cam] = auc([s[f"{olcu}_{cam}"] for s in kusur], [s[f"{olcu}_{cam}"] for s in dogru])
        # capraz: olcu diger hataya tepki vermemeli
        diger = 4 if lab == 6 else 6
        v["3b_diger_hataya"] = auc([s[f"{olcu}_3b"] for s in satirlar if s["etiket"] == diger],
                                   [s[f"{olcu}_3b"] for s in dogru])
        sonuc["auc"][ad] = {k: (round(x, 3) if x is not None else None) for k, x in v.items()}
    sonuc["kapsama"] = {cam: round(float(np.mean([s[f"kapsama_{cam}"] for s in satirlar])), 3)
                        for cam in KAMERALAR}
    (KOK / "out/ec3d_lunge.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("tekrar:", sonuc["n"], "kisi:", sonuc["n_kisi"])
    for ad, v in sonuc["auc"].items():
        print(ad, v)
    print("kapsama:", sonuc["kapsama"])


if __name__ == "__main__":
    main()
