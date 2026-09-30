"""EC3D squat: gercek dedektorun (OpenPose) 2B'sinden "dizler ice" ayrilabiliyor mu?

Fitness-AQA'da (gercek salon videosu) MediaPipe noktalarindan hesaplanan her
valgus olcusu AUC 0,52-0,60'ta kaldi. Iki aciklama var: olcu yanlis, ya da o
videolardaki anahtar noktalar yetersiz. EC3D bunu ayirir: laboratuvar
goruntusu (4 GoPro), gercek bir dedektorun (OpenPose BODY_25) 2B ciktisi, 3B
yer gercegi ve "dizler ice" etiketi ayni yerde. Goruntu dosyalari pakette yok;
yalnizca OpenPose noktalari var -- ama olculen de tam olarak odur.

- Kiyas: etiket 3 (dizler ice) vs 1 (dogru), tekrar basina.
- 2B olculer (kare basina, sonra 0,2 s kesintisiz tutulan en kotu deger):
  FPPA = kalca-diz-ayak bilegi acisinin 180'den sapmasi (buyuk, iki taraf);
  KASR = diz arasi / ayak bilegi arasi (kucuk = ice; skor -KASR).
- Referans: ayni tekrarda 3B iskelette 0014 valgus (`deney_ec3d` ile ayni).
- Bakis acisi 2B'den: kalca genisligi / govde boyu (onden buyuk, yandan kucuk).
- Ayak ucuna gore diz sapmasi (medial diz sapmasi, Cotic ve ark. 2020; 0014):
  3B'de dizin kalca-ayak bilegi-ayak basparmagi duzlemine isaretli uzakligi,
  2B'de kalca-ayak basparmagi cizgisine isaretli dik uzakligi; ice (diger
  bacaga dogru) pozitif, uyluk / bacak boyuna oranla.

    python scripts/deney_ec3d_2b.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.form import form_degerlendir  # noqa: E402
from veri.ec3d import kareyi_cevir, tekrarlar  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
KAMERALAR = ("6_1", "6_2", "6_3", "6_4")
GEREKLI = 6            # 0,2 s, 30 fps
# BODY_25: 1 boyun, 8 orta kalca, 9/12 kalca, 10/13 diz, 11/14 ayak bilegi
KALCA, DIZ, AYAK = (9, 12), (10, 13), (11, 14)
# BODY_25 ayak basparmagi: 22 sag, 19 sol (sag bacak once, KALCA ile ayni sira)
AYAK_UCU = (22, 19)


def surekli_en_buyuk(v: np.ndarray, g: int = GEREKLI) -> float:
    """`g` kare kesintisiz tutulan en buyuk deger; NaN seriyi keser."""
    en = -np.inf
    for i in range(len(v) - g + 1):
        w = v[i:i + g]
        if np.isfinite(w).all():
            en = max(en, float(w.min()))
    return en


def fppa(P: dict) -> float:
    """Iki bacagin FPPA'sinin buyugu (derece); eksik bacak atlanir."""
    en = np.nan
    for h, k, a in zip(KALCA, DIZ, AYAK):
        if not all(j in P for j in (h, k, a)):
            continue
        u, v = P[h] - P[k], P[a] - P[k]
        c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        d = 180 - np.degrees(np.arccos(np.clip(c, -1, 1)))
        en = d if not np.isfinite(en) else max(en, d)
    return en


def diz_sapmasi_2b(P: dict) -> float:
    """Iki bacagin buyugu: dizin kalca-ayak basparmagi cizgisine dik uzakligi,
    ice (diger kalcaya dogru) pozitif, kalca-ayak ucu boyuna oranla."""
    en = np.nan
    for i, (h, k, t) in enumerate(zip(KALCA, DIZ, AYAK_UCU)):
        diger = KALCA[1 - i]
        if not all(j in P for j in (h, k, t, diger)):
            continue
        u = P[t] - P[h]
        boy = np.linalg.norm(u)
        if boy < 1:
            continue
        n = np.array([-u[1], u[0]]) / boy                  # cizgiye dik birim
        if n @ (P[diger] - P[h]) < 0:                      # ice bakacak sekilde
            n = -n
        d = float(n @ (P[k] - P[h]) / boy)
        en = d if not np.isfinite(en) else max(en, d)
    return en


def diz_sapmasi_3b(poz: np.ndarray) -> float:
    """(3, 25) BODY_25 3B: dizin kalca-ayak bilegi-ayak basparmagi duzlemine
    isaretli uzakligi (ice pozitif), uyluk boyuna oranla; iki bacagin buyugu."""
    X = np.asarray(poz, float).T
    en = np.nan
    for i, (h, k, a, t) in enumerate(zip(KALCA, DIZ, AYAK, AYAK_UCU)):
        diger = KALCA[1 - i]
        n = np.cross(X[a] - X[h], X[t] - X[h])
        uyluk = np.linalg.norm(X[k] - X[h])
        if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-9 or uyluk < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if n @ (X[diger] - X[h]) < 0:
            n = -n
        d = float(n @ (X[k] - X[h]) / uyluk)
        en = d if not np.isfinite(en) else max(en, d)
    return en


def kasr(P: dict) -> float:
    if not all(j in P for j in DIZ + AYAK):
        return np.nan
    ayak = np.linalg.norm(P[AYAK[0]] - P[AYAK[1]])
    return float(np.linalg.norm(P[DIZ[0]] - P[DIZ[1]]) / ayak) if ayak > 1 else np.nan


def onden_orani(P: dict) -> float:
    """Kalca genisligi / govde boyu: onden ~0,5+, yandan ~0."""
    if not all(j in P for j in (1, 8) + KALCA):
        return np.nan
    boy = np.linalg.norm(P[1] - P[8])
    return float(np.linalg.norm(P[KALCA[0]] - P[KALCA[1]]) / boy) if boy > 1 else np.nan


def auc(pozitif, negatif) -> float | None:
    p, n = np.asarray(pozitif, float), np.asarray(negatif, float)
    if not p.size or not n.size:
        return None
    return float(((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])).mean())


def main() -> None:
    reps = tekrarlar(KOK / "data/dis/ec3d/data_3D.pickle", "SQUAT")
    with open(KOK / "data/dis/ec3d/data_3D.pickle", "rb") as f:
        etiket = pickle.load(f)["labels"]
    with open(KOK / "data/dis/ec3d/data.pickle", "rb") as f:
        kareler2b = pickle.load(f)["frames"]["SQUAT"]
    # (kisi, etiket, tekrar) -> kare numaralari (data_3D sirasiyla)
    kare_no: dict = {}
    for eg, kisi, lab, tek, kare in etiket:
        if eg == "SQUAT":
            kare_no.setdefault((str(kisi), int(lab), int(tek)), []).append(int(kare))

    satirlar = []
    for (_, kisi, lab, tek), poz in reps.items():
        if lab not in (1, 3):
            continue
        nolar = sorted(kare_no[(kisi, lab, tek)])
        deg3 = np.array([[form_degerlendir(kareyi_cevir(p)).olcumler[a].deger
                          for a in ("diz_valgusu_sag", "diz_valgusu_sol")] for p in poz])
        s = {"kisi": kisi, "etiket": lab, "tekrar": tek, "n_kare": len(nolar),
             "valgus_3b": max(surekli_en_buyuk(deg3[:, 0]), surekli_en_buyuk(deg3[:, 1])),
             "sapma_3b": surekli_en_buyuk(np.array([diz_sapmasi_3b(p) for p in poz]))}
        grup = kareler2b.get(kisi, {}).get(lab, {})
        for cam in KAMERALAR:
            fp, ks, on, sp = [], [], [], []
            for n in nolar:
                op = grup.get(f"frame_{n:06d}", {}).get("2D_op", {}).get(cam, {})
                P = {int(j): np.asarray(v, float) for j, v in op.items()}
                fp.append(fppa(P))
                sp.append(diz_sapmasi_2b(P))
                ks.append(kasr(P))
                on.append(onden_orani(P))
            fp, ks, on, sp = map(lambda x: np.array(x, float), (fp, ks, on, sp))
            s[f"sapma_{cam}"] = surekli_en_buyuk(sp)
            s[f"fppa_{cam}"] = surekli_en_buyuk(fp)
            s[f"kasr_{cam}"] = surekli_en_buyuk(-ks)
            s[f"onden_{cam}"] = float(np.nanmedian(on)) if np.isfinite(on).any() else np.nan
            s[f"kapsama_{cam}"] = float(np.isfinite(fp).mean())
        satirlar.append(s)

    poz = [s for s in satirlar if s["etiket"] == 3]
    neg = [s for s in satirlar if s["etiket"] == 1]

    def a(anahtar):
        v = auc([s[anahtar] for s in poz], [s[anahtar] for s in neg])
        return round(v, 3) if v is not None else None

    kameralar = {}
    for cam in KAMERALAR:
        kameralar[cam] = {
            "onden_orani_medyan": round(float(np.nanmedian([s[f"onden_{cam}"] for s in satirlar])), 3),
            "kapsama": round(float(np.mean([s[f"kapsama_{cam}"] for s in satirlar])), 3),
            "auc_fppa": a(f"fppa_{cam}"), "auc_kasr": a(f"kasr_{cam}"),
            "auc_diz_sapmasi": a(f"sapma_{cam}")}
    sonuc = {"deney": "ec3d_2b_valgus", "n_dizler_ice": len(poz), "n_dogru": len(neg),
             "n_kisi": len({s["kisi"] for s in satirlar}),
             "auc_valgus_3b": a("valgus_3b"), "auc_diz_sapmasi_3b": a("sapma_3b"),
             "kameralar": kameralar,
             "not": ("OpenPose BODY_25 2B (yazarlarin ciktisi). Kamera 6_1 dunya cercevesi; "
                     "bakis acisi 2B orandan. 4 kisi: on kanit; tanim (0014) bu veride secildi.")}
    (KOK / "out/ec3d_2b_valgus.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dizler ice {len(poz)}, dogru {len(neg)}; 3B valgus AUC {sonuc['auc_valgus_3b']}, "
          f"3B diz sapmasi AUC {sonuc['auc_diz_sapmasi_3b']}")
    for cam, v in kameralar.items():
        print(cam, v)


if __name__ == "__main__":
    main()
