"""REHAB24-6 lunge (Ex5): fizyoterapistin "yanlis" kararini hangi olcu acikliyor?

Telefon yok: mocap iskeleti (3B yer gercegi) ve fizyoterapist etiketi. EC3D
lunge'da (`2026-09-28-ec3d-lunge.md`) diz onde ve derinlik etiketli hatalari
ayirdi; burada ayni olculer bagimsiz veride, uzmanin kendi kararina karsi.
Fizyoterapist her kisiye farkli hata yaptirdi ve turunu etiketlemedi: AUC
toplamda ve kisi basina verilir.

Olculer (ondeki bacak; tekrar skoru 0,2 s kesintisiz tutulan en kotu deger):
- diz_onde: dizin ayak parmaginin (ToeBase) onunde kalan mesafesi, ayak
  yonunde (ayak bilegi -> parmak, yatay), kaval boyuna oranla
- sig: -(ondeki dizin en derin kesintisiz fleksiyonu)
- derin: ondeki dizin en derin kesintisiz fleksiyonu (asiri inme de hata olabilir)
- govde_egimi: kalca -> boyun ile dikey arasindaki en buyuk aci
- valgus: ondeki dizin kalca-ayak bilegi cizgisinden ice sapmasi (uyluk boyuna oranla)

REHAB24 26 eklem: sag (UpLeg 21, Leg 22, Foot 23, ToeBase 24), sol (16-19),
Hips 0, Neck 3. Dikey eksen kayit basina, ilk karelerden (ayak -> boyun).

    python scripts/deney_rehab24_lunge_tanim.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from veri.rehab24 import tekrar_kareleri, tekrarlar_oku  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("el", KOK / "scripts/deney_ec3d_lunge.py")
el = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(el)

VERI = KOK / "data/dis/rehab24_6"
LUNGE = 5
BACAK = {"sag": (21, 22, 23, 24), "sol": (16, 17, 18, 19)}   # kalca, diz, ayak bilegi, parmak
OLCULER = ("diz_onde", "sig", "derin", "govde_egimi", "valgus")


def _birim(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def kare_olculeri(X: np.ndarray, yukari: np.ndarray) -> dict:
    """X: (26, 3) mocap. Ondeki bacak: ayak bilegi ileri yonde daha onde olan."""
    def yatay(v):
        return _birim(v - (v @ yukari) * yukari)

    ayak = {ad: yatay(X[t] - X[a]) for ad, (h, k, a, t) in BACAK.items()}
    if any(v is None for v in ayak.values()):
        return {m: np.nan for m in OLCULER}
    ileri = yatay(ayak["sag"] + ayak["sol"])
    if ileri is None:
        return {m: np.nan for m in OLCULER}
    on = max(BACAK, key=lambda ad: X[BACAK[ad][2]] @ ileri)
    diger = "sol" if on == "sag" else "sag"
    h, k, a, t = BACAK[on]
    kaval = np.linalg.norm(X[k] - X[a])
    fl = el._fleksiyon(X[h], X[k], X[a])
    govde = _birim(X[3] - X[0])
    # valgus: dizin kalca-ayak bilegi cizgisine dik uzakligi, diger kalcaya dogru pozitif
    u = X[a] - X[h]
    dik = (X[k] - X[h]) - ((X[k] - X[h]) @ u) / (u @ u) * u
    ice = X[BACAK[diger][0]] - X[h]
    ice = _birim(ice - (ice @ u) / (u @ u) * u)
    uyluk = np.linalg.norm(X[k] - X[h])
    return {"diz_onde": float((X[k] - X[t]) @ ayak[on] / kaval),
            "sig": -fl, "derin": fl,
            "govde_egimi": float(np.degrees(np.arccos(np.clip(govde @ yukari, -1, 1)))),
            "valgus": float(dik @ ice / uyluk) if ice is not None else np.nan}


def dikey(X0: np.ndarray) -> np.ndarray:
    ayak = (X0[:, 18] + X0[:, 23]) / 2
    return _birim(np.nanmean(X0[:, 3] - ayak, axis=0))


def tekrar_skoru(kareler: list[dict], olcu: str) -> float:
    v = np.array([k[olcu] for k in kareler], float)
    if olcu == "sig":                       # en derin kesintisiz bukulmenin eksisi
        return -el.surekli_en_buyuk(-v)
    return el.surekli_en_buyuk(v)


def main() -> None:
    tekrarlar = [t for t in tekrarlar_oku(VERI / "Segmentation.csv", egzersiz=LUNGE)
                 if not t.mocap_hatali]
    yukari: dict = {}
    satirlar, atlanan = [], []
    for t in tekrarlar:
        try:
            X = np.asarray(tekrar_kareleri(VERI, t, egzersiz=LUNGE), float)[:, :, :3]
        except ValueError as e:          # segment dizinin disina tasiyor (veri seti)
            atlanan.append(str(e))
            continue
        if t.video not in yukari:
            yukari[t.video] = dikey(X[:10])
        kk = [kare_olculeri(x, yukari[t.video]) for x in X]
        satirlar.append({"kisi": t.kisi, "dogru": t.dogru,
                         **{m: tekrar_skoru(kk, m) for m in OLCULER}})

    def ayir(ss):
        return {m: (round(a, 3) if (a := el.auc([s[m] for s in ss if not s["dogru"]],
                                                [s[m] for s in ss if s["dogru"]])) is not None
                    else None) for m in OLCULER}

    kisiler = sorted({s["kisi"] for s in satirlar}, key=int)
    kisi_basina = {}
    for k in kisiler:
        ss = [s for s in satirlar if s["kisi"] == k]
        a = {m: v for m, v in ayir(ss).items() if v is not None}
        kisi_basina[k] = {"n_dogru": sum(s["dogru"] for s in ss),
                          "n_yanlis": sum(not s["dogru"] for s in ss), "auc": a,
                          "en_iyi_olcu": max(a, key=lambda m: abs(a[m] - 0.5)) if a else None}
    sonuc = {"deney": "rehab24_lunge_tanim", "n_tekrar": len(satirlar),
             "n_yanlis": sum(not s["dogru"] for s in satirlar), "n_kisi": len(kisiler),
             "atlanan_tekrar": atlanan,
             "auc_toplam": ayir(satirlar), "kisi_basina": kisi_basina,
             "not": "AUC: 'yanlis' tekrarda olcunun buyuk olma olasiligi; <0,5 ters yon."}
    (KOK / "out/rehab24_lunge_tanim.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tekrar {sonuc['n_tekrar']}, yanlis {sonuc['n_yanlis']}, kisi {sonuc['n_kisi']}, "
          f"atlanan {len(atlanan)}: {atlanan}")
    print("AUC toplam:", sonuc["auc_toplam"])
    for k, v in kisi_basina.items():
        print(f"kisi {k:>2}: dogru {v['n_dogru']:2d} yanlis {v['n_yanlis']:2d} | en iyi "
              f"{v['en_iyi_olcu']} |", v["auc"])


if __name__ == "__main__":
    main()
