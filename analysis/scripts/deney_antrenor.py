"""0040: antrenor bildirimi, uctan uca ve kisi-disarida (REHAB24-6, telefon).

Her kisi icin: hareket tanima modeli, form modelleri ve tekrar sayaci esikleri
**o kisi olmadan** egitilir; o kisinin videolari (iki kamera) canli akis gibi kare
kare `mono.antrenor.Antrenor`'a verilir. Olculen:

- kilit: videoda ilk kilitlenen hareket dogru mu
- tekrar bulma: gercek tekrarin, dogru harekette bildirilen bir tekrarla ortusmesi
- karar: bulunan tekrarda antrenorun karari (iyi / duzelt) fizyoterapistle ayni mi;
  kapsama = "olculemez" denmeyen tekrar orani

    .venv/bin/python scripts/deney_antrenor.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import deney_hareket_formu as hf  # noqa: E402
import deney_hareket_tanima as ht  # noqa: E402
from eval.hareket import HareketTanima, siniflandirici  # noqa: E402
from mono.antrenor import Antrenor  # noqa: E402

ADLAR = hf.ADLAR


def form_modelleri(haric: str, tel_hepsi, sayim_hepsi):
    modeller = {}
    for ex, ad in ADLAR.items():
        tel = [s for s in tel_hepsi[ex] if s["kisi"] != haric]
        olculer = [k for k in tel[0] if k not in hf._ALAN]
        gruplar = {}
        for grup in ("front", "half-profile", "profile"):
            g = [s for s in tel if s["yon"] == grup]
            deg = hf.birlesik_kisi_disarida(g, olculer) if len(g) >= 20 else None
            kayit = {"auc": None if deg is None else deg["auc"], "n": len(g)}
            if deg and deg["auc"] is not None and deg["auc"] >= hf.OLCULEBILIR_AUC:
                X = np.array([[s[o] for o in olculer] for s in g], float)
                y = np.array([s["yanlis"] for s in g], bool)
                kayit["model"] = hf.birlesik_model().fit(X, y)
            gruplar[grup] = kayit
        sayac = hf._esikler([v for v in sayim_hepsi[ex] if v["kisi"] != haric])
        modeller[ex] = {"ad": ad, "olculer": olculer, "sayac": sayac, "gruplar": gruplar}
    return modeller


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kisiler", nargs="*", help="yalniz bu kisiler (paralel kosum icin)")
    ap.add_argument("--birlestir", action="store_true", help="parca sonuclarini birlestir")
    arg = ap.parse_args()
    if arg.birlestir:
        return ozetle(*parcalari_oku())
    alt = hf.alt_turler()
    seg = ht.segmentler()
    tel_hepsi = {ex: hf.telefon(ex, alt) for ex in ADLAR}
    sayim_hepsi = {ex: hf.sayim_verisi(ex) for ex in ADLAR}
    X, y, kisi, *_ = ht.rehab24_pencereleri()
    print(f"{len(X)} pencere hazir", flush=True)
    top = defaultdict(lambda: defaultdict(int))
    ornek = []
    for k in sorted(set(kisi)):
        if arg.kisiler and k not in arg.kisiler:
            continue
        tanima_modeli = siniflandirici().fit(X[kisi != k], y[kisi != k])
        modeller = form_modelleri(k, tel_hepsi, sayim_hepsi)
        for video, (kisi_id, ex, reps) in sorted(seg.items()):
            if kisi_id != k or ex not in ADLAR:
                continue
            for kam in ("c17", "c18"):
                f = ht.TESPIT / f"{video}-{kam}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                a = Antrenor(HareketTanima(tanima_modeli), modeller)
                mesaj = []
                for i, kare in enumerate(d["kare"]):
                    mesaj += a.adim(1, kare / 30.0, d["noktalar"][i], d["gorunur"][i],
                                    d["dunya_tam"][i], d["ayak_dunya"][i])
                mesaj += a.bitir()
                ad = ADLAR[ex]
                t = top[ad]
                ilk = next((m["hareket"] for m in mesaj if m["tur"] == "hareket"), None)
                t["video"] += 1
                t["kilit_dogru"] += int(ilk == ad)
                bildirilen = [m for m in mesaj if m["tur"] in ("tekrar", "olculemez")]
                dogru_hareket = [m for m in bildirilen if m["hareket"] == ad]
                t["yanlis_harekette_tekrar"] += len(bildirilen) - len(dogru_hareket)
                kullanildi = set()
                for ilk_k, son_k, _, _, dogru in reps:
                    t["gercek_tekrar"] += 1
                    es = [i for i, m in enumerate(dogru_hareket) if i not in kullanildi
                          and m["bas"] * 30 <= son_k and m["son"] * 30 >= ilk_k]
                    if not es:
                        continue
                    kullanildi.add(es[0])
                    m = dogru_hareket[es[0]]
                    t["bulunan"] += 1
                    if m["karar"] == "olculemez":
                        continue
                    t["karar"] += 1
                    t["karar_dogru"] += int((m["karar"] == "yanlis") == (not dogru))
                    t["yanlis_etiket"] += int(not dogru)
                    t["yanlis_yakalanan"] += int(not dogru and m["karar"] == "yanlis")
                    t["dogru_etiket"] += int(dogru)
                    t["dogru_onaylanan"] += int(dogru and m["karar"] == "dogru")
                t["fazla_tekrar"] += len(dogru_hareket) - len(kullanildi)
                if len(ornek) < 12:
                    ornek += [f"{video}-{kam} t={m['t']:.1f}: {m['metin']}" for m in mesaj
                              if m["tur"] in ("ipucu", "set", "olculemez")][:2]
        print(f"kisi {k} bitti", flush=True)
    ad_ = "_".join(arg.kisiler) if arg.kisiler else "hepsi"
    Path(f"out/antrenor_parca_{ad_}.json").write_text(json.dumps(
        {"top": {a: dict(v) for a, v in top.items()}, "ornek": ornek}, ensure_ascii=False))
    if not arg.kisiler:
        ozetle(top, ornek)


def parcalari_oku():
    top, ornek = defaultdict(lambda: defaultdict(int)), []
    for f in sorted(Path("out").glob("antrenor_parca_*.json")):
        d = json.loads(f.read_text())
        for a, v in d["top"].items():
            for k, x in v.items():
                top[a][k] += x
        ornek += d["ornek"]
    return top, ornek[:12]


def ozetle(top, ornek):
    def oran(a, b):
        return round(a / b, 3) if b else None
    sonuc = {ad: {"video": t["video"], "kilit_dogru": oran(t["kilit_dogru"], t["video"]),
                  "tekrar_bulma": oran(t["bulunan"], t["gercek_tekrar"]),
                  "fazla_tekrar": t["fazla_tekrar"],
                  "yanlis_harekette_tekrar": t["yanlis_harekette_tekrar"],
                  "kapsama": oran(t["karar"], t["bulunan"]),
                  "karar_dogrulugu": oran(t["karar_dogru"], t["karar"]),
                  "duzelt_duyarliligi": oran(t["yanlis_yakalanan"], t["yanlis_etiket"]),
                  "iyi_ozgullugu": oran(t["dogru_onaylanan"], t["dogru_etiket"]),
                  "karar_verilen_tekrar": t["karar"]}
             for ad, t in top.items()}
    Path("out/antrenor.json").write_text(json.dumps({"ozet": sonuc, "ornek_bildirim": ornek},
                                                    ensure_ascii=False, indent=1))
    for ad, v in sonuc.items():
        print(ad, v)
    print("\n".join(ornek))


if __name__ == "__main__":
    main()
