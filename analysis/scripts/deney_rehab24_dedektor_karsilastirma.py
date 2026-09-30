"""REHAB24-6 squat: MediaPipe, RTMPose-m ve RTMW-x ayni karelerde (ortak manifest).

Copilot incelemesi (28 Eylul): "yeni model eklemeden once mevcut uc dedektoru
ayni kilitli kare kumesinde karsilastirin; eklem merkezleme egilimini olcun."

Kare kumesi: squat tekrarlarinin karelerinden, uc dedektorun de dosyasi olan ve
mocap 2B izdusumu bulunan kareler (RTMW her 3 karede bir kosuldu; digerleri de
ayni karelerle sinirlanir). Yer gercegi: REHAB24'un mocap 2B izdusumu.

Olculer (bakis acisi x dedektor x eklem):

- **hata**: |tespit - mocap| / mocap uyluk uzunlugu (kalca-diz, 2B). Piksel yerine
  oran: iki kamera ve bakis acilari arasinda karsilastirilabilir.
- **ice kayma** (yalniz onden ve yarim profil): (tespit - mocap), mocap'te karsi
  kalcaya dogru birim vektore izdusumu / uyluk; pozitif = eklem merkezinin icine.
- **kapsama**: `gorunur` eklem orani (her dedektor kendi esigiyle).
- **ofset duzeltmesi (kisi-disarida)**: tespitin kendi govde cercevesinde
  (dikey = kalca ortasi -> omuz ortasi, yana = ona dik; olcek = tespit uyluk
  uzunlugu) eklem basina medyan ofset diger kisilerden ogrenilir, test kisisine
  uygulanir; duzeltme sonrasi hata. Yer gercegi yalniz egitim kisilerinde kullanilir.

    .venv/bin/python scripts/deney_rehab24_dedektor_karsilastirma.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pose3d.iskelet import REFERANS_ISKELET  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tg", KOK / "scripts/deney_rehab24_tek_gorus.py")
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)

I = REFERANS_ISKELET.indeks  # noqa: E741
DEDEKTORLER = ("mediapipe", "rtmpose", "rtmw")
EKLEMLER = ("omuz", "kalca", "diz", "ayak_bilegi")
DUZELTILEN = ("kalca", "diz", "ayak_bilegi")
YONLER = ("front", "half-profile", "profile")


def _orta(P, e):
    return (P[I(f"sag_{e}")] + P[I(f"sol_{e}")]) / 2


def govde_cercevesi(P: np.ndarray):
    """(merkez, dikey birim, yana birim, olcek) tespitin kendi noktalarindan; yoksa None."""
    kalca, omuz = _orta(P, "kalca"), _orta(P, "omuz")
    uyluk = np.array([np.linalg.norm(P[I(f"{t}_diz")] - P[I(f"{t}_kalca")])
                      for t in ("sag", "sol")])
    olcek = float(np.nanmean(uyluk)) if np.isfinite(uyluk).any() else np.nan
    v = omuz - kalca
    if not (np.isfinite(v).all() and np.isfinite(olcek)) or np.linalg.norm(v) < 1e-6 \
            or olcek < 1e-6:
        return None
    v = v / np.linalg.norm(v)
    return kalca, v, np.array([-v[1], v[0]]), olcek


def mocap_uyluk(G: np.ndarray) -> float:
    u = [np.linalg.norm(G[I(f"{t}_diz")] - G[I(f"{t}_kalca")]) for t in ("sag", "sol")]
    return float(np.nanmean(u))


def ornekler() -> list[dict]:
    """Ortak karelerde tespit (uc dedektor) ve mocap 2B."""
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv") if not t.mocap_hatali]
    out = []
    for t in tekrarlar:
        for cam in tg.KAMERALAR:
            tesp = {b: tg._tespit_yukle(b, t.video, cam) for b in DEDEKTORLER}
            if any(d is None for d in tesp.values()):
                continue
            yon = t.yon if cam == "c17" else tg.C18_YONU[t.yon]
            gt2 = np.load(tg.VERI / f"2d_joints/Ex6/{t.video}-{cam}-30fps.npy")
            for kare in range(t.ilk, t.son + 1):
                if kare >= len(gt2) or not all(kare in d["indeks"] for d in tesp.values()):
                    continue
                G = tg._referans_2b(gt2[kare])
                if not np.isfinite(mocap_uyluk(G)):
                    continue
                kayit = {"yon": yon, "kisi": t.kisi, "G": G}
                for b, d in tesp.items():
                    i = d["indeks"][kare]
                    P = np.asarray(d["noktalar"][i], float)
                    kayit[b] = (P, d["gorunur"][i].astype(bool) & np.isfinite(P).all(1))
                out.append(kayit)
    return out


def _ozet(v) -> dict | None:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {"medyan": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
            "n": int(len(v))}


def _yerel(Q, k, cerceve):
    merkez, v, y, olcek = cerceve
    d = Q[k] - merkez
    return np.array([d @ y, d @ v]) / olcek


def ofset_kisi_disarida(R: list[dict], b: str) -> dict:
    """Eklem basina medyan ofset (tespitin govde cercevesinde) diger kisilerden."""
    once, sonra = defaultdict(list), defaultdict(list)
    cerceveler = [govde_cercevesi(np.where(r[b][1][:, None], r[b][0], np.nan)) for r in R]
    for kisi in sorted({r["kisi"] for r in R}):
        egitim = defaultdict(list)
        for r, c in zip(R, cerceveler):
            if r["kisi"] == kisi or c is None:
                continue
            P, g = r[b]
            for e in DUZELTILEN:
                for t in ("sag", "sol"):
                    k = I(f"{t}_{e}")
                    if g[k] and np.isfinite(r["G"][k]).all():
                        egitim[k].append(_yerel(r["G"], k, c) - _yerel(P, k, c))
        ofset = {k: np.median(np.array(v), axis=0) for k, v in egitim.items() if len(v) >= 30}
        for r, c in zip(R, cerceveler):
            if r["kisi"] != kisi or c is None:
                continue
            P, g = r[b]
            _, v, y, olcek = c
            u = mocap_uyluk(r["G"])
            for e in DUZELTILEN:
                for t in ("sag", "sol"):
                    k = I(f"{t}_{e}")
                    if not (g[k] and k in ofset and np.isfinite(r["G"][k]).all()):
                        continue
                    duz = P[k] + (ofset[k][0] * y + ofset[k][1] * v) * olcek
                    once[e].append(np.linalg.norm(P[k] - r["G"][k]) / u)
                    sonra[e].append(np.linalg.norm(duz - r["G"][k]) / u)
    return {e: {"once": _ozet(once[e]), "sonra": _ozet(sonra[e])} for e in DUZELTILEN}


def dedektor_olcu(R: list[dict], b: str, yon: str) -> dict:
    hata, ice, kaps = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in R:
        P, g = r[b]
        G, u = r["G"], mocap_uyluk(r["G"])
        for e in EKLEMLER:
            for t, karsi in (("sag", "sol"), ("sol", "sag")):
                k = I(f"{t}_{e}")
                kaps[e].append(bool(g[k]))
                if not g[k] or not np.isfinite(G[k]).all():
                    continue
                d = P[k] - G[k]
                hata[e].append(np.linalg.norm(d) / u)
                m = G[I(f"{karsi}_kalca")] - G[I(f"{t}_kalca")]
                if yon != "profile" and e in ("diz", "ayak_bilegi") and np.linalg.norm(m) > 1e-6:
                    ice[e].append(float(d @ m / np.linalg.norm(m)) / u)
    return {"hata_uyluk": {e: _ozet(hata[e]) for e in EKLEMLER},
            "ice_kayma_uyluk": ({e: _ozet(ice[e]) for e in ("diz", "ayak_bilegi")}
                                if yon != "profile" else None),
            "kapsama": {e: round(float(np.mean(kaps[e])), 3) for e in EKLEMLER},
            "ofset_duzeltmesi": ofset_kisi_disarida(R, b)}


def main() -> None:
    veri = ornekler()
    sonuc: dict = {"deney": "rehab24_dedektor_karsilastirma", "ortak_kare": len(veri),
                   "kisi": len({r["kisi"] for r in veri}), "yonler": {},
                   "not": ("hata ve kayma mocap uyluk uzunluguna oranla (2B); ice kayma "
                           "pozitif = eklem merkezinin icine; ofset kisi-disarida.")}
    for yon in YONLER:
        R = [r for r in veri if r["yon"] == yon]
        if R:
            sonuc["yonler"][yon] = {"kare": len(R),
                                    **{b: dedektor_olcu(R, b, yon) for b in DEDEKTORLER}}
    yol = KOK / "out/rehab24_dedektor_karsilastirma.json"
    yol.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    yazdir(sonuc)


def yazdir(s: dict) -> None:
    print(f"ortak kare-kamera {s['ortak_kare']}, kisi {s['kisi']}")
    for yon, bolum in s["yonler"].items():
        print(f"== {yon} ({bolum['kare']} kare)")
        for b in DEDEKTORLER:
            h = bolum[b]["hata_uyluk"]
            print(f"  {b:9s} hata   " + "  ".join(
                f"{e} {h[e]['medyan']:.3f}" for e in EKLEMLER if h[e]))
            if bolum[b]["ice_kayma_uyluk"]:
                ic = bolum[b]["ice_kayma_uyluk"]
                print(f"  {'':9s} ice    " + "  ".join(
                    f"{e} {ic[e]['medyan']:+.3f}" for e in ic if ic[e]))
            print(f"  {'':9s} kapsama " + "  ".join(
                f"{e} {v:.2f}" for e, v in bolum[b]["kapsama"].items()))
            o = bolum[b]["ofset_duzeltmesi"]
            print(f"  {'':9s} ofset  " + "  ".join(
                f"{e} {o[e]['once']['medyan']:.3f}->{o[e]['sonra']['medyan']:.3f}"
                for e in o if o[e]["once"] and o[e]["sonra"]))


if __name__ == "__main__":
    main()
