"""REHAB24-6 squat: melez iskelet -- goruntu duzlemi RTMW'den, derinlik MediaPipe'tan (0041).

Soru: onden valgus belirsizligi (kare sigma) 3,5 dereceye inebilir mi? Hata
butcesine gore onden hata goruntu duzleminden geliyor
(`docs/deney/2026-09-28-hata-butcesi.md`); RTMW 2B'de en iyi dedektor (0041).

Kare basina:
1. MediaPipe dunya iskeleti M (kalca merkezli, kamera eksenli, metre) yalniz
   **goreli derinlik** icin kullanilir.
2. s*M + T'nin izdusumu 2B noktalara en kucuk kareler ile oturtulur (s, T:
   4 parametre); boylece kok derinligi ve olcek goruntuden gelir.
3. Her eklem: Z = s*M_z + T_z; X, Y ise 2B noktadan geri izdusum (u, v, Z).
   2B'de gorunmeyen eklem oturtulmus s*M + T'den alinir.

2B kaynaklari: MediaPipe (yalniz yontemin etkisi), RTMW, RTMW + kisi-disarida
2B eklem ofseti (0041). Karsilastirma: ham MediaPipe dunya. Olcu: valgus artigi
(telefon - mocap) saglam SD'si (1,4826*MAD) ve mutlak medyani, bakis acisi grubuna
gore (etiket yalniz raporlama; ofset tahmini govde acisi dilimiyle ogrenilir).
Kamera ic parametreleri mocap-video eslesmesinden kestirilir (`kamera_kestir`):
telefonda bilinen/ kalibre edilen K varsayimi.

    .venv/bin/python scripts/deney_rehab24_melez_iskelet.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mono.canli import govde_acisi  # noqa: E402
from mono.melez import melez_iskelet as melez  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("hb", KOK / "scripts/deney_rehab24_hata_butcesi.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)
tg = hb.tg
_spec2 = importlib.util.spec_from_file_location(
    "dk", KOK / "scripts/deney_rehab24_dedektor_karsilastirma.py")
dk = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(dk)

I = REFERANS_ISKELET.indeks  # noqa: E741
DUZELTILEN = [I(f"{t}_{e}") for t in ("sag", "sol") for e in ("kalca", "diz", "ayak_bilegi")]
DILIMLER = ((0, 30), (30, 60), (60, 90.01))


def dilim(aci: float) -> int:
    return next((i for i, (a, b) in enumerate(DILIMLER) if a <= aci < b), -1)


def rsd(x) -> dict | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {"saglam_sd": round(float(1.4826 * np.median(np.abs(x - np.median(x)))), 2),
            "mutlak_medyan": round(float(np.median(np.abs(x))), 2),
            "kayma": round(float(np.median(x)), 2), "n": int(len(x))}


def main() -> None:
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv") if not t.mocap_hatali]
    ornek = tekrarlar[0].video
    X0 = np.load(tg.VERI / f"3d_joints/Ex6/{ornek}-30fps.npy")[::10, :, :3]
    kameralar = {c: tg.kamera_kestir(
        X0.reshape(-1, 3), np.load(tg.VERI / f"2d_joints/Ex6/{ornek}-{c}-30fps.npy")[::10]
        .reshape(-1, 2)) for c in tg.KAMERALAR}
    kayit = []
    for t in tekrarlar:
        kareler = tg.tekrar_kareleri(tg.VERI, t)
        for cam in tg.KAMERALAR:
            kam = kameralar[cam]
            mp = tg._tespit_yukle("mediapipe", t.video, cam)
            rw = tg._tespit_yukle("rtmw", t.video, cam)
            if mp is None or rw is None or "dunya_tam" not in mp:
                continue
            gt2 = np.load(tg.VERI / f"2d_joints/Ex6/{t.video}-{cam}-30fps.npy")
            for j in range(len(kareler)):
                kare = t.ilk + j
                if kare not in mp["indeks"] or kare not in rw["indeks"] or kare >= len(gt2):
                    continue
                M = np.asarray(mp["dunya_tam"][mp["indeks"][kare]], float)
                Pg = tg.kareyi_cevir(kareler[j]).noktalar
                G = (kam.R @ Pg.T + kam.t[:, None]).T
                if not (np.isfinite(M).all() and np.isfinite(G).all()):
                    continue
                i_mp, i_rw = mp["indeks"][kare], rw["indeks"][kare]
                aci = govde_acisi(M)
                kayit.append({
                    "kisi": t.kisi, "yon": t.yon if cam == "c17" else tg.C18_YONU[t.yon],
                    "dilim": dilim(aci) if np.isfinite(aci) else -1, "M": M, "K": kam.K,
                    "mp2": (np.asarray(mp["noktalar"][i_mp], float),
                            mp["gorunur"][i_mp].astype(bool)),
                    "rw2": (np.asarray(rw["noktalar"][i_rw], float),
                            rw["gorunur"][i_rw].astype(bool)),
                    "G2": tg._referans_2b(gt2[kare]), "v_gt": hb.valgus(G), "v_mp": hb.valgus(M)})
    print(f"{len(kayit)} kare", flush=True)

    # 2B ofset kisi-disarida, kaynak basina (MediaPipe, RTMW); govde acisi dilimi x eklem
    for kaynak in ("mp2", "rw2"):
        for kisi in sorted({k["kisi"] for k in kayit}, key=int):
            ogren = defaultdict(list)
            for k in kayit:
                if k["kisi"] == kisi:
                    continue
                P, g = k[kaynak]
                c = dk.govde_cercevesi(np.where(g[:, None], P, np.nan))
                if c is None:
                    continue
                for e in DUZELTILEN:
                    if g[e] and np.isfinite(k["G2"][e]).all():
                        ogren[(k["dilim"], e)].append(dk._yerel(k["G2"], e, c) - dk._yerel(P, e, c))
            ofset = {a: np.median(v, axis=0) for a, v in ogren.items() if len(v) >= 30}
            for k in (x for x in kayit if x["kisi"] == kisi):
                P, g = k[kaynak]
                Q = P.copy()
                c = dk.govde_cercevesi(np.where(g[:, None], P, np.nan))
                if c is not None:
                    _, v, y, olcek = c
                    for e in DUZELTILEN:
                        o = ofset.get((k["dilim"], e))
                        if o is not None and g[e]:
                            Q[e] = P[e] + (o[0] * y + o[1] * v) * olcek
                k[f"{kaynak}_ofset"] = (Q, g)

    for k in kayit:
        for ad, anahtar in (("melez_mp", "mp2"), ("melez_mp_ofset", "mp2_ofset"),
                            ("melez_rtmw", "rw2"), ("melez_rtmw_ofset", "rw2_ofset")):
            H = melez(k["M"], *k[anahtar], k["K"])
            k[f"v_{ad}"] = hb.valgus(H) if H is not None else np.full(2, np.nan)

    sonuc = {"deney": "rehab24_melez_iskelet", "kare": len(kayit), "yonler": {}}
    for yon in ("front", "half-profile", "profile"):
        s = [k for k in kayit if k["yon"] == yon]
        sonuc["yonler"][yon] = {ad: rsd(np.concatenate([k[f"v_{ad}"] - k["v_gt"] for k in s]))
                                for ad in ("mp", "melez_mp", "melez_mp_ofset", "melez_rtmw",
                                           "melez_rtmw_ofset")}
    (KOK / "out/rehab24_melez_iskelet.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    for yon, v in sonuc["yonler"].items():
        print(f"== {yon}")
        for ad, o in v.items():
            if o:
                print(f"   {ad:18s} saglam sd {o['saglam_sd']:6.2f}  "
                      f"|medyan| {o['mutlak_medyan']:5.2f}  kayma {o['kayma']:+6.2f}  n {o['n']}")


if __name__ == "__main__":
    main()
