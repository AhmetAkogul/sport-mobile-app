"""REHAB24-6: telefon valgus hatasinin halka halka butcesi (2B -> 3B -> aci).

Soru: MediaPipe world'un valgus hatasi 2B tespitten mi, 3B'ye cikarken
derinlik tahmininden mi geliyor? Her halkanin yer gercegi var: mocap'in
kameraya izdusumu (2B), mocap iskeleti kamera cercevesinde (3B).

1. **2B:** dedektor noktasi - mocap izdusumu, piksel; kisinin derinliginde
   milimetreye cevrilmis (px * Z / f). Diz, kalca, ayak bilegi.
2. **3B bilesenleri:** kok (kalca ortasi) merkezli, olcegi mocap'e eslenmis
   MediaPipe world ile mocap arasindaki fark X (yana), Y (dikey), Z (derinlik).
   MediaPipe world kamera eksenleriyle hizalidir; dondurme uygulanmaz.
3. **Karsi-olgusal valgus:** ayni karede valgus
   - tam MediaPipe,
   - derinligi mocap'ten (X,Y MediaPipe): kalan hata goruntu duzleminden,
   - X,Y mocap'ten (derinlik MediaPipe): kalan hata derinlikten.

Isaretli fark (telefon - mocap) ve mutlak farkin medyani raporlanir.

    python scripts/deney_rehab24_hata_butcesi.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tg", KOK / "scripts/deney_rehab24_tek_gorus.py")
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)

I = REFERANS_ISKELET.indeks  # noqa: E741
BACAK = ("sag_kalca", "sol_kalca", "sag_diz", "sol_diz", "sag_ayak_bilegi", "sol_ayak_bilegi")
ADIM = 3          # her tekrarda 3 karede bir


def kok_merkezle(P: np.ndarray) -> np.ndarray:
    return P - (P[I("sag_kalca")] + P[I("sol_kalca")]) / 2


def olcek_esle(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P'yi (kok merkezli) Q'nun olcegine getirir: bacak kemik boylarinin orani."""
    kemik = [(a, b) for a, b in REFERANS_ISKELET.baglantilar
             if a in BACAK and b in BACAK]
    lp = [np.linalg.norm(P[I(a)] - P[I(b)]) for a, b in kemik]
    lq = [np.linalg.norm(Q[I(a)] - Q[I(b)]) for a, b in kemik]
    s = np.nanmedian(np.array(lq) / np.array(lp))
    return P * s if np.isfinite(s) and s > 0 else P


def melez(xy_kaynak: np.ndarray, z_kaynak: np.ndarray) -> np.ndarray:
    return np.column_stack([xy_kaynak[:, :2], z_kaynak[:, 2]])


def _iskelet(P: np.ndarray) -> Iskelet3B:
    g = np.isfinite(P).all(axis=1)
    n = len(REFERANS_ISKELET)
    return Iskelet3B(REFERANS_ISKELET, P, g, g.astype(int), np.full(n, np.nan),
                     cerceve="kamera_kok", kaynak="hata_butcesi")


def valgus(P: np.ndarray) -> np.ndarray:
    return np.array(tg._valgus_kararlari(_iskelet(P))[1], float)


def _ozet(v) -> dict | None:
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not v.size:
        return None
    return {"n": int(v.size), "medyan": round(float(np.median(v)), 2),
            "ortalama": round(float(v.mean()), 2), "sd": round(float(v.std()), 2),
            "mutlak_medyan": round(float(np.median(np.abs(v))), 2)}


def main() -> None:
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv") if not t.mocap_hatali]
    ornek = tekrarlar[0].video
    X0 = np.load(tg.VERI / f"3d_joints/Ex6/{ornek}-30fps.npy")[::10, :, :3]
    kameralar = {c: tg.kamera_kestir(
        X0.reshape(-1, 3), np.load(tg.VERI / f"2d_joints/Ex6/{ornek}-{c}-30fps.npy")[::10]
        .reshape(-1, 2)) for c in tg.KAMERALAR}

    # topla[(yon, olcu)] -> liste
    topla: dict = defaultdict(list)
    for t in tekrarlar:
        kareler = tg.tekrar_kareleri(tg.VERI, t)
        for cam in tg.KAMERALAR:
            kam = kameralar[cam]
            yon = t.yon if cam == "c17" else tg.C18_YONU[t.yon]
            gt2 = np.load(tg.VERI / f"2d_joints/Ex6/{t.video}-{cam}-30fps.npy")
            tesp = {b: tg._tespit_yukle(b, t.video, cam) for b in ("mediapipe", "rtmpose")}
            for j in range(0, len(kareler), ADIM):
                kare = t.ilk + j
                Pg = tg.kareyi_cevir(kareler[j]).noktalar
                Pc = (kam.R @ Pg.T + kam.t[:, None]).T          # mocap, kamera cercevesi
                x_gt = tg._referans_2b(gt2[kare])
                f = kam.K[0, 0]
                # 1) 2B piksel hatasi ve kisinin derinliginde mm karsiligi
                for b, d in tesp.items():
                    if d is None or kare not in d["indeks"]:
                        continue
                    i = d["indeks"][kare]
                    g = d["gorunur"][i].astype(bool)
                    for e in ("diz", "kalca", "ayak_bilegi"):
                        for taraf in ("sag", "sol"):
                            k = I(f"{taraf}_{e}")
                            if not g[k] or not np.isfinite(x_gt[k]).all():
                                continue
                            px = float(np.linalg.norm(d["noktalar"][i][k] - x_gt[k]))
                            topla[(yon, f"2b_px_{b}_{e}")].append(px)
                            topla[(yon, f"2b_mm_{b}_{e}")].append(px * Pc[k, 2] / f * 1000)
                # 2-3) MediaPipe world: bilesen hatasi ve karsi-olgusal valgus
                d = tesp["mediapipe"]
                if d is None or kare not in d["indeks"] or "dunya_tam" not in d:
                    continue
                M = np.asarray(d["dunya_tam"][d["indeks"][kare]], float)
                if not np.isfinite(M).all() or not np.isfinite(Pc).all():
                    continue
                G = kok_merkezle(Pc)
                M = olcek_esle(kok_merkezle(M), G)
                bac = [I(e) for e in BACAK]
                for eksen, ad in enumerate(("x_yana", "y_dikey", "z_derinlik")):
                    topla[(yon, f"3b_mm_{ad}")].extend(
                        np.abs(M[bac, eksen] - G[bac, eksen]) * 1000)
                v_gt = valgus(G)
                for ad, P in (("tam_mediapipe", M), ("derinlik_mocap", melez(M, G)),
                              ("xy_mocap", melez(G, M))):
                    v = valgus(P)
                    topla[(yon, f"valgus_{ad}")].extend(v - v_gt)

    sonuc = {"deney": "rehab24_hata_butcesi", "adim_kare": ADIM,
             "yonler": {yon: {olcu: _ozet(v) for (y, olcu), v in sorted(topla.items())
                              if y == yon}
                        for yon in ("front", "half-profile", "profile")},
             "not": ("2B mm: piksel hatasi * mocap derinligi / f. 3B: kok merkezli, bacak "
                     "kemik oranina gore olceklenmis, dondurmesiz (MediaPipe world kamera "
                     "eksenli). Valgus: isaretli fark telefon - mocap, derece.")}
    (KOK / "out/rehab24_hata_butcesi.json").write_text(
        json.dumps(tg._temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    for yon, olculer in sonuc["yonler"].items():
        print(f"== {yon}")
        for olcu, o in olculer.items():
            if o:
                print(f"  {olcu:32s} medyan {o['medyan']:8.2f}  ort {o['ortalama']:8.2f}  "
                      f"sd {o['sd']:7.2f}  |medyan| {o['mutlak_medyan']:7.2f}  n {o['n']}")


if __name__ == "__main__":
    main()
