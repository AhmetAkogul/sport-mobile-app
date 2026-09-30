"""REHAB24-6 kopru pilotu: MediaPipe pose_world'u mocap'le duzeltmeyi ogrenmek.

Soru: tek kameranin valgus olcumu, cok kamerali/mocap referansla egitilen bir
duzeltmeyle **daha once gorulmemis kiside** referansla ayni karari verir mi?

- Girdi: MediaPipe `pose_world` (13x3, kalca merkezli, kamera yonunde).
- Hedef: mocap iskeleti kamera cercevesinde, kalca merkezine gore (13x3).
- Model: artik ridge regresyonu, Y = X + [X, 1] W (numpy, kapali form).
  Duzenleme katsayisi yalnizca egitim kisileri icinde, ic kisi-disari
  dogrulamayla secilir.
- Degerlendirme: kisi-disari (9 kat). Test kisisi egitime ve lambda secimine
  girmez (0070 §3, 0071 §6).
- Zaman filtresi: tekrar icinde merkezli hareketli ortalama (gecmis + gelecek
  kareler; canli kullanimda gecikme demek -- ayrica raporlanir).

Kollar ayni tekrarlarda, ayni tekrar kuraliyla (`deney_rehab24_tek_gorus`).

    python scripts/deney_rehab24_kopru.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.protokol import Oge, Sayim, kisi_agirlikli, kisi_bootstrap  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tg", KOK / "scripts/deney_rehab24_tek_gorus.py")
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)

N = len(REFERANS_ISKELET)
_KOK = [REFERANS_ISKELET.indeks("sag_kalca"), REFERANS_ISKELET.indeks("sol_kalca")]
LAMBDALAR = (0.1, 1.0, 10.0, 100.0)
PENCERE = 7          # kare (~0,23 s), merkezli


def _kok_goreli(P: np.ndarray) -> np.ndarray:
    return P - P[_KOK].mean(axis=0)


def _yumusat(seri: np.ndarray, w: int) -> np.ndarray:
    """(T, 13, 3) merkezli hareketli ortalama; NaN'lar ortalamaya girmez."""
    if w <= 1:
        return seri
    h = w // 2
    cikti = np.full_like(seri, np.nan)
    for i in range(len(seri)):
        pencere = seri[max(0, i - h): i + h + 1]
        if np.isfinite(pencere).any():
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # tumu NaN eklem
                cikti[i] = np.nanmean(pencere, axis=0)
    return cikti


def veri_topla() -> list[dict]:
    """Tekrar x kamera basina kare dizileri: MediaPipe world ve mocap (kamera yonu)."""
    V = tg.VERI
    tekrarlar = [t for t in tg.tekrarlar_oku(V / "Segmentation.csv") if not t.mocap_hatali]
    X0 = np.load(V / f"3d_joints/Ex6/{tekrarlar[0].video}-30fps.npy")[::10, :, :3]
    kams = {c: tg.kamera_kestir(X0.reshape(-1, 3), np.load(
        V / f"2d_joints/Ex6/{tekrarlar[0].video}-{c}-30fps.npy")[::10].reshape(-1, 2))
        for c in tg.KAMERALAR}
    diziler = []
    for t in tekrarlar:
        kareler = tg.tekrar_kareleri(V, t)
        for c in tg.KAMERALAR:
            d = tg._tespit_yukle("mediapipe", t.video, c)
            if d is None:
                continue
            mp, ref = [], []
            for j, q in enumerate(kareler):
                P_ref = (kams[c].R @ tg.kareyi_cevir(q).noktalar.T + kams[c].t[:, None]).T
                ref.append(_kok_goreli(P_ref))
                i = d["indeks"].get(t.ilk + j)
                mp.append(tg._dunya_iskeleti(d, i).noktalar if i is not None
                          else np.full((N, 3), np.nan))
            diziler.append({"kisi": t.kisi, "oge": f"{t.video}-{t.tekrar_no}-{c}",
                            "yon": t.yon if c == "c17" else tg.C18_YONU[t.yon],
                            "mp": np.array(mp), "ref": np.array(ref)})
    return diziler


def _ozellik(mp: np.ndarray) -> np.ndarray:
    return mp.reshape(len(mp), -1)


def ridge_egit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Artik ridge: [X, 1] W ~ (Y - X). Donus W (40, 39)."""
    A = np.c_[X, np.ones(len(X))]
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0                                  # sabit terim cezasiz
    return np.linalg.solve(A.T @ A + reg, A.T @ (Y - X))


def ridge_uygula(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return X + np.c_[X, np.ones(len(X))] @ W


def _egitim_kumesi(diziler, kisiler, w):
    Xs, Ys = [], []
    for s in diziler:
        if s["kisi"] not in kisiler:
            continue
        X = _ozellik(_yumusat(s["mp"], w))
        Y = _ozellik(s["ref"])
        m = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        Xs.append(X[m])
        Ys.append(Y[m])
    if not Xs:
        return np.zeros((0, 3 * N)), np.zeros((0, 3 * N))
    return np.concatenate(Xs), np.concatenate(Ys)


def lambda_sec(diziler, kisiler: set, w: int) -> float:
    """Ic kisi-disari dogrulama: yalnizca egitim kisileri."""
    en_iyi, en_iyi_hata = LAMBDALAR[0], np.inf
    for lam in LAMBDALAR:
        hatalar = []
        for k in kisiler:
            X, Y = _egitim_kumesi(diziler, kisiler - {k}, w)
            Xv, Yv = _egitim_kumesi(diziler, {k}, w)
            if len(Xv) == 0:
                continue
            P = ridge_uygula(Xv, ridge_egit(X, Y, lam))
            hatalar.append(np.mean(np.linalg.norm((P - Yv).reshape(-1, N, 3), axis=2)))
        if hatalar and np.mean(hatalar) < en_iyi_hata:
            en_iyi, en_iyi_hata = lam, float(np.mean(hatalar))
    return en_iyi


def _iskelet(P: np.ndarray) -> Iskelet3B:
    g = np.isfinite(P).all(axis=1)
    P = P.copy()
    P[~g] = np.nan
    return Iskelet3B(REFERANS_ISKELET, P, g, g.astype(int), np.full(N, np.nan),
                     cerceve="telefon_kamera_kok_goreli", kaynak="kopru")


def kol_degerlendir(dizi_tahmin: dict[str, np.ndarray], diziler) -> dict:
    satir = []
    for s in diziler:
        P = dizi_tahmin[s["oge"]]
        ref_k = [tg._valgus_kararlari(_iskelet(r)) for r in s["ref"]]
        tel_k = [tg._valgus_kararlari(_iskelet(p)) if np.isfinite(p).all()
                 else ((tg.Karar.BELIRSIZ,) * 2, (np.nan, np.nan)) for p in P]
        fark = [abs(a - b) for rk, tk in zip(ref_k, tel_k)
                for a, b in zip(rk[1], tk[1]) if np.isfinite(a) and np.isfinite(b)]
        m = np.isfinite(P).all(axis=(1, 2))
        mpjpe = (float(np.mean(np.linalg.norm(P[m] - s["ref"][m], axis=2)) * 1000)
                 if m.any() else np.nan)
        satir.append({"kisi": s["kisi"], "oge": s["oge"], "yon": s["yon"],
                      "referans": tg.tekrar_karari([r[0] for r in ref_k]),
                      "telefon": tg.tekrar_karari([k[0] for k in tel_k]),
                      "valgus_fark": float(np.median(fark)) if fark else np.nan,
                      "mpjpe_kok_goreli_mm": mpjpe})
    yonler = {}
    for yon in ("front", "half-profile", "profile"):
        r = [x for x in satir if x["yon"] == yon]
        og = [Oge(x["kisi"], x["oge"], x["referans"], x["telefon"]) for x in r]
        yonler[yon] = {
            "n": len(r),
            "protokol_0070": Sayim.ciftlerden((o.referans, o.telefon) for o in og).oranlar(),
            "kisi_agirlikli": kisi_agirlikli(og), "kisi_bootstrap": kisi_bootstrap(og),
            "valgus_fark_medyan": tg._ozet([x["valgus_fark"] for x in r]),
            "mpjpe_kok_goreli_mm": tg._ozet([x["mpjpe_kok_goreli_mm"] for x in r]),
        }
    return yonler


def main() -> None:
    diziler = veri_topla()
    kisiler = sorted({s["kisi"] for s in diziler})
    kollar: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    secilen_lambda: dict = {}
    for w, ad in ((1, "ham"), (PENCERE, "filtreli")):
        for s in diziler:
            kollar[f"mediapipe_{ad}"][s["oge"]] = _yumusat(s["mp"], w)
        for k in kisiler:
            egitim = set(kisiler) - {k}
            lam = lambda_sec(diziler, egitim, w)
            secilen_lambda[f"{ad}:{k}"] = lam
            W = ridge_egit(*_egitim_kumesi(diziler, egitim, w), lam)
            for s in diziler:
                if s["kisi"] != k:
                    continue
                X = _ozellik(_yumusat(s["mp"], w))
                P = np.full_like(X, np.nan)
                m = np.isfinite(X).all(axis=1)
                P[m] = ridge_uygula(X[m], W)
                kollar[f"kopru_{ad}"][s["oge"]] = P.reshape(-1, N, 3)

    sonuc = {
        "deney": "rehab24_kopru_pilotu",
        "yontem": "artik ridge, MediaPipe pose_world -> mocap (kamera yonu, kok goreli)",
        "degerlendirme": "kisi-disari (9 kat); lambda ic kisi-disari ile yalniz egitimde",
        "pencere_kare": PENCERE, "lambdalar": list(LAMBDALAR), "secilen_lambda": secilen_lambda,
        "n_dizi": len(diziler), "n_kisi": len(kisiler),
        "kollar": {ad: kol_degerlendir(t, diziler) for ad, t in kollar.items()},
        "not": ("9 kisi < 10: on kanit. Filtre merkezli (gelecek kare kullanir). "
                "Kok goreli MPJPE mutlak konum degildir."),
    }
    (KOK / "out/rehab24_kopru.json").write_text(json.dumps(
        tg._temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for ad, yonler in sonuc["kollar"].items():
        for yon, y in yonler.items():
            o = y["protokol_0070"]
            print(f"{ad:18s} {yon:12s}", {k: o[k] for k in ("N", "D", "C", "A", "K", "Y")},
                  "valgus_fark", (y["valgus_fark_medyan"] or {}).get("medyan"),
                  "mpjpe", (y["mpjpe_kok_goreli_mm"] or {}).get("medyan"))


if __name__ == "__main__":
    main()
