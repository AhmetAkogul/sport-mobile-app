"""0038: REHAB24'un kalan dort hareketinde fizyoterapistin "yanlis"ini ne acikliyor?

Squat/lunge yontemi: aday olculer (`eval.hareket_formu`, hepsi "buyukse yanlis"
yonunde, onceden sabit) once mocap'ta (yer gercegi), sonra ayni fonksiyonla
telefonda (MediaPipe dunya noktalari, `scripts/hareket_tespit.py` ciktisi).
Fizyoterapist hata turunu etiketlemedi; AUC toplamda ve kisi basina verilir.

Karar kurali (olcu basina): esik, egitim kisilerinde Youden J'yi en buyuten
deger; test kisisinde uygulanir (kisi-disarida). Telefonda ayni kural, mocap'ta
secilen olcuyle, telefon verisinde yeniden kisi-disarida ogrenilir.

    .venv/bin/python scripts/deney_hareket_formu.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.hareket_formu import (TekrarSayaci, aci_grubu, ana_sinyal, auc,  # noqa: E402
                                bakis_acisi, esikler_ogren, tekrar_olculeri)
from veri.rehab24 import kareyi_cevir, tekrar_kareleri, tekrarlar_oku  # noqa: E402

VERI = Path("data/dis/rehab24_6")
TESPIT = Path("out/hareket_tespit/mediapipe")
ADLAR = {1: "kol_abduksiyonu", 2: "kol_vw", 3: "sinav", 4: "bacak_abduksiyonu",
         5: "lunge", 6: "squat"}
MOCAP_YUKARI = np.array([0.0, 1.0, 0.0])
MP_YUKARI = np.array([0.0, -1.0, 0.0])
_ALAN = ("kisi", "video", "tekrar", "yon", "yanlis", "kamera")
OLCULEBILIR_AUC = 0.70


def alt_turler():
    with (VERI / "Segmentation.csv").open() as f:
        return {(r["video_id"], int(r["repetition_number"])): r["exercise_subtype"]
                for r in csv.DictReader(f, delimiter=";")}


def taraf_bul(alt: str) -> str | None:
    if "right" in alt:
        return "sag"
    if "left" in alt:
        return "sol"
    return None


def esik_kurali(egitim_x, egitim_y):
    """Youden J'yi en buyuten esik (x > esik -> yanlis)."""
    x, y = np.asarray(egitim_x, float), np.asarray(egitim_y, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if not len(x) or y.all() or not y.any():
        return np.inf
    aday = np.unique(x)
    j = [np.mean(x[y] > t) - np.mean(x[~y] > t) for t in aday]
    return float(aday[int(np.argmax(j))])


def kisi_disarida_dogruluk(satirlar, olcu):
    """Her kisi icin esik diger kisilerden; dogruluk, duyarlilik, ozgulluk."""
    tahmin, gercek = [], []
    for k in sorted({s["kisi"] for s in satirlar}):
        eg = [s for s in satirlar if s["kisi"] != k]
        te = [s for s in satirlar if s["kisi"] == k and np.isfinite(s[olcu])]
        t = esik_kurali([s[olcu] for s in eg], [s["yanlis"] for s in eg])
        tahmin += [s[olcu] > t for s in te]
        gercek += [s["yanlis"] for s in te]
    tahmin, gercek = np.array(tahmin, bool), np.array(gercek, bool)
    if not len(gercek):
        return None
    return {"dogruluk": round(float(np.mean(tahmin == gercek)), 3), "n": int(len(gercek)),
            "duyarlilik": round(float(np.mean(tahmin[gercek])), 3) if gercek.any() else None,
            "ozgulluk": round(float(np.mean(~tahmin[~gercek])), 3) if (~gercek).any() else None}


def birlesik_model():
    """Butun olculer: eksik -> medyan, olcekleme, dengeli lojistik regresyon."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced",
                                                              max_iter=1000))


def birlesik_kisi_disarida(satirlar, olculer):
    """Butun olculeri birlestiren model, kisi-disarida: AUC ve 0,5 esikte dogruluk."""
    X = np.array([[s[o] for o in olculer] for s in satirlar], float)
    y = np.array([s["yanlis"] for s in satirlar], bool)
    kisi = np.array([s["kisi"] for s in satirlar])
    p = np.full(len(y), np.nan)
    for k in sorted(set(kisi)):
        te = kisi == k
        if y[~te].all() or not y[~te].any():
            continue
        p[te] = birlesik_model().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    ok = np.isfinite(p)
    if not ok.any():
        return None
    a = auc(p[ok & y], p[ok & ~y])
    t = p[ok] > 0.5
    return {"auc": None if a is None else round(a, 3),
            "dogruluk": round(float(np.mean(t == y[ok])), 3), "n": int(ok.sum()),
            "duyarlilik": round(float(np.mean(t[y[ok]])), 3),
            "ozgulluk": round(float(np.mean(~t[~y[ok]])), 3)}


def ozet(satirlar, olculer):
    out = {}
    for o in olculer:
        kisi = {}
        for k in sorted({s["kisi"] for s in satirlar}):
            a = auc([s[o] for s in satirlar if s["kisi"] == k and s["yanlis"]],
                    [s[o] for s in satirlar if s["kisi"] == k and not s["yanlis"]])
            if a is not None:
                kisi[k] = round(a, 2)
        a = auc([s[o] for s in satirlar if s["yanlis"]], [s[o] for s in satirlar if not s["yanlis"]])
        out[o] = {"auc": None if a is None else round(a, 3), "kisi_auc": kisi,
                  "kapsama": round(float(np.mean([np.isfinite(s[o]) for s in satirlar])), 3)}
    return out


def mocap(ex, alt):
    satirlar = []
    for t in tekrarlar_oku(VERI / "Segmentation.csv", egzersiz=ex):
        if t.mocap_hatali:
            continue
        try:
            ham = tekrar_kareleri(VERI, t, egzersiz=ex)
        except ValueError:          # segment mocap dizisinden tasiyor (PM_117a, 0038)
            continue
        X = np.stack([kareyi_cevir(k).noktalar for k in ham])
        # mocap ayak: topuk yok (NaN), ayak ucu = ToeBase (sag 24, sol 19)
        ayak = np.full((len(ham), 4, 3), np.nan)
        ayak[:, 1], ayak[:, 3] = ham[:, 24, :3], ham[:, 19, :3]
        o = tekrar_olculeri(X, MOCAP_YUKARI, ex, taraf_bul(alt[(t.video, t.tekrar_no)]), 30.0,
                            ayak=ayak)
        satirlar.append({"kisi": t.kisi, "video": t.video, "tekrar": t.tekrar_no, "yon": t.yon,
                         "yanlis": not t.dogru, **o})
    return satirlar


def telefon(ex, alt):
    satirlar = []
    for t in tekrarlar_oku(VERI / "Segmentation.csv", egzersiz=ex):
        for kam in ("c17", "c18"):
            f = TESPIT / f"{t.video}-{kam}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            sec = (d["kare"] >= t.ilk) & (d["kare"] <= t.son)
            if sec.sum() < 5:
                continue
            # telefonda taraf etiketten degil hareketten (canlida etiket yok; 0038)
            o = tekrar_olculeri(d["dunya_tam"][sec], MP_YUKARI, ex, None, 10.0,
                                ayak=d["ayak_dunya"][sec])
            # grup kamera etiketiyle degil tahmini govde acisiyla: canlida da bilinen bu
            aci = float(np.nanmedian([bakis_acisi(P) for P in d["dunya_tam"][sec]]))
            satirlar.append({"kisi": t.kisi, "video": t.video, "tekrar": t.tekrar_no,
                             "kamera": kam, "yon": aci_grubu(aci),
                             "yanlis": not t.dogru, **o})
    return satirlar


def yumusat(x, n: int = 3):
    """Nedensel medyan (son n kare): canlida da ayni sekilde uygulanir."""
    out = np.array(x, float)
    for i in range(len(x)):
        w = x[max(0, i - n + 1):i + 1]
        w = w[np.isfinite(w)]
        out[i] = np.median(w) if len(w) else np.nan
    return out


def sayim_verisi(ex):
    """Video-kamera basina: kisi, zaman, ana sinyal, gercek tekrar araliklari (30 fps kare)."""
    out = []
    reps = tekrarlar_oku(VERI / "Segmentation.csv", egzersiz=ex)
    for video in sorted({t.video for t in reps}):
        gercek = [(t.ilk, t.son) for t in reps if t.video == video]
        kisi = next(t.kisi for t in reps if t.video == video)
        for kam in ("c17", "c18"):
            f = TESPIT / f"{video}-{kam}.npz"
            if f.exists():
                d = np.load(f)
                out.append({"kisi": kisi, "kare": d["kare"], "gercek": gercek,
                            "sinyal": ana_sinyal(d["dunya_tam"], MP_YUKARI, ex)})
    return out


def _esikler(veri):
    dinlenme, tepe, sure = [], [], []
    for v in veri:
        ic = np.zeros(len(v["kare"]), bool)
        for a, b in v["gercek"]:
            sec = (v["kare"] >= a) & (v["kare"] <= b)
            ic |= sec
            sure.append((b - a) / 30.0)
            if sec.any() and np.isfinite(v["sinyal"][sec]).any():
                tepe.append(np.nanmax(v["sinyal"][sec]))
        dinlenme += v["sinyal"][~ic].tolist()
    return esikler_ogren(np.array(dinlenme), np.array(tepe), np.array(sure))


def sayim_kisi_disarida(veri):
    """Esikler diger kisilerden; video basina sayim hatasi ve tekrar eslesme orani."""
    hata, bulunan, toplam, fazla = [], 0, 0, 0
    for k in sorted({v["kisi"] for v in veri}):
        parametre = _esikler([v for v in veri if v["kisi"] != k])
        for v in (v for v in veri if v["kisi"] == k):
            sayac = TekrarSayaci(*parametre)
            biten = [r for r in (sayac.ekle(n / 30.0, x)
                                 for n, x in zip(v["kare"], yumusat(v["sinyal"]))) if r]
            hata.append(abs(len(biten) - len(v["gercek"])))
            eslesen = set()
            for bas, son in biten:
                # tekrar, onayindan once gercek bir tekrarla ortusuyorsa eslesir
                j = [i for i, (a, b) in enumerate(v["gercek"]) if a <= son * 30 and bas * 30 <= b]
                j = [i for i in j if i not in eslesen]
                if j:
                    eslesen.add(j[0])
                else:
                    fazla += 1
            bulunan += len(eslesen)
            toplam += len(v["gercek"])
    return {"video": len(hata), "ortalama_mutlak_hata": round(float(np.mean(hata)), 2),
            "tam_dogru_video": round(float(np.mean(np.array(hata) == 0)), 3),
            "tekrar_bulma": round(bulunan / max(toplam, 1), 3), "fazla_tekrar": fazla}


def main():
    alt = alt_turler()
    sonuc, modeller = {}, {}
    for ex, ad in ADLAR.items():
        m = mocap(ex, alt)
        olculer = [k for k in m[0] if k not in _ALAN]
        mo = ozet(m, olculer)
        en_iyi = max(olculer, key=lambda o: mo[o]["auc"] or 0)
        tel = telefon(ex, alt)
        tel_ozet = {}
        for grup in sorted({s["yon"] for s in tel}):
            g = [s for s in tel if s["yon"] == grup]
            tel_ozet[grup] = {"n": len(g), "olculer": ozet(g, olculer),
                              "kural": kisi_disarida_dogruluk(g, en_iyi),
                              "birlesik": birlesik_kisi_disarida(g, olculer)}
        sayim = sayim_verisi(ex)
        sonuc_sayim = sayim_kisi_disarida(sayim)
        # canli mod icin son modeller: aci grubu basina; kisi-disarida AUC >= OLCULEBILIR_AUC
        # olmayan grupta model yok -> canlida "bu acidan olculemez" (0038)
        gruplar = {}
        for grup in ("front", "half-profile", "profile"):
            g = [s_ for s_ in tel if s_["yon"] == grup]
            deg = birlesik_kisi_disarida(g, olculer) if len(g) >= 20 else None
            kayit = {"auc": None if deg is None else deg["auc"], "n": len(g)}
            if deg and deg["auc"] is not None and deg["auc"] >= OLCULEBILIR_AUC:
                X = np.array([[s_[o] for o in olculer] for s_ in g], float)
                y = np.array([s_["yanlis"] for s_ in g], bool)
                kayit["model"] = birlesik_model().fit(X, y)
            gruplar[grup] = kayit
        modeller[ex] = {"ad": ad, "olculer": olculer, "sayac": _esikler(sayim),
                        "gruplar": gruplar}
        sonuc[ad] = {"tekrar": len(m), "yanlis": int(sum(s["yanlis"] for s in m)),
                     "sayim": sonuc_sayim,
                     "mocap": mo, "en_iyi_mocap": en_iyi,
                     "mocap_kural": kisi_disarida_dogruluk(m, en_iyi),
                     "mocap_birlesik": birlesik_kisi_disarida(m, olculer), "telefon": tel_ozet}
        print(f"\n== {ad}: {len(m)} tekrar, {sonuc[ad]['yanlis']} yanlis")
        for o in sorted(olculer, key=lambda o: -(mo[o]["auc"] or 0)):
            satir = f"  {o:15s} mocap {mo[o]['auc']}"
            for grup, v in tel_ozet.items():
                satir += f" | {grup} {v['olculer'][o]['auc']}"
            print(satir)
        print(f"  sayim (telefon, kisi-disarida): {sonuc_sayim}")
        print(f"  kural ({en_iyi}): mocap {sonuc[ad]['mocap_kural']}")
        print(f"  birlesik: mocap {sonuc[ad]['mocap_birlesik']}")
        for grup, v in tel_ozet.items():
            print(f"    telefon {grup} (n={v['n']}): kural {v['kural']}")
            print(f"    telefon {grup} (n={v['n']}): birlesik {v['birlesik']}")
    Path("out/hareket_formu.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1))
    import joblib
    joblib.dump(modeller, "out/hareket_formu_modelleri.joblib")


if __name__ == "__main__":
    main()
