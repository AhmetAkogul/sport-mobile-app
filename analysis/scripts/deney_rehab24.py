"""REHAB24-6 squat: 0014 valgus kurali bagimsiz 9 kiside, marker'li referansla.

EC3D'de secilen tanimin (0014) ilk bagimsiz sinamasi. Iki soru:

1. **Ozgulluk** -- fizyoterapistin "dogru" dedigi squat tekrarlarinda kural ne
   siklikla yanlis alarm veriyor? Dogru tekrar kusursuzdur (valgus dahil), bu
   yuzden referans burada tam: 0070 A = C/D dogrudan ozgulluktur.
2. **Betimsel** -- "yanlis" tekrarlarin ne kadari valgus olarak isaretleniyor?
   REHAB24-6'da yanlis etiketi tek bir kusur degil (her kisiye farkli hata
   yaptirildi); duyarlilik **olculemez**, yalnizca kisi basina tablo verilir.

Karsilastirma: eski tanim (0010'daki frontal izdusum acisi, FPPA) ayni
tekrarlarda; yalnizca bu betikte, raporlama icin yeniden yazildi.

    python scripts/deney_rehab24.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.form import Karar, form_degerlendir, govde_cercevesi  # noqa: E402
from eval.protokol import Oge, Sayim, kisi_agirlikli, kisi_bootstrap  # noqa: E402
from veri.rehab24 import kareyi_cevir, tekrar_kareleri, tekrarlar_oku  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data/dis/rehab24_6"
ESIK = 10.0


def _eski_fppa(isk, taraf: str) -> float:
    """0010'daki eski tanim: govde frontal duzlemine izdusum acisinin sapmasi."""
    c = govde_cercevesi(isk)
    if c is None:
        return float("nan")
    i, P = isk.tanim.indeks, isk.noktalar
    h, k, a = (P[i(f"{taraf}_{e}")] for e in ("kalca", "diz", "ayak_bilegi"))
    y = [np.array([(p - c.merkez) @ c.yanal, (p - c.merkez) @ c.yukari]) for p in (h, k, a)]
    u, v = y[0] - y[1], y[2] - y[1]
    if min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-9:
        return float("nan")
    sapma = 180.0 - np.degrees(np.arccos(np.clip(
        u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    bacak = y[2] - y[0]
    t = (y[1] - y[0]) @ bacak / (bacak @ bacak)
    yanal = (y[1] - (y[0] + t * bacak))[0]
    return float(np.sign(yanal * (1.0 if taraf == "sag" else -1.0)) * sapma)


def tekrar_olc(kareler: np.ndarray) -> dict:
    yeni, eski, kusurlu, oran = -np.inf, -np.inf, False, 1.0
    yeni_min, eski_min = np.inf, np.inf
    for q in kareler:
        isk = kareyi_cevir(q)
        rapor = form_degerlendir(isk)
        P, i = isk.noktalar, isk.tanim.indeks
        for s in ("sag", "sol"):
            o = rapor.olcumler[f"diz_valgusu_{s}"]
            if np.isfinite(o.deger):
                yeni, yeni_min = max(yeni, float(o.deger)), min(yeni_min, float(o.deger))
            kusurlu |= o.karar is Karar.KUSURLU
            e = _eski_fppa(isk, s)
            if np.isfinite(e):
                eski, eski_min = max(eski, e), min(eski_min, e)
            h, k, a = (P[i(f"{s}_{x}")] for x in ("kalca", "diz", "ayak_bilegi"))
            oran = min(oran, float(np.linalg.norm(a - h)
                                   / (np.linalg.norm(k - h) + np.linalg.norm(a - k))))
    return {"yeni": yeni, "eski": eski, "yeni_min": yeni_min, "eski_min": eski_min, "karar": "kusurlu" if kusurlu else "dogru",
            "eski_karar": "kusurlu" if eski > ESIK else "dogru", "en_derin_oran": oran}


def _ozet(v) -> dict:
    v = np.asarray(v, float)
    return {"n": int(v.size), "medyan": round(float(np.median(v)), 2),
            "p10": round(float(np.percentile(v, 10)), 2),
            "p90": round(float(np.percentile(v, 90)), 2)}


def main() -> None:
    tekrarlar = tekrarlar_oku(VERI / "Segmentation.csv")
    satir = []
    for t in tekrarlar:
        if t.mocap_hatali:          # 0070 R: referans bozuk, disarida ama sayilir
            satir.append({"kisi": t.kisi, "dogru": t.dogru, "mocap_hatali": True})
            continue
        satir.append({"kisi": t.kisi, "video": t.video, "tekrar": t.tekrar_no, "yon": t.yon,
                      "dogru": t.dogru, "mocap_hatali": False,
                      **tekrar_olc(tekrar_kareleri(VERI, t))})
    gecerli = [s for s in satir if not s["mocap_hatali"]]
    dogrular = [s for s in gecerli if s["dogru"]]
    yanlislar = [s for s in gecerli if not s["dogru"]]

    def ogeler(anahtar: str) -> list[Oge]:
        return [Oge(s["kisi"], s["video"], "dogru", s[anahtar]) for s in dogrular]

    ozgulluk = {}
    for ad, anahtar in (("yeni_0014", "karar"), ("eski_fppa", "eski_karar")):
        og = ogeler(anahtar)
        ozgulluk[ad] = {
            "sayim": Sayim.ciftlerden((o.referans, o.telefon) for o in og).oranlar(),
            "kisi_agirlikli": kisi_agirlikli(og),
            "kisi_bootstrap": kisi_bootstrap(og),
        }

    kisi_tablosu = defaultdict(dict)
    for s in gecerli:
        k = kisi_tablosu[s["kisi"]].setdefault("dogru" if s["dogru"] else "yanlis",
                                               {"n": 0, "kusurlu_denen": 0, "valgus": []})
        k["n"] += 1
        k["kusurlu_denen"] += s["karar"] == "kusurlu"
        k["valgus"].append(s["yeni"])
    for kisi in kisi_tablosu.values():
        for k in kisi.values():
            v = k.pop("valgus")
            k["en_buyuk_valgus_medyan"] = round(float(np.median(v)), 2)

    pos = [s["yeni"] for s in yanlislar]
    neg = [s["yeni"] for s in dogrular]
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

    sonuc = {
        "deney": "rehab24_squat_valgus",
        "kaynak": "REHAB24-6 Ex6, 3d_joints 30 fps (OptiTrack, 41 marker -> 26 eklem, metre)",
        "valgus_tanimi": "0014", "esik_derece": ESIK,
        "n_tekrar": len(satir), "mocap_hatali_R": len(satir) - len(gecerli),
        "n_dogru": len(dogrular), "n_yanlis": len(yanlislar),
        "n_kisi": len({s["kisi"] for s in gecerli}),
        "ozgulluk_dogru_tekrarlarda": ozgulluk,
        "valgus_derece": {"dogru": _ozet(neg), "yanlis": _ozet(pos),
                          "eski_fppa_dogru": _ozet([s["eski"] for s in dogrular]),
                          # Isaretli en buyuk deger iki tanimda da ayni ozgullugu
                          # veriyor; eski tanimin bukulmede bozulmasi varus
                          # yonunde (EC3D ile ayni): tekrar basina en kucuk deger.
                          "en_kucuk_dogru_yeni": _ozet([s["yeni_min"] for s in dogrular]),
                          "en_kucuk_dogru_eski_fppa": _ozet([s["eski_min"] for s in dogrular])},
        "en_derin_bacak_orani": {"dogru": _ozet([s["en_derin_oran"] for s in dogrular]),
                                 "yanlis": _ozet([s["en_derin_oran"] for s in yanlislar])},
        "yanlis_tekrar_kusurlu_denen_oran": round(
            float(np.mean([s["karar"] == "kusurlu" for s in yanlislar])), 4),
        "auc_yanlis_vs_dogru_betimsel": round(auc, 3),
        "kisi_tablosu": dict(sorted(kisi_tablosu.items())),
        "not": ("9 kisi < 10: 0070'e gore on kanit. Yanlis etiketi tek kusur degil; "
                "duyarlilik olculemez. Tanim EC3D'de secildi, bu veri secimde kullanilmadi."),
        "physical_validation": "gercek hareket; referans marker'li mocap iskeleti",
    }
    (KOK / "out").mkdir(exist_ok=True)
    (KOK / "out/rehab24_squat.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: sonuc[k] for k in ("n_tekrar", "mocap_hatali_R", "n_dogru", "n_yanlis",
                                           "n_kisi", "valgus_derece",
                                           "yanlis_tekrar_kusurlu_denen_oran",
                                           "auc_yanlis_vs_dogru_betimsel")}, ensure_ascii=False))
    for ad, o in ozgulluk.items():
        print(ad, {k: o["sayim"][k] for k in ("D", "C", "A")}, "kisi A:",
              o["kisi_agirlikli"]["A"], "CI:", o["kisi_bootstrap"].get("aralik_95", {}).get("A"))


if __name__ == "__main__":
    main()
