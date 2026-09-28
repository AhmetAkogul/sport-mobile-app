"""Fitness-AQA squat: telefon hattinin "dizler ice" kararinin bagimsiz sinamasi.

REHAB24-6'da valgus kaymasi bakis acisinin dogrusal fonksiyonu olarak
ogrenildi (`deney_rehab24_tek_gorus.py` -> `aci_kayma_dogrusu`). Burada o
dogru **donmus** olarak uygulanir: Fitness-AQA'da hicbir sey ogrenilmez,
esik de kural da degismez. Gercek spor salonu videolari, bilinmeyen kamera
acisi, uzman etiketi; mocap yok -- olculen karar dogrulugudur, milimetre degil.

Kol: MediaPipe `pose_world` maskesiz (`mediapipe_world_tam`), ham ve
aciya gore duzeltilmis. Tekrar karari REHAB24 ile ayni kural (0014, 10 derece,
0,2 s kesintisiz; kapsama %80). Esikten bagimsiz skor: 0,2 s boyunca
kesintisiz tutulan en buyuk valgus (iki dizin buyugu); kural "skor > esik"
ile esdegerdir.

Etiket: `error_knees_inward.json` bos degilse kusurlu. Kisi kimligi yok
(anahtar onekleri tekil); kisi-kume guven araligi verilemez.

    python scripts/deney_fitness_aqa_valgus.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.form import Karar  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
SQUAT = KOK / "data/dis/fitness_aqa/Squat/Labeled_Dataset"
TESPIT = KOK / "out/fitness_aqa_tespit/mediapipe"
_spec = importlib.util.spec_from_file_location("tg", KOK / "scripts/deney_rehab24_tek_gorus.py")
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)

YONTEM = "mediapipe_world_tam"
ACI_DILIMLERI = ((0, 30), (30, 60), (60, 90.01))


def surekli_tepe(deg: np.ndarray, kk: list[tuple[Karar, Karar]], kayma: float,
                 gerekli: int) -> float:
    """`gerekli` kare kesintisiz tutulan en buyuk (deg - kayma), iki taraf.

    BELIRSIZ kare seriyi keser (tekrar_karari ile ayni). Kural "kusurlu" ancak
    ve ancak bu deger esigi asarsa verir. Hic yeterli seri yoksa -inf.
    """
    en = -np.inf
    for s in range(2):
        v = np.array([d - kayma if k[s] is not Karar.BELIRSIZ and np.isfinite(d) else np.nan
                      for d, k in zip(deg[:, s], kk)])
        for i in range(len(v) - gerekli + 1):
            w = v[i:i + gerekli]
            if np.isfinite(w).all():
                en = max(en, float(w.min()))
    return en


def auc(pozitif, negatif) -> float | None:
    """Mann-Whitney AUC; -inf (skor yok) en dusuk sayilir."""
    p, n = np.asarray(pozitif, float), np.asarray(negatif, float)
    if not p.size or not n.size:
        return None
    return float(((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])).mean())


def ozet(satirlar: list[dict], anahtar: str) -> dict:
    poz = [s for s in satirlar if s["etiket_kusurlu"]]
    neg = [s for s in satirlar if not s["etiket_kusurlu"]]
    karar = f"karar_{anahtar}"
    kararli_poz = [s for s in poz if s[karar] != "belirsiz"]
    kararli_neg = [s for s in neg if s[karar] != "belirsiz"]
    return {
        "n": len(satirlar), "n_kusurlu": len(poz),
        "belirsiz": sum(s[karar] == "belirsiz" for s in satirlar),
        "duyarlilik": (round(sum(s[karar] == "kusurlu" for s in kararli_poz)
                             / len(kararli_poz), 3) if kararli_poz else None),
        "ozgulluk": (round(sum(s[karar] == "dogru" for s in kararli_neg)
                           / len(kararli_neg), 3) if kararli_neg else None),
        "yakalanan": f'{sum(s[karar] == "kusurlu" for s in kararli_poz)}/{len(kararli_poz)}',
        "yanlis_alarm": f'{sum(s[karar] == "kusurlu" for s in kararli_neg)}/{len(kararli_neg)}',
        "auc": (round(a, 3) if (a := auc([s[f"skor_{anahtar}"] for s in poz],
                                         [s[f"skor_{anahtar}"] for s in neg])) is not None
                else None),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--son-olcum", action="store_true",
                    help="resmi test bolumunu da isle (0034 Karar 4: yalniz bir kez, son olcumde)")
    son_olcum = ap.parse_args().son_olcum
    rehab = json.loads((KOK / "out/rehab24_tek_gorus.json").read_text())
    dogru = rehab.get("aci_kayma_dogrusu", {}).get(YONTEM)
    if dogru is None:
        raise RuntimeError(f"REHAB24 ciktisinda {YONTEM} aci dogrusu yok; once "
                           "deney_rehab24_tek_gorus.py kosun")
    etiket = json.loads((SQUAT / "Labels/error_knees_inward.json").read_text())
    test = set(json.loads((SQUAT / "Splits/test_keys.json").read_text()))
    gerekli = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))

    satirlar, eksik = [], 0
    for anahtar, araliklar in sorted(etiket.items()):
        if anahtar in test and not son_olcum:
            continue                     # 0034 Karar 4: test bolumu son olcume saklanir
        yol = TESPIT / f"{anahtar}.npz"
        if not yol.exists():
            eksik += 1
            continue
        d = dict(np.load(yol))
        if abs(float(d["fps"]) - tg.FPS) > 0.5:
            raise RuntimeError(f"{anahtar}: fps {d['fps']} != {tg.FPS}; sure kurali kayar")
        kk, deg, aci = [], [], []
        for i in range(len(d["tespit"])):
            isk = tg._dunya_tam_iskeleti(d, i)
            k, v = tg._valgus_kararlari(isk)
            kk.append(k)
            deg.append(v)
            aci.append(tg.govde_acisi(isk.noktalar))
        deg, aci = np.array(deg, float), np.array(aci, float)
        aci_med = float(np.median(aci[np.isfinite(aci)])) if np.isfinite(aci).any() else np.nan
        kayma = (dogru["b0"] + dogru["b1"] * aci_med) if np.isfinite(aci_med) else dogru["ortalama"]
        satirlar.append({
            "anahtar": anahtar, "test_bolumu": anahtar in test,
            "etiket_kusurlu": bool(araliklar), "govde_acisi_derece": aci_med,
            "kayma_derece": float(kayma),
            "karar_ham": tg.tekrar_karari(tg.kaydirilmis_kararlar(kk, deg, 0.0)),
            "karar_aci": tg.tekrar_karari(tg.kaydirilmis_kararlar(kk, deg, kayma)),
            "skor_ham": surekli_tepe(deg, kk, 0.0, gerekli),
            "skor_aci": surekli_tepe(deg, kk, kayma, gerekli),
        })

    # Kural ile skorun esdegerligi: karar "kusurlu" <=> skor > esik (her satirda).
    esik = tg.VARSAYILAN_ESIKLER[tg.VALGUS[0]].deger
    for s in satirlar:
        for a in ("ham", "aci"):
            if (s[f"karar_{a}"] == "kusurlu") != (s[f"skor_{a}"] > esik):
                raise RuntimeError(f"{s['anahtar']}: skor ile karar tutarsiz ({a})")

    def kume(ss):
        return {a: ozet(ss, a) for a in ("ham", "aci")}

    sonuc = {
        "deney": "fitness_aqa_valgus",
        "yontem": YONTEM, "aci_kayma_dogrusu_rehab24": dogru,
        "esik_derece": esik, "kusur_suresi_kare": gerekli,
        "eksik_tespit": eksik,
        "tamami": kume(satirlar),
        "test_bolumu": (kume([s for s in satirlar if s["test_bolumu"]]) if son_olcum
                        else "saklandi (0034 Karar 4; --son-olcum)"),
        "aci_dilimi": {f"{lo}-{min(hi, 90):.0f}": kume(
            [s for s in satirlar if lo <= s["govde_acisi_derece"] < hi])
            for lo, hi in ACI_DILIMLERI},
        "govde_acisi_derece": tg._ozet([s["govde_acisi_derece"] for s in satirlar]),
        "not": ("Fitness-AQA'da hicbir sey ogrenilmedi: aci dogrusu REHAB24'ten donmus. "
                "Kisi kimligi yok: kisi-kume araligi verilemez. Etiket uzman gozlemi, "
                "0014 tanimiyla ayni kavram olmayabilir."),
    }
    (KOK / "out/fitness_aqa_valgus.json").write_text(
        json.dumps(tg._temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (KOK / "out/fitness_aqa_valgus_satirlar.json").write_text(
        json.dumps(tg._temiz(satirlar), ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8")
    print("eksik tespit:", eksik, "| govde acisi:", sonuc["govde_acisi_derece"])
    for ad in ("tamami", "test_bolumu") if son_olcum else ("tamami",):
        for a, o in sonuc[ad].items():
            print(f"{ad:12s} {a:4s}", o)
    for dilim, v in sonuc["aci_dilimi"].items():
        for a, o in v.items():
            print(f"aci {dilim:6s} {a:4s}", o)


if __name__ == "__main__":
    main()
