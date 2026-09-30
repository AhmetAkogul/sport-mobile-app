"""EC3D squat: geometrik diz valgusu karari, yazarlarin dogru/yanlis etiketine karsi.

Soru (0070 iddia 2'nin karar kismi, gercek hareket verisinde): coklu kameradan
kurulmus 3B iskeletle **bizim** valgus kurali, "dogru yapilmis" ile "yanlis
yapilmis" squat tekrarlarini ne kadar ayiriyor?

- Tekrar karari: tekrarin herhangi bir karesinde sag ya da sol diz valgusu
  esigi (0010: 10 derece) asarsa KUSURLU, yoksa DOGRU (yalin esik).
- Referans: EC3D talimat etiketi. Adlar yazarlarin makalesinden (Zhao ve ark.,
  ACCV 2022, Tablo 1; "dizler ice" 23 dizi, bizim etiket 3 sayimiyla ayni):
  1 dogru, 2 ayaklar cok genis, 3 dizler ice, 4 yeterince inmiyor, 5 govde
  one egik; 10 makalede adlandirilmiyor. Valgus kuralinin asil kiyasi
  **3'e karsi 1**dir; diger etiketler valgus disi hatalardir ve etiket basina
  AUC kuralin onlara tepki vermedigini gosterir.
- Tekrar degeri isaretli en buyuk valgustur: dizin disa acilmasi (varus,
  negatif) kusur sayilmaz (0010 tek yonlu esik).
- Birim normalize oldugu icin yalnizca acilar kullanilir.

    python scripts/deney_ec3d.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.form import Karar, form_degerlendir  # noqa: E402
from eval.protokol import Oge, Sayim, kisi_agirlikli  # noqa: E402
from veri.ec3d import kareyi_cevir, tekrarlar  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
ETIKETLER = {1: "dogru", 2: "ayaklar_cok_genis", 3: "dizler_ice", 4: "yeterince_inmiyor",
             5: "govde_one_egik", 10: "adsiz"}
DIZLER_ICE = 3
VALGUS = ("diz_valgusu_sag", "diz_valgusu_sol")


def tekrar_karari(kareler: np.ndarray) -> tuple[str, float]:
    """(kare, 3, 25) -> (karar, tekrar boyunca isaretli en buyuk valgus derece)."""
    en_buyuk, kusurlu = -np.inf, False
    for poz in kareler:
        rapor = form_degerlendir(kareyi_cevir(poz))
        for ad in VALGUS:
            o = rapor.olcumler[ad]
            if np.isfinite(o.deger):
                en_buyuk = max(en_buyuk, float(o.deger))
            kusurlu |= o.karar is Karar.KUSURLU
    return ("kusurlu" if kusurlu else "dogru"), en_buyuk


def main() -> None:
    reps = tekrarlar(KOK / "data/dis/ec3d/data_3D.pickle", "SQUAT")
    ogeler, satir = [], []
    for (_, kisi, lab, tek), kareler in sorted(reps.items()):
        karar, deger = tekrar_karari(kareler)
        if lab in (1, DIZLER_ICE):
            ref = "dogru" if lab == 1 else "kusurlu"
            ogeler.append(Oge(kisi, f"{kisi}-{lab}", ref, karar))
        satir.append({"kisi": kisi, "etiket": lab, "tekrar": tek, "kare": len(kareler),
                      "karar": karar, "en_buyuk_valgus_derece": round(deger, 2)})

    sayim = Sayim.ciftlerden((o.referans, o.telefon) for o in ogeler)
    etiket_basina = {}
    for lab in sorted({s["etiket"] for s in satir}):
        v = [s for s in satir if s["etiket"] == lab]
        d = np.array([s["en_buyuk_valgus_derece"] for s in v])
        etiket_basina[str(lab)] = {
            "ad": ETIKETLER.get(lab, "bilinmiyor"), "tekrar": len(v),
            "kusurlu_denen_oran": round(float(np.mean([s["karar"] == "kusurlu" for s in v])), 3),
            "valgus_medyan": round(float(np.median(d)), 2),
            "valgus_p90": round(float(np.percentile(d, 90)), 2)}
    # Esikten bagimsiz ayirma gucu: etiket basina, dogru (1) tekrarlara karsi AUC.
    def _auc(lab: int) -> float | None:
        pos = [s["en_buyuk_valgus_derece"] for s in satir if s["etiket"] == lab]
        neg = [s["en_buyuk_valgus_derece"] for s in satir if s["etiket"] == 1]
        if not pos or not neg:
            return None
        return round(float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])), 3)

    for lab in etiket_basina:
        if int(lab) != 1:
            etiket_basina[lab]["auc_dogruya_karsi"] = _auc(int(lab))

    sonuc = {
        "deney": "ec3d_squat_valgus",
        "kaynak": "EC3D data_3D.pickle (4 kisi, 4 GoPro, normalize 3B)",
        "esik_derece": 10.0, "n_tekrar": len(satir), "n_kiyas": len(ogeler), "n_kisi": len({s["kisi"] for s in satir}),
        "protokol_0070": sayim.oranlar(), "kisi_agirlikli": kisi_agirlikli(ogeler),
        "kiyas": "dizler_ice (3) vs dogru (1); valgus tanimi 0014",
        "etiket_basina": etiket_basina,
        "not": "4 kisi: 0070'e gore genelleme yok (on kanit). Tanim bu veride secildi.",
    }
    (KOK / "out").mkdir(exist_ok=True)
    (KOK / "out/ec3d_squat.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: sonuc[k] for k in ("n_tekrar", "n_kiyas", "n_kisi", "etiket_basina")},
                     ensure_ascii=False))
    o = sonuc["protokol_0070"]
    print({k: o[k] for k in ("M", "N", "D", "C", "A", "K", "A_dengeli")})


if __name__ == "__main__":
    main()
