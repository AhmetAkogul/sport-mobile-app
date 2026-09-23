#!/usr/bin/env python3
"""Deney: kemik uzunlugu oncusundeki hata manset egrisini ne kadar bozuyor?

`2026-09-22-aci-taramasi.md` deneyinde oncular **yer gercegindendi**: telefonun
kullanicinin kemik uzunluklarini tam bildigi varsayildi. Gercekte bilmez. En iyi
durumda boy bilgisinden antropometrik tablolarla kestirilir ve segment basina
birkac yuzde hata kalir.

Bu varsayim tasiyici: eger %5 oncu hatasi frontal dogrulugu da yikiyorsa,
"telefonu +-35 derece icinde tut" kurali gecersizdir ve telefon hicbir acida
kullanilamaz demektir. Bu betik o soruyu olcer.

Oncu hatasi kemik basina **bagimsiz** ve durus basina yeniden cekiliyor:
kullanicidan kullaniciya degisen bir kestirim hatasi boyle davranir. Ortak bir
olcek hatasi (herkesin boyu %5 yanlis) farkli bir seydir ve tek gorus
kestiriminde yalnizca genel olcegi kaydirir, aci olcumlerini bozmaz.

Donanim gerekmez, deterministiktir (tohum sabit). Cikti: out/ + docs/deney/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.aci_taramasi import aci_taramasi, kemik_onculeri, sentetik_poz_ureteci
from eval.form import sentetik_durus

ACILAR = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
# Oncu hatasi (1-sigma, bagil). %4-5 antropometrik tablolarin boy bilgisinden
# segment uzunlugu kestirirken tipik olarak biraktigi mertebedir.
HATALAR = [0.0, 0.02, 0.05, 0.10]
N_DURUS = 300
TOHUM = 20260922
VALGUS = "diz_valgusu_sag"


def duruslar_uret() -> list:
    rng = np.random.default_rng(TOHUM)
    return [
        sentetik_durus(
            valgus_sag=float(rng.uniform(0.0, 24.0)),
            kalca_hizasi=float(rng.uniform(-16.0, 16.0)),
            govde_rotasyonu=float(rng.uniform(-30.0, 30.0)),
        )
        for _ in range(N_DURUS)
    ]


def bozuk_oncu_ureteci(bagil_hata: float, seed: int):
    """Her durus icin kemik basina bagimsiz bozulmus oncu uretir.

    Hata carpimsal: uzunluk *= (1 + N(0, bagil_hata)). Toplamsal olsaydi kisa
    kemikler (onkol) uzun kemiklerden (uyluk) orantisiz etkilenirdi; kestirim
    hatasi gercekte uzunlukla olcekleniyor.
    """
    rng = np.random.default_rng(seed)

    def uret(durus):
        gercek = kemik_onculeri(durus)
        if bagil_hata <= 0.0:
            return gercek
        return {
            cift: max(1e-3, uzunluk * (1.0 + rng.normal(0.0, bagil_hata)))
            for cift, uzunluk in gercek.items()
        }

    return uret


def cizim(satirlar: list[dict], hedef: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    renkler = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for s, renk in zip(satirlar, renkler):
        x = [a["aci_derece"] for a in s["acilar"]]
        y = [a["dogruluk"] for a in s["acilar"]]
        ax.plot(x, y, marker="o", color=renk,
                label=f"öncü hatası %{s['oncu_hatasi'] * 100:.0f}")
    ax.axhline(0.5, color="#888", ls=":", lw=1)
    ax.text(1.5, 0.52, "yazı tura", fontsize=8, color="#666")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlabel("kamera açısı (0° = önden, 90° = yandan)")
    ax.set_ylabel("doğru karar oranı")
    ax.set_title("Kemik uzunluğu öncüsü bilinmiyorsa eğri ne oluyor?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(hedef, dpi=150)
    plt.close(fig)


def main() -> None:
    kok = Path(__file__).resolve().parent.parent
    duruslar = duruslar_uret()

    satirlar = []
    for hata in HATALAR:
        uret, K = sentetik_poz_ureteci(gurultu_px=0.0, seed=TOHUM)
        sonuc = aci_taramasi(duruslar, ACILAR, uret, K,
                             onculer=bozuk_oncu_ureteci(hata, TOHUM))
        satirlar.append({
            "oncu_hatasi": hata,
            "acilar": [
                {
                    "aci_derece": n.aci_derece,
                    "dogruluk": round(n.sayimlar[VALGUS].dogruluk, 4),
                    "yanlis_karar_orani": round(
                        n.sayimlar[VALGUS].yanlis_karar_orani, 4),
                    "aci_hatasi_derece": round(n.aci_hatasi_derece[VALGUS], 3),
                }
                for n in sonuc.noktalar
            ],
        })

    frontal = {s["oncu_hatasi"]: s["acilar"][0]["dogruluk"] for s in satirlar}
    sonuc_json = {
        "experiment": "oncu_hatasi_duyarliligi",
        "description": (
            "Kemik uzunlugu oncusundeki hatanin form karari dogruluguna etkisi; "
            "manset egrisinin en tasiyici varsayiminin sinanmasi."
        ),
        "physical_validation": False,
        "n_durus": N_DURUS,
        "tohum": TOHUM,
        "olcum": VALGUS,
        "tarama": satirlar,
        "ozet": {
            "frontal_dogruluk": {f"{k:.0%}": v for k, v in frontal.items()},
            "frontal_dusus_yuzde_puan": round(
                (frontal[0.0] - frontal[0.05]) * 100, 1),
        },
    }

    cikti = kok / "out"
    cikti.mkdir(exist_ok=True)
    (cikti / "oncu_hatasi.json").write_text(
        json.dumps(sonuc_json, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    cizim(satirlar, kok / "docs/deney/2026-09-22-oncu-hatasi.png")
    print(json.dumps(sonuc_json["ozet"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
