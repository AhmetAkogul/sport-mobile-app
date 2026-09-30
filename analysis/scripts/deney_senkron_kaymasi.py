#!/usr/bin/env python3
"""Deney: senkron kaymasi kac mm hata demek?

Kapi 2 olcutu ("Senkron kaymasi -- olculmus ve mm cinsinden ifade edilmis") ve
Risk 3'un azaltma adimi ("kaymayi olc, mm'ye cevir, sinir olarak raporla").

Ciktisi tek bir sayi degil, bir **gereksinim**: Kapi 2'nin 10 mm butcesini
tutturmak icin kameralar arasi kayma kac ms'nin altinda kalmali? Cevap harekete
bagli oldugu icin uc hizda ayri ayri veriliyor.

Donanim gerekmez, deterministiktir (tohum sabit). Cikti: out/ + docs/deney/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import rig_yay, sahne_uret
from eval.senkron import REJIMLER, azami_kayma_ms, kayma_taramasi

# Kare periyotlari referans olarak tarandi: 120 fps = 8,3 ms, 60 fps = 16,7 ms,
# 30 fps = 33,3 ms. Serbest kosan kameralarda kayma kare periyoduna kadar cikar.
KAYMALAR_MS = [0.0, 2.0, 5.0, 8.3, 10.0, 16.7, 20.0, 33.3, 50.0]

# Squat/lunge sirasinda diz ve kalca ~0,5-1 m/s; hizli tekrarlarda bilek 2 m/s'yi
# bulur. Ust sinir kasitli olarak zorlayici secildi.
HIZLAR_M_S = [0.5, 1.0, 2.0]

ESIK_MM = 10.0            # Kapi 2'nin 3B konum hatasi butcesi
N_ORNEK = 300
TOHUM = 20260922


def duzenek():
    """Gercekci kalibrasyon (0,25 px) ile 4 kameralik referans rig."""
    kameralar = rig_yay(odak_px=900.0)
    s = sahne_uret(BoardSpec(5, 7), kameralar=kameralar, n_kare=20,
                   gurultu_px=0.25, dagilim="genis", seed=8080)
    kalib = kalibre_et(s.obj_noktalari, s.img_noktalari,
                       [c.boyut for c in kameralar], s.mask)
    return kalib, kameralar


def cizim(sonuclar, hedef: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    renkler = {0.5: "#2ca02c", 1.0: "#1f77b4", 2.0: "#d62728"}
    basliklar = {"tek_kamera": "bir kamera geç kalıyor",
                 "dagilmis": "kameralar serbest koşuyor"}

    for ax, rejim in zip(eksenler, REJIMLER):
        for hiz in HIZLAR_M_S:
            nokta = sorted((s for s in sonuclar
                            if s.rejim == rejim and s.hiz_m_s == hiz),
                           key=lambda s: s.kayma_ms)
            ax.plot([s.kayma_ms for s in nokta], [s.p95_mm for s in nokta],
                    marker="o", ms=4, color=renkler[hiz], label=f"{hiz} m/s")
        ax.axhline(ESIK_MM, color="#333", ls=":", lw=1.5,
                   label=f"Kapı 2 bütçesi ({ESIK_MM:.0f} mm)")
        ax.axvline(33.3, color="#888", ls="--", lw=1)
        ax.text(33.8, 0.6, "30 fps", fontsize=8, color="#666", rotation=90)
        ax.axvline(16.7, color="#bbb", ls="--", lw=1)
        ax.text(17.2, 0.6, "60 fps", fontsize=8, color="#999", rotation=90)
        ax.set_xlabel("kameralar arası kayma (ms)")
        ax.set_title(basliklar[rejim], fontsize=10)
        ax.grid(alpha=0.25)

    eksenler[0].set_ylabel("3B konum hatası, p95 (mm)")
    eksenler[0].set_yscale("log")
    eksenler[0].legend(fontsize=8)
    fig.suptitle("Senkron kayması kaç mm hata demek?", fontsize=12)
    fig.tight_layout()
    fig.savefig(hedef, dpi=150)
    plt.close(fig)


def main() -> None:
    kok = Path(__file__).resolve().parent.parent
    kalib, kameralar = duzenek()
    sonuclar = kayma_taramasi(kalib, kameralar, KAYMALAR_MS, HIZLAR_M_S,
                              n_ornek=N_ORNEK, seed=TOHUM)

    gereksinim = {
        rejim: {
            f"{hiz} m/s": azami_kayma_ms(sonuclar, rejim, hiz, esik_mm=ESIK_MM)
            for hiz in HIZLAR_M_S
        }
        for rejim in REJIMLER
    }

    def bul(rejim, hiz, kayma):
        return next(s for s in sonuclar if s.rejim == rejim
                    and s.hiz_m_s == hiz and s.kayma_ms == kayma)

    cikti_json = {
        "experiment": "senkron_kaymasi_mm",
        "description": (
            "Kameralar arasi zaman kaymasinin 3B konum hatasina cevrimi ve "
            "Kapi 2 butcesinden turetilen azami kayma gereksinimi."
        ),
        "physical_validation": False,
        "esik_mm": ESIK_MM,
        "n_ornek": N_ORNEK,
        "tohum": TOHUM,
        "tarama": [s.ozet() for s in sonuclar],
        "azami_kayma_ms": gereksinim,
        "ozet": {
            "serbest_30fps_1ms_p95_mm": bul("dagilmis", 1.0, 33.3).p95_mm,
            "serbest_60fps_1ms_p95_mm": bul("dagilmis", 1.0, 16.7).p95_mm,
            "azami_kayma_1ms_dagilmis": gereksinim["dagilmis"]["1.0 m/s"],
            "azami_kayma_2ms_dagilmis": gereksinim["dagilmis"]["2.0 m/s"],
        },
    }

    cikti = kok / "out"
    cikti.mkdir(exist_ok=True)
    (cikti / "senkron_kaymasi.json").write_text(
        json.dumps(cikti_json, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    cizim(sonuclar, kok / "docs/deney/2026-09-22-senkron-kaymasi.png")
    print(json.dumps(cikti_json["ozet"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
