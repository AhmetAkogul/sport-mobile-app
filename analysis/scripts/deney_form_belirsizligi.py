#!/usr/bin/env python3
"""Deney: eklem konum belirsizligi form kararini ne zaman imkansiz kilar?

Tezin tam ortasindaki soru. Duzenek 2,8 mm'de olcuyor, telefon literaturde
146-249 mm'de. Bu milimetreler **dereceye** cevrilmeden form karari hakkinda
konusulamaz: karar acida veriliyor, hata konumda olculuyor.

Bu betik ikisi arasindaki donusumu olcer ve her iki katmanin karar
cozunurlugunu cikarir -- yani "esikten ne kadar uzakta olursa karar
verilebilir" sorusunun sayisal cevabini.

Donanim gerekmez, deterministiktir (tohum sabit). Cikti: out/ + docs/deney/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.form import (
    VARSAYILAN_ESIKLER, VARSAYILAN_K, form_degerlendir, sentetik_durus,
)

# Taranan eklem konum belirsizlikleri (1-sigma, mm).
# 2,8 mm duzenegin olculmus degeri (docs/deney/2026-09-22-triangulasyon-butcesi.md),
# 146 ve 249 mm literaturdeki tek kamera 3B poz hatasinin alt ve ust ucu.
SIGMALAR_MM = [0.5, 1.0, 2.8, 5.0, 10.0, 20.0, 50.0, 146.0, 249.0]
DUZENEK_MM = 2.8
TELEFON_MM = (146.0, 249.0)

# Sinanan durus: esigi 2 derece asan bir valgus. Esige yakin secildi, cunku
# kararin zorlastigi yer tam olarak sinirin yanidir.
GERCEK_VALGUS = 12.0
ESIK = VARSAYILAN_ESIKLER["diz_valgusu_sag"].deger

N_ORNEK = 2000
TOHUM = 20260922

# Bu esigin uzerindeki sacilim sayi olarak okunmamali: acilar -180..180
# araliginda sarmalandigi icin dogrusal standart sapma anlamini yitirir.
# Pratikte "olcum bilgi tasimiyor" demektir.
DOYGUNLUK_DERECE = 45.0


def tara() -> list[dict]:
    """Her belirsizlik degeri icin acisal sacilim ve karar."""
    iskelet = sentetik_durus(valgus_sag=GERCEK_VALGUS)
    satirlar = []
    for sigma_mm in SIGMALAR_MM:
        rapor = form_degerlendir(iskelet, konum_belirsizligi_m=sigma_mm / 1000.0,
                                 n_ornek=N_ORNEK, seed=TOHUM)
        olcumler = {
            ad: {
                "belirsizlik_derece": round(o.belirsizlik, 3),
                "doygun": bool(o.belirsizlik > DOYGUNLUK_DERECE),
            }
            for ad, o in rapor.olcumler.items()
        }
        valgus = rapor.olcumler["diz_valgusu_sag"]
        satirlar.append({
            "eklem_belirsizligi_mm": sigma_mm,
            "olcumler": olcumler,
            # Karar cozunurlugu: esikten en az bu kadar uzakta olan bir kusur
            # karara baglanabilir. k=2 guvenlik payindan geliyor.
            "karar_cozunurlugu_derece": round(VARSAYILAN_K * valgus.belirsizlik, 3),
            "valgus_karari": str(valgus.karar),
        })
    return satirlar


def cizim(satirlar: list[dict], hedef: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [s["eklem_belirsizligi_mm"] for s in satirlar]
    adlar = list(satirlar[0]["olcumler"])
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(*TELEFON_MM, color="#d62728", alpha=0.12,
               label="telefon (literatür, 146–249 mm)")
    ax.axvline(DUZENEK_MM, color="#2ca02c", ls="--", lw=1.5,
               label=f"düzenek (ölçülmüş, {DUZENEK_MM} mm)")
    ax.axhline(ESIK, color="#333", ls=":", lw=1.5,
               label=f"valgus eşiği ({ESIK:.0f}°)")

    for ad in adlar:
        y = [s["olcumler"][ad]["belirsizlik_derece"] for s in satirlar]
        ax.plot(x, y, marker="o", ms=4, label=ad.replace("_", " "))

    ax.axhspan(DOYGUNLUK_DERECE, 1e4, color="#888", alpha=0.10)
    ax.text(0.6, DOYGUNLUK_DERECE * 1.15, "doygun — açı sarmalanıyor, bilgi yok",
            fontsize=8, color="#555")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 320)
    ax.set_ylim(0.05, 200)
    ax.set_xlabel("eklem konum belirsizliği (1σ, mm)")
    ax.set_ylabel("açısal belirsizlik (1σ, derece)")
    ax.set_title("Form kararı ne zaman verilemez?")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(hedef, dpi=150)
    plt.close(fig)


def main() -> None:
    kok = Path(__file__).resolve().parent.parent
    satirlar = tara()

    duzenek = next(s for s in satirlar if s["eklem_belirsizligi_mm"] == DUZENEK_MM)
    telefon = next(s for s in satirlar if s["eklem_belirsizligi_mm"] == TELEFON_MM[0])

    sonuc = {
        "experiment": "form_karari_belirsizligi",
        "description": (
            "Eklem konum belirsizligi acisal belirsizlige tasinir; form kararinin "
            "hangi olcum kalitesinde verilebildigi cikarilir."
        ),
        "physical_validation": False,
        "gercek_valgus_derece": GERCEK_VALGUS,
        "esik_derece": ESIK,
        "guvenlik_payi_k": VARSAYILAN_K,
        "n_ornek": N_ORNEK,
        "tohum": TOHUM,
        "doygunluk_esigi_derece": DOYGUNLUK_DERECE,
        "tarama": satirlar,
        "ozet": {
            "duzenek_karar_cozunurlugu_derece": duzenek["karar_cozunurlugu_derece"],
            "duzenek_karari": duzenek["valgus_karari"],
            "telefon_karar_cozunurlugu_derece": telefon["karar_cozunurlugu_derece"],
            "telefon_karari": telefon["valgus_karari"],
        },
    }

    cikti = kok / "out"
    cikti.mkdir(exist_ok=True)
    (cikti / "form_belirsizligi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    cizim(satirlar, kok / "docs/deney/2026-09-22-form-karari-belirsizligi.png")
    print(json.dumps(sonuc["ozet"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
