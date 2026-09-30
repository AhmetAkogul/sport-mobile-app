#!/usr/bin/env python3
"""Deney: gurultu ve board poz cesitliligi, kalibrasyon belirsizligini nasil etkiliyor?

KOD-PLANI.md Adim 2'nin ciktisi. Donanim gerektirmez, deterministiktir
(tohumlar sabit), `make reproduce` ile yeniden uretilir.

Iki soru:
  1. Gurultusuz sentetik veride parametreleri geri buluyor muyuz?  (kabul: hata < %0.1)
  2. Gurultu ve poz cesitliligi, parametre sacilimini nasil degistiriyor?

Cikti: out/sentetik_kalibrasyon.json + docs/deney/ altina sekil.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Hem "python3 scripts/deney_...py" hem "python3 -m scripts.deney_..." calissin diye
# proje kokunu path'e al. Betigi calistiran kisi calisma dizinini dusunmek zorunda kalmasin.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calib.board import BoardSpec
from calib.synthetic import rig_yay, sahne_uret
from uncertainty.montecarlo import sacilim_olc

SPEC = BoardSpec(5, 7)
GERCEK_ODAK = 900.0
GURULTULER = (0.0, 0.1, 0.25, 0.5, 1.0)
DAGILIMLAR = ("dar", "genis")
N_TEKRAR = 15
N_KARE = 20
TOHUM = 4242


def kosu(gurultu: float, dagilim: str, kameralar) -> dict:
    def uretec(seed):
        return sahne_uret(SPEC, kameralar=kameralar, n_kare=N_KARE,
                          gurultu_px=gurultu, dagilim=dagilim, seed=TOHUM + seed)
    sonuc = sacilim_olc(uretec, n_tekrar=N_TEKRAR,
                        gercekler={"fx": GERCEK_ODAK, "fy": GERCEK_ODAK})
    return sonuc.sozluk() | {"gurultu_px": gurultu, "dagilim": dagilim}


def sekil_ciz(satirlar: list[dict], hedef: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (sol, sag) = plt.subplots(1, 2, figsize=(11, 4.2))
    renk = {"dar": "#c0392b", "genis": "#2471a3"}
    etiket = {"dar": "dar poz dağılımı", "genis": "geniş poz dağılımı"}

    for dag in DAGILIMLAR:
        alt = [s for s in satirlar if s["dagilim"] == dag]
        g = [s["gurultu_px"] for s in alt]
        std = [s["parametreler"]["fx"]["std"] for s in alt]
        rms = [s["rms_ortalama"] for s in alt]
        sol.plot(g, std, "o-", color=renk[dag], label=etiket[dag])
        sag.plot(g, rms, "o-", color=renk[dag], label=etiket[dag])

    # Log olcek sart: degerler dort buyukluk mertebesine yayiliyor. Dogrusal
    # olcekte tek bir aykiri deger (dar/1.0 px) paneli eziyor ve asil bulgu --
    # dusuk gurultude dar dagilimin RMS'inin DAHA IYI gorunmesi -- kayboluyor.
    for eksen in (sol, sag):
        eksen.set_yscale("log")
        eksen.set_xlabel("tespit gürültüsü (piksel)")
        eksen.grid(alpha=0.3, which="both")
        eksen.legend()

    sol.set_ylabel("odak uzaklığı saçılımı, std (piksel, log)")
    sol.set_title("Belirsizlik — düşük olan iyi")

    sag.set_ylabel("yeniden izdüşüm RMS (piksel, log)")
    sag.set_title("Uyum — düşük RMS iyi kalibrasyon DEĞİL")

    # Bulguyu sekil uzerinde isaretle: 0.25 px'te dar dagilim daha iyi "uyuyor"
    # ama belirsizligi kat kat yuksek.
    dar_025 = next(s for s in satirlar if s["dagilim"] == "dar" and s["gurultu_px"] == 0.25)
    genis_025 = next(s for s in satirlar if s["dagilim"] == "genis" and s["gurultu_px"] == 0.25)
    sag.annotate(
        f"0,25 px'te dar daha düşük RMS\n({dar_025['rms_ortalama']:.2f} < "
        f"{genis_025['rms_ortalama']:.2f}) ama\nbelirsizliği "
        f"{dar_025['parametreler']['fx']['std'] / genis_025['parametreler']['fx']['std']:.0f}× daha kötü",
        xy=(0.25, dar_025["rms_ortalama"]), xytext=(0.30, 0.02),
        fontsize=8, color="#7b241c",
        arrowprops=dict(arrowstyle="->", color="#7b241c", lw=0.8))

    fig.suptitle("Sentetik kalibrasyon — board poz çeşitliliği belirsizliği belirliyor",
                 fontsize=11)
    fig.tight_layout()
    hedef.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hedef, dpi=150)
    plt.close(fig)
    return hedef


def main() -> int:
    kameralar = rig_yay(odak_px=GERCEK_ODAK)
    satirlar = [kosu(g, d, kameralar) for d in DAGILIMLAR for g in GURULTULER]

    print(f"{'dagilim':8} {'gurultu':>8} {'rms px':>9} {'fx ort':>9} "
          f"{'fx std':>9} {'sapma %':>9} {'yakinsayan':>11}")
    for s in satirlar:
        fx = s["parametreler"]["fx"]
        print(f"{s['dagilim']:8} {s['gurultu_px']:8.2f} {s['rms_ortalama']:9.4f} "
              f"{fx['ortalama']:9.2f} {fx['std']:9.3f} {fx['sapma_yuzde']:9.4f} "
              f"{s['basarili']:>7}/{s['n_tekrar']}")

    gurultusuz = [s for s in satirlar if s["gurultu_px"] == 0.0]
    kabul = all(s["parametreler"]["fx"]["sapma_yuzde"] < 0.1 for s in gurultusuz)
    print(f"\nKABUL OLCUTU (gurultusuz fx hatasi < %0.1): {'GECTI' if kabul else 'KALDI'}")

    kazanc = []
    for g in GURULTULER[1:]:
        d = next(s for s in satirlar if s["dagilim"] == "dar" and s["gurultu_px"] == g)
        e = next(s for s in satirlar if s["dagilim"] == "genis" and s["gurultu_px"] == g)
        kazanc.append(d["parametreler"]["fx"]["std"] / e["parametreler"]["fx"]["std"])
    print(f"Poz cesitliligi kazanci (dar std / genis std): ortalama {np.mean(kazanc):.2f}x")

    cikti = Path("out/sentetik_kalibrasyon.json")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps({
        "deney": "sentetik_coklu_kamera_kalibrasyon",
        "donanim_gerekli": False,
        "tohum": TOHUM, "n_tekrar": N_TEKRAR, "n_kare": N_KARE,
        "gercek_odak_px": GERCEK_ODAK,
        "kabul_olcutu_gecti": bool(kabul),
        "poz_cesitliligi_kazanci": float(np.mean(kazanc)),
        "kosular": satirlar,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"veri : {cikti}")
    print(f"sekil: {sekil_ciz(satirlar, Path('docs/deney/2026-09-22-sentetik-kalibrasyon.png'))}")
    return 0 if kabul else 1


if __name__ == "__main__":
    raise SystemExit(main())
