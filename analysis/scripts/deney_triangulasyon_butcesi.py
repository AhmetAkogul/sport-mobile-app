#!/usr/bin/env python3
"""Deney: 3B konum hatasini ne belirliyor -- kalibrasyon mu, tespit gurultusu mu?

PROJE-PLANI.md Kapi 2 olcutu 2 m mesafede < 10 mm hata istiyor. Bu deney o
olcute hangi kosullarda ulasildigini sentetik olarak cikarir ve hatayi iki
kaynagina ayirir:

  * kalibrasyon hatasi  -- kamera parametreleri ne kadar dogru bilindigi
  * tespit gurultusu    -- her karede 2B nokta ne kadar hassas bulundugu

Ayrim onemli: ikisi ayni sonuca yol acar ama cozumleri tamamen farklidir.
Kalibrasyon baskinsa cozum daha iyi bir kalibrasyon oturumu (bedava); tespit
baskinsa cozum daha iyi kamera/objektif/isik (pahali).

OLCUM UYARISI -- bu deneyin ilk surumu yanlis olcuyordu. Mutlak 3B konumu yer
gercegiyle dogrudan karsilastirmak hatalidir: kalibrasyon kamera 0'i referans
alir, ama kestirilen kamera 0 ile gercek kamera 0 ayni cerceve degildir. Ilk
kosuda bu fark 208 mm "hata" gibi gorundu; oysa cubugun uzunlugu 2 mm hatayla
dogru cikiyordu. Cerceveden bagimsiz buyuklukler olculur:
  * bilinen mesafe hatasi          -- Kapi 2'nin tanimladigi olcut
  * rijit hizalama sonrasi artik   -- gercek sekil bozulmasi
  * olcek hatasi                   -- ayri raporlanir

Donanim gerekmez, deterministiktir. Cikti: out/ + docs/deney/ altina sekil.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calib.board import BoardSpec
from calib.multiview import kalibre_et
from calib.synthetic import (
    dunyadan_kamera0, nokta_gozlemleri, olculu_cubuk, rig_yay, sahne_uret,
)
from pose3d.hizalama import rijit_hizala
from pose3d.triangulate import bilinen_mesafe_hatasi, ucgenle_toplu

SPEC = BoardSpec(5, 7)
ODAK = 900.0
KALIB_GURULTULERI = (0.0, 0.25)          # mukemmel kalibrasyon vs gerçekçi
GOZLEM_GURULTULERI = (0.1, 0.25, 0.5, 1.0)
KAMERA_SETLERI = {2: (0, 3), 3: (0, 2, 3), 4: (0, 1, 2, 3)}
N_KALIB = 5
N_GOZLEM = 6
TOHUM = 8080
KAPI_2_ESIK_MM = 10.0


def kalibrasyonlar(kameralar, gurultu):
    cikti = []
    for i in range(N_KALIB):
        s = sahne_uret(SPEC, kameralar=kameralar, n_kare=20, gurultu_px=gurultu,
                       dagilim="genis", seed=TOHUM + i)
        cikti.append(kalibre_et(s.obj_noktalari, s.img_noktalari,
                                [c.boyut for c in kameralar], s.mask))
    return cikti


def hucre(kalibler, kameralar, noktalar, gercek, gercek_mesafeler,
          gozlem_gurultu, kamera_seti):
    """Cerceveden bagimsiz uc olcu: mesafe hatasi, sekil artigi, olcek hatasi."""
    mesafe_hatalari, sekil_rms, olcek_hatalari = [], [], []
    for kalib in kalibler:
        for j in range(N_GOZLEM):
            goz = nokta_gozlemleri(kameralar, noktalar, gurultu_px=gozlem_gurultu,
                                   seed=TOHUM + 500 + j)
            secili = [{k: v for k, v in g.items() if k in kamera_seti} for g in goz]
            kest = ucgenle_toplu(kalib, secili)
            if not all(k.gecerli for k in kest):
                continue
            r = bilinen_mesafe_hatasi(kest, gercek_mesafeler)
            mesafe_hatalari.extend(abs(o["hata_mm"]) for o in r["olcumler"])
            h = rijit_hizala(np.array([k.nokta for k in kest]), gercek,
                             olcek_serbest=True)
            sekil_rms.append(h.rms_mm)
            olcek_hatalari.append(abs(1.0 - h.olcek) * 100.0)
    m = np.asarray(mesafe_hatalari)
    return {
        "mesafe_ortalama_mm": float(m.mean()),
        "mesafe_p95_mm": float(np.percentile(m, 95)),
        "mesafe_en_buyuk_mm": float(m.max()),
        "sekil_rms_mm": float(np.mean(sekil_rms)),
        "olcek_hatasi_yuzde": float(np.mean(olcek_hatalari)),
        "n": int(m.size),
    }


def sekil_ciz(satirlar, hedef: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    renk = {2: "#c0392b", 3: "#b9770e", 4: "#2471a3"}
    baslik = {0.0: "Mükemmel kalibrasyon", 0.25: "Gerçekçi kalibrasyon (0,25 px)"}

    for eksen, kg in zip(eksenler, KALIB_GURULTULERI):
        for n_kam in KAMERA_SETLERI:
            alt = [s for s in satirlar if s["kalib_gurultu"] == kg and s["n_kamera"] == n_kam]
            eksen.plot([s["gozlem_gurultu"] for s in alt],
                       [s["mesafe_ortalama_mm"] for s in alt],
                       "o-", color=renk[n_kam], label=f"{n_kam} kamera")
        eksen.axhline(KAPI_2_ESIK_MM, color="black", ls="--", lw=1)
        eksen.text(0.12, KAPI_2_ESIK_MM * 1.15, "Kapı 2 eşiği: 10 mm", fontsize=8)
        eksen.set_yscale("log")
        eksen.set_xlabel("tespit gürültüsü (piksel)")
        eksen.set_title(baslik[kg])
        eksen.grid(alpha=0.3, which="both")
        eksen.legend()
    eksenler[0].set_ylabel("bilinen mesafe hatası (mm, log)")
    fig.suptitle("Triangulation hata bütçesi — kalibrasyon mu, tespit mi baskın?",
                 fontsize=11)
    fig.tight_layout()
    hedef.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hedef, dpi=150)
    plt.close(fig)
    return hedef


def main() -> int:
    kameralar = rig_yay(odak_px=ODAK)
    noktalar, gercek_mesafeler = olculu_cubuk(uzunluk_m=1.0, n_isaret=5)
    gercek = dunyadan_kamera0(kameralar, noktalar)

    satirlar = []
    for kg in KALIB_GURULTULERI:
        kalibler = kalibrasyonlar(kameralar, kg)
        for gg in GOZLEM_GURULTULERI:
            for n_kam, seti in KAMERA_SETLERI.items():
                s = hucre(kalibler, kameralar, noktalar, gercek, gercek_mesafeler,
                          gg, seti)
                satirlar.append(s | {"kalib_gurultu": kg, "gozlem_gurultu": gg,
                                     "n_kamera": n_kam,
                                     "kapi_2_gecti": s["mesafe_p95_mm"] < KAPI_2_ESIK_MM})

    print(f"{'kalib':>6} {'gozlem':>7} {'kam':>4} | {'mesafe ort':>11} {'p95':>8} "
          f"{'max':>8} | {'sekil rms':>10} {'olcek %':>8}  Kapi2")
    for s in satirlar:
        print(f"{s['kalib_gurultu']:6.2f} {s['gozlem_gurultu']:7.2f} {s['n_kamera']:4d} | "
              f"{s['mesafe_ortalama_mm']:11.3f} {s['mesafe_p95_mm']:8.3f} "
              f"{s['mesafe_en_buyuk_mm']:8.3f} | {s['sekil_rms_mm']:10.3f} "
              f"{s['olcek_hatasi_yuzde']:8.3f}  {'GECTI' if s['kapi_2_gecti'] else 'KALDI'}")

    def bul(kg, gg, n):
        return next(s for s in satirlar if s["kalib_gurultu"] == kg
                    and s["gozlem_gurultu"] == gg and s["n_kamera"] == n)

    print("\n--- Hata butcesi: bilinen mesafe hatasi, 4 kamera ---")
    for gg in GOZLEM_GURULTULERI:
        mukemmel = bul(0.0, gg, 4)["mesafe_ortalama_mm"]
        gercekci = bul(0.25, gg, 4)["mesafe_ortalama_mm"]
        pay = (gercekci - mukemmel) / gercekci * 100 if gercekci else 0.0
        print(f"  gozlem {gg:.2f} px: sadece tespit {mukemmel:6.3f} mm | "
              f"kalibrasyonla birlikte {gercekci:6.3f} mm | "
              f"kalibrasyonun payi ~%{pay:.0f}")

    gecen = [s for s in satirlar if s["kalib_gurultu"] == 0.25 and s["kapi_2_gecti"]]
    if gecen:
        en_kotu = max(s["gozlem_gurultu"] for s in gecen if s["n_kamera"] == 4)
        print(f"\nGercekci kalibrasyonla 4 kamerada Kapi 2'yi geciren en yuksek "
              f"tespit gurultusu: {en_kotu} px")

    cikti = Path("out/triangulasyon_butcesi.json")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps({
        "deney": "triangulasyon_hata_butcesi", "donanim_gerekli": False,
        "tohum": TOHUM, "kapi_2_esik_mm": KAPI_2_ESIK_MM,
        "n_kalibrasyon": N_KALIB, "n_gozlem": N_GOZLEM,
        "kosular": satirlar,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"veri : {cikti}")
    print(f"sekil: {sekil_ciz(satirlar, Path('docs/deney/2026-09-22-triangulasyon-butcesi.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
