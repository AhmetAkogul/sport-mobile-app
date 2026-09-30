#!/usr/bin/env python3
"""Deney: olcum dogrulugu calisma hacminde nasil degisiyor?

PROJE-PLANI.md Faz 2 maddesi. Bilinen mesafe deneyi hacmin **merkezinde**
yapildi; bu deney o sonucun mekanda ne kadar gecerli oldugunu cikarir.

Soru: "Sistem 10 mm dogrulukta" demek yeterli degil -- NEREDE 10 mm?
Cevap, veri toplama sirasinda denegin nerede duracagini belirler.

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
from calib.synthetic import rig_yay, sahne_uret
from eval.hacim import hacim_haritasi, kullanilabilir_bolge

SPEC = BoardSpec(5, 7)
ODAK = 900.0
KALIB_GURULTU = 0.25          # gerçekçi kalibrasyon
GOZLEM_GURULTU = 0.25
ADIM_M = 0.2
PROB_M = 0.20                 # onkol mertebesinde; eklem-eklem mesafesi
KAPI_2_ESIK_MM = 10.0
TOHUM = 8080


def sekil_ciz(harita, kameralar, bolge, hedef: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, (sol, sag) = plt.subplots(1, 2, figsize=(12, 4.6))
    sinirlar = [harita.x_m[0], harita.x_m[-1], harita.z_m[0], harita.z_m[-1]]

    gecerli = harita.hata_mm[np.isfinite(harita.hata_mm)]
    im = sol.imshow(harita.hata_mm, origin="lower", extent=sinirlar, aspect="auto",
                    cmap="viridis_r",
                    norm=LogNorm(vmin=max(gecerli.min(), 0.1), vmax=gecerli.max()))
    cs = sol.contour(harita.x_m, harita.z_m, harita.hata_mm,
                     levels=[KAPI_2_ESIK_MM], colors="red", linewidths=2)
    sol.clabel(cs, fmt={KAPI_2_ESIK_MM: "10 mm"}, fontsize=9)
    fig.colorbar(im, ax=sol, label="prob uzunluk hatası (mm, log)")
    sol.set_title(f"Hata haritası — hacmin %{bolge['gecen_oran']*100:.0f}'i 10 mm altında")

    im2 = sag.imshow(harita.gorunurluk, origin="lower", extent=sinirlar, aspect="auto",
                     cmap="YlGnBu", vmin=0, vmax=len(kameralar))
    fig.colorbar(im2, ax=sag, label="probu tam gören kamera sayısı")
    sag.set_title("Görünürlük — 2'nin altı ölçülemez")

    for eksen in (sol, sag):
        for i, kam in enumerate(kameralar):
            m = kam.merkez
            eksen.plot(m[0], m[2], "^", color="white", markeredgecolor="black",
                       markersize=9, zorder=5)
            eksen.annotate(f"K{i}", (m[0], m[2]), textcoords="offset points",
                           xytext=(0, -14), ha="center", fontsize=8, color="black")
        eksen.set_xlabel("yatay konum X (m)")
        eksen.set_ylabel("derinlik Z (m)")

    fig.suptitle(f"Çalışma hacmi — {PROB_M*100:.0f} cm prob, "
                 f"{GOZLEM_GURULTU} px tespit gürültüsü (üstten görünüm)", fontsize=11)
    fig.tight_layout()
    hedef.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hedef, dpi=150)
    plt.close(fig)
    return hedef


def main() -> int:
    kameralar = rig_yay(odak_px=ODAK)
    s = sahne_uret(SPEC, kameralar=kameralar, n_kare=20, gurultu_px=KALIB_GURULTU,
                   dagilim="genis", seed=TOHUM)
    kalib = kalibre_et(s.obj_noktalari, s.img_noktalari,
                       [c.boyut for c in kameralar], s.mask)
    print(f"kalibrasyon: rms={kalib.rms_px:.3f} px  fx={kalib.odak(0)[0]:.1f}")

    # Calisma hacmi kameralarin birlestigi yerdir: rig_yay orijine baktirir,
    # kameralar Z<0 tarafinda durur. Aralik orijini kapsayacak sekilde secilir.
    harita = hacim_haritasi(kalib, kameralar, x_araligi=(-1.6, 1.6),
                            z_araligi=(-1.4, 1.6), adim_m=ADIM_M, prob_m=PROB_M,
                            gurultu_px=GOZLEM_GURULTU, seed=TOHUM + 7)
    ozet = harita.ozet()
    bolge = kullanilabilir_bolge(harita, KAPI_2_ESIK_MM)

    print("\n--- Hacim ozeti ---")
    for k, v in ozet.items():
        print(f"  {k:24s} {v}")
    print("\n--- Kapi 2 olcutunu saglayan bolge ---")
    for k, v in bolge.items():
        print(f"  {k:24s} {v}")

    print("\n--- Kamera mesafesine gore hata (satir ortalamasi) ---")
    kam_z = float(np.mean([k.merkez[2] for k in kameralar]))
    # Ortalama birkac dejenere hucre tarafindan cekiliyor; medyan da yazilir.
    print(f"{'Z (m)':>8} {'~kamera uzakligi':>17} {'medyan mm':>11} "
          f"{'ortalama mm':>12} {'en iyi mm':>11} {'olculebilir':>12}")
    for iz, z in enumerate(harita.z_m):
        satir = harita.hata_mm[iz]
        g = satir[np.isfinite(satir)]
        uzaklik = z - kam_z
        if g.size == 0:
            print(f"{z:8.2f} {uzaklik:17.2f} {'-':>11} {'-':>12} {'-':>11} {'0%':>12}")
            continue
        print(f"{z:8.2f} {uzaklik:17.2f} {np.median(g):11.2f} {g.mean():12.2f} "
              f"{g.min():11.2f} {np.isfinite(satir).mean()*100:11.0f}%")

    cikti = Path("out/hacim_haritasi.json")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps({
        "deney": "calisma_hacmi_hata_haritasi", "donanim_gerekli": False,
        "tohum": TOHUM, "kalib_gurultu_px": KALIB_GURULTU,
        "gozlem_gurultu_px": GOZLEM_GURULTU, "prob_m": PROB_M, "adim_m": ADIM_M,
        "kamera_merkezleri": [k.merkez.tolist() for k in kameralar],
        "ozet": ozet, "kapi_2_bolgesi": bolge,
        "x_m": harita.x_m.tolist(), "z_m": harita.z_m.tolist(),
        "hata_mm": [[None if not np.isfinite(v) else float(v) for v in satir]
                    for satir in harita.hata_mm],
        "gorunurluk": harita.gorunurluk.tolist(),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"\nveri : {cikti}")
    print(f"sekil: {sekil_ciz(harita, kameralar, bolge, Path('docs/deney/2026-09-22-hacim-haritasi.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
