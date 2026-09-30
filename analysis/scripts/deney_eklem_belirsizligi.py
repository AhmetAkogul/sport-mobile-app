"""0013'un sayilarini (2,20 mm; 4,5x) yeniden ureten deney -- sayi kilidine girer.

Onceden bu sayilar yalnizca karar kaydinda yaziliydi ve `make reproduce`
tarafindan uretilmiyordu (rapor iskeleti C5, M13). Kurulum `tests/
test_uncertainty_eklem.py` ile aynidir: rig_yay (4 kamera, odak 900 px),
gurultusuz kalibrasyon (tohum 8080), tek sabit durus, 4 kamera gorusu.

    python scripts/deney_eklem_belirsizligi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from calib.board import BoardSpec  # noqa: E402
from calib.multiview import kalibre_et  # noqa: E402
from calib.synthetic import rig_yay, sahne_uret  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, iskelet_ucgenle  # noqa: E402
from pose3d.pose2d import sentetik_poz  # noqa: E402
from uncertainty.eklem import eklem_belirsizligi, eklem_kovaryanslari  # noqa: E402

TOHUM = 8080
SIGMALAR_PX = [0.25, 1.0, 2.0]
DURUS = {
    "boyun": (0, 0.55, 0), "sag_omuz": (-0.18, 0.45, 0),
    "sag_dirsek": (-0.22, 0.18, 0.05), "sag_bilek": (-0.24, -0.08, 0.08),
    "sol_omuz": (0.18, 0.45, 0), "sol_dirsek": (0.22, 0.18, 0.05),
    "sol_bilek": (0.24, -0.08, 0.08), "sag_kalca": (-0.11, 0, 0),
    "sag_diz": (-0.12, -0.42, 0.02), "sag_ayak_bilegi": (-0.12, -0.85, 0),
    "sol_kalca": (0.11, 0, 0), "sol_diz": (0.12, -0.42, 0.02),
    "sol_ayak_bilegi": (0.12, -0.85, 0),
}


def main() -> None:
    kok = Path(__file__).resolve().parent.parent
    kameralar = rig_yay(odak_px=900.0)
    s = sahne_uret(BoardSpec(5, 7), kameralar=kameralar, n_kare=20,
                   gurultu_px=0.0, dagilim="genis", seed=TOHUM)
    kalib = kalibre_et(s.obj_noktalari, s.img_noktalari,
                       [c.boyut for c in kameralar], s.mask)
    P3 = np.array([DURUS[e] for e in REFERANS_ISKELET.eklemler])
    goz = {i: sentetik_poz(k, P3, REFERANS_ISKELET, gurultu_px=0.0, seed=i, kamera_indeksi=i)
           for i, k in enumerate(kameralar)}
    isk = iskelet_ucgenle(kalib, goz, REFERANS_ISKELET)

    satirlar = []
    for sp in SIGMALAR_PX:
        sig = eklem_belirsizligi(kalib, isk, goz, sigma_px=sp) * 1000.0
        satirlar.append({"sigma_px": sp,
                         "medyan_mm": round(float(np.nanmedian(sig)), 3),
                         "en_buyuk_mm": round(float(np.nanmax(sig)), 3)})
    oranlar = [float(np.linalg.eigvalsh(k).max() / np.linalg.eigvalsh(k).min())
               for k in eklem_kovaryanslari(kalib, isk, goz, sigma_px=1.0) if k is not None]
    sonuc = {
        "experiment": "eklem_belirsizligi_0013",
        "description": "Ucgenlemeden eklem konum belirsizligi (yalniz 2B tespit payi).",
        "physical_validation": False, "tohum": TOHUM, "n_kamera": len(kameralar),
        "satirlar": satirlar,
        "ozet": {
            "medyan_mm_1px": next(r["medyan_mm"] for r in satirlar if r["sigma_px"] == 1.0),
            "ozdeger_orani_medyan": round(float(np.median(oranlar)), 2),
            "ozdeger_orani_en_buyuk": round(float(np.max(oranlar)), 2),
            "n_eklem": len(oranlar),
        },
    }
    (kok / "out").mkdir(exist_ok=True)
    (kok / "out" / "eklem_belirsizligi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(sonuc["ozet"] | {"satirlar": satirlar}, ensure_ascii=False))


if __name__ == "__main__":
    main()
