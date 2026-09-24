"""Gercek cok kamerali kayitta ilk olcum: CMU Panoptic 171204_pose1.

Iki soru, ayni karelerde:

1. **2B dogruluk** -- Panoptic'in 3B iskeleti her kameraya izdusurulur; modelin
   buldugu eklemle piksel farki. Referans, Panoptic'in 480 VGA + 31 HD kameradan
   kurdugu iskelettir: bizim modellerimizden ve ucgenlememizden bagimsizdir
   (0031'de etiketli veri olmadigi icin olculemeyen kiyasin ilk gercek hali).
2. **3B dogruluk** -- secilen HD kameralardaki tespitlerden **bizim**
   `iskelet_ucgenle`'miz ile kurulan iskeletin Panoptic iskeletine hatasi (mm),
   ikisi de kamera 0 cercevesinde (rijit hizalama yok: mutlak konum).

Model ortaminda kosulur (proje kodu PYTHONPATH ile):

    PYTHONPATH=. data/mediapipe-env/bin/python scripts/deney_panoptic.py \\
        --backend mediapipe --model <pose_landmarker_full.task>
    PYTHONPATH=. data/rtmpose-env/bin/python scripts/deney_panoptic.py \\
        --backend rtmpose --model <pose.onnx> --detector <detector.onnx>

Kare numarasi = HD video kare numarasi = body3DScene_<kare>.json (toolbox README).
Yalnizca tek kisilik kareler kullanilir (coklu kisi eslestirmesi kapsam disi).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mono.backend import kestirici_olustur  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, iskelet_ucgenle  # noqa: E402
from veri.panoptic import (  # noqa: E402
    dunyadan_kamera0, iskeletler_oku, kalibrasyon_oku, referansa_tasi,
)

KOK = Path(__file__).resolve().parent.parent
DIZI = KOK / "data/dis/cmu_panoptic/171204_pose1"
KAMERALAR = ["00_00", "00_01", "00_02", "00_03"]
# Sol <-> sag esleme (ayna): tani icin, modelin sol/sag karistirip karistirmadigi.
_AYNA = [REFERANS_ISKELET.indeks(e.replace("sol_", "@").replace("sag_", "sol_").replace("@", "sag_"))
         for e in REFERANS_ISKELET.eklemler]


def _video_yolu(ad: str) -> Path:
    """Tam video yoksa bastan indirilmis parca (`*_bas.mp4`) kullanilir."""
    tam, bas = DIZI / f"hdVideos/hd_{ad}.mp4", DIZI / f"hdVideos/hd_{ad}_bas.mp4"
    if bas.exists() and (not tam.exists() or bas.stat().st_size > tam.stat().st_size):
        return bas
    return tam


def _kareleri_topla(yol: Path, kareler: set[int]) -> dict[int, np.ndarray]:
    """Sirayla okuyup istenen kareleri al (H.264'te konum atlama kare-kesin degil)."""
    cap = cv2.VideoCapture(str(yol))
    sonuc, i, son = {}, 0, max(kareler)
    try:
        while i <= son:
            ok, im = cap.read()
            if not ok:
                break
            if i in kareler:
                sonuc[i] = im
            i += 1
    finally:
        cap.release()
    return sonuc


def _ozet(v) -> dict | None:
    v = np.asarray(v, float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "medyan": round(float(np.median(v)), 2),
            "ortalama": round(float(v.mean()), 2),
            "p95": round(float(np.percentile(v, 95)), 2)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("mediapipe", "rtmpose"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--detector")
    p.add_argument("--ilk", type=int, default=300)
    p.add_argument("--son", type=int, default=3000)
    p.add_argument("--adim", type=int, default=30)
    p.add_argument("--esik", type=float, default=None,
                   help="guven esigi (varsayilan modele gore); duyarlilik icin")
    a = p.parse_args()

    pk = kalibrasyon_oku(DIZI / "calibration_171204_pose1.json", KAMERALAR)
    istenen = list(range(a.ilk, a.son + 1, a.adim))
    gt = iskeletler_oku(DIZI / "hdPose3d_stage1_coco19.tar", istenen)
    tek = {k for k, bs in gt.items() if len(bs) == 1}
    goruntuler = {ci: _kareleri_topla(_video_yolu(ad), tek) for ci, ad in enumerate(KAMERALAR)}
    ortak = sorted(k for k in tek if all(k in goruntuler[ci] for ci in goruntuler))

    hata2b: dict[int, list[float]] = {ci: [] for ci in range(len(KAMERALAR))}
    hata3b: list[float] = []
    eklem3b: dict[str, list[float]] = {e: [] for e in REFERANS_ISKELET.eklemler}
    ucgenlenen: list[float] = []
    tani: list[dict] = []
    kadraj_disi = {"gt_disarida": 0, "model_gorunur_dedi": 0}
    # RTMPose dedektoru kubbedeki birden fazla kisiyi bulabiliyor; en buyuk kutu
    # secilir (yer gercegine bakmaz; MediaPipe'in tek-kisi secimine denk).
    ek = {"coklu_kisi": "en_buyuk"} if a.backend == "rtmpose" else {}
    with kestirici_olustur(a.backend, model=a.model, detector=a.detector,
                           threshold=a.esik, **ek) as model:
        model_id = model.model_id
        for k in ortak:
            P_dunya, g_ref = referansa_tasi(gt[k][0])
            P_ref = np.full_like(P_dunya, np.nan)
            P_ref[g_ref] = dunyadan_kamera0(P_dunya[g_ref], pk)
            gozlem = {}
            for ci in range(len(KAMERALAR)):
                poz = replace(model(goruntuler[ci][k]), kamera=ci, kare=k)
                gozlem[ci] = poz
                # Kadraj disi: referans eklem bu kameranin goruntusunun disina dusuyorsa
                # model onu "gorunur" diyor mu? (SIMCC skoru bunu ayirt etmeyebilir.)
                tum, _ = cv2.projectPoints(
                    P_ref[g_ref], cv2.Rodrigues(pk.kalib.Rs[ci])[0], pk.kalib.Ts[ci],
                    pk.kalib.Ks[ci], pk.kalib.bozulmalar[ci])
                tum = tum.reshape(-1, 2)
                w, h = pk.boyutlar[ci]
                disarida = np.zeros(len(g_ref), bool)
                disarida[g_ref] = ~((tum[:, 0] >= 0) & (tum[:, 0] < w)
                                    & (tum[:, 1] >= 0) & (tum[:, 1] < h))
                kadraj_disi["gt_disarida"] += int(disarida.sum())
                kadraj_disi["model_gorunur_dedi"] += int((disarida & poz.gorunur).sum())
                # 2B: referansi bu kameraya izdusur (bozulma dahil), gorunenlerle kiyasla.
                # Kadraj disindaki referans eklem 2B kiyastan cikarilir (piksel hatasi
                # tanimsiz); ama 3B'ye giren "gorunur" tahmin olarak ayrica sayilir.
                sec = g_ref & poz.gorunur & ~disarida
                if sec.any():
                    izd, _ = cv2.projectPoints(
                        P_ref[sec], cv2.Rodrigues(pk.kalib.Rs[ci])[0], pk.kalib.Ts[ci],
                        pk.kalib.Ks[ci], pk.kalib.bozulmalar[ci])
                    fark = np.linalg.norm(izd.reshape(-1, 2) - poz.noktalar[sec], axis=1)
                    hata2b[ci].extend(fark.tolist())
                    # Tani: ayna (sol/sag takas) ile hata ne olurdu?
                    ayna = poz.noktalar[_AYNA]
                    sec_a = sec & poz.gorunur[_AYNA]
                    ayna_medyan = None
                    if sec_a.any():          # bos girdide projectPoints None doner
                        izd_a, _ = cv2.projectPoints(
                            P_ref[sec_a], cv2.Rodrigues(pk.kalib.Rs[ci])[0], pk.kalib.Ts[ci],
                            pk.kalib.Ks[ci], pk.kalib.bozulmalar[ci])
                        ayna_medyan = round(float(np.median(np.linalg.norm(
                            izd_a.reshape(-1, 2) - ayna[sec_a], axis=1))), 1)
                    tani.append({"kare": k, "kamera": KAMERALAR[ci],
                                 "medyan_px": round(float(np.median(fark)), 1),
                                 "ayna_medyan_px": ayna_medyan,
                                 "bulunan_kisi": int(poz.ek.get("bulunan_kisi", 1))})
            isk = iskelet_ucgenle(pk.kalib, gozlem, REFERANS_ISKELET)
            sec = g_ref & isk.gorunur
            ucgenlenen.append(float(sec.sum()) / float(g_ref.sum()))
            d = np.linalg.norm(isk.noktalar[sec] - P_ref[sec], axis=1) * 1000.0
            hata3b.extend(d.tolist())
            for j, e in enumerate(np.array(REFERANS_ISKELET.eklemler)[sec]):
                eklem3b[str(e)].append(float(d[j]))

    sonuc = {
        "deney": "panoptic_171204_pose1", "backend": a.backend, "model": model_id,
        "kameralar": KAMERALAR,
        "kareler": {"ilk": a.ilk, "son": a.son, "adim": a.adim,
                    "tek_kisilik": len(tek), "kullanilan": len(ortak)},
        "referans": "Panoptic hdPose3d_stage1_coco19 (480 VGA + 31 HD'den; bizden bagimsiz)",
        "hata_2b_px": {KAMERALAR[ci]: _ozet(v) for ci, v in hata2b.items()},
        "hata_2b_px_tum": _ozet([x for v in hata2b.values() for x in v]),
        "hata_3b_mm": _ozet(hata3b),
        "hata_3b_mm_eklem": {e: _ozet(v) for e, v in eklem3b.items()},
        "ucgenlenen_eklem_orani": round(float(np.mean(ucgenlenen)), 4) if ucgenlenen else None,
        "physical_validation": "gercek goruntu; referans Panoptic'in kendi 3B'si",
        "esik": getattr(model, "threshold", None),
        "kadraj_disi": {**kadraj_disi, "gorunur_dedi_orani": (
            round(kadraj_disi["model_gorunur_dedi"] / kadraj_disi["gt_disarida"], 4)
            if kadraj_disi["gt_disarida"] else None)},
        "tani": {
            "coklu_kisi_orani": round(float(np.mean([t["bulunan_kisi"] > 1 for t in tani])), 4)
            if tani else None,
            "ayna_daha_iyi_orani": round(float(np.mean(
                [t["ayna_medyan_px"] is not None and t["ayna_medyan_px"] < 0.5 * t["medyan_px"]
                 for t in tani])), 4) if tani else None,
            "en_kotu_10": sorted(tani, key=lambda t: -t["medyan_px"])[:10],
        },
    }
    ek_ad = "" if a.esik is None else f"_esik{a.esik:g}"
    cikti = KOK / "out" / f"panoptic_{a.backend}{ek_ad}.json"
    cikti.parent.mkdir(exist_ok=True)
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False),
                     encoding="utf-8")
    print(json.dumps({k: sonuc[k] for k in ("esik", "kareler", "hata_2b_px_tum", "hata_3b_mm",
                                           "ucgenlenen_eklem_orani", "kadraj_disi")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
