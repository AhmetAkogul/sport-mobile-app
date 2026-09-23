"""Tek komutla telefon demosu: video -> 2B poz -> tek gorus 3B -> form karari -> isaretli video.

Her kare icin **iki karar** yan yana verilir, cunku tezin mesaji bu farktir:

- **Yalin esik** -- telefon uygulamalarinin yaptigi: olculen aci esigi gectiyse
  "kusurlu". Kesin konusur, ama tek gorus 3B'nin hatasini hesaba katmaz.
- **Belirsizligi bilen** -- ayni olcum, eklem konum belirsizligi bandiyla.
  Band esigi kesiyorsa "degerlendirilemiyor" (BELIRSIZ) der.

Varsayilan konum belirsizligi 76 mm: sentetik deneyde tek gorus kestiriminin
**onden bakista, mukemmel kemik onculeri ve sifir tespit gurultusuyle** olculen
model hatasi (`docs/deney/2026-09-23-derinlik-duyarliligi.md`). Gercek kosulda
hata bundan buyuktur; yani bu varsayilan iyimser bir alt sinirdir ve belirsizligi
bilen kolun "degerlendirilemiyor" demesi bir arizaya degil, olcumun gercek
sinirina isaret eder.

Cikti (`--output` altinda):
    hat/            mono.run_phone ciktisi (results.jsonl, summary.json)
    form.jsonl      kare basina iki form raporu
    overlay.avi     iskelet + iki karar isaretli video (MJPG)
    summary.json    karar sayimlari, kullanilan belirsizlik ve kaynagi

    python -m mono.demo --video v.mp4 --model m.task --intrinsics K.json \
        --lengths lengths.json --output out/demo
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from capture.record import _write_json
from eval.form import Karar, form_degerlendir
from mono import run_phone
from mono.phone import json_uyumlu
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B

# Tek gorus model hatasi, onden, ideal kosul (derinlik duyarliligi deneyi).
VARSAYILAN_KONUM_BELIRSIZLIGI_M = 0.076
BELIRSIZLIK_KAYNAGI = (
    "docs/deney/2026-09-23-derinlik-duyarliligi.md: onden bakista sentetik "
    "model hatasi (mukemmel oncu, sifir gurultu) -- gercek hatanin alt siniri")

_RENK = {Karar.DOGRU: (80, 200, 80), Karar.KUSURLU: (60, 60, 230),
         Karar.BELIRSIZ: (0, 190, 240)}
_ETIKET = {Karar.DOGRU: "DOGRU", Karar.KUSURLU: "KUSURLU",
           Karar.BELIRSIZ: "DEGERLENDIRILEMIYOR"}


def satirdan_iskelet(satir: dict) -> Iskelet3B:
    """`results.jsonl` satirindan tek gorus iskeletini geri kur (null -> NaN)."""
    noktalar = np.array([[np.nan if v is None else v for v in p]
                         for p in satir["points_3d_m"]], dtype=np.float64)
    gorunur = np.array(satir["visible_3d"], dtype=bool)
    return Iskelet3B(tanim=REFERANS_ISKELET, noktalar=noktalar, gorunur=gorunur,
                     goren_kamera=gorunur.astype(int),
                     artik_px=np.full(len(REFERANS_ISKELET), np.nan))


def _ciz(image: np.ndarray, satir: dict, yalin, bilen) -> np.ndarray:
    out = image.copy()
    pts = satir["points_px"]
    gor = satir["visible"]
    for a, b in REFERANS_ISKELET.baglantilar:
        ia, ib = REFERANS_ISKELET.indeks(a), REFERANS_ISKELET.indeks(b)
        if gor[ia] and gor[ib]:
            cv2.line(out, tuple(int(round(v)) for v in pts[ia]),
                     tuple(int(round(v)) for v in pts[ib]), (255, 213, 0), 2, cv2.LINE_AA)
    for p, g in zip(pts, gor):
        if g:
            cv2.circle(out, tuple(int(round(v)) for v in p), 4, (255, 255, 255), -1, cv2.LINE_AA)
    olcek = max(0.4, out.shape[1] / 1600)
    for i, (baslik, rapor) in enumerate((("Yalin esik", yalin), ("Belirsizligi bilen", bilen))):
        karar = rapor.genel_karar
        metin = f"{baslik}: {_ETIKET[karar]}"
        if rapor.kusurlar:
            metin += f" ({', '.join(rapor.kusurlar)})"
        y = int(30 * olcek * 1.6) * (i + 1)
        cv2.putText(out, metin, (12, y), cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, metin, (12, y), cv2.FONT_HERSHEY_SIMPLEX, olcek, _RENK[karar], 2,
                    cv2.LINE_AA)
    return out


def run(video, model, output, K, lengths, *, max_frames=300, backend="mediapipe",
        detector=None, threshold=None,
        konum_belirsizligi_m: float = VARSAYILAN_KONUM_BELIRSIZLIGI_M) -> dict:
    if not (isinstance(konum_belirsizligi_m, (int, float))
            and math.isfinite(konum_belirsizligi_m) and konum_belirsizligi_m > 0):
        raise ValueError("konum_belirsizligi_m sonlu ve pozitif olmali (metre)")
    output = Path(output)
    if output.exists():
        raise ValueError("cikti dizini yeni olmali")
    output.mkdir(parents=True)

    hat = run_phone.run(video, model, output / "hat", K, lengths, max_frames,
                        backend=backend, detector=detector, threshold=threshold)
    satirlar = [json.loads(s) for s in
                (output / "hat" / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    yalin_say, bilen_say, kusur_say = Counter(), Counter(), Counter()
    raporlar = []
    with (output / "form.jsonl").open("x", encoding="utf-8") as f:
        for satir in satirlar:
            iskelet = satirdan_iskelet(satir)
            yalin = form_degerlendir(iskelet)
            bilen = form_degerlendir(iskelet, konum_belirsizligi_m=konum_belirsizligi_m)
            raporlar.append((yalin, bilen))
            yalin_say[str(yalin.genel_karar)] += 1
            bilen_say[str(bilen.genel_karar)] += 1
            kusur_say.update(yalin.kusurlar)
            f.write(json.dumps(json_uyumlu({
                "frame_index": satir["frame_index"], "tespit": satir["tespit"],
                "yalin_esik": yalin.ozet(), "belirsizligi_bilen": bilen.ozet()}),
                ensure_ascii=False, allow_nan=False) + "\n")

    cap = cv2.VideoCapture(str(Path(video).resolve()))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError("video cizim icin acilamadi")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 10.0
        for satir, (yalin, bilen) in zip(satirlar, raporlar):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"cizim sirasinda kare {satir['frame_index']} okunamadi")
            kare = _ciz(image, satir, yalin, bilen)
            if writer is None:
                writer = cv2.VideoWriter(str(output / "overlay.avi"),
                                         cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                         (kare.shape[1], kare.shape[0]))
                if not writer.isOpened():
                    raise ValueError("overlay yazicisi acilamadi")
            writer.write(kare)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    ozet = {
        "status": "completed", "frames": len(satirlar),
        "detected_frames": hat.get("detected_frames"),
        "model": hat.get("model"), "backend": backend, "threshold": hat.get("threshold"),
        "konum_belirsizligi_m": konum_belirsizligi_m,
        "belirsizlik_kaynagi": BELIRSIZLIK_KAYNAGI,
        "yalin_esik_kararlari": dict(yalin_say),
        "belirsizligi_bilen_kararlari": dict(bilen_say),
        "yalin_esigin_buldugu_kusurlar": dict(kusur_say),
        "physical_validation": False,
        "not": ("Tek gorus 3B; duzeltme katmani yok. 'Degerlendirilemiyor' bir "
                "ariza degil, olcum hatasinin karar sinirini astigi anlamina gelir."),
    }
    _write_json(output / "summary.json", ozet)
    return ozet


def main():
    p = argparse.ArgumentParser(description="Telefon demosu: video -> form karari -> isaretli video")
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--backend", choices=("mediapipe", "rtmpose"), default="mediapipe")
    p.add_argument("--detector", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--konum-belirsizligi-m", type=float,
                   default=VARSAYILAN_KONUM_BELIRSIZLIGI_M,
                   help="eklem konum belirsizligi, metre (varsayilan: tek gorus model hatasi)")
    a = p.parse_args()
    print(json.dumps(run(a.video, a.model, a.output,
                         json.loads(Path(a.intrinsics).read_text(encoding="utf-8")),
                         run_phone.load_lengths(a.lengths), max_frames=a.max_frames,
                         backend=a.backend, detector=a.detector, threshold=a.threshold,
                         konum_belirsizligi_m=a.konum_belirsizligi_m),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
