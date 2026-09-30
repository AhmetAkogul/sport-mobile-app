"""REHAB24-6: eklem tanim ofseti duzeltmesi -- kaymayi nedeninden duzeltmek.

Hata butcesi (`2026-09-28-hata-butcesi.md`): dedektorler dizi ve ayak bilegini
eklem merkezinin icine koyuyor; aciya bagli valgus kaymasinin mekanizmasi bu.
Aci-kayma duzeltmesi (0034 Karar 2) sonucu duzeltir; burada nedeni duzeltilir:

- Her eklem icin ofset, kisinin **kendi govde cercevesinde** (yanal, yukari,
  ileri; MediaPipe iskeletinden kurulur, test aninda da kurulabilir) ogrenilir:
  o_j = R^T (mocap_j - telefon_j), kok merkezli ve olcek eslenmis.
- Ofset kisi-disari ogrenilir (egitim kisilerinin medyani), govde acisi
  dilimine gore (0-30, 30-60, 60-90) ayri; ikinci kol ayrica diz fleksiyonu
  dilimine gore (aci x fleksiyon, en az MIN_ORNEK kare; azsa aci dilimi).
- Duzeltilmis iskeletten valgus yeniden hesaplanir; mocap valgusuyla fark:
  kayma (ortalama), sd, |medyan|. Aci-kayma duzeltmesiyle yan yana.

Yalniz MediaPipe world (maskesiz). Her tekrarda 3 karede bir.

    python scripts/deney_rehab24_eklem_ofseti.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.egzersiz import diz_fleksiyonu  # noqa: E402
from eval.form import govde_cercevesi  # noqa: E402
from mono.canli import govde_acisi  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("hb", KOK / "scripts/deney_rehab24_hata_butcesi.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)
tg = hb.tg
DILIMLER = ((0, 30), (30, 60), (60, 90.01))
FLEKS_DILIMLERI = ((0, 30), (30, 60), (60, 90), (90, 180.01))
MIN_ORNEK = 30         # aci x fleksiyon hucresinde en az kare; azsa aci dilimine duser


def cerceve(P: np.ndarray) -> np.ndarray | None:
    """(3, 3) sutunlari yanal, yukari, ileri; kurulamazsa None."""
    c = govde_cercevesi(hb._iskelet(P))
    return None if c is None else np.column_stack([c.yanal, c.yukari, c.ileri])


def dilim(aci: float, dilimler=DILIMLER) -> int:
    for i, (lo, hi) in enumerate(dilimler):
        if lo <= aci < hi:
            return i
    return -1


def ofset_ogren(kayitlar: list[dict], anahtar: str = "dilim",
                min_ornek: int = 1) -> dict:
    """Hucre basina (13, 3) govde cercevesinde medyan ofset (en az `min_ornek` kare)."""
    out = {}
    for d in {k[anahtar] for k in kayitlar}:
        o = [k["ofset"] for k in kayitlar if k[anahtar] == d]
        if len(o) >= min_ornek:
            out[d] = np.nanmedian(np.stack(o), axis=0)
    return out


def ofset_uygula(M: np.ndarray, R: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Govde cercevesindeki ofseti (13, 3) kamera cercevesine cevirip ekler."""
    return M + o @ R.T


def main() -> None:
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv") if not t.mocap_hatali]
    ornek = tekrarlar[0].video
    X0 = np.load(tg.VERI / f"3d_joints/Ex6/{ornek}-30fps.npy")[::10, :, :3]
    kameralar = {c: tg.kamera_kestir(
        X0.reshape(-1, 3), np.load(tg.VERI / f"2d_joints/Ex6/{ornek}-{c}-30fps.npy")[::10]
        .reshape(-1, 2)) for c in tg.KAMERALAR}
    kayit = []
    for t in tekrarlar:
        kareler = tg.tekrar_kareleri(tg.VERI, t)
        for cam in tg.KAMERALAR:
            kam = kameralar[cam]
            d = tg._tespit_yukle("mediapipe", t.video, cam)
            if d is None or "dunya_tam" not in d:
                continue
            for j in range(0, len(kareler), hb.ADIM):
                kare = t.ilk + j
                if kare not in d["indeks"]:
                    continue
                M = np.asarray(d["dunya_tam"][d["indeks"][kare]], float)
                Pg = tg.kareyi_cevir(kareler[j]).noktalar
                G = hb.kok_merkezle((kam.R @ Pg.T + kam.t[:, None]).T)
                if not (np.isfinite(M).all() and np.isfinite(G).all()):
                    continue
                M = hb.olcek_esle(hb.kok_merkezle(M), G)
                R = cerceve(M)
                aci = govde_acisi(M)
                if R is None or not np.isfinite(aci):
                    continue
                fl = diz_fleksiyonu(hb._iskelet(M))
                kayit.append({"kisi": t.kisi, "yon": t.yon if cam == "c17" else tg.C18_YONU[t.yon],
                              "aci": aci, "dilim": dilim(aci), "M": M, "R": R,
                              "hucre": (dilim(aci), dilim(-1 if fl is None else fl,
                                                         FLEKS_DILIMLERI)),
                              "ofset": (G - M) @ R, "v_gt": hb.valgus(G), "v_tel": hb.valgus(M)})

    kisiler = sorted({k["kisi"] for k in kayit}, key=int)
    for kisi in kisiler:
        egitim = [k for k in kayit if k["kisi"] != kisi]
        tablo = ofset_ogren(egitim)
        tablo_fl = ofset_ogren(egitim, "hucre", MIN_ORNEK)
        # aci-kayma dogrusu (0034), ayni kareler ve ayni kisi-disari bolmeyle
        a = np.array([k["aci"] for k in egitim])
        f = np.array([np.nanmean(k["v_tel"] - k["v_gt"]) for k in egitim])
        ok = np.isfinite(f)
        b1, b0 = np.polyfit(a[ok], f[ok], 1)
        for k in (x for x in kayit if x["kisi"] == kisi):
            k["v_aci"] = k["v_tel"] - (b0 + b1 * k["aci"])
            o = tablo.get(k["dilim"])
            k["v_ofset"] = (hb.valgus(ofset_uygula(k["M"], k["R"], o)) if o is not None
                            else np.full(2, np.nan))
            o2 = tablo_fl.get(k["hucre"], o)
            k["v_ofset_fl"] = (hb.valgus(ofset_uygula(k["M"], k["R"], o2)) if o2 is not None
                               else np.full(2, np.nan))

    sonuc = {"deney": "rehab24_eklem_ofseti", "adim_kare": hb.ADIM, "yonler": {}}
    for yon in ("front", "half-profile", "profile"):
        s = [k for k in kayit if k["yon"] == yon]
        sonuc["yonler"][yon] = {
            ad: hb._ozet(np.concatenate([k[anahtar] - k["v_gt"] for k in s]))
            for ad, anahtar in (("ham", "v_tel"), ("aci_kaymasi", "v_aci"),
                                ("eklem_ofseti", "v_ofset"),
                                ("eklem_ofseti_fleks", "v_ofset_fl"))}
    (KOK / "out/rehab24_eklem_ofseti.json").write_text(
        json.dumps(tg._temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    for yon, v in sonuc["yonler"].items():
        print(f"== {yon}")
        for ad, o in v.items():
            print(f"   {ad:13s} kayma {o['ortalama']:+6.2f}  sd {o['sd']:6.2f}  "
                  f"|medyan| {o['mutlak_medyan']:5.2f}  n {o['n']}")


if __name__ == "__main__":
    main()
