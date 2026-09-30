"""REHAB24-6: fizyoterapistin "yanlis" karari hangi olcuyle aciklaniyor?

Telefonu tamamen disarida birakir: mocap iskeletinde (3B yer gercegi) aday
form olculeri hesaplanir ve fizyoterapist etiketini (correctness) ne kadar
ayirdiklari olculur. Soru, 3B mukemmel olsa bile kuralimizin (0014 valgus,
10 derece) uzmanla ayni karari verip vermedigidir.

Veri seti makalesine gore fizyoterapist **her kisiye farkli bir hata**
yaptirdi ve hata turu etiketlenmedi: tek bir olcunun her seyi aciklamasi
beklenmez. Bu yuzden AUC hem toplamda hem kisi basina verilir; kisinin en iyi
olcusu, o kisiye yaptirilan hatanin adayidir (cikarim, etiket degil).

Olculer (tekrar basina, mocap):
- valgus: 0014, 0,2 s kesintisiz tutulan en buyuk deger (iki diz)
- derinlik: en buyuk diz fleksiyonu
- govde_egimi: govde (kalca ortasi -> boyun) ile dikey arasindaki en buyuk aci
- diz_onde: diz, ayak bileginin ne kadar onunde (en buyuk, kaval boyuna oranla)
- kalca_kaymasi: kalca ortasinin ayak bilekleri ortasindan yana sapmasi
  (en buyuk, kalca genisligine oranla)
- asimetri: sag-sol diz fleksiyon farkinin en buyugu

Dikey eksen kayit basina, tekrarin ilk karelerinden (ayakta) kestirilir.

    python scripts/deney_rehab24_tanim.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.form import govde_cercevesi  # noqa: E402
from mono.canli import sagital_olculer  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tg", KOK / "scripts/deney_rehab24_tek_gorus.py")
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)
I = REFERANS_ISKELET.indeks  # noqa: E741
OLCULER = ("valgus", "derinlik", "govde_egimi", "diz_onde", "kalca_kaymasi", "asimetri")


def _birim(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def dikey_kestir(P0: np.ndarray) -> np.ndarray:
    """Ayakta karelerden (T, 13, 3) yukari birim vektor: ayak bilegi ortasi -> boyun."""
    ayak = (P0[:, I("sag_ayak_bilegi")] + P0[:, I("sol_ayak_bilegi")]) / 2
    return _birim(np.nanmean(P0[:, I("boyun")] - ayak, axis=0))


def _sag_sol_fleksiyon(P: np.ndarray) -> tuple[float, float]:
    out = []
    for taraf in ("sag", "sol"):
        h, k, a = P[I(f"{taraf}_kalca")], P[I(f"{taraf}_diz")], P[I(f"{taraf}_ayak_bilegi")]
        u, v = h - k, a - k
        c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        out.append(180 - np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out[0], out[1]


def tekrar_olculeri(iskeletler: list, yukari: np.ndarray, gerekli: int) -> dict:
    deg = np.array([tg._valgus_kararlari(s)[1] for s in iskeletler], float)
    valgus = -np.inf
    for s in range(2):
        v = deg[:, s]
        for i in range(len(v) - gerekli + 1):
            w = v[i:i + gerekli]
            if np.isfinite(w).all():
                valgus = max(valgus, float(w.min()))
    derinlik, egim, onde, kayma, asim = [], [], [], [], []
    for s in iskeletler:
        P = s.noktalar
        sg = sagital_olculer(s, yukari)          # canli modla ayni tanim
        derinlik.append(sg["derinlik"])
        egim.append(sg["govde_egimi"])
        onde.append(sg["diz_onde"])
        kalca = (P[I("sag_kalca")] + P[I("sol_kalca")]) / 2
        c = govde_cercevesi(s)
        if c is not None:
            yanal = _birim(c.yanal - (c.yanal @ yukari) * yukari)
            ayak = (P[I("sag_ayak_bilegi")] + P[I("sol_ayak_bilegi")]) / 2
            genislik = np.linalg.norm(P[I("sol_kalca")] - P[I("sag_kalca")])
            kayma.append(abs((kalca - ayak) @ yanal) / genislik)
        a, b = _sag_sol_fleksiyon(P)
        asim.append(abs(a - b))

    def enb(x):
        x = [v for v in x if np.isfinite(v)]
        return float(max(x)) if x else float("nan")
    return {"valgus": valgus if np.isfinite(valgus) else float("nan"),
            "derinlik": enb(derinlik), "govde_egimi": enb(egim), "diz_onde": enb(onde),
            "kalca_kaymasi": enb(kayma), "asimetri": enb(asim)}


def auc(pozitif, negatif) -> float | None:
    p = np.asarray([x for x in pozitif if np.isfinite(x)], float)
    n = np.asarray([x for x in negatif if np.isfinite(x)], float)
    if not p.size or not n.size:
        return None
    return float(((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])).mean())


def main() -> None:
    tekrarlar = [t for t in tg.tekrarlar_oku(tg.VERI / "Segmentation.csv") if not t.mocap_hatali]
    gerekli = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))
    yukari_kayit: dict[str, np.ndarray] = {}
    satirlar = []
    for t in tekrarlar:
        kareler = tg.tekrar_kareleri(tg.VERI, t)
        isk = [tg.kareyi_cevir(q) for q in kareler]
        if t.video not in yukari_kayit:
            yukari_kayit[t.video] = dikey_kestir(np.array([s.noktalar for s in isk[:10]]))
        o = tekrar_olculeri(isk, yukari_kayit[t.video], gerekli)
        karar = tg.tekrar_karari([tg._valgus_kararlari(s)[0] for s in isk])
        satirlar.append({"kisi": t.kisi, "video": t.video, "tekrar": t.tekrar_no,
                         "fizyoterapist_dogru": t.dogru, "kural_0014": karar, **o})

    def ayir(ss):
        return {m: (round(a, 3) if (a := auc([s[m] for s in ss if not s["fizyoterapist_dogru"]],
                                             [s[m] for s in ss if s["fizyoterapist_dogru"]]))
                    is not None else None) for m in OLCULER}

    kisiler = sorted({s["kisi"] for s in satirlar}, key=int)
    kisi_basina = {}
    for k in kisiler:
        ss = [s for s in satirlar if s["kisi"] == k]
        a = ayir(ss)
        # AUC 0,5'ten en uzak olcu: yon ne olursa olsun en cok ayiran
        gecerli = {m: v for m, v in a.items() if v is not None}
        en_iyi = max(gecerli, key=lambda m: abs(gecerli[m] - 0.5)) if gecerli else None
        kisi_basina[k] = {"n_dogru": sum(s["fizyoterapist_dogru"] for s in ss),
                          "n_yanlis": sum(not s["fizyoterapist_dogru"] for s in ss),
                          "auc": a, "en_iyi_olcu": en_iyi,
                          "kural_kusurlu_yanlista": sum(s["kural_0014"] == "kusurlu"
                                                        for s in ss if not s["fizyoterapist_dogru"]),
                          "kural_kusurlu_dogruda": sum(s["kural_0014"] == "kusurlu"
                                                       for s in ss if s["fizyoterapist_dogru"])}
    capraz = {f"fizyo_{'dogru' if d else 'yanlis'}->kural_{k}": sum(
        s["fizyoterapist_dogru"] == d and s["kural_0014"] == k for s in satirlar)
        for d in (True, False) for k in ("dogru", "kusurlu", "belirsiz")}
    sonuc = {"deney": "rehab24_tanim", "n_tekrar": len(satirlar), "n_kisi": len(kisiler),
             "kural_vs_fizyoterapist": capraz, "auc_toplam": ayir(satirlar),
             "kisi_basina": kisi_basina,
             "not": ("AUC: fizyoterapistin 'yanlis' dedigi tekrarda olcunun daha buyuk olma "
                     "olasiligi; <0,5 ters yon. Kisiye yaptirilan hata turu etiketli degil; "
                     "en_iyi_olcu bir cikarimdir.")}
    (KOK / "out/rehab24_tanim.json").write_text(
        json.dumps(tg._temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print("kural vs fizyoterapist:", capraz)
    print("AUC toplam:", sonuc["auc_toplam"])
    for k, v in kisi_basina.items():
        print(f"kisi {k:>2}: dogru {v['n_dogru']:2d} yanlis {v['n_yanlis']:2d} | en iyi "
              f"{v['en_iyi_olcu']:13s} | kural kusurlu: yanlista {v['kural_kusurlu_yanlista']}, "
              f"dogruda {v['kural_kusurlu_dogruda']} |", v["auc"])


if __name__ == "__main__":
    main()
