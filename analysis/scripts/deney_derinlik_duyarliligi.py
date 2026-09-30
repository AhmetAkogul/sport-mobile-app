"""Tek gorus 3B hatasinin ayristirilmasi: model hatasi vs 2B gurultu buyutmesi.

Uctan uca kosuda (2026-09-23) iki gozlem vardi: MJPG sikistirmasi 2B'de 3,1 px
oynatip 3B'de 82 mm; MediaPipe ile RTMPose 2B'de medyan 3,8 px ayrisip 3B'de
medyan 103 mm. Soru: bu buyutme nereden geliyor ve telefonun belirsizligini
yalnizca 2B gurultuden turetmek yeterli mi?

`tek_gorus_3b` derinligi s = f*L/l ile bulur (L oncu, l izdusum boyu, piksel).
Turevden: ds/s = -dl/l -- derinligin goreli hatasi, 2B kemik boyunun goreli
hatasina esittir (Taylor 2000'in tek goruntu yeniden kurulumunun bilinen
ozelligi). Bu betik iki bileseni **ayri** olcer:

1. Model hatasi -- gurultusuz 2B ile kestirim vs yer gercegi (sekil RMS).
   "Kemik kamera duzlemine paralel" varsayiminin bedeli.
2. Gurultu buyutmesi -- ayni karede gurultulu vs gurultusuz kestirim farki
   (ortak kare, model hatasi farkta sadelesir) ve analitik tahminle kiyasi.

    python scripts/deney_derinlik_duyarliligi.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from deney_aci_taramasi import duruslar_uret  # noqa: E402
from eval.aci_taramasi import kemik_onculeri, sentetik_poz_ureteci  # noqa: E402
from mono.phone import tek_gorus_3b  # noqa: E402
from pose3d.hizalama import rijit_hizala  # noqa: E402

ACILAR = [0.0, 45.0, 90.0]
SIGMALAR_PX = [0.5, 1.0, 2.0, 4.0]
TOHUM = 20260923


def _analitik_derinlik_sapmasi(poz, kestirim, sigma_px: float) -> np.ndarray:
    """Eklem basina ongorulen derinlik sapmasi (metre).

    Kemik basina: sigma_s = s * sqrt(2) * sigma_px / l_px (iki uc bagimsiz
    gurultulu; kemik yonundeki bilesen sqrt(2)*sigma). Eklem derinligi bagli
    kemiklerin ortalamasi oldugu icin ongoru de ortalama alinir (bagimsizlik
    varsayilmaz -- komsu kemikler ayni eklemi paylasir, kotumser tarafta kalir).
    """
    tanim = kestirim.tanim
    n = len(tanim)
    birikim: dict[int, list[float]] = {j: [] for j in range(n)}
    for a, b in tanim.baglantilar:
        ia, ib = tanim.indeks(a), tanim.indeks(b)
        if not (kestirim.gorunur[ia] and kestirim.gorunur[ib]):
            continue
        l_px = float(np.linalg.norm(poz.noktalar[ia] - poz.noktalar[ib]))
        if l_px < 1e-9:
            continue
        for j in (ia, ib):
            s = float(kestirim.noktalar[j, 2])
            birikim[j].append(s * np.sqrt(2.0) * sigma_px / l_px)
    return np.array([np.mean(v) if v else np.nan for v in birikim.values()])


def olc(duruslar, aci: float, rng) -> dict:
    uret, K = sentetik_poz_ureteci(gurultu_px=0.0, seed=TOHUM)
    model_rms, kemik_px = [], []
    buyutme = {s: {"gozlenen": [], "ongorulen": [], "derinlik": []} for s in SIGMALAR_PX}
    for d in duruslar:
        poz0 = uret(aci, d)
        if poz0 is None:                  # sentetik uretec her zaman poz dondurur
            continue
        onculer = kemik_onculeri(d)
        e0 = tek_gorus_3b(poz0, K, onculer)
        maske = e0.gorunur & d.gorunur
        if int(maske.sum()) >= 4:
            model_rms.append(rijit_hizala(e0.noktalar[maske], d.noktalar[maske]).rms_mm)
        for a, b in d.tanim.baglantilar:
            ia, ib = d.tanim.indeks(a), d.tanim.indeks(b)
            if poz0.gorunur[ia] and poz0.gorunur[ib]:
                kemik_px.append(float(np.linalg.norm(poz0.noktalar[ia] - poz0.noktalar[ib])))
        for sigma in SIGMALAR_PX:
            gurultu = rng.normal(0.0, sigma, poz0.noktalar.shape)
            e = tek_gorus_3b(replace(poz0, noktalar=poz0.noktalar + gurultu), K, onculer)
            ortak = e.gorunur & e0.gorunur
            dz = np.abs(e.noktalar[ortak, 2] - e0.noktalar[ortak, 2])
            d3 = np.linalg.norm(e.noktalar[ortak] - e0.noktalar[ortak], axis=1)
            ong = _analitik_derinlik_sapmasi(poz0, e0, sigma)[ortak]
            buyutme[sigma]["gozlenen"].extend(d3.tolist())
            # |N(0, s)|'nin medyani 0.674*s: ongoru medyan mutlak farka cevrilir.
            buyutme[sigma]["ongorulen"].extend((0.674 * ong).tolist())
            buyutme[sigma]["derinlik"].extend(dz.tolist())
    uzunluk = np.array(kemik_px)
    return {
        "aci_derece": aci,
        "model_hatasi_mm": round(float(np.mean(model_rms)), 1),
        "kemik_izdusumu_px_medyan": round(float(np.median(uzunluk)), 1),
        "kemik_izdusumu_px_p10": round(float(np.percentile(uzunluk, 10)), 1),
        "gurultu": [
            {
                "sigma_px": s,
                "eklem_3b_fark_mm_medyan": round(1000 * float(np.median(v["gozlenen"])), 1),
                "eklem_3b_fark_mm_p95": round(1000 * float(np.percentile(v["gozlenen"], 95)), 1),
                "derinlik_fark_mm_medyan": round(1000 * float(np.median(v["derinlik"])), 1),
                "analitik_derinlik_mm_medyan": round(1000 * float(np.nanmedian(v["ongorulen"])), 1),
            }
            for s, v in buyutme.items()
        ],
    }


def cizim(satirlar: list[dict], hedef: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (sol, sag) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in satirlar:
        x = [g["sigma_px"] for g in s["gurultu"]]
        cizgi, = sol.plot(x, [g["eklem_3b_fark_mm_medyan"] for g in s["gurultu"]],
                          marker="o", label=f"{s['aci_derece']:.0f}°: gürültü payı (medyan)")
        sol.axhline(s["model_hatasi_mm"], linestyle="--", alpha=0.6, color=cizgi.get_color())
    sol.set_xlabel("2B tespit gürültüsü σ (px)")
    sol.set_ylabel("3B fark (mm)")
    sol.set_title("Kesikli: gürültüsüz model hatası (şekil RMS)", fontsize=10)
    sol.legend(fontsize=8)
    sol.grid(alpha=0.3)

    tum_g = [g for s in satirlar for g in s["gurultu"]]
    sag.scatter([g["analitik_derinlik_mm_medyan"] for g in tum_g],
                [g["derinlik_fark_mm_medyan"] for g in tum_g])
    ust = max(max(g["analitik_derinlik_mm_medyan"] for g in tum_g),
              max(g["derinlik_fark_mm_medyan"] for g in tum_g)) * 1.1
    sag.plot([0, ust], [0, ust], color="gray", linestyle=":")
    sag.set_xlabel("analitik öngörü: 0,674·s·√2·σ/l (mm, medyan)")
    sag.set_ylabel("gözlenen derinlik farkı (mm, medyan)")
    sag.set_title("Derinlik hatası ≈ 2B kemik boyunun göreli hatası", fontsize=10)
    sag.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(hedef, dpi=150)
    plt.close(fig)


def main() -> None:
    kok = Path(__file__).resolve().parent.parent
    duruslar = duruslar_uret()
    rng = np.random.default_rng(TOHUM)
    satirlar = [olc(duruslar, aci, rng) for aci in ACILAR]

    on = satirlar[0]
    bir_px = next(g for g in on["gurultu"] if g["sigma_px"] == 1.0)
    sonuc = {
        "experiment": "tek_gorus_derinlik_duyarliligi",
        "description": ("Tek gorus 3B hatasinin model hatasi ve 2B gurultu "
                        "buyutmesi olarak ayristirilmasi."),
        "physical_validation": False,
        "n_durus": len(duruslar),
        "tohum": TOHUM,
        "satirlar": satirlar,
        "ozet": {
            "frontal_model_hatasi_mm": on["model_hatasi_mm"],
            "frontal_1px_gurultu_payi_mm": bir_px["eklem_3b_fark_mm_medyan"],
            "frontal_model_gurultu_orani_1px": round(
                on["model_hatasi_mm"] / bir_px["eklem_3b_fark_mm_medyan"], 1),
            "analitik_ongoru_orani_medyan": round(float(np.median([
                g["derinlik_fark_mm_medyan"] / g["analitik_derinlik_mm_medyan"]
                for s in satirlar for g in s["gurultu"]
                if g["analitik_derinlik_mm_medyan"] > 0])), 3),
        },
    }
    cikti = kok / "out"
    cikti.mkdir(exist_ok=True)
    (cikti / "derinlik_duyarliligi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    cizim(satirlar, kok / "docs/deney/2026-09-23-derinlik-duyarliligi.png")
    print(json.dumps({"ozet": sonuc["ozet"], "satirlar": satirlar}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
