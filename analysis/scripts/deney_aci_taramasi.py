#!/usr/bin/env python3
"""Deney: form karari dogrulugu kamera acisina gore -- tezin manset grafigi.

Duzenek yer gercegini uretiyor, telefon tek kameradan kestiriyor. Soru: telefon
hangi acilarda dogru karar veriyor, ve yanlis karar verirken bunu bilebiliyor mu?

Iki kol kosuluyor:

- **Bilmeyen telefon:** kendi kestirim hatasini bilmiyor, esigi yalin
  karsilastiriyor. Telefon uygulamalarinin yaptigi sey budur.
- **Bilen telefon:** ayni kestirim, ama olculmus kendi hatasi belirsizlik
  olarak veriliyor. Karar verebilecegi yerde veriyor, veremeyecegi yerde susuyor.

Ikisinin farki, projenin "belirsizligi olcuyoruz" iddiasinin bedelini ve
karsiligini gosterir.

Donanim gerekmez, deterministiktir (tohum sabit). Cikti: out/ + docs/deney/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.aci_taramasi import aci_taramasi, kemik_onculeri, sentetik_poz_ureteci
from eval.form import Karar, form_degerlendir, sentetik_durus
from mono.phone import tek_gorus_3b
from pose3d.hizalama import rijit_hizala

ACILAR = [0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0]
N_DURUS = 300
TOHUM = 20260922
VALGUS = "diz_valgusu_sag"

# 2B tespit gurultusu. 0 px kasitli: bu deneyin olctugu sey **geometrik**
# bozulma. Gurultu eklenince egri daha da duser; bkz. Sinirliliklar.
GURULTU_PX = 0.0


def duruslar_uret() -> list:
    """Esigin iki yanina dagilmis, uc kusuru birlikte tasiyan duruslar."""
    rng = np.random.default_rng(TOHUM)
    return [
        sentetik_durus(
            valgus_sag=float(rng.uniform(0.0, 24.0)),
            kalca_hizasi=float(rng.uniform(-16.0, 16.0)),
            govde_rotasyonu=float(rng.uniform(-30.0, 30.0)),
        )
        for _ in range(N_DURUS)
    ]


def sekil_hatasi(duruslar: list, aci: float) -> float:
    """Telefonun kendi eklem hatasi (rijit hizalama sonrasi sekil RMS, metre).

    Mutlak konum degil sekil hatasi olculuyor: tek gorus kestirimi zaten
    telefonun kendi koordinatinda ve olcegi oncu uzunluklardan geliyor, yani
    konum farki karari etkilemez. Karari bozan sey iskeletin **bicimidir**.
    """
    uret, K = sentetik_poz_ureteci(gurultu_px=GURULTU_PX, seed=TOHUM)
    artiklar = []
    for d in duruslar:
        kestirim = tek_gorus_3b(uret(aci, d), K, kemik_onculeri(d))
        maske = kestirim.gorunur & d.gorunur
        if int(maske.sum()) >= 4:
            artiklar.append(
                rijit_hizala(kestirim.noktalar[maske], d.noktalar[maske]).rms_mm
            )
    return float(np.mean(artiklar)) / 1000.0 if artiklar else float("nan")


def eksen_hatalari(duruslar: list, aci: float) -> np.ndarray:
    """Telefonun kendi hatasi, **kamera ekseni basina** (sx, sy, sz; metre).

    Yer gercegi iskelet kestirime rijit hizalanir (kestirim telefon kamerasi
    cercevesinde oldugu icin artiklar da o cercevede kalir), sonra eksen basina
    RMS alinir. Tek gorus hatasinin derinlikte (Z) yogunlastigini sayisal
    gosterir; kovaryansli kahin kolunun girdisidir.
    """
    uret, K = sentetik_poz_ureteci(gurultu_px=GURULTU_PX, seed=TOHUM)
    artiklar = []
    for d in duruslar:
        kestirim = tek_gorus_3b(uret(aci, d), K, kemik_onculeri(d))
        maske = kestirim.gorunur & d.gorunur
        if int(maske.sum()) >= 4:
            h = rijit_hizala(d.noktalar[maske], kestirim.noktalar[maske])
            artiklar.append(kestirim.noktalar[maske] - h.uygula(d.noktalar[maske]))
    a = np.concatenate(artiklar)
    return np.sqrt(np.mean(a ** 2, axis=0))


def cizim(satirlar: list[dict], hedef: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [s["aci_derece"] for s in satirlar]
    fig, (ust, alt) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   height_ratios=[3, 2])

    ust.plot(x, [s["bilmeyen"]["dogruluk"] for s in satirlar],
             marker="o", color="#1f77b4", label="doğru karar oranı")
    ust.plot(x, [s["bilmeyen"]["yanlis_karar_orani"] for s in satirlar],
             marker="s", color="#d62728",
             label="yanlış karar oranı (karar verilenler içinde)")
    ust.plot(x, [s["bilen"]["karar_verilen_oran"] for s in satirlar],
             marker="^", ls="--", color="#2ca02c",
             label="belirsizliğini bilen telefon: karar verdiği oran")
    ust.axhline(0.5, color="#888", ls=":", lw=1)
    ust.text(1.5, 0.52, "yazı tura", fontsize=8, color="#666")
    ust.set_ylim(-0.03, 1.05)
    ust.set_ylabel("oran")
    ust.set_title("Form kararı doğruluğu vs kamera açısı (diz valgusu)")
    ust.grid(alpha=0.25)
    ust.legend(fontsize=8, loc="upper right", framealpha=0.9)

    alt.plot(x, [s["sekil_hatasi_mm"] for s in satirlar], marker="o", color="#9467bd")
    alt.set_yscale("log")
    alt.set_xlabel("kamera açısı (0° = önden, 90° = yandan)")
    alt.set_ylabel("tek görüş şekil hatası (mm)")
    alt.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(hedef, dpi=150)
    plt.close(fig)


def main() -> None:
    kok = Path(__file__).resolve().parent.parent
    duruslar = duruslar_uret()

    # Kol 1: telefon kendi hatasini bilmiyor.
    uret, K = sentetik_poz_ureteci(gurultu_px=GURULTU_PX, seed=TOHUM)
    bilmeyen = aci_taramasi(duruslar, ACILAR, uret, K)

    # Kol 2: her acida once telefonun kendi hatasi olculuyor, sonra ayni tarama
    # o hatayi belirsizlik olarak bilerek kosuluyor.
    satirlar = []
    for aci in ACILAR:
        sigma_m = sekil_hatasi(duruslar, aci)
        uret2, K2 = sentetik_poz_ureteci(gurultu_px=GURULTU_PX, seed=TOHUM)
        bilen = aci_taramasi(duruslar, [aci], uret2, K2,
                             konum_belirsizligi_m=sigma_m)
        b = bilen.noktalar[0].sayimlar[VALGUS]
        # Kol 3: ayni kahin, ama belirsizlik eksen basina (kovaryans). Izotrop
        # sigma derinlikteki hatayi goruntu duzlemine de yayar; bu kol "0,000
        # karar" sonucunun izotrop varsayimdan ne kadar kaynaklandigini olcer.
        eksen = eksen_hatalari(duruslar, aci)
        kov = np.tile(np.diag(eksen ** 2), (len(duruslar[0].tanim), 1, 1))
        uret3, K3 = sentetik_poz_ureteci(gurultu_px=GURULTU_PX, seed=TOHUM)
        c = aci_taramasi(duruslar, [aci], uret3, K3,
                         konum_belirsizligi_m=kov).noktalar[0].sayimlar[VALGUS]
        a = next(n for n in bilmeyen.noktalar if n.aci_derece == aci).sayimlar[VALGUS]
        satirlar.append({
            "aci_derece": aci,
            "sekil_hatasi_mm": round(sigma_m * 1000.0, 1),
            "bilmeyen": {
                "dogruluk": round(a.dogruluk, 4),
                "yanlis_karar_orani": round(a.yanlis_karar_orani, 4),
                "yanlis": a.yanlis, "belirsiz": a.belirsiz,
                "kacirma": a.kacirma, "yanlis_alarm": a.yanlis_alarm,
            },
            "bilen": {
                "dogruluk": round(b.dogruluk, 4),
                "yanlis": b.yanlis, "belirsiz": b.belirsiz,
                "karar_verilen_oran": round(b.karar_verilen_oran, 4),
            },
            "eksen_hatasi_mm": [round(float(v) * 1000.0, 1) for v in eksen],
            "bilen_anizotrop": {
                "dogruluk": round(c.dogruluk, 4),
                "yanlis": c.yanlis, "kacirma": c.kacirma, "belirsiz": c.belirsiz,
                "karar_verilen_oran": round(c.karar_verilen_oran, 4),
                "yanlis_karar_orani": (None if not np.isfinite(c.yanlis_karar_orani)
                                       else round(c.yanlis_karar_orani, 4)),
            },
        })

    # Egrinin dirsegi: dogrulugun frontal degerinin %90'inin altina ilk dustugu
    # aci. Mutlak esik kullanilmiyor, cunku frontal dogruluk da 1 degil -- tek
    # gorus kestirimi onden bakarken bile kusursuz degil.
    n_kusurlu = sum(form_degerlendir(d).olcumler[VALGUS].karar is Karar.KUSURLU
                    for d in duruslar)
    frontal = satirlar[0]["bilmeyen"]["dogruluk"]
    dirsek = next((s["aci_derece"] for s in satirlar
                   if s["bilmeyen"]["dogruluk"] < 0.90 * frontal), None)

    sonuc = {
        "experiment": "form_karari_aci_taramasi",
        "description": (
            "Tek kamera form karari dogrulugunun kamera acisina bagimliligi; "
            "belirsizligini bilen ve bilmeyen telefon karsilastirmasi."
        ),
        "physical_validation": False,
        "n_durus": N_DURUS,
        "tohum": TOHUM,
        "gurultu_px": GURULTU_PX,
        "olcum": VALGUS,
        "tarama": satirlar,
        "ozet": {
            "dirsek_aci_derece": dirsek,
            "frontal_dogruluk": satirlar[0]["bilmeyen"]["dogruluk"],
            "frontal_sekil_hatasi_mm": satirlar[0]["sekil_hatasi_mm"],
            "sagital_dogruluk": satirlar[-1]["bilmeyen"]["dogruluk"],
            "sagital_yanlis_karar_orani": satirlar[-1]["bilmeyen"]["yanlis_karar_orani"],
            "bilen_toplam_yanlis": sum(s["bilen"]["yanlis"] for s in satirlar),
            "anizotrop_frontal_karar_orani": satirlar[0]["bilen_anizotrop"]["karar_verilen_oran"],
            "anizotrop_frontal_dogruluk": satirlar[0]["bilen_anizotrop"]["dogruluk"],
            "anizotrop_toplam_yanlis": sum(s["bilen_anizotrop"]["yanlis"] for s in satirlar),
            "frontal_eksen_hatasi_mm": satirlar[0]["eksen_hatasi_mm"],
            "bilmeyen_toplam_yanlis": sum(s["bilmeyen"]["yanlis"] for s in satirlar),
            # Yanlis kararin turu (dis inceleme A.2). Payda: yer gercegine gore
            # gercekten kusurlu olan duruslar -- "kusurlarin kacta kacini kacirdi".
            "gercek_kusurlu_durus": n_kusurlu,
            "frontal_kacirilan_kusur_orani": round(satirlar[0]["bilmeyen"]["kacirma"] / n_kusurlu, 4),
            "sagital_kacirilan_kusur_orani": round(satirlar[-1]["bilmeyen"]["kacirma"] / n_kusurlu, 4),
            "bilmeyen_yanlislarin_kacirma_payi": round(
                sum(s["bilmeyen"]["kacirma"] for s in satirlar)
                / sum(s["bilmeyen"]["yanlis"] for s in satirlar), 4),
        },
    }

    cikti = kok / "out"
    cikti.mkdir(exist_ok=True)
    (cikti / "aci_taramasi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    cizim(satirlar, kok / "docs/deney/2026-09-22-aci-taramasi.png")
    print(json.dumps(sonuc["ozet"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
