#!/usr/bin/env python3
"""Deney: kalibrasyon suruklenmesi ne zaman fark edilir?

`uncertainty/drift.py` kayitlardan egri ciziyor. Bu deney, aracin **enjekte
edilen fiziksel kaymayi geri bulup bulmadigini** ve -- daha onemlisi -- ne
kadar kucuk bir kaymanin artik olcum saçilimindan ayirt edilemedigini olcer.
Donanim gerekmez; `KOD-PLANI.md` Adim 2 doktrininin suruklenmeye uygulanisi.

Ilk taslakta gozlenen ve tasarimi degistiren iki sey:

1. **Saçilim, kucuk suruklenmelerden kat kat buyuk.** 0,25 px tespit
   gurultusunde oturumlar arasi taban_01 saçilimi ~40 mm; 2 mm/hafta kayma
   8 haftada ancak 12 mm birikir. Yani "goruyor muyuz?" sorusunun cevabi
   `hayir, saçilimin altinda`. Asil olculebilir soru: **tespit esigi nerede?**
2. **Saçilimin kendisi de guvenilir olculecek.** 6 tekrarla std tahmini
   saçilimi kucuk gosteriyor (taban_01 icin 24,6 mm'ye karsi 30 tekrarda
   39,8 mm) ve esik yanlis yere konunca **kontrol kolu bile "suruklenme"
   gosteriyor**. Bu yuzden sigma, tekrar sayisina gore ayri raporlanir.

Olcum tasarimi:

1. **Sigma calismasi:** sabit duzenekte N_SIGMA=30 kalibrasyon; fx ve taban_01
   saçilimi ile bunun tekrar sayisina bagimliligi (5, 10, 15, 20, 30) olculur.
2. **Kontrol (sifir suruklenme):** 8 haftalik oturum, duzenek sabit. Egri
   tespit esiklerini asmamali. Falsifikasyon kolu budur.
3. **Kademeli suruklenme:** mekanik kayma (kamera 1 her hafta X mm yana kayar
   -- sarsinti/darbe; analitik beklenen taban_01 degisimi) ve termal odak
   suruklenmesi (kamera 0'in odagi %Y/hafta artar).
4. **Iki gurultu duzeyi (0,25 ve 0,10 px):** tespit esiginin tespit
   gurultusuyle olceklendigini gostermek icin.

**Tespit olcutu:** baz (0. hafta) ile hafta w farki, iki bagimsiz oturumun
farkidir; saçilimi sqrt(2)*sigma'dir. Esik = 3*sqrt(2)*sigma. Esigi sigma
sanan (3*sigma) koymak kontrol kolunda yanlis pozitif uretir.

Cikti: out/suruklenme_sentetik.json + docs/deney/ altina sekil. Deterministiktir
(tohumlar sabit), `make reproduce` ile yeniden uretilir.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Hem "python3 scripts/deney_...py" hem "python3 -m scripts.deney_...": proje kokunu
# path'e al ki betigi calistiran calisma dizinini dusunmek zorunda kalmasin.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calib.board import BoardSpec
from calib.io import kalibrasyondan
from calib.multiview import kalibre_et
from calib.synthetic import Kamera, rig_yay, sahne_uret
from uncertainty.drift import egri_olustur

SPEC = BoardSpec(5, 7)
GERCEK_ODAK = 900.0
N_KARE = 20
HAFTA = 8
SIGMA_KATSAYISI = 3.0
KAREKOK_IKI = math.sqrt(2.0)
TOHUM = 7311
BASLANGIC = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)

GURULTULER = (0.25, 0.10)
N_SIGMA = 30
TEKRAR_SAYILARI = (5, 10, 15, 20, 30)
MEKANIK_MM = (2.0, 5.0, 25.0)              # haftalik kayma
TERMAL_ORAN_YUZDE = (0.1, 0.25, 1.0)       # haftalik odak artisi (%)
YETERLILIK_ESIGI_YUZDE = 25.0              # "geri bulundu" sayilan bagil hata


@dataclass(frozen=True)
class Senaryo:
    ad: str
    takip: str          # izlenen parametre (drift modulu adi)
    birim: str          # "mm" veya "yuzde"
    buyuklukler: tuple[float, ...]


MEKANIK = Senaryo("mekanik_kayma", "taban_01", "mm", MEKANIK_MM)
TERMAL = Senaryo("termal_odak", "fx", "yuzde", TERMAL_ORAN_YUZDE)
SENARYOLAR = (MEKANIK, TERMAL)


# ---------------------------------------------------------------------------
# Duzenek degistirme yardimcilari
# ---------------------------------------------------------------------------

def _kamera_kaydir(kameralar: list[Kamera], indeks: int, kayma_m: float) -> list[Kamera]:
    """Bir kamerayi dunya X ekseninde kaydirir; merkez = -R^T t."""
    eksen = np.array([1.0, 0.0, 0.0])
    yeni = []
    for i, kam in enumerate(kameralar):
        if i != indeks:
            yeni.append(kam)
            continue
        merkez = kam.merkez + eksen * kayma_m
        yeni.append(Kamera(K=kam.K.copy(), R=kam.R.copy(),
                           t=(-kam.R @ merkez).reshape(3, 1), boyut=kam.boyut))
    return yeni


def _kamera_odak(kameralar: list[Kamera], indeks: int, oran: float) -> list[Kamera]:
    """Bir kameranin odak uzakligini oran kadar degistirir (termal genlesme)."""
    yeni = []
    for i, kam in enumerate(kameralar):
        if i != indeks:
            yeni.append(kam)
            continue
        K = kam.K.copy()
        K[0, 0] *= 1.0 + oran
        K[1, 1] *= 1.0 + oran
        yeni.append(Kamera(K=K, R=kam.R.copy(), t=kam.t.copy(), boyut=kam.boyut))
    return yeni


def _duzenek(senaryo: Senaryo, hafta: int, buyukluk: float, temel: list[Kamera]) -> list[Kamera]:
    """hafta. haftadaki fiziksel duzenek. buyukluk=0 -> kontrol."""
    if buyukluk == 0:
        return temel
    if senaryo.ad == MEKANIK.ad:
        return _kamera_kaydir(temel, 1, buyukluk / 1000.0 * hafta)
    return _kamera_odak(temel, 0, buyukluk / 100.0 * hafta)


def _beklenen(senaryo: Senaryo, buyukluk: float, temel: list[Kamera]) -> np.ndarray:
    """Ayni gurultude beklenen sapma (birim: senaryo.birim).

    Mekanikte beklenti analitiktir: kamera 0-1 merkez mesafesinin degisimi (mm).
    Termal'de enjekte edilen odak orani dogrudan fx sapmasidir (%).
    """
    if buyukluk == 0:
        return np.zeros(HAFTA)
    if senaryo.ad == MEKANIK.ad:
        cikti = []
        for w in range(HAFTA):
            kam = _kamera_kaydir(temel, 1, buyukluk / 1000.0 * w)
            cikti.append((float(np.linalg.norm(kam[1].merkez - kam[0].merkez))
                          - float(np.linalg.norm(temel[1].merkez - temel[0].merkez))) * 1000.0)
        return np.asarray(cikti)
    return np.asarray([buyukluk * w for w in range(HAFTA)])


# ---------------------------------------------------------------------------
# Sigma calismasi: saçilim kac tekrarla olculebilir?
# ---------------------------------------------------------------------------

def _tek_kalibrasyon(seed: int, gurultu: float) -> tuple[float, float]:
    """(fx piksel, taban_01 mm) -- sabit duzenekte tek oturum."""
    temel = rig_yay(odak_px=GERCEK_ODAK)
    sahne = sahne_uret(SPEC, kameralar=temel, n_kare=N_KARE,
                       gurultu_px=gurultu, dagilim="genis", seed=seed)
    kalib = kalibre_et(sahne.obj_noktalari, sahne.img_noktalari,
                       [c.boyut for c in sahne.kameralar], sahne.mask)
    return float(kalib.Ks[0][0, 0]), float(np.linalg.norm(kalib.Ts[1])) * 1000.0


def sigma_calismasi(gurultu: float) -> dict:
    """N_SIGMA kalibrasyon; saçilimin tekrar sayisina bagimliligini da verir."""
    degerler = [_tek_kalibrasyon(TOHUM + s, gurultu) for s in range(N_SIGMA)]
    fx = np.asarray([d[0] for d in degerler])
    taban = np.asarray([d[1] for d in degerler])
    return {
        "gurultu_px": gurultu,
        "fx_px": fx.tolist(),
        "taban_01_mm": taban.tolist(),
        "ortalama": {"fx_px": float(fx.mean()), "taban_01_mm": float(taban.mean())},
        "tekrar_bagimliligi": {
            str(n): {
                "fx_yuzde": float(fx[:n].std(ddof=1) / fx[:n].mean() * 100.0),
                "taban_01_mm": float(taban[:n].std(ddof=1)),
            } for n in TEKRAR_SAYILARI if n <= N_SIGMA
        },
    }


def sigma_sec(calisma: dict, senaryo: Senaryo) -> float:
    """Tam orneklemden sigma, senaryonun biriminde (mm veya %)."""
    fx = np.asarray(calisma["fx_px"])
    taban = np.asarray(calisma["taban_01_mm"])
    if senaryo.birim == "mm":
        return float(taban.std(ddof=1))
    return float(fx.std(ddof=1) / abs(fx.mean()) * 100.0)


# ---------------------------------------------------------------------------
# Haftalik kosular
# ---------------------------------------------------------------------------

def _egri(senaryo: Senaryo, buyukluk: float, gurultu: float, temel: list[Kamera]):
    """HAFTA haftalik oturum; her hafta farkli board pozu, ayni gurultu."""
    kayitlar = []
    for w in range(HAFTA):
        kameralar = _duzenek(senaryo, w, buyukluk, temel)
        sahne = sahne_uret(SPEC, kameralar=kameralar, n_kare=N_KARE,
                           gurultu_px=gurultu, dagilim="genis", seed=TOHUM + 100 * w)
        kalib = kalibre_et(sahne.obj_noktalari, sahne.img_noktalari,
                           [c.boyut for c in sahne.kameralar], sahne.mask)
        kayitlar.append(kalibrasyondan(
            kalib, n_kare=N_KARE, gorunurluk=sahne.gorunurluk,
            etiketler={"senaryo": senaryo.ad, "hafta": str(w)},
            tarih_iso=(BASLANGIC + timedelta(weeks=w)).isoformat()))
    return egri_olustur(kayitlar)


def _geri_bulunan(egri, senaryo: Senaryo) -> np.ndarray:
    """Izlenen parametrenin 0. haftaya gore sapmasi (senaryo.birim)."""
    seri = egri.seri(senaryo.takip)
    if senaryo.birim == "mm":
        return (seri - seri[0]) * 1000.0
    return (seri - seri[0]) / abs(seri[0]) * 100.0


def kosu(senaryo: Senaryo, buyukluk: float, gurultu: float,
         temel: list[Kamera], sigma: float) -> dict:
    """Tek (senaryo, buyukluk, gurultu) kosusu."""
    egri = _egri(senaryo, buyukluk, gurultu, temel)
    bulunan = _geri_bulunan(egri, senaryo)
    beklenen = _beklenen(senaryo, buyukluk, temel)

    # Baz ile w. hafta farki: iki bagimsiz oturumun farki -> sqrt(2)*sigma.
    esik = SIGMA_KATSAYISI * KAREKOK_IKI * sigma
    tespit = None
    for w in range(1, HAFTA):
        if abs(bulunan[w]) > esik:
            tespit = w
            break

    hafta_ramp = np.arange(HAFTA, dtype=float)
    trend = _korelasyon(hafta_ramp, bulunan)
    egilim = _trend_testi(bulunan, sigma)
    son_hata = (float(abs(bulunan[-1] - beklenen[-1]) / abs(beklenen[-1]) * 100.0)
                if abs(beklenen[-1]) > 1e-12 else None)
    return {
        "enjekte_egim": float(buyukluk),
        "egim": egilim["egim"],
        "egim_se": egilim["se"],
        "egim_t": egilim["t"],
        "trend_tespit": egilim["tespit"],
        "senaryo": senaryo.ad,
        "buyukluk": buyukluk,
        "birim": senaryo.birim,
        "gurultu_px": gurultu,
        "sigma": float(sigma),
        "tespit_esigi": float(esik),
        "geri_bulunan": bulunan.tolist(),
        "beklenen": beklenen.tolist(),
        "trend_korelasyon": float(trend),
        "tespit_haftasi": tespit,
        "son_hafta_beklenen": float(beklenen[-1]),
        "son_hafta_bagil_hata_yuzde": son_hata,
        "sinyal_esik_orani": (float(abs(beklenen[-1]) / esik) if esik else None),
        "geri_bulundu": bool(egilim["tespit"] and son_hata is not None
                             and son_hata < YETERLILIK_ESIGI_YUZDE),
    }


def _korelasyon(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _trend_testi(bulunan: np.ndarray, sigma: float) -> dict:
    """Egim testi: suruklenme karari tek hafta degil, **egilim** ile verilir.

    Tek haftalik esik asimi agir kuyruklu saçilimda yanlis alarm uretir (bu
    deneyde kontrol kolunda gozlendi). Dogru istatistik, tum haftalari kullanan
    egimdir: hafta farklarinin saçilimi sqrt(2)*sigma oldugu icin egimin standart
    hatasi SE = sqrt(2)*sigma / sqrt(sum((w-w_ort)^2)). |egim| > 3*SE ise
    suruklenme var.
    """
    hafta = np.arange(len(bulunan), dtype=float)
    merkez = hafta - hafta.mean()
    payda = float(np.sum(merkez ** 2))
    egim = float(np.sum(merkez * bulunan) / payda)
    se = KAREKOK_IKI * sigma / math.sqrt(payda)
    t = egim / se if se > 0 else 0.0
    return {"egim": egim, "se": se, "t": t, "tespit": bool(abs(t) > SIGMA_KATSAYISI)}


def _esik_buyuklukleri(kosular: list[dict], senaryo: Senaryo, gurultu: float):
    """Kabul edilebilir tespit edilen en kucuk suruklenme buyuklugu."""
    adaylar = [k["buyukluk"] for k in kosular
               if k["senaryo"] == senaryo.ad and k["gurultu_px"] == gurultu
               and k["buyukluk"] > 0 and k["geri_bulundu"]]
    return min(adaylar) if adaylar else None


def _kucuk_veya_esit(a, b) -> bool:
    """None = 'test edilen aralikta tespit edilemedi' (sonsuz esik)."""
    if a is None:
        return b is not None
    if b is None:
        return True
    return a <= b


# ---------------------------------------------------------------------------
# Sekil
# ---------------------------------------------------------------------------

def sekil_ciz(kosular: list[dict], sigmalar: dict, hedef: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12, 8.5))
    haftalar = np.arange(HAFTA)

    # (a) sigma tahmini tekrar sayisina bagli -- kucuk orneklem saçilimi kucuk gosterir
    for gurultu, renk in ((0.25, "#c0392b"), (0.10, "#2471a3")):
        calisma = sigmalar[gurultu]
        n_listesi = sorted(int(n) for n in calisma["tekrar_bagimliligi"])
        fx = [calisma["tekrar_bagimliligi"][str(n)]["fx_yuzde"] for n in n_listesi]
        tb = [calisma["tekrar_bagimliligi"][str(n)]["taban_01_mm"]
              / calisma["ortalama"]["taban_01_mm"] * 100.0 for n in n_listesi]
        a.plot(n_listesi, fx, "o-", color=renk, label=f"fx, {gurultu} px")
        a.plot(n_listesi, tb, "s--", color=renk, alpha=0.6,
               label=f"taban_01, {gurultu} px")
    a.set_yscale("log")
    a.set_xlabel("tekrar sayısı")
    a.set_ylabel("σ, % (log)")
    a.set_title("Sürüklenme eşiğini σ belirler;\nσ'yı az tekrarla ölçmek onu küçük gösterir",
                fontsize=9.5)
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=7.5)

    # (b)-(c) kontrol ve guclu suruklenme, tespit esigi bandiyla
    for eksen, senaryo, goster in ((b, MEKANIK, 25.0), (c, TERMAL, 1.0)):
        alt = [k for k in kosular if k["senaryo"] == senaryo.ad and k["gurultu_px"] == 0.25]
        if not alt:
            continue
        esik = alt[0]["tespit_esigi"]
        eksen.fill_between(haftalar, -esik, esik, color="#95a5a6", alpha=0.25,
                           label=f"±3√2σ tespit eşiği ({esik:.1f})")
        eksen.plot(haftalar, [k for k in alt if k["buyukluk"] == 0][0]["geri_bulunan"],
                   "o-", color="#7f8c8d", label="kontrol (0)")
        guclu = [k for k in alt if k["buyukluk"] == goster][0]
        etiket = (f"{goster:g} mm/hafta" if senaryo.birim == "mm" else f"%{goster:g}/hafta")
        eksen.plot(haftalar, guclu["geri_bulunan"], "o-", color="#2471a3", label=etiket)
        eksen.plot(haftalar, guclu["beklenen"], "k--", lw=1.0, label="beklenen")
        eksen.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        eksen.set_xticks(haftalar)
        eksen.set_xlabel("hafta")
        eksen.set_ylabel("sapma (mm)" if senaryo.birim == "mm" else "sapma (%)")
        eksen.set_title(("Mekanik kayma → taban uzunluğu" if senaryo.birim == "mm"
                         else "Termal/odak sürüklenmesi → fx") + ", 0,25 px", fontsize=9.5)
        eksen.grid(alpha=0.3)
        eksen.legend(fontsize=7.5)

    # (d) karar kurali: egim testi. Sinyal/SE buyudukce |t| buyur; |t| > 3 tespit.
    for senaryo, isaret in ((MEKANIK, "o"), (TERMAL, "s")):
        for gurultu, renk in ((0.25, "#c0392b"), (0.10, "#2471a3")):
            alt = [k for k in kosular if k["senaryo"] == senaryo.ad
                   and k["gurultu_px"] == gurultu and k["buyukluk"] > 0]
            if not alt:
                continue
            d.scatter([k["enjekte_egim"] / k["egim_se"] for k in alt],
                      [abs(k["egim_t"]) for k in alt],
                      marker=isaret, s=28, color=renk, alpha=0.85,
                      label=f"{senaryo.ad}, {gurultu} px")
    d.set_xscale("log")
    d.axhline(SIGMA_KATSAYISI, color="#7b241c", ls="--", lw=0.9)
    d.text(0.02, 0.9, "|t| = 3 → tespit", fontsize=7, transform=d.transAxes,
           color="#7b241c")
    d.set_xlabel("enjekte edilen eğim / SE  (log)")
    d.set_ylabel("|eğim t|")
    d.set_title("Karar kuralı: eğim testi (kontrol kolları |t| < 1)", fontsize=9.5)
    d.grid(alpha=0.3, which="both")
    d.legend(fontsize=7)

    fig.suptitle("Sentetik sürüklenme — tespit eşiği ölçüm saçılımı tarafından belirleniyor",
                 fontsize=11)
    fig.tight_layout()
    hedef.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hedef, dpi=150)
    plt.close(fig)
    return hedef


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main() -> int:
    temel = rig_yay(odak_px=GERCEK_ODAK)
    sigmalar, kosular = {}, []

    for gurultu in GURULTULER:
        calisma = sigma_calismasi(gurultu)
        sigmalar[gurultu] = calisma
        print(f"\n=== tespit gurultusu {gurultu:.2f} px | "
              + ", ".join(f"sigma({s.ad})={sigma_sec(calisma, s):.3f} {s.birim}"
                          for s in SENARYOLAR) + " ===")
        print("  sigma'nin tekrar sayisina bagimliligi:")
        for n in sorted(int(x) for x in calisma["tekrar_bagimliligi"]):
            d = calisma["tekrar_bagimliligi"][str(n)]
            print(f"    n={n:2}  fx %{d['fx_yuzde']:6.3f}   taban_01 {d['taban_01_mm']:7.2f} mm")

        print(f"{'senaryo':16} {'büyüklük':>10} {'sinyal/eşik':>11} {'eğim t':>8} "
              f"{'trend':>7} {'uyarı':>7} {'son hata %':>11}")
        for senaryo in SENARYOLAR:
            sigma = sigma_sec(calisma, senaryo)
            for buyukluk in (0.0,) + senaryo.buyuklukler:
                k = kosu(senaryo, buyukluk, gurultu, temel, sigma)
                kosular.append(k)
                etiket = "kontrol" if buyukluk == 0 else (
                    f"{buyukluk:g} mm" if senaryo.birim == "mm" else f"%{buyukluk:g}")
                oran = k["sinyal_esik_orani"]
                hata = k["son_hafta_bagil_hata_yuzde"]
                print(f"{senaryo.ad:16} {etiket:>10} "
                      f"{(f'{oran:11.2f}' if oran is not None else '          -')} "
                      f"{k['egim_t']:8.2f} "
                      f"{('OK' if k['trend_tespit'] else 'YOK'):>7} "
                      f"{str(k['tespit_haftasi']):>7} "
                      f"{(f'{hata:11.1f}' if hata is not None else '          -')}")

    # --- Kabul olcutleri ---------------------------------------------------
    # Karar kurali egim testidir; haftalik bant yalnizca erken uyari icin.
    kontrol_temiz = all(not k["trend_tespit"] for k in kosular if k["buyukluk"] == 0)
    uyari_yanlis_pozitif = sum(1 for k in kosular
                               if k["buyukluk"] == 0 and k["tespit_haftasi"] is not None)
    uyari_toplam = sum(1 for k in kosular if k["buyukluk"] == 0)
    en_kucuk_025 = {s.ad: _esik_buyuklukleri(kosular, s, GURULTULER[0]) for s in SENARYOLAR}
    en_kucuk_010 = {s.ad: _esik_buyuklukleri(kosular, s, GURULTULER[1]) for s in SENARYOLAR}
    kucuk_sinyal_gorunmez = all(en_kucuk_025[s.ad] != s.buyuklukler[0]
                                for s in SENARYOLAR)
    guclu = [k for k in kosular if k["senaryo"] == MEKANIK.ad
             and k["gurultu_px"] == GURULTULER[1] and k["buyukluk"] == MEKANIK_MM[-1]][0]
    guclu_geri_bulunuyor = bool(guclu["geri_bulundu"])
    esik_dusuyor = all(_kucuk_veya_esit(en_kucuk_010[s.ad], en_kucuk_025[s.ad])
                       for s in SENARYOLAR)

    kabul = {
        "kontrol_uydurmuyor": bool(kontrol_temiz),
        "kucuk_suruklenme_esik_altinda_kaliyor": bool(kucuk_sinyal_gorunmez),
        "guclu_suruklenme_geri_bulunuyor": bool(guclu_geri_bulunuyor),
        "esik_gurultuyle_dusuyor": bool(esik_dusuyor),
    }
    gecti = all(kabul.values())
    print(f"\nHaftalik bant uyarisi (kontrol kollari): "
          f"{uyari_yanlis_pozitif}/{uyari_toplam} yanlis pozitif "
          "-> tek haftayla karar verilmez, egim testi kullanilir")
    print(f"Kabul edilebilir tespit edilen en kucuk suruklenme (birim/hafta):")
    for s in SENARYOLAR:
        print(f"  {s.ad:16} 0,25 px: {en_kucuk_025[s.ad]}   0,10 px: {en_kucuk_010[s.ad]}")
    print(f"\nKABUL OLCUTLERI: {'GECTI' if gecti else 'KALDI'}")
    for ad, x in kabul.items():
        print(f"  {ad:40} {'OK' if x else 'YOK'}")

    cikti = Path("out/suruklenme_sentetik.json")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps({
        "deney": "sentetik_suruklenme_tespit_esigi",
        "donanim_gerekli": False,
        "tohum": TOHUM,
        "n_kare": N_KARE,
        "hafta": HAFTA,
        "n_sigma": N_SIGMA,
        "sigma_katsayisi": SIGMA_KATSAYISI,
        "karekok_iki_duzeltmesi": True,
        "yeterlilik_esigi_yuzde": YETERLILIK_ESIGI_YUZDE,
        "gurultuler_px": list(GURULTULER),
        "kabul_olcutleri_gecti": bool(gecti),
        "kabul": kabul,
        "karar_kurali": "egim testi: SE = sqrt(2)*sigma/sqrt(28); |t| > 3",
        "haftalik_uyari_kurali": "|sapma| > 3*sqrt(2)*sigma (erken uyari; tek basina karar degil)",
        "haftalik_uyari_yanlis_pozitif": {"sayi": uyari_yanlis_pozitif,
                                          "toplam": uyari_toplam},
        "sigma_calismalari": {str(g): sigmalar[g] for g in GURULTULER},
        "en_kucuk_tespit_edilen": {
            "0.25": en_kucuk_025, "0.10": en_kucuk_010},
        "kosular": kosular,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"veri : {cikti}")
    print(f"sekil: {sekil_ciz(kosular, sigmalar, Path('docs/deney/2026-09-22-suruklenme-sentetik.png'))}")
    return 0 if gecti else 1


if __name__ == "__main__":
    raise SystemExit(main())
