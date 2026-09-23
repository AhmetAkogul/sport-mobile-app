"""Telefon hatti: kayit -> 2B poz -> 3B kestirim (iskelet).

Tuketici katmaninin kod karsiligi. Bu modulun iki siniri vardir ve ikisi de
bilincli kararidir:

1. **2B kestirimi bu modul yazmaz.** MediaPipe / RTMPose / Roboflow gibi hazir
   modeller takilir (`HAZIR-TEKNOLOJILER.md`); burada yalnizca **adaptor
   sozlesmesi** sabitlenir: cagiran, goruntuden `Poz2B` donduren bir cagrilabilir
   verir. Adaptorun sorumluluklari madde 2 ve 3'te yazan her seydir -- eklem
   isimlerini tasinmasi, letterbox geri donusumu, guven/maske bildirimi.
2. **Tek gorus 3B kestirimi kasitli olarak ilkel.** Tek kameradan derinlik
   cikmaz; bu projenin tum tezi bu. Baseline, kemik uzunlugu onculu ile olcekli
   tek gorus kestirimidir (asagida belgelenmis diklem varsayimiyla).Bu kestirimin hatasi, Faz 4'te duzenek yer gercegine karsi olculecek
      nesnedir. Duzelten sey `correction/` olacak (Kapi 4).

Karsilastirma notu: duzenek hatti 3B'yi kamera 0 koordinatinda uretir; bu modul
telefon kamerasinin kendi koordinatinda. Ikisi `pose3d.hizalama` ile rijit
hizalanmadan karsilastirilamaz -- sabit bir donusum farki hata gibi gorunur.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable
import math
import warnings

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B, IskeletTanimi, eslestir
from pose3d.pose2d import Poz2B, letterbox_donusumu

# Kestirici sozlesmesi: goruntu (BGR uint8) -> ozgun goruntu uzayinda Poz2B.
Kestirici = Callable[[np.ndarray], Poz2B]

# Kemik uzunlugu onculeri: {eklem cifti: metre}. Cift sirasi onemli degil.
UzunlukOnculeri = dict[tuple[str, str], float]


def json_uyumlu(deger):
    """NumPy dizilerini JSON'a cevrilecek hale getir; NaN/Inf -> None.

    Politikayi tek yerde tutar: eksik eklem NaN'dir, JSON'da `null` olur. 0.0
    gecerli bir piksel koordinatidir ve "yok" bilgisini saklamaz. `allow_nan=False`
    ile yazan her cikti bu fonksiyondan gecmelidir (dis inceleme U.3/U.6/X.4).
    """
    if isinstance(deger, np.ndarray):
        return [json_uyumlu(v) for v in deger.tolist()]
    if isinstance(deger, np.generic):
        return json_uyumlu(deger.item())
    if isinstance(deger, dict):
        return {json_uyumlu(k): json_uyumlu(v) for k, v in deger.items()}
    if isinstance(deger, (list, tuple)):
        return [json_uyumlu(v) for v in deger]
    if isinstance(deger, float) and not math.isfinite(deger):
        return None
    return deger


@dataclass(frozen=True)
class KareKaydi:
    """Telefon kaydindan gelen tek kare. Zaman bilgisi varsa tasinir."""

    kare: int
    image_bgr: np.ndarray
    zaman_ms: float | None = None
    kamera_id: str = "telefon"


@dataclass
class TelefonSonucu:
    """Hat ciktisi: 2B pozlar + tek gorus 3B iskeletler + ozet."""

    pozlar: list[Poz2B]
    iskeletler: list[Iskelet3B]
    tanim: IskeletTanimi

    def ozet(self) -> dict:
        """JSON'a donebilir ozet -- determinism kabul testinin olcutu."""
        n = len(self.pozlar)
        goren = [p.gorunur_oran for p in self.pozlar]
        eklem_gorunurluk = {}
        for j, eklem in enumerate(self.tanim.eklemler):
            eklem_gorunurluk[eklem] = float(
                np.mean([s.gorunur[j] for s in self.iskeletler])) if n else 0.0
        modeller = sorted({p.model for p in self.pozlar})
        zamanlar = [p.ek.get("zaman_ms") for p in self.pozlar]
        return {
            "n_kare": n,
            "modeller": modeller,
            "iskelet": self.tanim.ad,
            "ortalama_gorunur_oran": float(np.mean(goren)) if goren else 0.0,
            "eklem_gorunurluk": eklem_gorunurluk,
            "zamani_bilinen_kare": sum(z is not None for z in zamanlar),
        }


# ---------------------------------------------------------------------------
# Tek gorus 3B baseline
# ---------------------------------------------------------------------------

def tek_gorus_3b(
    poz: Poz2B,
    K: np.ndarray,
    uzunluklar_m: UzunlukOnculeri,
    tanim: IskeletTanimi = REFERANS_ISKELET,
) -> Iskelet3B:
    """Tek kameradan 2B pozu, kemik uzunlugu onculeriyle 3B'ye olcekle.

    Yontem ve varsayimlar (belgelenmesi sart, cunku hata analizinin nesnesi):

    - Pinhole ve **kare piksel** varsayimi; telefon kalibrasyonu Faz 1'de gelir.
    - Derinlik, gorunen her kemic icin ``s = f * L / l`` ile kestirilir:
      l goruntudeki kemik piksel uzunlugu, L gercek uzunluk, f odak.
      Bu, kemiğin **kamera duzlemine paralel** oldugu varsayimidir; egik kemik
      icin derinligi sistematik olarak sasar. Sasma buyuklugu tezin olcecegi
      seyin ta kendisi.
    - Eklem derinligi, kendine bagli kemiklerin derinliklerinin ortalamasi;
      kemiklerin onerdigi derinlikler birbirini tutmuyorsa bu **saklanir**
      (`ek["derinlik_sacilimi_m"]`), cunku buyuk acilarda uyusmazlik buyur ve
      "bu eklemin derinligi guvenilir mi" sorusunun cevabi budur.
    - Hicbir kemige bagli olmayan eklem uydurulmaz, gorunmez isaretlenir.
    - Onculu olmayan kemik atlanir ve atlananlar **kaydedilir**
      (`ek["atlanan_kemikler"]`): hata analizinde "kac eklem eksik oncu yuzunden
      dustu" sorusu bu alandan cevaplanir. Eksik veri sessizce doldurulmaz.

    Cikti telefon kamerasinin koordinat sistemindedir (metre).
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K (3, 3) olmali, {K.shape} geldi")
    if not np.isfinite(K).all():
        raise ValueError("K sonlu olmali")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("odak uzakliklari pozitif olmali")
    if abs(fx - fy) / max(fx, fy) > 0.02:
        # Kare piksel varsayimi bozulmus: f ortalama alinir, ama bu sistematik
        # bir hata kaynagidir ve sessizce gecmemeli.
        warnings.warn(
            f"fx={fx:.2f} ve fy={fy:.2f} farkli (>{0.02:.0%}); tek gorus 3B kare "
            "piksel varsayar, sonuc sistematik olarak sapabilir",
            stacklevel=2)
    f_ort = 0.5 * (fx + fy)

    # Model iskeletini referans tanima isimle tasi; indeks esitligi varsayilmaz.
    harita = eslestir(poz.iskelet, tanim)
    noktalar2 = np.full((len(tanim), 2), np.nan)
    gorunur = np.zeros(len(tanim), dtype=bool)
    for j, k in enumerate(harita):
        if k >= 0 and poz.gorunur[k]:
            noktalar2[j] = poz.noktalar[k]
            gorunur[j] = True

    if not isinstance(uzunluklar_m, dict):
        raise ValueError("oncu uzunluklari sozluk olmali: {(eklem, eklem): metre}")
    normallestirilmis = {}
    for (a, b), uzunluk in uzunluklar_m.items():
        if not np.isfinite(uzunluk) or uzunluk <= 0:
            raise ValueError(f"oncu uzunluk sonlu ve pozitif olmali: {a}-{b}={uzunluk}")
        normallestirilmis[frozenset((a, b))] = float(uzunluk)

    # Iskelette karsiligi olmayan oncu (orn. sag_bilek-sol_ayak_bilegi) hicbir
    # kemige uymaz ve sessizce "hic eklem uretilmedi" sonucuna doner: reddedilir.
    kemikler = {frozenset(c) for c in tanim.baglantilar}
    tanimsiz = sorted(tuple(sorted(k)) for k in normallestirilmis if k not in kemikler)
    if tanimsiz:
        raise ValueError(f"iskelette tanimli olmayan kemik icin oncu: {tanimsiz}")

    # Onculu olmayan kemikler atlanir; hangileri oldugu kaydedilir (dis inceleme P.2).
    atlanan = [(a, b) for a, b in tanim.baglantilar
               if frozenset((a, b)) not in normallestirilmis]

    derinlik_toplam: dict[int, list[float]] = {}
    for a, b in tanim.baglantilar:
        anahtar = frozenset((a, b))
        L = normallestirilmis.get(anahtar)
        if L is None:
            continue
        ia, ib = tanim.indeks(a), tanim.indeks(b)
        if not (gorunur[ia] and gorunur[ib]):
            continue
        l_px = float(np.linalg.norm(noktalar2[ia] - noktalar2[ib]))
        if l_px < 1e-9:
            continue                      # sifir uzunluklu izdusum derinlik vermez
        s = f_ort * L / l_px
        derinlik_toplam.setdefault(ia, []).append(s)
        derinlik_toplam.setdefault(ib, []).append(s)

    n = len(tanim)
    noktalar3 = np.full((n, 3), np.nan)
    goren = np.zeros(n, dtype=int)
    artik = np.full(n, np.nan)            # tek goruste yeniden izdusum artigi tanimsiz
    sacilim = {}
    for j in range(n):
        liste = derinlik_toplam.get(j)
        if not liste:
            continue
        s = float(np.mean(liste))
        u, v = noktalar2[j]
        noktalar3[j] = ((u - cx) * s / fx, (v - cy) * s / fy, s)
        goren[j] = 1                      # tek gorus: goren kamera sayisi en fazla 1
        # Kemiklerin onerdigi derinlikler uyusmuyorsa sakla: derinlik guvenilirligi.
        sacilim[tanim.eklemler[j]] = float(np.std(liste))
    gorunur = goren.astype(bool)
    return Iskelet3B(tanim=tanim, noktalar=noktalar3, gorunur=gorunur,
                     goren_kamera=goren, artik_px=artik,
                     ek={"atlanan_kemikler": atlanan, "derinlik_sacilimi_m": sacilim})


# ---------------------------------------------------------------------------
# Hat
# ---------------------------------------------------------------------------

def hatti_kostur(
    kareler: Iterable[KareKaydi],
    kestirici: Kestirici,
    K: np.ndarray,
    uzunluklar_m: UzunlukOnculeri,
    tanim: IskeletTanimi = REFERANS_ISKELET,
    guven_esigi: float | None = None,
) -> TelefonSonucu:
    """Kayittan 3B iskelete kadar telefon hatti.

    Sozlesme denetimleri burada toplanir; adaptor bozuk cikti verirse hata
    sessizligi burada yakalanir:

    - Adaptor `uzay="ozgun"` dondurmek zorunda (HAZIR-TEKNOLOJILER madde 3).
      Letterbox'li adaptor donusumu kendisi geri almali; hat model uzayindaki
      pozu kabul etmez.
    - Esik **adaptorun isidir** ve modele ozgudur (MediaPipe 0.5, SIMCC 0.3;
      `mono/backend.py`). Hat varsayilan olarak yeniden esiklemez: sabit bir
      0.5 burada RTMPose'un 0.3 maskesini sessizce ezer (0.4 skorlu 13 eklem
      sifira duser). `guven_esigi` verilirse ek bir **daraltma** olarak
      uygulanir; nokta silinmez, maskelenir.
    """
    pozlar: list[Poz2B] = []
    iskeletler: list[Iskelet3B] = []
    for kayit in kareler:
        if kayit.image_bgr.ndim != 3:
            raise ValueError(f"kare {kayit.kare}: goruntu 3 kanalli olmali")
        poz = kestirici(kayit.image_bgr)
        if poz.uzay != "ozgun":
            raise ValueError(
                f"kare {kayit.kare}: adaptor model uzayinda poz dondurdu -- "
                "sozlesme geregi ozgun goruntu koordinatina geri cevrilmeli")
        if kayit.zaman_ms is not None:
            poz = replace(poz, ek={**poz.ek, "zaman_ms": float(kayit.zaman_ms)})
        if guven_esigi is not None:
            poz = poz.guven_esikle(guven_esigi)
        pozlar.append(poz)
        iskeletler.append(tek_gorus_3b(poz, K, uzunluklar_m, tanim=tanim))
    return TelefonSonucu(pozlar=pozlar, iskeletler=iskeletler, tanim=tanim)


# ---------------------------------------------------------------------------
# Test/dev kestirici -- gercek model takilana kadar hatti uçtan uca kosturur
# ---------------------------------------------------------------------------

def sentetik_kestirici(
    kamera,
    iskeletler_3b: list[np.ndarray],
    tanim: IskeletTanimi = REFERANS_ISKELET,
    model_boyutu: tuple[int, int] | None = None,
    gurultu_px: float = 0.0,
    seed: int = 0,
) -> Kestirici:
    """Bilinen 3B iskeletleri izdusuren deterministik sahte model.

    Gercek MediaPipe takilana kadar hatti test etmek icin var; goruntunun
    icerigine bakmaz, kare sirasina gore `iskeletler_3b` listesinden izdusurur.
    `model_boyutu` verilirse poz once letterbox'lanmis model uzayinda
    uretilip geri donusturulur -- sozlesme maddesi 3'un yolunu calistirir.
    """
    from pose3d.pose2d import sentetik_poz

    durum = {"kare": 0}

    def kestir(image_bgr: np.ndarray) -> Poz2B:
        i = durum["kare"]
        durum["kare"] += 1
        if i >= len(iskeletler_3b):
            raise ValueError(f"kare {i}: sentetik iskelet listesi bitti")
        poz = sentetik_poz(kamera, iskeletler_3b[i], tanim=tanim,
                           gurultu_px=gurultu_px, model="sentetik-baseline")
        if model_boyutu is not None:
            don = letterbox_donusumu(kamera.boyut, model_boyutu)
            poz = replace(poz, noktalar=don.ileri(poz.noktalar), uzay="model")
            poz = poz.ozgun_uzaya(don)
        return poz

    return kestir
