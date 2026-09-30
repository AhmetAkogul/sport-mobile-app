"""Senkron kaymasinin 3B konum hatasina cevrilmesi.

`PROJE-PLANI.md` Kapi 2: "Senkron kaymasi -- **olculmus ve mm cinsinden ifade
edilmis**". Risk 3'un azaltma adimi da ayni: "kaymayi olc, mm'ye cevir, sinir
olarak raporla". Bu modul o cevrimi yapar.

Fizik basit ama sonucu belirleyici: kameralar ayni ani goruntulemezse, hareketli
bir eklemi **farkli konumlarda** gorurler. Ucgenleme bu tutarsiz isinlari
kesistirmeye calisir ve ortaya, hicbir kameranin gormedigi bir nokta cikar.
Hata kaymanin kendisiyle degil, kameralar **arasindaki** kayma farkiyla olusur:
butun kameralar 10 ms gec tetiklenirse hata yoktur, yalnizca zaman otelenir.

Iki rejim ayri ayri olculuyor, cunku donanim secimi ikisinden hangisine
dustugune bagli:

- **tek_kamera** -- bir kamera gec kaliyor (dusen kare, yavas tetik). Digerleri
  hizali.
- **dagilmis** -- kameralar serbest kosuyor, faz iliskisi yok. Kayma
  [-T/2, +T/2] araliginda duzgun dagilir; T kare periyodudur (30 fps icin
  33 ms). Donanim senkronu olmayan kurulumun tam karsiligi budur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calib.multiview import Kalibrasyon
from calib.synthetic import Kamera, nokta_gozlemleri
from pose3d.triangulate import ucgenle

REJIMLER = ("tek_kamera", "dagilmis")


@dataclass(frozen=True)
class KaymaSonucu:
    """Bir (kayma, hiz, rejim) uclusunun 3B hata ozeti."""

    kayma_ms: float
    hiz_m_s: float
    rejim: str
    ortalama_mm: float
    medyan_mm: float
    p95_mm: float
    n_ornek: int
    n_basarisiz: int          # iki gorusun altina dusup ucgenlenemeyen ornek

    def ozet(self) -> dict:
        return {
            "kayma_ms": self.kayma_ms,
            "hiz_m_s": self.hiz_m_s,
            "rejim": self.rejim,
            "ortalama_mm": round(self.ortalama_mm, 3),
            "medyan_mm": round(self.medyan_mm, 3),
            "p95_mm": round(self.p95_mm, 3),
            "n_ornek": self.n_ornek,
            "n_basarisiz": self.n_basarisiz,
        }


def _kamera_kaymalari(rejim: str, n_kamera: int, kayma_ms: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Rejime gore kamera basina zaman kaymasi (saniye).

    Kaymalar ortalamasi cikarilarak merkezlenir: ortak kayma hata uretmez,
    yalnizca zamani oteler. Olculen sey kameralar **arasindaki** farktir.

    `kayma_ms`'nin anlami rejime gore degisir (dis inceleme S.2):

    - `tek_kamera`: bir kamera digerlerinden tam `kayma_ms` geride (tek bozuk
      tetik hatti). Merkezleme sonrasi en buyuk iki-kamera farki `kayma_ms`.
    - `dagilmis`: her kamera [-kayma_ms/2, +kayma_ms/2] araliginda tekduze
      (serbest calisan kameralar). En buyuk fark **en fazla** `kayma_ms`,
      tipik olarak daha az. Iki rejim ayni sayida ayni "kotuluk" degildir;
      raporda rejim adi her zaman sayinin yaninda yazilir.
    """
    if rejim == "tek_kamera":
        k = np.zeros(n_kamera)
        k[-1] = kayma_ms / 1000.0
    elif rejim == "dagilmis":
        yari = kayma_ms / 2000.0
        k = rng.uniform(-yari, yari, size=n_kamera)
    else:
        raise ValueError(f"bilinmeyen rejim: {rejim} (beklenen: {REJIMLER})")
    return k - k.mean()


def kayma_hatasi(
    kalib: Kalibrasyon,
    kameralar: list[Kamera],
    nokta_dunya: np.ndarray,
    hiz_dunya: np.ndarray,
    kaymalar_s: np.ndarray,
    gurultu_px: float = 0.0,
    seed: int = 0,
) -> float | None:
    """Tek bir noktanin, verilen kamera kaymalariyla olusan 3B hatasi (metre).

    Her kamera noktayi **kendi aninda** gorur: p(t_i) = p0 + v * t_i. Ucgenleme
    bu tutarsiz gozlemlerden yapilir.

    Olculen sey, **ayni kalibrasyonla** kaymasiz ucgenlemeye gore kayma:
    referans, butun kameralarin ortalama yakalama aninda goruntuledigi durumun
    kestirimidir. Iki sebeple boyle:

    1. **Cerceveden bagimsiz.** Mutlak konum, kestirilen kalibrasyonun
       cercevesi gercek cerceveden biraz kaydigi icin kalibrasyon hatasiyla
       kirlenir; hacim haritasi deneyi ayni tuzagi kaydetmisti. Iki kestirim
       ayni kalibrasyondan gectigi icin o bilesen sadelesir ve geriye yalnizca
       senkronun katkisi kalir.
    2. **Ortak gecikme zararsizdir.** Butun kameralar birlikte gec tetiklenirse
       iskelet bozulmaz, yalnizca zamanda otelenir; eklem acilari -- yani form
       karari -- etkilenmez. Gereksinim ortak gecikmeye gore sikilastirilmamali.
    3. **Gurultu farkta sadelesir (bilincli).** Iki kestirim ayni tohumla
       (`seed + i`) ayni piksel gurultusunu alir; bu "ortak rastgele sayilar"
       teknigidir. Sonuc yalnizca **kaymanin yalitilmis katkisidir**; gurultunun
       katkisi hata butcesi ve hacim haritasinda ayrica olculur. Toplam hata
       istenirse iki katki ayri raporlanir, burada karistirilmaz (dis inceleme S.1).

    Iki gorusun altina duserse None doner -- basarisizlik sessizce sifir hata
    sayilmaz.
    """
    p0 = np.asarray(nokta_dunya, float)
    v = np.asarray(hiz_dunya, float)
    referans_t = float(np.mean(kaymalar_s))

    def kestir(zamanlar: np.ndarray):
        gozlem: dict[int, np.ndarray] = {}
        for i, (kam, dt) in enumerate(zip(kameralar, zamanlar)):
            tek = nokta_gozlemleri([kam], (p0 + v * dt).reshape(1, 3),
                                   gurultu_px=gurultu_px, seed=seed + i)[0]
            if 0 in tek:
                gozlem[i] = tek[0]
        if len(gozlem) < 2:
            return None
        sonuc = ucgenle(kalib, gozlem)
        return sonuc.nokta if sonuc.gecerli else None

    kaymali = kestir(np.asarray(kaymalar_s, float))
    kaymasiz = kestir(np.full(len(kameralar), referans_t))
    if kaymali is None or kaymasiz is None:
        return None
    return float(np.linalg.norm(kaymali - kaymasiz))


def kayma_taramasi(
    kalib: Kalibrasyon,
    kameralar: list[Kamera],
    kaymalar_ms: list[float],
    hizlar_m_s: list[float],
    rejimler: tuple[str, ...] = REJIMLER,
    n_ornek: int = 200,
    hacim_yaricap_m: float = 0.6,
    gurultu_px: float = 0.0,
    seed: int = 20260922,
) -> list[KaymaSonucu]:
    """Kayma x hiz x rejim taramasi.

    Her ornekte nokta hacimde rastgele bir yere, hiz **rastgele bir yone**
    konur: hata geometriye bagli oldugu icin tek bir yon yaniltir. Derinlik
    ekseni boyunca hareket, yanal harekete gore farkli hata uretir.
    """
    rng = np.random.default_rng(seed)
    sonuclar: list[KaymaSonucu] = []

    for rejim in rejimler:
        for hiz in hizlar_m_s:
            for kayma in kaymalar_ms:
                hatalar: list[float] = []
                basarisiz = 0
                for j in range(n_ornek):
                    nokta = rng.uniform(-hacim_yaricap_m, hacim_yaricap_m, size=3)
                    yon = rng.normal(size=3)
                    yon /= np.linalg.norm(yon)
                    h = kayma_hatasi(
                        kalib, kameralar, nokta, yon * hiz,
                        _kamera_kaymalari(rejim, len(kameralar), kayma, rng),
                        gurultu_px=gurultu_px, seed=seed + j,
                    )
                    if h is None:
                        basarisiz += 1
                    else:
                        hatalar.append(h * 1000.0)
                dizi = np.asarray(hatalar) if hatalar else np.array([np.nan])
                sonuclar.append(KaymaSonucu(
                    kayma_ms=kayma, hiz_m_s=hiz, rejim=rejim,
                    ortalama_mm=float(np.mean(dizi)),
                    medyan_mm=float(np.median(dizi)),
                    p95_mm=float(np.percentile(dizi, 95)),
                    n_ornek=len(hatalar), n_basarisiz=basarisiz,
                ))
    return sonuclar


def azami_kayma_ms(
    sonuclar: list[KaymaSonucu],
    rejim: str,
    hiz_m_s: float,
    esik_mm: float = 10.0,
    olcu: str = "p95_mm",
) -> float | None:
    """Esigin altinda kalinan en buyuk kaymayi dogrusal ara degerle bulur.

    Kapi 2'nin 3B hata butcesi 10 mm; bu fonksiyon o butceyi **zaman**
    cinsinden bir donanim gereksinimine cevirir. Varsayilan olcu p95: ortalama
    iyimserdir, kotu vakalar karari bozar.

    Esik hicbir kaymada asilmiyorsa taranan en buyuk kayma doner; ilk noktada
    bile asiliyorsa None doner (kayma toleransi yok demektir).
    """
    ilgili = sorted(
        (s for s in sonuclar if s.rejim == rejim and s.hiz_m_s == hiz_m_s),
        key=lambda s: s.kayma_ms,
    )
    if not ilgili:
        raise ValueError(
            f"tarama bu rejim/hiz icin sonuc icermiyor: {rejim}, {hiz_m_s}")

    onceki = None
    for s in ilgili:
        deger = getattr(s, olcu)
        if deger > esik_mm:
            if onceki is None:
                return None
            y0, y1 = getattr(onceki, olcu), deger
            if y1 == y0:
                return onceki.kayma_ms
            oran = (esik_mm - y0) / (y1 - y0)
            return float(onceki.kayma_ms + oran * (s.kayma_ms - onceki.kayma_ms))
        onceki = s
    return ilgili[-1].kayma_ms
