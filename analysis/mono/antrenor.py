"""Antrenor bildirimi: hareket tanima + tekrar sayimi + tekrar basina karar + ipucu (0040).

Butun katmanlari birlestiren son katman. Kamera ve modelden bagimsiz saf
mantiktir; canli modda (`mono.canli`) ve cevrimdisi uctan uca degerlendirmede
(`scripts/deney_antrenor.py`) ayni kod calisir. Kisi basina:

1. **Hareketi kilitle:** hareket tanima (`eval.hareket`) ayni hareketi
   `KILIT_S` saniye boyunca >= `KILIT_P` olasilikla verince hareket kilitlenir.
   `BIRAK_S` saniye "yok" gelirse set biter, ozet bildirilir.
2. **Tekrar say:** kilitli hareketin ana sinyali (`eval.hareket_formu.ana_sinyal`,
   nedensel 3 karelik medyan) `TekrarSayaci`'na verilir.
3. **Tekrar karari:** biten tekrarin olculeri, tahmini bakis acisi grubunun
   modeline verilir; olasilik 0,5'e `kararsiz_bant`'tan yakinsa karar verilmez
   ("emin degilim"). O acida olculebilir model yoksa (0038) karar verilmez;
   "bu acidan olculemez" ve hangi aciya gecilecegi soylenir.
4. **Ipucu:** yanlis tekrarda, modelin kararina en cok katki veren olcunun
   Turkce ipucu. Ayni ipucu `TEKRAR_S` saniye icinde yinelenmez.

Arastirma denemesidir; klinik ya da antrenor karari degildir.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from eval.hareket import HAREKETLER
from eval.hareket_formu import (TekrarSayaci, aci_grubu, ana_sinyal, bakis_acisi,
                                tekrar_olculeri)

MP_YUKARI = np.array([0.0, -1.0, 0.0])
KILIT_S, KILIT_P, BIRAK_S, TEKRAR_S = 1.5, 0.6, 2.0, 10.0
# Kilitlenince tampondaki son GERI_S saniye sayaca geriye donuk verilir: tanima
# 2 s pencere + 1,5 s kilit bekledigi icin ilk tekrar kilitten once biter
# (REHAB24 squat videosunda 5 tekrarin ilki boyle kaciyordu).
GERI_S = 4.0
# Sayac sinyalinin nedensel medyani saniye cinsinden: modeller 10 kare/s veride
# 3 karelik (0,3 s) medyanla ayarlandi; canli 30 kare/s'de "3 kare" 0,1 s'ye
# iniyor ve titresim cift tekrar uretiyordu (29 Eylul, mobil kip demosu).
# Pencere (t - YUMUSATMA_S, t]: 10 kare/s'de tam 3 ornek (eski davranis birebir),
# 30 kare/s'de 8 ornek. 0,3 yerine 0,25: kayan noktada 4. ornegi almasin.
YUMUSATMA_S = 0.25
_AD_NO = {v: k for k, v in HAREKETLER.items()}

# Olcu -> ipucu (ekranda cv2 ile cizildigi icin ASCII).
IPUCLARI = {
    "kol_yetersiz": "Kolu biraz daha yukari kaldir",
    "kol_asiri": "Kolu omuz hizasinda durdur",
    "dirsek_bukulme": "Dirsegini duz tut",
    "govde_yana": "Govdeni yana egme, dik dur",
    "govde_egimi": "Govdeni dik tut",
    "kol_one": "Kolu yana kaldir, one degil",
    "omuz_kalkma": "Omzunu kulagina dogru kaldirma",
    "asimetri": "Iki kolu esit kaldir",
    "kalca_sarkma": "Karnini sik, kalcan sarkmasin",
    "kalca_pike": "Kalcani indir, vucudun duz bir cizgi olsun",
    "sig": "Daha derine in",
    "derin": "Bu kadar derine inme",
    "dirsek_acilma": "Dirseklerini govdene yakin tut",
    "bacak_yetersiz": "Bacagi biraz daha yana ac",
    "bacak_asiri": "Bacagi bu kadar yuksege kaldirma",
    "bacak_one": "Bacagi yana ac, one degil",
    "diz_bukulme": "Kaldirdigin bacagin dizini duz tut",
    "destek_diz": "Destek bacagini sabit tut",
    "pelvis_egimi": "Kalcani yukari kaldirma, pelvis duz kalsin",
    "diz_onde": "Dizlerin one kaciyor, kalcani geriye it",
    "diz_parmak_onde": "Dizin ayak ucunu geciyor, kalcani geriye it",
    "diz_parmak_onde_2b": "Dizin ayak ucunu geciyor, kalcani geriye it",
    "diz_ice": "Dizlerin ice kapaniyor, disari it",
}
ACI_TARIFI = {"front": "karsidan", "half-profile": "capraz (45 derece)", "profile": "yandan"}


def en_buyuk_katki(pipeline, x: np.ndarray, olculer: list[str]) -> str | None:
    """Lojistik modelde "yanlis" yonune en cok katki veren olcu (katsayi x olcekli deger)."""
    try:
        imp, olc, lr = (pipeline.named_steps[k] for k in
                        ("simpleimputer", "standardscaler", "logisticregression"))
    except (AttributeError, KeyError):
        return None
    z = olc.transform(imp.transform(x[None]))[0]
    katki = lr.coef_[0] * z
    i = int(np.argmax(katki))
    return olculer[i] if katki[i] > 0 else None


@dataclass
class KisiDurumu:
    zaman: deque = field(default_factory=lambda: deque(maxlen=600))
    P2: deque = field(default_factory=lambda: deque(maxlen=600))
    G2: deque = field(default_factory=lambda: deque(maxlen=600))
    X3: deque = field(default_factory=lambda: deque(maxlen=600))
    ayak: deque = field(default_factory=lambda: deque(maxlen=600))
    sinyal: deque = field(default_factory=lambda: deque(maxlen=128))   # (t, deger)
    etiketler: deque = field(default_factory=lambda: deque(maxlen=600))   # (t, ad, p); 30 kare/s'de 20 s
    hareket: int | None = None
    sayac: TekrarSayaci | None = None
    tekrarlar: list = field(default_factory=list)
    son_ipucu: dict = field(default_factory=dict)


class Antrenor:
    """Kisi basina durum; `adim` her karede cagrilir, bildirim listesi dondurur.

    `tanima`: `eval.hareket.HareketTanima`; `modeller`: `scripts/deney_hareket_formu.py`
    ciktisi {hareket_no: {"ad", "olculer", "sayac", "gruplar": {aci: {"model"?, "auc"}}}}.
    """

    def __init__(self, tanima, modeller: dict, yukari=MP_YUKARI, kararsiz_bant: float = 0.0):
        self.tanima, self.modeller, self.yukari = tanima, modeller, np.asarray(yukari, float)
        self.kararsiz_bant = kararsiz_bant
        self.kisiler: dict[int, KisiDurumu] = {}

    def _kilit(self, d: KisiDurumu, t: float) -> list[dict]:
        out = []
        if d.hareket is None:
            pencere = [e for e in d.etiketler if e[0] >= t - KILIT_S]
            dolu = bool(pencere) and d.etiketler[0][0] <= t - KILIT_S + 0.3
            if (dolu and len({ad for _, ad, _ in pencere}) == 1
                    and min(p for *_, p in pencere) >= KILIT_P):
                no = _AD_NO.get(pencere[0][1])
                if no and no in self.modeller:
                    d.hareket, d.tekrarlar = no, []
                    d.sayac = TekrarSayaci(*self.modeller[no]["sayac"])
                    d.sinyal.clear()
                    out.append({"tur": "hareket", "hareket": pencere[0][1],
                                "metin": f"Hareket: {pencere[0][1]}"})
        else:
            yok = [e for e in d.etiketler if e[0] >= t - BIRAK_S]
            dolu = bool(yok) and d.etiketler[0][0] <= t - BIRAK_S + 0.3
            if dolu and all(ad == "yok" for _, ad, _ in yok):
                out.append(self._set_ozeti(d))
                d.hareket, d.sayac = None, None
        return out

    def _set_ozeti(self, d: KisiDurumu) -> dict:
        n = len(d.tekrarlar)
        dogru = sum(r["karar"] == "dogru" for r in d.tekrarlar)
        yanlis = sum(r["karar"] == "yanlis" for r in d.tekrarlar)
        ad = self.modeller[d.hareket]["ad"]
        return {"tur": "set", "metin": f"Set bitti ({ad}): {n} tekrar, {dogru} iyi, "
                                       f"{yanlis} duzeltilecek, {n - dogru - yanlis} karar yok",
                "hareket": ad, "tekrar": n, "dogru": dogru, "yanlis": yanlis}

    def _tekrar(self, d: KisiDurumu, bas: float, son: float, t: float) -> list[dict]:
        m = self.modeller[d.hareket]
        z = np.array(d.zaman)
        sec = (z >= bas) & (z <= son)
        if sec.sum() < 3:
            return []
        X = np.array(d.X3)[sec]
        hiz = max((sec.sum() - 1) / max(son - bas, 1e-6), 1.0)
        olc = tekrar_olculeri(X, self.yukari, d.hareket, None, hiz, ayak=np.array(d.ayak)[sec],
                              P2=np.array(d.P2)[sec])
        aci = float(np.nanmedian([bakis_acisi(P) for P in X]))
        grup = aci_grubu(aci)
        kayit = {"hareket": m["ad"], "tekrar": len(d.tekrarlar) + 1, "bas": bas, "son": son,
                 "aci": aci, "grup": grup, "olculer": olc}
        g = m["gruplar"].get(grup, {})
        if "model" not in g:
            iyi = [k for k, v in m["gruplar"].items() if "model" in v]
            kayit["karar"] = "olculemez"
            d.tekrarlar.append(kayit)
            oneri = (f"; {ACI_TARIFI[max(iyi, key=lambda k: m['gruplar'][k]['auc'])]} dur"
                     if iyi else "")
            return [{"tur": "olculemez",
                     "metin": f"Tekrar {kayit['tekrar']}: bu acidan olculemez{oneri}", **kayit}]
        x = np.array([olc[o] for o in m["olculer"]], float)
        p = float(g["model"].predict_proba(x[None])[0, 1])
        kayit["p_yanlis"] = p
        if abs(p - 0.5) < self.kararsiz_bant:
            kayit["karar"] = "belirsiz"
            d.tekrarlar.append(kayit)
            return [{"tur": "belirsiz", "metin": f"Tekrar {kayit['tekrar']}: emin degilim",
                     **kayit}]
        kayit["karar"] = "yanlis" if p > 0.5 else "dogru"
        d.tekrarlar.append(kayit)
        durum = "DUZELT" if kayit["karar"] == "yanlis" else "iyi"
        out = [{"tur": "tekrar", "metin": f"Tekrar {kayit['tekrar']}: {durum}", **kayit}]
        if kayit["karar"] == "yanlis":
            neden = en_buyuk_katki(g["model"], x, m["olculer"])
            if neden and t - d.son_ipucu.get(neden, -np.inf) >= TEKRAR_S:
                d.son_ipucu[neden] = t
                out.append({"tur": "ipucu", "olcu": neden, "metin": IPUCLARI.get(neden, neden)})
        return out

    def adim(self, kimlik: int, t: float, P2: np.ndarray, G2: np.ndarray,
             X3: np.ndarray | None, ayak: np.ndarray | None = None) -> list[dict]:
        """Bir kisinin bir karesi: 2B tam vucut (65, 2) + dunya referans iskeleti (13, 3)."""
        d = self.kisiler.setdefault(kimlik, KisiDurumu())
        d.zaman.append(t)
        d.P2.append(P2)
        d.G2.append(G2)
        d.X3.append(np.full((13, 3), np.nan) if X3 is None else X3)
        d.ayak.append(np.full((4, 3), np.nan) if ayak is None else ayak)
        r = self.tanima.tahmin(kimlik, np.array(d.zaman), np.array(d.P2), np.array(d.G2))
        if r is not None:
            d.etiketler.append((t, r[0], r[1]))
        onceki = d.hareket
        out = self._kilit(d, t)
        if d.hareket is not None and d.sayac is not None:
            yeni = onceki is None
            kareler = ([(tt, X) for tt, X in zip(d.zaman, d.X3) if tt >= t - GERI_S] if yeni
                       else [(t, d.X3[-1])])
            for tt, X in kareler:
                d.sinyal.append((tt, ana_sinyal(X[None], self.yukari, d.hareket)[0]))
                v = np.array([x for ts, x in d.sinyal if ts > tt - YUMUSATMA_S])
                v = v[np.isfinite(v)]
                biten = d.sayac.ekle(tt, float(np.median(v)) if len(v) else np.nan)
                if biten:
                    out += self._tekrar(d, *biten, t)
        return [{"kimlik": kimlik, "t": t, **b} for b in out]

    def bitir(self) -> list[dict]:
        """Akis bitti: acik setlerin ozeti."""
        out = []
        for k, d in self.kisiler.items():
            if d.hareket is not None:
                t = d.zaman[-1] if d.zaman else 0.0
                out.append({"kimlik": k, "t": t, **self._set_ozeti(d)})
                d.hareket = None
        return out

    def unut(self, kimlik: int) -> None:
        self.kisiler.pop(kimlik, None)
        self.tanima.unut(kimlik)
