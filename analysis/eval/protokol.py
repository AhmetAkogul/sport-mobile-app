"""Degerlendirme protokolunun kod karsiligi (`docs/kararlar/0070`).

Bu modul 0070'in uc bolumunu uygular; tanimlar oradan birebir alinir:

- §1 paydalar: M (plan), R (referanssiz), N = M - R, D (telefon karar verdi),
  C (dogru karar), W = D - C, U = N - D ve oranlar A = C/D, K = D/N, Y = C/N.
  Tanimsiz oran **None**'dir; asla 0 ya da 1 degil.
- §2 birim: kare olcum birimi, genellemede bagimsiz ornek **kisi**. Oturum
  oranlari esit agirlikla, sonra kisiler esit agirlikla ortalanir; guven araligi
  kisi kumesi bootstrap'i (2000 tekrar, tohum 70).
- §3 bolme: SHA-256(`0070|kisi_id`) sirasiyla test/dogrulama/egitim; manifest
  sonuclar gorulmeden kilitlenir.

Karar etiketleri `eval.form.Karar` degerleridir ("dogru", "kusurlu",
"belirsiz"). Referansi hic olmayan oge icin `None` verilir (R'ye girer);
telefon tarafinda `None` teknik basarisizliktir (gozlemsiz, model hatasi) ve
U'ya girer -- N kuculmez.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

DOGRU, KUSURLU, BELIRSIZ = "dogru", "kusurlu", "belirsiz"
_KESIN = (DOGRU, KUSURLU)
_ETIKETLER = (DOGRU, KUSURLU, BELIRSIZ, None)

BOOTSTRAP_TEKRAR = 2000
BOOTSTRAP_TOHUM = 70


def _oran(pay: int, payda: int) -> float | None:
    return pay / payda if payda else None


@dataclass(frozen=True)
class Sayim:
    """Bir kusur ve kosul grubunun (referans, telefon) sayim matrisi."""

    matris: dict = field(default_factory=dict)   # {(referans, telefon): adet}

    def __post_init__(self) -> None:
        for (r, t), n in self.matris.items():
            if r not in _ETIKETLER or t not in _ETIKETLER:
                raise ValueError(f"gecersiz karar cifti: {(r, t)}")
            if type(n) is not int or n < 0:
                raise ValueError(f"sayim negatif olmayan tamsayi olmali: {(r, t)}={n}")

    @classmethod
    def ciftlerden(cls, ciftler: Iterable[tuple]) -> "Sayim":
        matris: dict = defaultdict(int)
        for r, t in ciftler:
            matris[(r, t)] += 1
        return cls(dict(matris))

    def _topla(self, kosul) -> int:
        return sum(n for (r, t), n in self.matris.items() if kosul(r, t))

    # --- 0070 §1 paydalari ---------------------------------------------------
    @property
    def M(self) -> int:
        return self._topla(lambda r, t: True)

    @property
    def R(self) -> int:
        """Gecerli referansi olmayan: referans yok ya da BELIRSIZ."""
        return self._topla(lambda r, t: r not in _KESIN)

    @property
    def N(self) -> int:
        return self.M - self.R

    @property
    def D(self) -> int:
        return self._topla(lambda r, t: r in _KESIN and t in _KESIN)

    @property
    def C(self) -> int:
        return self._topla(lambda r, t: r in _KESIN and t == r)

    @property
    def W(self) -> int:
        return self.D - self.C

    @property
    def U(self) -> int:
        return self.N - self.D

    def sinif(self, referans: str) -> "Sayim":
        """Yalniz bir referans sinifinin alt matrisi."""
        return Sayim({k: n for k, n in self.matris.items() if k[0] == referans})

    def oranlar(self) -> dict:
        """0070 §1'in birlikte raporlanacak oranlari; tanimsiz -> None."""
        siniflar = [self.sinif(s) for s in _KESIN]
        a_s = [_oran(x.C, x.D) for x in siniflar]
        k_s = [_oran(x.D, x.N) for x in siniflar]
        return {
            "M": self.M, "R": self.R, "N": self.N, "D": self.D,
            "C": self.C, "W": self.W, "U": self.U,
            "A": _oran(self.C, self.D),
            "K": _oran(self.D, self.N),
            "Y": _oran(self.C, self.N),
            "W_N": _oran(self.W, self.N),
            "U_N": _oran(self.U, self.N),
            # Sinif dengeli: iki referans sinifinin ortalamasi; biri yoksa None.
            "A_dengeli": None if None in a_s else float(np.mean(a_s)),
            "K_dengeli": None if None in k_s else float(np.mean(k_s)),
            "matris": {f"{r}|{t}": n for (r, t), n in sorted(
                self.matris.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))},
        }


# --- 0070 §2: kisi ve oturum agirligi, kisi kumesi bootstrap ----------------------

@dataclass(frozen=True)
class Oge:
    """0070 §1'in degerlendirme ogesi (bir kusur turu icin)."""

    kisi: str
    oturum: str
    referans: str | None
    telefon: str | None


def _ortala(degerler: list[float | None]) -> float | None:
    gecerli = [v for v in degerler if v is not None]
    return float(np.mean(gecerli)) if gecerli else None


def kisi_agirlikli(ogeler: Iterable[Oge]) -> dict:
    """Oturum oranlari esit, sonra kisiler esit agirlikla (0070 §2).

    Hic D'si olmayan kisinin K'si 0, A'si None kalir; None sayilari ayrica verilir.
    """
    gruplar: dict = defaultdict(lambda: defaultdict(list))
    for o in ogeler:
        gruplar[o.kisi][o.oturum].append((o.referans, o.telefon))
    kisi_oranlari = {}
    for kisi, oturumlar in gruplar.items():
        o_or = [Sayim.ciftlerden(c).oranlar() for c in oturumlar.values()]
        kisi_oranlari[kisi] = {ad: _ortala([o[ad] for o in o_or]) for ad in ("A", "K", "Y")}
    sonuc: dict = {ad: _ortala([k[ad] for k in kisi_oranlari.values()])
                   for ad in ("A", "K", "Y")}
    sonuc["n_kisi"] = len(kisi_oranlari)
    sonuc["null_kisi"] = {ad: sum(k[ad] is None for k in kisi_oranlari.values())
                          for ad in ("A", "K", "Y")}
    return sonuc


def kisi_bootstrap(ogeler: Iterable[Oge], tekrar: int = BOOTSTRAP_TEKRAR,
                   tohum: int = BOOTSTRAP_TOHUM) -> dict:
    """Kisileri yerine koyarak ornekleyen %95 aralik (0070 §2).

    Bir kisinin butun oturumlari birlikte tasinir. Ikiden az kisi varsa aralik
    **dogrulanmadi**dir. Tanimsiz tekrarlar atilmaz, sayilari yazilir.
    """
    ogeler = list(ogeler)
    kisiler = sorted({o.kisi for o in ogeler})
    if len(kisiler) < 2:
        return {"durum": "dogrulanmadi", "neden": "ikiden az bagimsiz kisi",
                "n_kisi": len(kisiler)}
    kisiye_gore: dict = defaultdict(list)
    for o in ogeler:
        kisiye_gore[o.kisi].append(o)
    rng = np.random.default_rng(tohum)
    ornekler: dict = {ad: [] for ad in ("A", "K", "Y")}
    tanimsiz: dict = {ad: 0 for ad in ("A", "K", "Y")}
    for _ in range(tekrar):
        secim = rng.choice(len(kisiler), size=len(kisiler), replace=True)
        # Ayni kisi iki kez secilirse iki ayri kume sayilir (yeniden adlandirma).
        yeniden = [Oge(f"{kisiler[i]}#{j}", o.oturum, o.referans, o.telefon)
                   for j, i in enumerate(secim) for o in kisiye_gore[kisiler[i]]]
        sonuc = kisi_agirlikli(yeniden)
        for ad in ornekler:
            if sonuc[ad] is None:
                tanimsiz[ad] += 1
            else:
                ornekler[ad].append(sonuc[ad])
    return {
        "durum": "hesaplandi", "n_kisi": len(kisiler), "tekrar": tekrar, "tohum": tohum,
        "aralik_95": {ad: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                           if v else None) for ad, v in ornekler.items()},
        "tanimsiz_tekrar": tanimsiz,
    }


# --- 0070 §3: bolme ve manifest kilidi ----------------------------------------------

def kisi_bolmesi(kisi_idleri: Iterable[str]) -> dict:
    """SHA-256(`0070|kisi_id`) sirasiyla test / dogrulama / egitim (0070 §3).

    test = max(2, ceil(0,2P)), dogrulama = max(1, ceil(0,2P)), kalan egitim.
    P < 6 ya da egitim < 2 ise bu protokolde **genelleme kabulu verilmez**.
    """
    kimlikler = [str(k) for k in kisi_idleri]
    if len(set(kimlikler)) != len(kimlikler):
        raise ValueError("kisi kimlikleri benzersiz olmali")
    sira = sorted(kimlikler,
                  key=lambda k: hashlib.sha256(f"0070|{k}".encode("utf-8")).hexdigest())
    P = len(sira)
    n_test = max(2, math.ceil(0.2 * P))
    n_dog = max(1, math.ceil(0.2 * P))
    egitim = sira[n_test + n_dog:]
    return {
        "test": sira[:n_test], "dogrulama": sira[n_test:n_test + n_dog], "egitim": egitim,
        "P": P, "genelleme_kabulu_mumkun": P >= 6 and len(egitim) >= 2,
    }


def manifest_kilidi(manifest: dict) -> str:
    """Manifestin kanonik JSON'unun SHA-256'si; sonuclar gorulmeden kaydedilir."""
    metin = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(metin.encode("utf-8")).hexdigest()


# --- 0070 §5 hedef hukmu (0,80; ek: kucuk orneklemde on kanit) ---------------------

HEDEF = 0.80
ON_KANIT_KISI_SINIRI = 10


def hedef_hukmu(A_dengeli: float | None, K: float | None,
                A_alt: float | None, K_alt: float | None, n_test_kisi: int,
                hedef: float = HEDEF) -> str:
    """Bir kusur/kosul icin 0070 §5 hukmu.

    - Nokta tahmini (sinif dengeli A ya da K) hedefin altinda -> "basarisiz".
    - Iki %95 alt sinir de hedefin ustunde -> "basarili".
    - Aralik hedefi kesiyor, sinif yok ya da aralik hesaplanamadi -> "dogrulanmadi";
      ancak test kisisi 10'dan azsa ve nokta tahminleri hedefi gecerse
      "on_kanit" (0070 eki, 23 Eylul): az kisiyle aralik hedefi neredeyse hic
      gecemez; bu basamak basari degildir, raporda boyle adlandirilir.
    """
    if A_dengeli is None or K is None:
        return "dogrulanmadi"
    if A_dengeli < hedef or K < hedef:
        return "basarisiz"
    if A_alt is not None and K_alt is not None and A_alt >= hedef and K_alt >= hedef:
        return "basarili"
    if n_test_kisi < ON_KANIT_KISI_SINIRI:
        return "on_kanit"
    return "dogrulanmadi"
