"""Aci taramasi: form karari dogrulugu kamera acisina gore nasil degisiyor?

Tezin manset grafigi. Duzenek yer gercegini uretiyor, telefon tek kameradan
kestiriyor; bu modul ikisinin **kararlarini** karsilastirip dogrulugu kamera
acisinin fonksiyonu olarak veriyor.

Neden aci? Diz valgusu **frontal duzlem** olcusudur. Telefon onden bakiyorsa
kusur goruntu duzleminde durur ve olculebilir; yandan bakiyorsa ayni kusur
derinlik eksenine duser ve tek kameranin en zayif oldugu yere gider. Beklenen
sonuc dogrulugun aciyla dusmesi; bu modul o dususu **olcer**.

Karsilastirma karar duzeyindedir, eklem konumu duzeyinde degil. Literatur zaten
eklem konum hatasini olcuyor; kullanicinin gordugu sey ise karardir ve 20 mm'lik
bir eklem hatasinin karari bozup bozmadigi ayri bir sorudur.

**Poz kaynagi disaridan verilir.** `poz_uret(aci, iskelet) -> Poz2B` sozlesmesi
sayesinde sentetik izdusum ile gercek model arasinda gecis tek argumandir:
gercek kayitlar geldiginde ayni tarama, ayni egriyi gercek veriyle uretir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from calib.synthetic import rig_yay
from eval.form import Esik, Karar, form_degerlendir
from mono.phone import UzunlukOnculeri, tek_gorus_3b
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B, IskeletTanimi
from pose3d.pose2d import Poz2B, sentetik_poz

# Bir acida bir durus icin 2B poz ureten sozlesme. None dondurmek "bu acidan
# gozlem yok" demektir (ornegin gercek kayitta o aci cekilmemis).
PozUreteci = Callable[[float, Iskelet3B], Poz2B | None]


@dataclass(frozen=True)
class KararSayimi:
    """Bir olcumun bir acidaki karar dagilimi."""

    dogru: int = 0
    yanlis: int = 0
    belirsiz: int = 0
    gozlemsiz: int = 0

    @property
    def toplam(self) -> int:
        return self.dogru + self.yanlis + self.belirsiz + self.gozlemsiz

    @property
    def dogruluk(self) -> float:
        """Dogru karar / tum durus. Belirsiz karar dogru sayilmaz."""
        return self.dogru / self.toplam if self.toplam else float("nan")

    @property
    def karar_verilen_oran(self) -> float:
        """Katmanin hic karar verebildigi durus orani."""
        n = self.dogru + self.yanlis
        return n / self.toplam if self.toplam else float("nan")

    @property
    def yanlis_karar_orani(self) -> float:
        """Karar verilenler icinde yanlis olanlar -- en tehlikeli sayi.

        Belirsiz demek zarar vermez; yanlis "formun dogru" demek verir.
        """
        n = self.dogru + self.yanlis
        return self.yanlis / n if n else float("nan")


@dataclass(frozen=True)
class TaramaNoktasi:
    """Tek bir kamera acisindaki sonuc."""

    aci_derece: float
    sayimlar: dict[str, KararSayimi]
    aci_hatasi_derece: dict[str, float]   # |telefon - yer gercegi| ortalamasi


@dataclass(frozen=True)
class TaramaSonucu:
    noktalar: tuple[TaramaNoktasi, ...]
    olcum_adlari: tuple[str, ...]

    def egri(self, olcum: str) -> tuple[list[float], list[float]]:
        """(aci, dogruluk) -- cizim icin."""
        return ([n.aci_derece for n in self.noktalar],
                [n.sayimlar[olcum].dogruluk for n in self.noktalar])

    def ozet(self) -> dict:
        return {
            "olcumler": list(self.olcum_adlari),
            "noktalar": [
                {
                    "aci_derece": n.aci_derece,
                    "olcumler": {
                        ad: {
                            "dogruluk": round(s.dogruluk, 4),
                            "karar_verilen_oran": round(s.karar_verilen_oran, 4),
                            "yanlis_karar_orani": (
                                None if not np.isfinite(s.yanlis_karar_orani)
                                else round(s.yanlis_karar_orani, 4)
                            ),
                            "dogru": s.dogru, "yanlis": s.yanlis,
                            "belirsiz": s.belirsiz, "gozlemsiz": s.gozlemsiz,
                            "aci_hatasi_derece": (
                                None if not np.isfinite(n.aci_hatasi_derece[ad])
                                else round(n.aci_hatasi_derece[ad], 3)
                            ),
                        }
                        for ad, s in n.sayimlar.items()
                    },
                }
                for n in self.noktalar
            ],
        }


def kemik_onculeri(iskelet: Iskelet3B) -> UzunlukOnculeri:
    """Yer gercegi iskeletten kemik uzunlugu onculeri.

    Bu **iyimser** bir varsayimdir: gercek kullanicinin kemik uzunluklari
    bilinmez. Burada kasitli olarak mukemmel oncu veriliyor ki olculen bozulma
    yalnizca **geometriden** gelsin, oncu hatasindan degil.
    """
    onculer: UzunlukOnculeri = {}
    for a, b in iskelet.tanim.baglantilar:
        uzunluk = iskelet.uzunluk(a, b)
        if np.isfinite(uzunluk) and uzunluk > 0:
            onculer[(a, b)] = float(uzunluk)
    return onculer


def sentetik_poz_ureteci(
    yaricap_m: float = 3.0,
    yukseklik_m: float = 0.9,
    odak_px: float = 1400.0,
    boyut: tuple[int, int] = (1080, 1920),
    gurultu_px: float = 0.0,
    seed: int = 0,
) -> tuple[PozUreteci, np.ndarray]:
    """Telefonu verilen acida konumlandirip bilinen iskeleti izdusuren uretec.

    Aci 0 = onden (frontal), 90 = yandan (sagital). Kamera merkeze bakar.
    Doner: (poz_ureteci, K). K, `tek_gorus_3b` icin gereken ic parametredir.

    **Sinirlilik:** 2B tespit gurultusu burada bakis acisindan bagimsiz kabul
    edilir. `docs/kararlar/0005` bu varsayimi zaten acik sinirlilik olarak
    kaydetti: gercek modelde uc acilardan tespit ayrica bozulur, yani bu
    uretecle cikan egri gercek bozulmanin **alt siniridir**.
    """
    K = np.array([[odak_px, 0.0, boyut[0] / 2],
                  [0.0, odak_px, boyut[1] / 2],
                  [0.0, 0.0, 1.0]])
    sayac = {"n": 0}

    def uret(aci: float, iskelet: Iskelet3B) -> Poz2B | None:
        kamera = rig_yay([aci], yaricap_m=yaricap_m, yukseklik_m=yukseklik_m,
                         odak_px=odak_px, boyut=boyut)[0]
        sayac["n"] += 1
        return sentetik_poz(kamera, iskelet.noktalar, iskelet.tanim,
                            gurultu_px=gurultu_px, seed=seed + sayac["n"])

    return uret, K


def _sonuc(dogru_karar: Karar, telefon_karar: Karar) -> str:
    if telefon_karar is Karar.BELIRSIZ:
        return "belirsiz"
    return "dogru" if telefon_karar is dogru_karar else "yanlis"


def aci_taramasi(
    duruslar: Iterable[Iskelet3B],
    acilar_derece: Iterable[float],
    poz_uret: PozUreteci,
    K: np.ndarray,
    onculer: UzunlukOnculeri | None = None,
    esikler: dict[str, Esik] | None = None,
    tanim: IskeletTanimi = REFERANS_ISKELET,
    konum_belirsizligi_m: float | None = None,
) -> TaramaSonucu:
    """Her aci icin telefon kararini yer gercegi karariyla karsilastirir.

    `duruslar` yer gercegi 3B iskeletlerdir (duzenekten ya da sentetik).
    Her durus icin once **yer gercegi karari** hesaplanir, sonra ayni durus
    verilen acidan telefon hattindan gecirilip karari alinir.

    `onculer` verilmezse her durus icin kendi yer gercegi kemik uzunluklari
    kullanilir (bkz. `kemik_onculeri` -- iyimser varsayim).
    """
    duruslar = list(duruslar)
    if not duruslar:
        raise ValueError("en az bir durus gerekli")

    gercek = [form_degerlendir(d, esikler=esikler) for d in duruslar]
    adlar = tuple(gercek[0].olcumler)

    noktalar: list[TaramaNoktasi] = []
    for aci in acilar_derece:
        sayim = {ad: {"dogru": 0, "yanlis": 0, "belirsiz": 0, "gozlemsiz": 0}
                 for ad in adlar}
        hatalar: dict[str, list[float]] = {ad: [] for ad in adlar}

        for durus, gercek_rapor in zip(duruslar, gercek):
            poz = poz_uret(float(aci), durus)
            if poz is None:
                for ad in adlar:
                    sayim[ad]["gozlemsiz"] += 1
                continue
            kestirim = tek_gorus_3b(poz, K, onculer or kemik_onculeri(durus), tanim)
            telefon_rapor = form_degerlendir(
                kestirim, esikler=esikler,
                konum_belirsizligi_m=konum_belirsizligi_m,
            )
            for ad in adlar:
                sayim[ad][_sonuc(gercek_rapor.olcumler[ad].karar,
                                 telefon_rapor.olcumler[ad].karar)] += 1
                a = telefon_rapor.olcumler[ad].deger
                b = gercek_rapor.olcumler[ad].deger
                if np.isfinite(a) and np.isfinite(b):
                    hatalar[ad].append(abs(a - b))

        noktalar.append(TaramaNoktasi(
            aci_derece=float(aci),
            sayimlar={ad: KararSayimi(**s) for ad, s in sayim.items()},
            aci_hatasi_derece={
                ad: float(np.mean(v)) if v else float("nan")
                for ad, v in hatalar.items()
            },
        ))

    return TaramaSonucu(noktalar=tuple(noktalar), olcum_adlari=adlar)
