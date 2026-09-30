"""Kalibrasyon kayitlarindan haftalik seri ve Kapi 2 olcutunun sinanmasi.

`PROJE-PLANI.md` Kapi 2 su satiri tasiyor: "Kalibrasyon suruklenme kaydi --
**>= 4 haftalik veri**". Bu modul o cumleyi koda cevirir.

Dikkat edilen sey sudur: **dort kayit dort hafta demek degildir.** Ayni gun
alinmis dort kalibrasyon, sayarak bakan bir kontrolden gecer ama suruklenme
hakkinda hicbir sey soylemez; suruklenme zamanla olusur. Bu yuzden olcut uc
parcali: kac farkli hafta, ilk ile son arasinda kac gun, ve ardisik kayitlar
arasindaki en buyuk bosluk. Ucuncusu olmadan "dort hafta" iki uctaki iki
kayitla da saglanir ve arada hicbir sey bilinmez.

Haftalar **ISO haftasidir** (pazartesi baslar). Pazar ile ertesi pazartesi bir
gun arayla olsa da ayri haftalardir; bu, takvim haftasi saymanin dogal sonucu
ve kayit protokolunde haftanin ayni gunune baglanarak yonetilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from calib.io import Kabin, oku
from uncertainty.drift import tarih_coz

# Kapi 2 varsayilanlari. Gerekce: docs/kararlar/0012.
EN_AZ_HAFTA = 4
EN_AZ_KAPSAM_GUN = 28.0
EN_FAZLA_BOSLUK_GUN = 10.0


@dataclass(frozen=True)
class Hafta:
    """Tek bir ISO haftasi ve icindeki kayitlar."""

    yil: int
    hafta: int
    pazartesi: date
    kayitlar: tuple[str, ...]          # o haftadaki kayit tarihleri (ISO)

    @property
    def etiket(self) -> str:
        return f"{self.yil}-H{self.hafta:02d}"


@dataclass(frozen=True)
class HaftalikSeri:
    """Kayitlarin takvim haftalarina dagilimi."""

    haftalar: tuple[Hafta, ...]
    ilk: datetime
    son: datetime
    bos_haftalar: tuple[str, ...]      # ilk ile son arasinda kaydi olmayan haftalar
    en_buyuk_bosluk_gun: float

    @property
    def n_hafta(self) -> int:
        return len(self.haftalar)

    @property
    def n_kayit(self) -> int:
        return sum(len(h.kayitlar) for h in self.haftalar)

    @property
    def kapsam_gun(self) -> float:
        return (self.son - self.ilk).total_seconds() / 86400.0

    def ozet(self) -> dict:
        return {
            "n_kayit": self.n_kayit,
            "n_hafta": self.n_hafta,
            "kapsam_gun": round(self.kapsam_gun, 2),
            "en_buyuk_bosluk_gun": round(self.en_buyuk_bosluk_gun, 2),
            "ilk": self.ilk.isoformat(),
            "son": self.son.isoformat(),
            "haftalar": [
                {"etiket": h.etiket, "n_kayit": len(h.kayitlar)} for h in self.haftalar
            ],
            "bos_haftalar": list(self.bos_haftalar),
        }


@dataclass(frozen=True)
class Kapi2Durumu:
    """Kapi 2'nin suruklenme satirinin sonucu.

    `gecti` tek basina okunmamali: `eksikler` neyin eksik oldugunu ayri ayri
    soyler, cunku "dort hafta oldu ama arada 20 gun bosluk var" ile "hic kayit
    yok" ayni sey degildir ve farkli mudahale gerektirir.
    """

    gecti: bool
    seri: HaftalikSeri
    eksikler: tuple[str, ...]
    en_az_hafta: int = EN_AZ_HAFTA
    en_az_kapsam_gun: float = EN_AZ_KAPSAM_GUN
    en_fazla_bosluk_gun: float = EN_FAZLA_BOSLUK_GUN

    def ozet(self) -> dict:
        return {
            "gecti": self.gecti,
            "eksikler": list(self.eksikler),
            "olcut": {
                "en_az_hafta": self.en_az_hafta,
                "en_az_kapsam_gun": self.en_az_kapsam_gun,
                "en_fazla_bosluk_gun": self.en_fazla_bosluk_gun,
            },
            "seri": self.seri.ozet(),
        }


def kayitlari_yukle(dizin: str | Path, desen: str = "*.json") -> list[Kabin]:
    """Bir klasordeki tum kalibrasyon kayitlarini okur, tarihe gore siralar.

    Bozuk dosya **sessizce atlanmaz**: hangi dosyanin neden okunamadigi hatada
    yazar. Kapi 2 kontrolunde eksik kayit fark edilmeden gecerse kapi yanlis
    yerde acilmis olur.
    """
    kok = Path(dizin)
    if not kok.is_dir():
        raise ValueError(f"kayit klasoru yok: {kok}")
    yollar = sorted(kok.glob(desen))
    if not yollar:
        raise ValueError(f"klasorde '{desen}' ile eslesen kayit yok: {kok}")

    kabinler = []
    for yol in yollar:
        try:
            kabinler.append(oku(yol))
        except Exception as e:
            raise ValueError(f"kayit okunamadi: {yol.name} -- {e}") from e
    return sorted(kabinler, key=lambda k: tarih_coz(k.meta.tarih_iso))


def _iso_hafta(t: datetime) -> tuple[int, int, date]:
    g = t.date()
    yil, hafta, gun = g.isocalendar()
    return yil, hafta, g - timedelta(days=gun - 1)


def haftalik_grupla(kabinler: list[Kabin]) -> HaftalikSeri:
    """Kayitlari ISO takvim haftalarina dagitir.

    Ayrica ilk ile son arasindaki **bos haftalari** ve ardisik kayitlar
    arasindaki en buyuk boslugu cikarir; ikisi de "seri duzenli mi" sorusunun
    cevabidir ve yalnizca hafta saymakla gorunmezler.
    """
    if not kabinler:
        raise ValueError("haftalik seri icin en az bir kayit gerekli")

    tarihler = sorted(tarih_coz(k.meta.tarih_iso) for k in kabinler)
    kovalar: dict[tuple[int, int], list[str]] = {}
    pazartesiler: dict[tuple[int, int], date] = {}
    for t in tarihler:
        yil, hafta, pzt = _iso_hafta(t)
        kovalar.setdefault((yil, hafta), []).append(t.isoformat())
        pazartesiler[(yil, hafta)] = pzt

    haftalar = tuple(
        Hafta(yil=y, hafta=h, pazartesi=pazartesiler[(y, h)],
              kayitlar=tuple(kovalar[(y, h)]))
        for y, h in sorted(kovalar)
    )

    # Ilk ve son hafta arasinda hic kaydi olmayan takvim haftalari.
    bos: list[str] = []
    imza = {(h.yil, h.hafta) for h in haftalar}
    yurur = haftalar[0].pazartesi
    while yurur <= haftalar[-1].pazartesi:
        y, h, _ = yurur.isocalendar()
        if (y, h) not in imza:
            bos.append(f"{y}-H{h:02d}")
        yurur += timedelta(days=7)

    bosluklar = [(b - a).total_seconds() / 86400.0
                 for a, b in zip(tarihler, tarihler[1:])]
    return HaftalikSeri(
        haftalar=haftalar, ilk=tarihler[0], son=tarihler[-1],
        bos_haftalar=tuple(bos),
        en_buyuk_bosluk_gun=max(bosluklar) if bosluklar else 0.0,
    )


def kapi2_degerlendir(
    seri: HaftalikSeri,
    en_az_hafta: int = EN_AZ_HAFTA,
    en_az_kapsam_gun: float = EN_AZ_KAPSAM_GUN,
    en_fazla_bosluk_gun: float = EN_FAZLA_BOSLUK_GUN,
) -> Kapi2Durumu:
    """Kapi 2'nin ">= 4 haftalik veri" satirini uc olcutle sinar.

    Uc kosul da saglanmadan gecilmez:

    1. **Farkli hafta sayisi** >= `en_az_hafta`. Ayni gun alinmis dort kayit
       tek haftadir ve suruklenme gostermez.
    2. **Kapsam** >= `en_az_kapsam_gun`. Dort ardisik takvim haftasi, pazardan
       pazartesiye 22 gun kadar kisa olabilir; kapsam bunu kapatir.
    3. **En buyuk bosluk** <= `en_fazla_bosluk_gun`. Yoksa "dort hafta" iki
       uctaki iki kayitla saglanir ve arada olan biten bilinmez.
    """
    eksikler: list[str] = []
    if seri.n_hafta < en_az_hafta:
        eksikler.append(
            f"farkli hafta sayisi {seri.n_hafta} < {en_az_hafta} "
            f"({seri.n_kayit} kayit tek basina yetmez)")
    if seri.kapsam_gun < en_az_kapsam_gun:
        eksikler.append(
            f"kapsam {seri.kapsam_gun:.1f} gun < {en_az_kapsam_gun:.0f} gun")
    if seri.en_buyuk_bosluk_gun > en_fazla_bosluk_gun:
        eksikler.append(
            f"en buyuk bosluk {seri.en_buyuk_bosluk_gun:.1f} gun > "
            f"{en_fazla_bosluk_gun:.0f} gun -- seri duzenli degil")
    return Kapi2Durumu(
        gecti=not eksikler, seri=seri, eksikler=tuple(eksikler),
        en_az_hafta=en_az_hafta, en_az_kapsam_gun=en_az_kapsam_gun,
        en_fazla_bosluk_gun=en_fazla_bosluk_gun,
    )


def kapi2_kontrol(dizin: str | Path, desen: str = "*.json", **olcut) -> Kapi2Durumu:
    """Klasorden dogrudan Kapi 2 sonucu -- gercek kayitlar gelince tek cagri."""
    return kapi2_degerlendir(haftalik_grupla(kayitlari_yukle(dizin, desen)), **olcut)
