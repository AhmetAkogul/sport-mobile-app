"""Rapordaki sayilarin kilidi: `make reproduce` ciktisini kayitli degerlerle karsilastirir.

`out/` git disinda (.gitignore). Bu yuzden bir duzeltme "14,1 mm"yi sessizce
15,3 mm yaparsa hicbir sey fark etmezdi; `git status` temiz gorunurdu. Bu betik
deney JSON'larindaki **butun sayisal alanlari** (listeler dahil) duzlestirip
izlenen `docs/deney/sayi-kilidi.json` ile karsilastirir.

    python scripts/sayi_kilidi.py            # denetle; sapma varsa cikis kodu 1
    python scripts/sayi_kilidi.py --guncelle # kilidi bilincli olarak yenile

Kilit yenilemek bir **karar**dir: degisen sayi bulgu tablosunda ve ilgili
deney kaydinda da guncellenmeli, commit mesajinda gerekcesi yazilmali.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
KILIT = KOK / "docs" / "deney" / "sayi-kilidi.json"

# `make reproduce`'un urettigi deney ciktilari. Yeni deney eklenirse buraya da eklenir.
DENEY_CIKTILARI = (
    "sentetik_kalibrasyon.json",
    "triangulasyon_butcesi.json",
    "hacim_haritasi.json",
    "suruklenme_sentetik.json",
    "form_belirsizligi.json",
    "aci_taramasi.json",
    "oncu_hatasi.json",
    "senkron_kaymasi.json",
    "derinlik_duyarliligi.json",
    "eklem_belirsizligi.json",
)

# Ayni makinede cikti bit bit aynidir; baska platformda BLAS/libm farki kucuk
# sapma uretir: GitHub'in Linux x86 makinesinde Mac ARM kilidine gore bagil
# 1e-6..5e-6 fark goruldu (30 Eylul). Raporlanan sayilar 2-3 anlamli basamakla
# yazildigi icin 1e-4 (%0,01) bagil tolerans rapora yansiyacak her degisikligi
# yine yakalar.
BAGIL_TOLERANS = 1e-4
MUTLAK_TOLERANS = 1e-9


def duzlestir(deger, on: str = "") -> dict[str, float | None]:
    """Ic ice sozluk/listeyi `a.b[3]` anahtarli sayi sozlugune indir.

    Metin alanlari atlanir; `None` (JSON null) korunur -- NaN'in yerine gecen
    eksik deger de kilitlenir, cunku "olculemedi"nin "olculdu"ya donmesi de
    bir degisikliktir.
    """
    cikti: dict[str, float | None] = {}
    if isinstance(deger, dict):
        for k, v in deger.items():
            cikti.update(duzlestir(v, f"{on}.{k}" if on else str(k)))
    elif isinstance(deger, list):
        for i, v in enumerate(deger):
            cikti.update(duzlestir(v, f"{on}[{i}]"))
    elif deger is None:
        cikti[on] = None
    elif isinstance(deger, bool):
        cikti[on] = float(deger)
    elif isinstance(deger, (int, float)):
        cikti[on] = float(deger)
    return cikti


def topla(out_dizini: Path) -> dict[str, dict[str, float | None]]:
    sonuc = {}
    for ad in DENEY_CIKTILARI:
        yol = out_dizini / ad
        if not yol.is_file():
            raise FileNotFoundError(f"{yol} yok -- once `make reproduce` calistirin")
        sonuc[ad] = duzlestir(json.loads(yol.read_text(encoding="utf-8")))
    return sonuc


def _esit(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=BAGIL_TOLERANS, abs_tol=MUTLAK_TOLERANS)


def karsilastir(beklenen: dict, gozlenen: dict) -> list[str]:
    """Farklari insan okunur satirlar olarak dondur; bos liste = kilit tutuyor."""
    farklar: list[str] = []
    for dosya in sorted(set(beklenen) | set(gozlenen)):
        b, g = beklenen.get(dosya), gozlenen.get(dosya)
        if b is None or g is None:
            farklar.append(f"{dosya}: {'kilitte yok' if b is None else 'ciktida yok'}")
            continue
        for anahtar in sorted(set(b) | set(g)):
            if anahtar not in b:
                farklar.append(f"{dosya} {anahtar}: yeni alan = {g[anahtar]}")
            elif anahtar not in g:
                farklar.append(f"{dosya} {anahtar}: alan kayboldu (kilit {b[anahtar]})")
            elif not _esit(b[anahtar], g[anahtar]):
                farklar.append(f"{dosya} {anahtar}: kilit {b[anahtar]} -> simdi {g[anahtar]}")
    return farklar


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rapordaki sayilarin kilidi")
    p.add_argument("--guncelle", action="store_true", help="kilidi mevcut ciktiyla yenile")
    p.add_argument("--out", type=Path, default=KOK / "out")
    p.add_argument("--kilit", type=Path, default=KILIT)
    a = p.parse_args(argv)

    gozlenen = topla(a.out)
    if a.guncelle and a.kilit.is_file():
        # Yenilemeden once dokum: "yeni alan" ile "degisen sayi" ayni sey
        # degildir. Ikincisi bulgu tablosunda ve deney kaydinda da guncellenmeli.
        farklar = karsilastir(json.loads(a.kilit.read_text(encoding="utf-8")), gozlenen)
        # Yeni alan ve kilitte hic olmayan yeni dosya "eklenen"dir; gerisi degisen.
        degisen = [f for f in farklar if "yeni alan" not in f and "kilitte yok" not in f]
        print(f"yenileme dokumu: {len(farklar) - len(degisen)} yeni alan, "
              f"{len(degisen)} degisen/kaybolan")
        for satir in degisen[:20]:
            print("  " + satir)
    if a.guncelle:
        a.kilit.write_text(json.dumps(gozlenen, ensure_ascii=False, indent=1,
                                      sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
        n = sum(len(v) for v in gozlenen.values())
        print(f"kilit yenilendi: {n} sayi, {len(gozlenen)} dosya -> {a.kilit}")
        return 0

    if not a.kilit.is_file():
        print(f"kilit yok: {a.kilit} (ilk kurulum icin --guncelle)", file=sys.stderr)
        return 1
    farklar = karsilastir(json.loads(a.kilit.read_text(encoding="utf-8")), gozlenen)
    if farklar:
        print(f"SAYI KILIDI KIRILDI: {len(farklar)} fark", file=sys.stderr)
        for satir in farklar[:40]:
            print("  " + satir, file=sys.stderr)
        if len(farklar) > 40:
            print(f"  ... ve {len(farklar) - 40} fark daha", file=sys.stderr)
        print("Degisiklik bilincliyse: bulgu tablosu + deney kaydini guncelleyin, "
              "sonra `python scripts/sayi_kilidi.py --guncelle`.", file=sys.stderr)
        return 1
    n = sum(len(v) for v in gozlenen.values())
    print(f"sayi kilidi tutuyor: {n} sayi, {len(gozlenen)} dosya")
    return 0


if __name__ == "__main__":
    sys.exit(main())
