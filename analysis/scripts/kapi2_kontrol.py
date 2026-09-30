#!/usr/bin/env python3
"""Kapi 2'nin suruklenme satirini bir kayit klasoru uzerinde sinar.

Kullanim:

    python scripts/kapi2_kontrol.py <kayit-klasoru> [--json]

Gercek kalibrasyon kayitlari gelmeye basladiginda calistirilacak tek komut.
`make reproduce` icine **konulmadi**: uretecek sentetik sayisi yok, gercek veri
olmadan calistirilamaz. Kapi 2 yaklastikca elle kosulur.

Cikis kodu 0 ise olcut saglandi, 1 ise saglanmadi -- betikten cagirilabilsin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uncertainty.haftalik import (
    EN_AZ_HAFTA, EN_AZ_KAPSAM_GUN, EN_FAZLA_BOSLUK_GUN, kapi2_kontrol,
)


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("klasor", help="kalibrasyon kayitlarinin klasoru")
    ayristirici.add_argument("--desen", default="*.json")
    ayristirici.add_argument("--en-az-hafta", type=int, default=EN_AZ_HAFTA)
    ayristirici.add_argument("--en-az-kapsam-gun", type=float,
                             default=EN_AZ_KAPSAM_GUN)
    ayristirici.add_argument("--en-fazla-bosluk-gun", type=float,
                             default=EN_FAZLA_BOSLUK_GUN)
    ayristirici.add_argument("--json", action="store_true",
                             help="insan okunur ozet yerine JSON yaz")
    a = ayristirici.parse_args()

    try:
        durum = kapi2_kontrol(
            a.klasor,
            desen=a.desen,
            en_az_hafta=a.en_az_hafta,
            en_az_kapsam_gun=a.en_az_kapsam_gun,
            en_fazla_bosluk_gun=a.en_fazla_bosluk_gun,
        )
    except ValueError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(durum.ozet(), ensure_ascii=False, indent=2))
        return 0 if durum.gecti else 1

    s = durum.seri
    print(f"Kayit klasoru  : {a.klasor}")
    print(f"Kayit sayisi   : {s.n_kayit}")
    print(f"Farkli hafta   : {s.n_hafta}  (en az {durum.en_az_hafta})")
    print(f"Kapsam         : {s.kapsam_gun:.1f} gun  "
          f"(en az {durum.en_az_kapsam_gun:.0f})")
    print(f"En buyuk bosluk: {s.en_buyuk_bosluk_gun:.1f} gun  "
          f"(en fazla {durum.en_fazla_bosluk_gun:.0f})")
    print(f"Ilk / son      : {s.ilk.date()} .. {s.son.date()}")
    if s.bos_haftalar:
        print(f"Bos haftalar   : {', '.join(s.bos_haftalar)}")
    print()
    if durum.gecti:
        print("KAPI 2 suruklenme olcutu: SAGLANDI")
    else:
        print("KAPI 2 suruklenme olcutu: SAGLANMADI")
        for e in durum.eksikler:
            print(f"  - {e}")
    return 0 if durum.gecti else 1


if __name__ == "__main__":
    raise SystemExit(main())
