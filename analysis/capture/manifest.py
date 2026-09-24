"""Kayit manifesti: eszamanli squat verisinin tek dogruluk kaynagi (0070, 0071).

Manifest, sonuclar gorulmeden kilitlenen bir JSON'dur. Her kaydin kim, ne zaman,
hangi kalibrasyon ve hangi kosulda cekildigini; videolarin SHA-256'sini; tekrar
sinirlarini ve teste ayrilan kosullari tasir. Etiketler ve model ciktilari
**ayri** dosyalardadir; manifest yalnizca ham kaydi tanimlar.

Denetlenen kurallar (her biri bir 0070/0071 maddesine karsilik gelir):

- Kimlikler benzersiz, her atif cozulur (oturum -> kisi, kalibrasyon).
- Her oturumda en az bir `telefon` ve en az iki `duzenek` cihazi var; telefon
  referans ucgenlemesine **katilamaz** -- aksi halde degerlendirilen tahmin kendi
  dogru cevabinin olusturulmasina karisir (0071 §5).
- Video hash'i 64 haneli SHA-256; fps ve cozunurluk pozitif.
- Tekrar sinirlari artan ve oturum icinde cakismaz.
- Kemik uzunlugu kullaniliyorsa kaynagi yazili.
- Bolme SHA-256(`0070|kisi`) ile deterministik hesaplanir (0070 §3); bir
  kisinin butun oturumlari tek bolumdedir.
- Teste ayrilan kosul degerleri egitim/dogrulama kisilerinde gorulurse o
  oturumlar **cikarilir ve sayilir** (0070 §3), sessizce kullanilmaz.

    python -m capture.manifest data/manifest.json   # denetle + kilidi yazdir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from eval.protokol import kisi_bolmesi, manifest_kilidi

SURUM = 1
_SHA = re.compile(r"^[0-9a-f]{64}$")
_KOSUL_ALANLARI = ("aci_derece", "mesafe_m", "isik", "ortulme")
_ROLLER = ("duzenek", "telefon")


def _benzersiz(ogeler, alan, ad, hatalar):
    sayac = Counter(o.get(alan) for o in ogeler)
    for deger, n in sayac.items():
        if n > 1:
            hatalar.append(f"{ad}: '{deger}' {n} kez tanimli")


def _oturum_denetle(o: dict, kisi_kodlari: set, kalib_idler: set, h: list) -> None:
    oid = o.get("id")
    if o.get("kisi") not in kisi_kodlari:
        h.append(f"oturum {oid}: bilinmeyen kisi {o.get('kisi')!r}")
    if o.get("kalibrasyon") not in kalib_idler:
        h.append(f"oturum {oid}: bilinmeyen kalibrasyon {o.get('kalibrasyon')!r}")
    if not o.get("tarih_iso"):
        h.append(f"oturum {oid}: 'tarih_iso' eksik")
    kosul = o.get("kosul") or {}
    for alan in _KOSUL_ALANLARI:
        if alan not in kosul:
            h.append(f"oturum {oid}: kosul.{alan} eksik")

    cihazlar = o.get("cihazlar") or []
    roller = Counter(c.get("rol") for c in cihazlar)
    if roller["telefon"] < 1:
        h.append(f"oturum {oid}: telefon kaydi yok")
    if roller["duzenek"] < 2:
        h.append(f"oturum {oid}: en az iki duzenek kamerasi gerekli")
    _benzersiz(cihazlar, "id", f"oturum {oid} cihaz", h)
    for c in cihazlar:
        cid = f"oturum {oid} cihaz {c.get('id')!r}"
        if c.get("rol") not in _ROLLER:
            h.append(f"{cid}: rol {c.get('rol')!r} ({'/'.join(_ROLLER)} olmali)")
        if not _SHA.match(str(c.get("video_sha256", ""))):
            h.append(f"{cid}: video_sha256 64 haneli kucuk harf SHA-256 olmali")
        if not (isinstance(c.get("fps"), (int, float)) and c["fps"] > 0):
            h.append(f"{cid}: fps pozitif olmali")
        boyut = c.get("cozunurluk")
        if not (isinstance(boyut, list) and len(boyut) == 2
                and all(isinstance(v, int) and v > 0 for v in boyut)):
            h.append(f"{cid}: cozunurluk [genislik, yukseklik] pozitif tamsayi olmali")
        for alan in ("model", "video_dosyasi"):
            if not c.get(alan):
                h.append(f"{cid}: '{alan}' eksik")

    referans = set(o.get("referans_ucgenleme") or [])
    telefonlar = {c.get("id") for c in cihazlar if c.get("rol") == "telefon"}
    if telefonlar & referans:
        h.append(f"oturum {oid}: telefon referans ucgenlemesine katilamaz "
                 f"({sorted(telefonlar & referans)}) -- 0071 §5")
    if len(referans) < 2:
        h.append(f"oturum {oid}: referans_ucgenleme en az iki duzenek kamerasi listelemeli")
    elif not referans <= {c.get("id") for c in cihazlar}:
        h.append(f"oturum {oid}: referans_ucgenleme bilinmeyen cihaz iceriyor")

    tekrarlar = o.get("tekrarlar") or []
    if not tekrarlar:
        h.append(f"oturum {oid}: en az bir tekrar gerekli")
    _benzersiz(tekrarlar, "no", f"oturum {oid} tekrar", h)
    onceki_bitis = None
    for t in tekrarlar:
        b, s = t.get("baslangic_ms"), t.get("bitis_ms")
        if not (isinstance(b, (int, float)) and isinstance(s, (int, float)) and b < s):
            h.append(f"oturum {oid} tekrar {t.get('no')}: baslangic < bitis olmali")
            continue
        if onceki_bitis is not None and b < onceki_bitis:
            h.append(f"oturum {oid} tekrar {t.get('no')}: onceki tekrarla cakisiyor")
        onceki_bitis = s
    if o.get("kemik_uzunlugu") is not None and not o.get("kemik_uzunlugu_kaynagi"):
        h.append(f"oturum {oid}: kemik uzunlugu var ama kaynagi yazilmamis")


def denetle(manifest: dict) -> list[str]:
    """Butun kurallari denetle; bos liste = gecerli. Ilk hatada durmaz."""
    if not isinstance(manifest, dict):
        return ["manifest bir JSON nesnesi olmali"]
    h: list[str] = []
    if manifest.get("surum") != SURUM:
        h.append(f"surum {SURUM} olmali, {manifest.get('surum')!r} geldi")
    kalibrasyonlar = manifest.get("kalibrasyonlar")
    kisiler = manifest.get("kisiler")
    oturumlar = manifest.get("oturumlar")
    for ad, liste in (("kalibrasyonlar", kalibrasyonlar), ("kisiler", kisiler),
                      ("oturumlar", oturumlar)):
        if not isinstance(liste, list) or not liste:
            h.append(f"{ad} bos olmayan bir liste olmali")
    if h:
        return h

    _benzersiz(kalibrasyonlar, "id", "kalibrasyon", h)
    _benzersiz(kisiler, "kod", "kisi", h)
    _benzersiz(oturumlar, "id", "oturum", h)
    for k in kalibrasyonlar:
        for alan in ("id", "tarih_iso", "kayit_dosyasi"):
            if not k.get(alan):
                h.append(f"kalibrasyon {k.get('id')!r}: '{alan}' eksik")
    kisi_kodlari = {k.get("kod") for k in kisiler}
    kalib_idler = {k.get("id") for k in kalibrasyonlar}
    for o in oturumlar:
        _oturum_denetle(o, kisi_kodlari, kalib_idler, h)
    for kisi in kisi_kodlari:
        if not any(o.get("kisi") == kisi for o in oturumlar):
            h.append(f"kisi {kisi}: hic oturumu yok")
    for alan in manifest.get("teste_ayrilan_kosullar") or {}:
        if alan not in _KOSUL_ALANLARI:
            h.append(f"teste_ayrilan_kosullar: bilinmeyen kosul alani {alan!r}")
    return h


def bolme_ve_cikarilanlar(manifest: dict) -> dict:
    """0070 §3 bolmesi + teste ayrilan kosul yuzunden cikarilan oturumlar."""
    bolme = kisi_bolmesi([k["kod"] for k in manifest["kisiler"]])
    bolum = {k: ad for ad in ("test", "dogrulama", "egitim") for k in bolme[ad]}
    tutulan = manifest.get("teste_ayrilan_kosullar") or {}
    cikarilan = []
    for o in manifest["oturumlar"]:
        if bolum[o["kisi"]] == "test":
            continue
        for alan, degerler in tutulan.items():
            if o["kosul"].get(alan) in degerler:
                cikarilan.append({"oturum": o["id"], "kisi": o["kisi"],
                                  "bolum": bolum[o["kisi"]], "kosul": alan})
                break
    return {**bolme, "cikarilan_oturumlar": cikarilan, "n_cikarilan": len(cikarilan)}


# --- etiketler (ayri JSONL; manifestten bagimsiz, ona atif yapar) -------------------

KUSURLAR = ("diz_valgusu_sag", "diz_valgusu_sol", "kalca_hizasi", "govde_rotasyonu")
_KAYNAKLAR = ("geometrik", "uzman")
_KARARLAR = ("dogru", "kusurlu", "belirsiz")


def etiketleri_denetle(etiketler: list[dict], manifest: dict) -> list[str]:
    """Etiket satirlarini manifeste karsi denetle.

    Geometrik esik etiketi ile uzman degerlendirmesi **ayri kaynaklardir**
    (0070 §5): ayni (oturum, tekrar, kusur) icin ikisi de bulunabilir ama biri
    digerinin yerine gecmez; ayni kaynaktan iki etiket celiskidir. "belirsiz"
    karar gerekce ister. Etiketleyen kimligi zorunlu (uzlasma analizi icin).
    """
    tekrarlar = {(o["id"], t["no"]) for o in manifest.get("oturumlar", [])
                 for t in o.get("tekrarlar", [])}
    h: list[str] = []
    gorulen: Counter = Counter()
    for i, e in enumerate(etiketler):
        yer = f"etiket {i}"
        if (e.get("oturum"), e.get("tekrar")) not in tekrarlar:
            h.append(f"{yer}: manifestte olmayan oturum/tekrar "
                     f"({e.get('oturum')!r}, {e.get('tekrar')!r})")
        if e.get("kusur") not in KUSURLAR:
            h.append(f"{yer}: bilinmeyen kusur {e.get('kusur')!r}")
        if e.get("kaynak") not in _KAYNAKLAR:
            h.append(f"{yer}: kaynak {'/'.join(_KAYNAKLAR)} olmali")
        if e.get("karar") not in _KARARLAR:
            h.append(f"{yer}: karar {'/'.join(_KARARLAR)} olmali")
        if e.get("karar") == "belirsiz" and not e.get("gerekce"):
            h.append(f"{yer}: 'belirsiz' karar gerekce ister")
        if not e.get("etiketleyen"):
            h.append(f"{yer}: 'etiketleyen' eksik")
        anahtar = (e.get("oturum"), e.get("tekrar"), e.get("kusur"), e.get("kaynak"),
                   e.get("etiketleyen"))
        gorulen[anahtar] += 1
    for anahtar, n in gorulen.items():
        if n > 1:
            h.append(f"ayni etiketleyenden tekrarli etiket: {anahtar}")
    return h


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kayit manifestini denetle ve kilidini yazdir")
    p.add_argument("manifest", type=Path)
    a = p.parse_args(argv)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    hatalar = denetle(manifest)
    if hatalar:
        print(f"MANIFEST GECERSIZ: {len(hatalar)} hata", file=sys.stderr)
        for satir in hatalar:
            print("  " + satir, file=sys.stderr)
        return 1
    print(json.dumps({"kilit_sha256": manifest_kilidi(manifest),
                      "bolme": bolme_ve_cikarilanlar(manifest)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
