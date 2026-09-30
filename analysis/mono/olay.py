"""Mobil ve salon icin ortak olay semasi (0073, 0042).

`mono.antrenor.Antrenor`'un bildirimleri (hareket, tekrar, belirsiz, olculemez,
ipucu, set) bu semaya cevrilip JSONL'e yazilir; `mono.olay_servisi` ayni
dosyayi izleyip antrenor paneline aktarir. Olay anonimdir: kisi, takipcinin
verdigi oturum ici kimliktir (uyelik/yuz tanima yok, 0073). Olculerin
ayrintisi ve goruntu olaya girmez.
"""

from __future__ import annotations

from datetime import datetime, timezone

OLAY_SURUMU = 1
TURLER = ("hareket", "tekrar", "belirsiz", "olculemez", "ipucu", "set")
YUZEYLER = ("mobil", "salon")
_ALANLAR = ("hareket", "tekrar", "karar", "p_yanlis", "grup", "aci", "olcu",
            "dogru", "yanlis")
# Antrenore "problemli" uyarisi: son PROBLEM_PENCERE karar verilen tekrarin en az
# PROBLEM_ESIK'i "yanlis". Tek tekrarlik hata uyari uretmez (yanlis alarm).
PROBLEM_PENCERE, PROBLEM_ESIK = 3, 2


def olay_yap(bildirim: dict, *, yuzey: str, istasyon: str, kamera: str,
             model_surumu: str, zaman: datetime | None = None) -> dict:
    """Antrenor bildirimi -> sema. Bilinmeyen tur ya da yuzey ValueError."""
    if bildirim.get("tur") not in TURLER:
        raise ValueError(f"bilinmeyen olay turu: {bildirim.get('tur')!r}")
    if yuzey not in YUZEYLER:
        raise ValueError(f"yuzey {YUZEYLER} icinden olmali: {yuzey!r}")
    z = zaman or datetime.now(timezone.utc)
    olay = {"surum": OLAY_SURUMU, "zaman": z.isoformat(timespec="milliseconds"),
            "t_s": round(float(bildirim["t"]), 3),
            "kaynak": {"yuzey": yuzey, "istasyon": istasyon, "kamera": kamera},
            "kisi": int(bildirim["kimlik"]), "tur": bildirim["tur"],
            "metin": bildirim["metin"], "model_surumu": model_surumu}
    for k in _ALANLAR:
        v = bildirim.get(k)
        if v is not None:
            olay[k] = round(float(v), 3) if isinstance(v, float) else v
    return olay


def olay_dogrula(olay: dict) -> None:
    """Semaya uymayan olay ValueError (servis bozuk satiri atlar)."""
    for k in ("surum", "zaman", "t_s", "kaynak", "kisi", "tur", "metin", "model_surumu"):
        if k not in olay:
            raise ValueError(f"eksik alan: {k}")
    if olay["surum"] != OLAY_SURUMU:
        raise ValueError(f"desteklenmeyen surum: {olay['surum']}")
    if olay["tur"] not in TURLER or olay["kaynak"].get("yuzey") not in YUZEYLER:
        raise ValueError("tur ya da yuzey gecersiz")


class PanelDurumu:
    """Olay akisindan kisi basina ozet: antrenor panelinin gosterdigi sey.

    Anahtar (istasyon, kisi): farkli istasyonlarda ayni takip kimligi farkli kisidir.
    """

    def __init__(self):
        self.kisiler: dict[tuple[str, int], dict] = {}

    def ekle(self, olay: dict) -> dict | None:
        """Olayi isler; kisi yeni "problemli" olduysa uyari olayi dondurur."""
        anahtar = (olay["kaynak"]["istasyon"], olay["kisi"])
        k = self.kisiler.setdefault(anahtar, {
            "istasyon": anahtar[0], "kisi": anahtar[1], "hareket": None, "tekrar": 0,
            "kararlar": [], "son_ipucu": None, "son_zaman": None, "problemli": False})
        k["son_zaman"] = olay["zaman"]
        if olay["tur"] == "hareket":
            k.update(hareket=olay.get("hareket"), tekrar=0, kararlar=[], problemli=False)
        elif olay["tur"] in ("tekrar", "belirsiz", "olculemez"):
            k["tekrar"] = olay.get("tekrar", k["tekrar"])
            k["hareket"] = olay.get("hareket", k["hareket"])
            if olay["tur"] == "tekrar":
                k["kararlar"].append(olay.get("karar"))
        elif olay["tur"] == "ipucu":
            k["son_ipucu"] = olay["metin"]
        elif olay["tur"] == "set":
            k["hareket"] = None
        once = k["problemli"]
        son = k["kararlar"][-PROBLEM_PENCERE:]
        k["problemli"] = sum(x == "yanlis" for x in son) >= PROBLEM_ESIK
        if k["problemli"] and not once:
            n_yanlis = sum(x == "yanlis" for x in son)
            ipucu = f" ({k['son_ipucu']})" if k["son_ipucu"] else ""
            return {"tur": "uyari", "istasyon": anahtar[0], "kisi": anahtar[1],
                    "hareket": k["hareket"], "zaman": olay["zaman"],
                    "metin": (f"{anahtar[0]} / kisi #{anahtar[1]}: {k['hareket']} son "
                              f"{len(son)} tekrarin {n_yanlis}'i duzeltilecek{ipucu}")}
        return None

    def ozet(self) -> list[dict]:
        satirlar = []
        for d in self.kisiler.values():
            s = {k: v for k, v in d.items() if k != "kararlar"}
            s["duzelt"] = sum(x == "yanlis" for x in d["kararlar"])
            s["iyi"] = sum(x == "dogru" for x in d["kararlar"])
            satirlar.append(s)
        return sorted(satirlar, key=lambda d: (not d["problemli"], d["istasyon"], d["kisi"]))
