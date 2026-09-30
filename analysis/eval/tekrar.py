"""Zaman damgali squat evreleri; gercek zamanla sure ve karar kapsama hesabi.

Gelecek kare kullanmaz, eksik kareyi enterpole etmez. Esikler arastirma
ayaridir; squat derinligi veya klinik form etiketi degildir.
"""
import math

from eval.egzersiz import SQUAT


class SquatTakip:
    def __init__(self, profil=SQUAT):
        self.profil = profil
        self.evre = "hazir_degil"
        self.onceki = None
        self.aktif = None
        self.sonuclar = []

    def _bitir(self, zaman, tam, neden):
        a = self.aktif
        sure = max(0.0, zaman - a["baslangic_s"])
        if tam and sure < self.profil.min_tekrar_suresi_s:
            # Tek karelik sicrama: tekrar degil, kayda gecmez.
            self.aktif = None
            return None
        kapsama = {k: a["kesin_s"][k] / sure if sure else 0.0 for k in self.profil.kapsam}
        kusurlar = [k for k in self.profil.kapsam
                    if a["en_uzun_kusur_s"][k] + 1e-12 >= self.profil.kusur_suresi_s]
        karar = "kusurlu" if kusurlar else (
            "dogru" if tam and sure > 0 and min(kapsama.values()) >= self.profil.min_kapsama
            else "belirsiz")
        sonuc = {"tekrar": len(self.sonuclar)+1, "baslangic_s": a["baslangic_s"],
                 "bitis_s": zaman, "tamamlandi": tam, "neden": neden,
                 "karar": karar, "kusurlar": kusurlar, "kapsama": kapsama,
                 "kusur_suresi_s": a["en_uzun_kusur_s"], "profil": self.profil.surum,
                 "aciklama": "Yalniz degerlendirilen olcutler; tum hareketin dogrulugu degil"}
        self.sonuclar.append(sonuc)
        self.aktif = None
        return sonuc

    def ekle(self, zaman_s, kare):
        if not math.isfinite(zaman_s) or zaman_s < 0:
            raise ValueError("zaman_s sonlu ve negatif olmayan olmali")
        if self.onceki is not None and zaman_s <= self.onceki[0]:
            raise ValueError("zaman damgalari kesin artmali")
        if kare["profil"] != self.profil.surum or set(kare["olcumler"]) != set(self.profil.kapsam):
            raise ValueError("profil/olcum kapsami uyusmuyor")
        f = kare["fleksiyon_derece"]
        if f is not None and (not math.isfinite(f) or not 0 <= f <= 180):
            raise ValueError("fleksiyon 0..180 veya None olmali")
        for o in kare["olcumler"].values():
            if o["karar"] not in ("dogru", "kusurlu", "belirsiz"):
                raise ValueError("gecersiz karar")
        once = self.onceki
        self.onceki = (zaman_s, kare)
        sonuc = None
        kopuk = once is not None and zaman_s-once[0] > self.profil.max_bosluk_s
        if kopuk:
            if self.aktif:
                # Bitis, bosluktan onceki son gozlenen karedir; kayit olmayan
                # zaman sureye ve kapsamaya katilmaz.
                sonuc = self._bitir(once[0], False, "zaman_boslugu")
            self.evre = "hazir_degil"
        if f is None:
            if self.aktif:
                for k in self.profil.kapsam:
                    self.aktif["seri_s"][k] = 0.0
            return {"evre": "gozlemsiz", "sonuc": sonuc}
        if self.aktif and once and not kopuk:
            dt = zaman_s-once[0]
            gecerli_evre = self.evre in self.profil.evreler
            for k in self.profil.kapsam:
                p, q = once[1]["olcumler"][k]["karar"], kare["olcumler"][k]["karar"]
                gecerli = gecerli_evre and once[1]["fleksiyon_derece"] is not None
                if gecerli and p != "belirsiz" and q != "belirsiz":
                    self.aktif["kesin_s"][k] += dt
                if gecerli and p == q == "kusurlu":
                    self.aktif["seri_s"][k] += dt
                    self.aktif["en_uzun_kusur_s"][k] = max(
                        self.aktif["en_uzun_kusur_s"][k], self.aktif["seri_s"][k])
                else:
                    self.aktif["seri_s"][k] = 0.0
        if self.evre == "hazir_degil" and f <= self.profil.bitis_fleksiyon:
            self.evre = "ayakta"
        elif self.evre == "ayakta" and f >= self.profil.baslama_fleksiyon:
            self.aktif = {"baslangic_s": zaman_s,
                          **{k: dict.fromkeys(self.profil.kapsam, 0.0)
                             for k in ("kesin_s", "seri_s", "en_uzun_kusur_s")}}
            self.evre = "dip" if f >= self.profil.dip_fleksiyon else "inis"
        elif self.evre == "inis":
            if f >= self.profil.dip_fleksiyon:
                self.evre = "dip"
            elif f <= self.profil.bitis_fleksiyon:
                sonuc = self._bitir(zaman_s, False, "dip_evresi_gozlenmedi")
                self.evre = "ayakta"
        elif self.evre == "dip" and f < self.profil.dip_fleksiyon - 5:
            self.evre = "cikis"
            if f <= self.profil.bitis_fleksiyon:
                sonuc = self._bitir(zaman_s, True, "tamamlandi")
                self.evre = "ayakta"
        elif self.evre == "cikis" and f <= self.profil.bitis_fleksiyon:
            sonuc = self._bitir(zaman_s, True, "tamamlandi")
            self.evre = "ayakta"
        elif self.evre == "cikis" and f >= self.profil.dip_fleksiyon:
            self.evre = "dip"
        return {"evre": self.evre, "sonuc": sonuc}

    def bitir(self):
        """Kayit ortasinda biten tekrar kaybolmaz; eksik olarak raporlanir."""
        if self.aktif:
            self._bitir(self.onceki[0], False, "kayit_sonu")
        return self.sonuclar
