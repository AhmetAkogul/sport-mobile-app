"""Canli kamera: bilgisayar kamerasindan squat valgus karari (arastirma denemesi).

Hat, 28 Eylul olcumlerinde en iyi cikan koldur:

1. MediaPipe `pose_world`, gorunurluge bakmadan (gorunmez eklem tahmini dahil).
2. Govde acisi (0 onden, 90 profil) kalca/omuz hattindan kestirilir.
3. Valgus kaymasi aciya gore cikarilir: kayma = B0 + B1 * aci. Katsayilar
   REHAB24-6'nin 9 kisisinden (mocap referansli) ogrenildi ve burada DONMUSTUR
   (`docs/deney/2026-09-28-profil-ve-etiketsiz-yon.md`).
4. Govde acisi ACI_SINIRI'ni gecerse valgus karari verilmez ("bu acidan
   olculemez"): profilde hat kusurlari goremiyor (REHAB24 profil 0/6).
5. Tekrar karari `eval.tekrar.SquatTakip` ile, projedeki kuralla ayni
   (0014, 10 derece, 0,2 s kesintisiz, kapsama %80), gercek zaman damgasiyla.
6. Kurulum rehberi: bacaklar kadrajda degilse ya da kisi sirti donukse (yuz
   gorunmuyor) valgus karari verilmez ve ekranda ne yapilacagi yazar.
7. Govde acisi SAGITAL_ACI'yi gecince derinlik, diz onde ve govde egimi
   gosterilir -- yalniz deger, karar yok (esik bagimsiz veriyle kalibre edilmedi).
   Onden valgus, yandan sagital: her acidan bir olcut (28 Eylul olcumleri).

**Guvenilirlik:** REHAB24'te (laboratuvar, onden/yarim profil) dogru tekrarlarda
dogru diyor; gercek salon videolarinda (Fitness-AQA) "dizler ice"yi ayiramadi
(AUC 0,57). Bu bir olcum denemesidir, klinik ya da antrenor karari degildir.

    PYTHONPATH=. <mediapipe-env>/bin/python -m mono.canli --model <pose_landmarker_full.task>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from eval.egzersiz import EgzersizProfili, diz_fleksiyonu
from eval.form import VARSAYILAN_ESIKLER, Karar, form_degerlendir, govde_cercevesi
from eval.tekrar import SquatTakip
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B, eslestir

VALGUS = ("diz_valgusu_sag", "diz_valgusu_sol")
# REHAB24-6, mediapipe_world_tam, 9 kisi, 382 tekrar x kamera (28 Eylul).
KAYMA_B0, KAYMA_B1 = 4.35, 0.105
# Sezgisel: REHAB24'te yarim profil (medyan 41 derece) calisiyor, profil (85)
# calismiyor. 10 kusurlu tekrarla kalibre edilemez; kalibre edilmis esik degil.
ACI_SINIRI = 60.0
PROFIL = EgzersizProfili(surum="squat-canli-valgus-v1", kapsam=VALGUS)
# Sagital olculer (derinlik, diz onde, govde egimi) bu acidan itibaren gosterilir:
# REHAB24'te yarim profil ve profilde telefon fizyoterapist kararini 0,73-0,84 AUC
# ile ayiriyor, onden ayirmiyor (`docs/deney/2026-09-28-sagital.md`). Esik yok:
# bagimsiz veriyle kalibre edilmedi, yalniz deger gosterilir.
SAGITAL_ACI = 30.0
# Yuz gorunmuyorsa (sirti donuk) MediaPipe 3B'si sag-sol/derinlik isaretini
# karistirabilir; valgus karari verilmez.
YUZ_ESIGI = 0.5
MP_YUKARI = np.array([0.0, -1.0, 0.0])      # MediaPipe dunya: Y asagi, kamera yatay


def _birim(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def sagital_olculer(iskelet: Iskelet3B, yukari: np.ndarray = MP_YUKARI) -> dict:
    """Tek kare: derinlik (diz fleksiyonu, derece), diz_onde (dizin ayak bileginin
    onunde kalan yatay mesafesi / kaval boyu, iki bacagin buyugu), govde_egimi
    (kalca ortasi -> boyun ile dikey arasi, derece). Hesaplanamayan NaN."""
    P, i = iskelet.noktalar, iskelet.tanim.indeks
    f = diz_fleksiyonu(iskelet)
    kalca = (P[i("sag_kalca")] + P[i("sol_kalca")]) / 2
    govde = P[i("boyun")] - kalca
    egim = (float(np.degrees(np.arccos(np.clip(_birim(govde) @ yukari, -1, 1))))
            if np.isfinite(govde).all() and np.linalg.norm(govde) > 0 else float("nan"))
    onde = float("nan")
    c = govde_cercevesi(iskelet)
    if c is not None:
        ileri = _birim(c.ileri - (c.ileri @ yukari) * yukari)
        for taraf in ("sag", "sol"):
            v = P[i(f"{taraf}_diz")] - P[i(f"{taraf}_ayak_bilegi")]
            if np.isfinite(v).all() and np.linalg.norm(v) > 0:
                d = float(v @ ileri / np.linalg.norm(v))
                onde = d if not np.isfinite(onde) else max(onde, d)
    return {"derinlik": float("nan") if f is None else float(f),
            "diz_onde": onde, "govde_egimi": egim}


def rehber(bacak_gorunur: bool, yuz_guveni: float | None, aci: float | None) -> tuple[str, str]:
    """(kip, mesaj). kip: kadraj / sirt / onden / ara / yandan."""
    if not bacak_gorunur:
        return "kadraj", "Tum vucut ve ayaklar kadraja girmeli"
    if aci is None:
        return "kadraj", "Kisi bulunamadi"
    if aci < ACI_SINIRI and yuz_guveni is not None and yuz_guveni < YUZ_ESIGI:
        return "sirt", "Sirtiniz donuk: yuzunuzu kameraya donun"
    if aci < SAGITAL_ACI:
        return "onden", "Onden: dizler ice olculuyor"
    if aci > ACI_SINIRI:
        return "yandan", "Yandan: derinlik, diz ve govde olculuyor"
    return "ara", "Capraz: dizler ice ve derinlik (ikisi de daha az kesin)"


def govde_acisi(P: np.ndarray) -> float:
    """Kamera cercevesinde govdenin bakis acisi, derece: 0 onden, 90 profil.

    Sol-sag kalca ve omuz vektorlerinin ortalamasinin X (sag) - Z (derinlik)
    duzlemindeki yonu. Kameraya donuk ile sirti donuk ayrilmaz (ikisi de 0);
    karar icin gereken de yalnizca profile ne kadar yakin oldugudur.
    `P`: (13, 3) REFERANS sirasinda, kamera eksenleriyle hizali (X sag, Z ileri).
    """
    i = REFERANS_ISKELET.indeks
    v = [P[i("sol_kalca")] - P[i("sag_kalca")], P[i("sol_omuz")] - P[i("sag_omuz")]]
    v = [x for x in v if np.isfinite(x).all()]
    if not v:
        return float("nan")
    x, z = np.mean(v, axis=0)[[0, 2]]
    if x == 0 and z == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(z), abs(x))))


def dunya_iskeleti(ek: dict) -> Iskelet3B | None:
    """Poz `ek`inden maskesiz MediaPipe dunya iskeleti; tespit yoksa None."""
    P = ek.get("world_points_all_m")
    if P is None:
        return None
    P = np.asarray(P, float)
    g = np.isfinite(P).all(axis=1)
    n = len(REFERANS_ISKELET)
    return Iskelet3B(REFERANS_ISKELET, P, g, g.astype(int), np.full(n, np.nan),
                     cerceve="mediapipe_kalca_merkezi", kaynak="mediapipe_world_tam")


def kare_hazirla(iskelet: Iskelet3B | None, bacak_gorunur: bool = True,
                 yuz_guveni: float | None = None) -> tuple[dict, dict]:
    """SquatTakip karesi ve ekran bilgisi.

    Kayma aciya gore cikarilir. Aci bilinmiyor, ACI_SINIRI'ni asiyor, bacaklar
    kadrajda degil ya da kisi sirti donukse valgus karari BELIRSIZ olur
    (duzeltme karar uydurmaz).
    """
    bos = {ad: {"karar": "belirsiz"} for ad in VALGUS}
    if iskelet is None:
        kip, mesaj = rehber(bacak_gorunur, yuz_guveni, None)
        return ({"profil": PROFIL.surum, "olcumler": bos, "fleksiyon_derece": None},
                {"aci": None, "kayma": None, "valgus": (None, None), "olculemez": False,
                 "kip": kip, "mesaj": mesaj, "sagital": None})
    aci = govde_acisi(iskelet.noktalar)
    kayma = KAYMA_B0 + KAYMA_B1 * aci if np.isfinite(aci) else None
    kip, mesaj = rehber(bacak_gorunur, yuz_guveni, aci if np.isfinite(aci) else None)
    olculemez = bool(np.isfinite(aci) and aci > ACI_SINIRI) or kip in ("kadraj", "sirt")
    rapor = form_degerlendir(iskelet)
    olcumler, valgus = {}, []
    for ad in VALGUS:
        o = rapor.olcumler[ad]
        if o.karar is Karar.BELIRSIZ or not np.isfinite(o.deger) or kayma is None:
            olcumler[ad] = {"karar": "belirsiz"}
            valgus.append(None)
            continue
        d = float(o.deger - kayma)
        valgus.append(d)
        karar = Karar.BELIRSIZ if olculemez else (
            Karar.KUSURLU if d > VARSAYILAN_ESIKLER[ad].deger else Karar.DOGRU)
        olcumler[ad] = {"karar": str(karar)}
    kare = {"profil": PROFIL.surum, "olcumler": olcumler,
            "fleksiyon_derece": diz_fleksiyonu(iskelet)}
    sagital = (sagital_olculer(iskelet)
               if np.isfinite(aci) and aci >= SAGITAL_ACI and kip != "kadraj" else None)
    return kare, {"aci": aci if np.isfinite(aci) else None, "kayma": kayma,
                  "valgus": tuple(valgus), "olculemez": olculemez,
                  "kip": kip, "mesaj": mesaj, "sagital": sagital}


def _tekrar_sagital(kareler: list[dict]) -> dict | None:
    """Tekrar boyunca her sagital olcunun en buyugu (NaN atlanir)."""
    if not kareler:
        return None
    out = {}
    for k in ("derinlik", "diz_onde", "govde_egimi"):
        v = [x[k] for x in kareler if np.isfinite(x[k])]
        out[k] = float(max(v)) if v else None
    return out


class CanliDegerlendirici:
    """Kare kare poz alir, tekrar kararlarini biriktirir."""

    def __init__(self):
        self.takip = SquatTakip(PROFIL)
        self.son_zaman: float | None = None
        self.acilar: list[float] = []
        self.sagital: list[dict] = []
        self.son_tekrar: dict | None = None

    def ekle(self, zaman_s: float, ek: dict, bacak_gorunur: bool = True) -> dict:
        if self.son_zaman is not None and zaman_s <= self.son_zaman:
            zaman_s = self.son_zaman + 1e-6        # SquatTakip kesin artan zaman ister
        self.son_zaman = zaman_s
        kare, bilgi = kare_hazirla(dunya_iskeleti(ek), bacak_gorunur, ek.get("yuz_guveni"))
        durum = self.takip.ekle(zaman_s, kare)
        if self.takip.aktif is not None:
            if bilgi["aci"] is not None:
                self.acilar.append(bilgi["aci"])
            if bilgi["sagital"] is not None:
                self.sagital.append(bilgi["sagital"])
        if durum.get("sonuc"):
            s = dict(durum["sonuc"])
            s["medyan_aci"] = float(np.median(self.acilar)) if self.acilar else None
            s["olculemez"] = s["medyan_aci"] is not None and s["medyan_aci"] > ACI_SINIRI
            s["sagital"] = _tekrar_sagital(self.sagital)
            self.son_tekrar = s
            self.acilar, self.sagital = [], []
        elif self.takip.aktif is None:
            self.acilar, self.sagital = [], []
        return {**bilgi, "evre": durum.get("evre", self.takip.evre),
                "tekrar_sayisi": len(self.takip.sonuclar), "son_tekrar": self.son_tekrar}


_BACAK = [REFERANS_ISKELET.indeks(e) for e in (
    "sag_kalca", "sol_kalca", "sag_diz", "sol_diz", "sag_ayak_bilegi", "sol_ayak_bilegi")]
_RENK = {"dogru": (60, 180, 60), "kusurlu": (40, 40, 220), "belirsiz": (0, 190, 230)}


def ciz(image: np.ndarray, noktalar_2b: np.ndarray | None, durum: dict) -> np.ndarray:
    import cv2
    out = image.copy()
    if noktalar_2b is not None:
        for a, b in REFERANS_ISKELET.baglantilar:
            pa = noktalar_2b[REFERANS_ISKELET.indeks(a)]
            pb = noktalar_2b[REFERANS_ISKELET.indeks(b)]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(out, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                         (255, 255, 255), 3, cv2.LINE_AA)
        for p in noktalar_2b:
            if np.isfinite(p).all():
                cv2.circle(out, tuple(int(v) for v in p), 5, (0, 140, 255), -1, cv2.LINE_AA)

    def yaz(metin, y, renk=(255, 255, 255), olcek=0.7):
        cv2.putText(out, metin, (14, y), cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, metin, (14, y), cv2.FONT_HERSHEY_SIMPLEX, olcek, renk, 2, cv2.LINE_AA)

    aci = durum.get("aci")
    f = lambda v, b="+.0f": "--" if v is None or not np.isfinite(v) else format(v, b)  # noqa: E731
    uyari = durum.get("kip") in ("kadraj", "sirt")
    yaz(durum.get("mesaj") or "", 30, _RENK["belirsiz"] if uyari else (255, 255, 255))
    yaz(f"Bakis acisi: {f(aci, '.0f')} derece (0 onden, 90 yandan)", 58)
    if not durum.get("olculemez"):
        sag, sol = durum.get("valgus", (None, None))
        yaz(f"Dizler ice (duzeltilmis) sag {f(sag)}  sol {f(sol)}  esik +10", 86)
    sg = durum.get("sagital")
    if sg:
        yaz(f"Derinlik {f(sg['derinlik'], '.0f')}  diz onde {f(sg['diz_onde'], '.2f')}  "
            f"govde {f(sg['govde_egimi'], '.0f')}", 114)
    yaz(f"Evre: {durum.get('evre')}   Tekrar: {durum.get('tekrar_sayisi', 0)}", 142)
    s = durum.get("son_tekrar")
    if s:
        karar = "dizler ice olculemedi" if s.get("olculemez") else s["karar"]
        yaz(f"Son tekrar #{s['tekrar']}: {karar.upper()}", 180,
            _RENK.get(s["karar"], (255, 255, 255)), 0.9)
        t = s.get("sagital")
        if t:
            yaz(f"  en derin {f(t['derinlik'], '.0f')}  diz onde {f(t['diz_onde'], '.2f')}  "
                f"govde {f(t['govde_egimi'], '.0f')} (esik yok)", 208)
    yaz("Arastirma denemesi - klinik karar degil.  q: cikis", out.shape[0] - 14,
        (200, 200, 200), 0.55)
    return out


def main(argv=None) -> None:
    import cv2

    from mono.backend import kestirici_olustur

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="MediaPipe pose_landmarker .task")
    p.add_argument("--kamera", type=int, default=0, help="kamera indeksi")
    p.add_argument("--aynalama", action=argparse.BooleanOptionalAction, default=True,
                   help="ekrani ayna gibi goster (hesap aynalanmamis goruntuyle yapilir)")
    p.add_argument("--kaydet", type=Path, help="islenmis goruntuyu bu .mp4'e de yaz")
    a = p.parse_args(argv)

    cap = cv2.VideoCapture(a.kamera)
    if not cap.isOpened():
        raise SystemExit(f"kamera {a.kamera} acilamadi (macOS: Sistem Ayarlari > Gizlilik "
                         "ve Guvenlik > Kamera'dan Terminal/VS Code'a izin verin)")
    yazici = None
    degerlendirici = CanliDegerlendirici()
    n = len(REFERANS_ISKELET)
    t0 = time.monotonic()
    try:
        with kestirici_olustur("mediapipe", model=a.model) as model:
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                poz = model(im)
                P2 = None
                if poz.tespit:
                    P2 = np.full((n, 2), np.nan)
                    for j, k in enumerate(eslestir(poz.iskelet, REFERANS_ISKELET)):
                        if k >= 0 and poz.gorunur[k]:
                            P2[j] = poz.noktalar[k]
                bacak = P2 is not None and bool(np.isfinite(P2[_BACAK]).all())
                durum = degerlendirici.ekle(time.monotonic() - t0, poz.ek, bacak)
                if a.aynalama:
                    im = cv2.flip(im, 1)
                    if P2 is not None:
                        P2[:, 0] = im.shape[1] - 1 - P2[:, 0]
                goster = ciz(im, P2, durum)
                if a.kaydet is not None:
                    if yazici is None:
                        yazici = cv2.VideoWriter(str(a.kaydet), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 30.0, (goster.shape[1], goster.shape[0]))
                    yazici.write(goster)
                cv2.imshow("Squat valgus - canli", goster)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if yazici is not None:
            yazici.release()
        cv2.destroyAllWindows()
    for s in degerlendirici.takip.sonuclar:
        print(f"tekrar {s['tekrar']}: {s['karar']} ({s['bitis_s'] - s['baslangic_s']:.1f} s)")


if __name__ == "__main__":
    main()
