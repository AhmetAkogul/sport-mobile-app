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
    """Kamera cercevesinde govdenin bakis acisi (0 onden, 90 profil); tek tanim
    `eval.hareket_formu.bakis_acisi`'dadir (hareket formu da ayni aciyi kullanir)."""
    from eval.hareket_formu import bakis_acisi
    return bakis_acisi(P)


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


# Tam vucut cizim renkleri (BGR): govde beyaz, ayak turuncu, eller parmak parmak.
_GRUP_RENK = {"bas": (255, 200, 120), "govde": (255, 255, 255), "ayak": (0, 160, 255),
              "sol_el": (120, 255, 120), "sag_el": (255, 120, 255)}


def iskelet_ciz(out: np.ndarray, poz, renk=None, kalinlik: int = 2) -> np.ndarray:
    """TAM_VUCUT (ya da herhangi bir iskelet) pozunu yerinde cizer.

    Yalniz iki ucu da gorunen baglanti cizilir (eksik = NaN politikasi).
    `renk` verilirse butun iskelet o renkte (cok kiside kisi rengi) cizilir.
    """
    import cv2

    from pose3d.tam_vucut import GRUPLAR, TAM_VUCUT
    grup = {}
    if poz.iskelet.ad == TAM_VUCUT.ad:
        grup = {i: g for g, ix in GRUPLAR.items() for i in ix}
    P = poz.noktalar
    for a, b in poz.iskelet.baglantilar:
        i, j = poz.iskelet.indeks(a), poz.iskelet.indeks(b)
        if poz.gorunur[i] and poz.gorunur[j]:
            r = renk or _GRUP_RENK.get(grup.get(j, "govde"), (255, 255, 255))
            el = grup.get(j, "").endswith("_el")
            cv2.line(out, tuple(int(v) for v in P[i]), tuple(int(v) for v in P[j]), r,
                     max(1, kalinlik - 1) if el else kalinlik + 1, cv2.LINE_AA)
    for i in np.flatnonzero(poz.gorunur):
        el = grup.get(i, "").endswith("_el")
        cv2.circle(out, tuple(int(v) for v in P[i]), 2 if el else 4,
                   renk or (0, 140, 255), -1, cv2.LINE_AA)
    return out


def aynala(poz):
    """Pozu yatay aynalanmis goruntuye tasir (yalniz gosterim; sol/sag adlari
    kisinin kendi sol/sagi olarak kalir)."""
    from dataclasses import replace
    P = poz.noktalar.copy()
    P[:, 0] = poz.goruntu_boyutu[0] - 1 - P[:, 0]
    return replace(poz, noktalar=P)


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


def iskelet_kutusu(poz, pay: float = 0.15):
    """Onceki karenin iskeletinden sonraki kare icin kisi kutusu (dedektor atlanir)."""
    from pose3d.tam_vucut import GRUPLAR
    g = list(GRUPLAR["bas"] + GRUPLAR["govde"] + GRUPLAR["ayak"])
    P = poz.noktalar[g][poz.gorunur[g]]
    if len(P) < 4:
        return None
    lo, hi = P.min(axis=0), P.max(axis=0)
    d = (hi - lo) * pay
    w, h = poz.goruntu_boyutu
    return [max(lo[0] - d[0], 0), max(lo[1] - d[1], 0), min(hi[0] + d[0], w - 1), min(hi[1] + d[1], h - 1)]


def coklu_kisi_karesi(model, hat, im, t: float, kare_no: int, tespit_sikligi: int,
                      son: list) -> list:
    """Bir kare: RTMW (dedektor her `tespit_sikligi` karede bir) + takip.

    Aradaki karelerde onceki karenin iskelet kutulari kullanilir; kutu listesi
    bosalirsa dedektor hemen calisir (yeni giren kisi en gec N karede bulunur).
    """
    kutular = None
    if tespit_sikligi > 1 and kare_no % tespit_sikligi and son:
        kutular = [k for k in (iskelet_kutusu(p) for _, p in son) if k is not None] or None
    return hat.adim(t, model.kisiler(im, kutular=kutular))


def kisileri_ciz(out, eslesen: list, etiketler: dict | None = None):
    import cv2

    from mono.coklu_kisi import kisi_rengi
    for kimlik, poz in eslesen:
        renk = kisi_rengi(kimlik)
        iskelet_ciz(out, poz, renk)
        P = poz.noktalar[poz.gorunur]
        if len(P):
            x, y = int(P[:, 0].min()), int(max(P[:, 1].min() - 12, 20))
            metin = f"#{kimlik}" + (f" {etiketler[kimlik]}" if etiketler and kimlik in etiketler else "")
            cv2.putText(out, metin, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, metin, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2, cv2.LINE_AA)
    return out


def hareket_etiketleri(tanima, hat, eslesen: list) -> dict[int, str]:
    """Kimlik -> "squat %87" gibi etiket (2 s gecmis dolmadiysa etiket yok)."""
    out = {}
    for kimlik, _ in eslesen:
        g = hat.kisiler.get(kimlik)
        if g is None:
            continue
        r = tanima.tahmin(kimlik, *g.dizi())
        if r is not None:
            out[kimlik] = f"{r[0]} %{100 * r[1]:.0f}"
    return out


def kutu_dunya(mp_model, im, kutu, pay: float = 0.15):
    """Kisinin kutusunu kirpip MediaPipe'le dunya iskeleti (13, 3) ve ayak (4, 3).

    MediaPipe dunya noktalari kalca merkezli ve metriktir; kirpinti konumu
    sonucu degistirmez. Cok kiside her kisiye tek kisi modeli boyle uygulanir.
    """
    h, w = im.shape[:2]
    x0, y0, x1, y1 = kutu
    dx, dy = (x1 - x0) * pay, (y1 - y0) * pay
    x0, y0 = int(max(x0 - dx, 0)), int(max(y0 - dy, 0))
    x1, y1 = int(min(x1 + dx, w)), int(min(y1 + dy, h))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None, None
    ek = mp_model(np.ascontiguousarray(im[y0:y1, x0:x1])).ek
    if "world_points_all_m" not in ek:
        return None, None
    return np.array(ek["world_points_all_m"], float), np.array(ek["ayak_dunya_m"], float)


def _bildir(b: dict, kayit) -> None:
    print(f"[{b['t']:7.1f} s] #{b['kimlik']} {b['metin']}", flush=True)
    if kayit is not None:
        import json
        kayit.write(json.dumps({k: v for k, v in b.items() if k != "olculer"},
                               ensure_ascii=False, default=float) + "\n")


def antrenor_karesi(ant, mp_model, im, t: float, eslesen: list, son_mesaj: dict,
                    kayit=None, gosterim_s: float = 4.0) -> dict[int, str]:
    """Her kisi icin antrenore bir kare; kisi basina ekranda gosterilecek etiket."""
    etiket = {}
    for kimlik, poz in eslesen:
        kutu = iskelet_kutusu(poz)
        X3, ayak = kutu_dunya(mp_model, im, kutu) if kutu is not None else (None, None)
        for b in ant.adim(kimlik, t, poz.noktalar, poz.gorunur, X3, ayak):
            _bildir(b, kayit)
            son_mesaj[kimlik] = (t, b["metin"])
        d = ant.kisiler.get(kimlik)
        ad = ant.modeller[d.hareket]["ad"] if d is not None and d.hareket else ""
        n = len(d.tekrarlar) if d is not None and d.hareket else 0
        metin = f"{ad} x{n}" if ad else ""
        if kimlik in son_mesaj and t - son_mesaj[kimlik][0] <= gosterim_s:
            metin = (metin + " | " if metin else "") + son_mesaj[kimlik][1]
        etiket[kimlik] = metin
    return etiket


def coklu_main(a) -> None:
    """--coklu-kisi: karedeki herkes, kimlikli ve tam vucut (RTMW + ByteTrack; 0036)."""
    import cv2

    from mono.coklu_kisi import CokKisiHatti, KisiTakip
    from mono.rtmw_model import RTMWEstimator

    cap = cv2.VideoCapture(str(a.video) if a.video else a.kamera)
    if not cap.isOpened():
        raise SystemExit(f"kaynak acilamadi: {a.video or a.kamera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hat = CokKisiHatti(KisiTakip(kare_hizi=fps if a.video else 15.0, kayip_s=2.0))
    tanima = None
    if a.hareket_modeli:
        import joblib

        from eval.hareket import HareketTanima
        tanima = HareketTanima(joblib.load(a.hareket_modeli))
    ant = mp_model = kayit = None
    if a.antrenor:
        import joblib

        from eval.hareket import HareketTanima
        from mono.antrenor import Antrenor
        from mono.backend import kestirici_olustur
        if not (a.hareket_modeli and a.form_modelleri):
            raise SystemExit("--antrenor icin --hareket-modeli ve --form-modelleri gerekli")
        ant = Antrenor(HareketTanima(joblib.load(a.hareket_modeli)), joblib.load(a.form_modelleri))
        mp_model = kestirici_olustur("mediapipe", model=a.model)
        kayit = open(a.bildirim_kaydi, "a", encoding="utf-8") if a.bildirim_kaydi else None
    son_mesaj: dict[int, tuple[float, str]] = {}
    yazici, son, kare_no = None, [], 0
    t0 = time.monotonic()
    try:
        with RTMWEstimator(a.dedektor, a.rtmw) as model:
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                t = kare_no / fps if a.video else time.monotonic() - t0
                son = coklu_kisi_karesi(model, hat, im, t, kare_no, a.tespit_sikligi, son)
                kare_no += 1
                etiketler = hareket_etiketleri(tanima, hat, son) if tanima and not ant else None
                if ant is not None:
                    etiketler = antrenor_karesi(ant, mp_model, im, t, son, son_mesaj, kayit)
                cizilecek = son
                if a.aynalama:
                    im = cv2.flip(im, 1)
                    cizilecek = [(k, aynala(p)) for k, p in son]
                goster = kisileri_ciz(im, cizilecek, etiketler)
                cv2.putText(goster, f"Kisi: {len(son)}  (arastirma denemesi)  q: cikis",
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if a.kaydet is not None:
                    if yazici is None:
                        yazici = cv2.VideoWriter(str(a.kaydet), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps if a.video else 15.0,
                                                 (goster.shape[1], goster.shape[0]))
                    yazici.write(goster)
                if a.pencere:
                    cv2.imshow("Cok kisi - canli", goster)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        cap.release()
        if yazici is not None:
            yazici.release()
        if a.pencere:
            cv2.destroyAllWindows()
        if ant is not None:
            for b in ant.bitir():
                _bildir(b, kayit)
            mp_model.close()
            if kayit:
                kayit.close()
    print(f"{kare_no} kare, gorulen kimlikler: {sorted(hat.kisiler)}")


def main(argv=None) -> None:
    import cv2

    from mono.backend import kestirici_olustur

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="MediaPipe pose_landmarker .task")
    p.add_argument("--kamera", type=int, default=0, help="kamera indeksi")
    p.add_argument("--video", type=Path, help="kamera yerine bu videodan oku (deneme icin)")
    p.add_argument("--pencere", action=argparse.BooleanOptionalAction, default=True,
                   help="goruntuyu pencerede goster (--no-pencere: yalniz --kaydet)")
    p.add_argument("--aynalama", action=argparse.BooleanOptionalAction, default=True,
                   help="ekrani ayna gibi goster (hesap aynalanmamis goruntuyle yapilir)")
    p.add_argument("--kaydet", type=Path, help="islenmis goruntuyu bu .mp4'e de yaz")
    p.add_argument("--el-modeli", type=Path,
                   help="MediaPipe hand_landmarker .task; verilirse parmak eklemleri de cizilir")
    p.add_argument("--coklu-kisi", action="store_true",
                   help="karedeki herkes (RTMW + takip); --dedektor ve --rtmw gerekli")
    p.add_argument("--dedektor", type=Path, help="YOLOX kisi dedektoru .onnx (--coklu-kisi)")
    p.add_argument("--rtmw", type=Path, help="RTMW tam vucut .onnx (--coklu-kisi)")
    p.add_argument("--hareket-modeli", type=Path,
                   help="hareket tanima modeli (.joblib, scripts/deney_hareket_tanima.py)")
    p.add_argument("--antrenor", action="store_true",
                   help="--coklu-kisi ile: tekrar sayimi, tekrar karari ve ipucu (0040)")
    p.add_argument("--form-modelleri", type=Path,
                   help="hareket formu modelleri (.joblib, scripts/deney_hareket_formu.py)")
    p.add_argument("--bildirim-kaydi", type=Path, help="bildirimleri bu JSONL dosyasina ekle")
    p.add_argument("--tespit-sikligi", type=int, default=3,
                   help="dedektor kac karede bir calissin (--coklu-kisi)")
    a = p.parse_args(argv)
    if a.coklu_kisi:
        if not (a.dedektor and a.rtmw):
            p.error("--coklu-kisi icin --dedektor ve --rtmw gerekli")
        return coklu_main(a)

    cap = cv2.VideoCapture(str(a.video) if a.video else a.kamera)
    if not cap.isOpened() and a.video:
        raise SystemExit(f"video acilamadi: {a.video}")
    if not cap.isOpened():
        raise SystemExit(f"kamera {a.kamera} acilamadi (macOS: Sistem Ayarlari > Gizlilik "
                         "ve Guvenlik > Kamera'dan Terminal/VS Code'a izin verin)")
    yazici = None
    degerlendirici = CanliDegerlendirici()
    n = len(REFERANS_ISKELET)
    t0 = time.monotonic()
    try:
        with kestirici_olustur("mediapipe", model=a.model, el_modeli=a.el_modeli) as model:
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                # Tam vucut (bas, ayaklar, parmaklar) yalniz cizim icin; olcum
                # referans iskeletten yapilir, degismedi.
                poz, tam = model.tam_vucut(im)
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
                    tam = aynala(tam)
                goster = ciz(iskelet_ciz(im, tam), None, durum)
                if a.kaydet is not None:
                    if yazici is None:
                        yazici = cv2.VideoWriter(str(a.kaydet), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 30.0, (goster.shape[1], goster.shape[0]))
                    yazici.write(goster)
                if a.pencere:
                    cv2.imshow("Squat valgus - canli", goster)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        cap.release()
        if yazici is not None:
            yazici.release()
        if a.pencere:
            cv2.destroyAllWindows()
    for s in degerlendirici.takip.sonuclar:
        print(f"tekrar {s['tekrar']}: {s['karar']} ({s['bitis_s'] - s['baslangic_s']:.1f} s)")


if __name__ == "__main__":
    main()
