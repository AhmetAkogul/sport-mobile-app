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
from eval.form import VARSAYILAN_ESIKLER, Karar, form_degerlendir
from eval.tekrar import SquatTakip
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B, eslestir

VALGUS = ("diz_valgusu_sag", "diz_valgusu_sol")
# REHAB24-6, mediapipe_world_tam, 9 kisi, 382 tekrar x kamera (28 Eylul).
KAYMA_B0, KAYMA_B1 = 4.35, 0.105
# Sezgisel: REHAB24'te yarim profil (medyan 41 derece) calisiyor, profil (85)
# calismiyor. 10 kusurlu tekrarla kalibre edilemez; kalibre edilmis esik degil.
ACI_SINIRI = 60.0
PROFIL = EgzersizProfili(surum="squat-canli-valgus-v1", kapsam=VALGUS)


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


def kare_hazirla(iskelet: Iskelet3B | None) -> tuple[dict, dict]:
    """SquatTakip karesi ve ekran bilgisi.

    Kayma aciya gore cikarilir; aci bilinmiyor ya da ACI_SINIRI'ni asiyorsa
    valgus karari BELIRSIZ olur (duzeltme karar uydurmaz).
    """
    bos = {ad: {"karar": "belirsiz"} for ad in VALGUS}
    if iskelet is None:
        return ({"profil": PROFIL.surum, "olcumler": bos, "fleksiyon_derece": None},
                {"aci": None, "kayma": None, "valgus": (None, None), "olculemez": False})
    aci = govde_acisi(iskelet.noktalar)
    kayma = KAYMA_B0 + KAYMA_B1 * aci if np.isfinite(aci) else None
    olculemez = bool(np.isfinite(aci) and aci > ACI_SINIRI)
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
    return kare, {"aci": aci if np.isfinite(aci) else None, "kayma": kayma,
                  "valgus": tuple(valgus), "olculemez": olculemez}


class CanliDegerlendirici:
    """Kare kare poz alir, tekrar kararlarini biriktirir."""

    def __init__(self):
        self.takip = SquatTakip(PROFIL)
        self.son_zaman: float | None = None
        self.acilar: list[float] = []
        self.son_tekrar: dict | None = None

    def ekle(self, zaman_s: float, ek: dict) -> dict:
        if self.son_zaman is not None and zaman_s <= self.son_zaman:
            zaman_s = self.son_zaman + 1e-6        # SquatTakip kesin artan zaman ister
        self.son_zaman = zaman_s
        kare, bilgi = kare_hazirla(dunya_iskeleti(ek))
        durum = self.takip.ekle(zaman_s, kare)
        if self.takip.aktif is not None and bilgi["aci"] is not None:
            self.acilar.append(bilgi["aci"])
        if durum.get("sonuc"):
            s = dict(durum["sonuc"])
            s["medyan_aci"] = float(np.median(self.acilar)) if self.acilar else None
            s["olculemez"] = s["medyan_aci"] is not None and s["medyan_aci"] > ACI_SINIRI
            self.son_tekrar = s
            self.acilar = []
        elif self.takip.aktif is None:
            self.acilar = []
        return {**bilgi, "evre": durum.get("evre", self.takip.evre),
                "tekrar_sayisi": len(self.takip.sonuclar), "son_tekrar": self.son_tekrar}


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
    yaz(f"Bakis acisi: {aci:.0f} derece (0 onden, 90 yandan)" if aci is not None
        else "Kisi bulunamadi", 30)
    if durum.get("olculemez"):
        yaz("Bu acidan olculemez: kameraya onden don", 60, _RENK["belirsiz"])
    else:
        sag, sol = durum.get("valgus", (None, None))
        f = lambda v: "--" if v is None else f"{v:+.0f}"  # noqa: E731
        yaz(f"Valgus (duzeltilmis) sag {f(sag)}  sol {f(sol)}  esik +10", 60)
    yaz(f"Evre: {durum.get('evre')}   Tekrar: {durum.get('tekrar_sayisi', 0)}", 90)
    s = durum.get("son_tekrar")
    if s:
        karar = "olculemez" if s.get("olculemez") else s["karar"]
        yaz(f"Son tekrar #{s['tekrar']}: {karar.upper()}", 130,
            _RENK.get(s["karar"], (255, 255, 255)), 0.9)
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
                durum = degerlendirici.ekle(time.monotonic() - t0, poz.ek)
                P2 = None
                if poz.tespit:
                    P2 = np.full((n, 2), np.nan)
                    for j, k in enumerate(eslestir(poz.iskelet, REFERANS_ISKELET)):
                        if k >= 0 and poz.gorunur[k]:
                            P2[j] = poz.noktalar[k]
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
