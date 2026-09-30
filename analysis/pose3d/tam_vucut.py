"""Tam vucut iskeleti: govde + bas + ayaklar + parmak eklemleri (65 nokta).

Duzen COCO-WholeBody'dir (RTMW'nin 133 noktasi) ama **yuzun 68 noktasi yoktur**:
egzersiz olcumunde kullanilmiyor ve KVKK acisindan gereksiz veri (0007'deki
referans iskeletle ayni gerekce). Bas; burun, gozler ve kulaklarla temsil edilir
-- bas yonu/egimi icin yeterli.

    0-16   govde (COCO-17, bas noktalari dahil)
    17-22  ayaklar: sol bas parmak, sol serce parmak, sol topuk, sag ...
    23-43  sol el (21): el kok, basparmak 1-4, isaret 1-4, orta 1-4, yuzuk 1-4, serce 1-4
    44-64  sag el (21): ayni sira

El sirasi MediaPipe Hand Landmarker'in 21 noktasiyla birebir aynidir; iki
kaynak (RTMW, MediaPipe poz + el) ayni sozlesmeye kayipsiz baglanir.

Eksik veri politikasi degismez: eksik nokta NaN ve `gorunur=False`.
"""

from __future__ import annotations

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, IskeletTanimi
from pose3d.pose2d import Poz2B

_GOVDE = ("burun", "sol_goz", "sag_goz", "sol_kulak", "sag_kulak",
          "sol_omuz", "sag_omuz", "sol_dirsek", "sag_dirsek", "sol_bilek", "sag_bilek",
          "sol_kalca", "sag_kalca", "sol_diz", "sag_diz", "sol_ayak_bilegi", "sag_ayak_bilegi")
_AYAK = ("sol_bas_parmak", "sol_serce_parmak", "sol_topuk",
         "sag_bas_parmak", "sag_serce_parmak", "sag_topuk")
PARMAKLAR = ("basparmak", "isaret", "orta", "yuzuk", "serce")


def _el(taraf: str) -> tuple[str, ...]:
    return (f"{taraf}_el_kok",) + tuple(f"{taraf}_{p}_{i}" for p in PARMAKLAR for i in range(1, 5))


def _el_baglantilari(taraf: str) -> tuple[tuple[str, str], ...]:
    out = []
    for p in PARMAKLAR:
        zincir = [f"{taraf}_el_kok"] + [f"{taraf}_{p}_{i}" for i in range(1, 5)]
        out += list(zip(zincir[:-1], zincir[1:]))
    # avuc ici: parmak koklerini birbirine bagla (MediaPipe cizimiyle ayni)
    kokler = [f"{taraf}_{p}_1" for p in PARMAKLAR[1:]]
    out += list(zip(kokler[:-1], kokler[1:]))
    # el kokunu govde bilegine bagla: iki model ayni noktayi ayri kestirir
    out.append((f"{taraf}_bilek", f"{taraf}_el_kok"))
    return tuple(out)


_GOVDE_BAGLANTI = (
    ("burun", "sol_goz"), ("burun", "sag_goz"), ("sol_goz", "sol_kulak"), ("sag_goz", "sag_kulak"),
    ("sol_omuz", "sag_omuz"), ("sol_omuz", "sol_dirsek"), ("sol_dirsek", "sol_bilek"),
    ("sag_omuz", "sag_dirsek"), ("sag_dirsek", "sag_bilek"),
    ("sol_omuz", "sol_kalca"), ("sag_omuz", "sag_kalca"), ("sol_kalca", "sag_kalca"),
    ("sol_kalca", "sol_diz"), ("sol_diz", "sol_ayak_bilegi"),
    ("sag_kalca", "sag_diz"), ("sag_diz", "sag_ayak_bilegi"),
    ("sol_ayak_bilegi", "sol_topuk"), ("sol_topuk", "sol_bas_parmak"),
    ("sol_bas_parmak", "sol_serce_parmak"),
    ("sag_ayak_bilegi", "sag_topuk"), ("sag_topuk", "sag_bas_parmak"),
    ("sag_bas_parmak", "sag_serce_parmak"),
)

TAM_VUCUT = IskeletTanimi(
    ad="bitirme-tam-65",
    eklemler=_GOVDE + _AYAK + _el("sol") + _el("sag"),
    baglantilar=_GOVDE_BAGLANTI + _el_baglantilari("sol") + _el_baglantilari("sag"),
)

GRUPLAR = {
    "bas": tuple(range(0, 5)),
    "govde": tuple(range(5, 17)),
    "ayak": tuple(range(17, 23)),
    "sol_el": tuple(range(23, 44)),
    "sag_el": tuple(range(44, 65)),
}

# COCO-WholeBody (133) -> TAM_VUCUT (65): govde+ayak 0..22 aynen, yuz 23..90
# atlanir, sol el 91..111, sag el 112..132.
COCO_WB_INDEKS = np.array(list(range(23)) + list(range(91, 133)))

# MediaPipe Pose (33) -> TAM_VUCUT govde + ayak; -1: MediaPipe'ta karsiligi yok
# (serce parmak ucu). Ayak "bas parmak" = MediaPipe foot_index.
MP_POZ_INDEKS = np.array([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
                          31, -1, 29, 32, -1, 30])


# Avuc boyu / omuz genisligi icin kabul araligi (cokmus el tespitini eler).
AVUC_ORANI = (0.12, 0.9)


def bos_poz(boyut, model: str) -> Poz2B:
    """Tespit yok: butun noktalar NaN, `tespit=False`."""
    n = len(TAM_VUCUT)
    return Poz2B(TAM_VUCUT, np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool),
                 False, boyut, model=model)


def coco_wholebody_poz(noktalar, skorlar, boyut, *, model: str, esik: float,
                       ek: dict | None = None) -> Poz2B:
    """RTMW'nin (133,2) nokta ve (133,) skorunu TAM_VUCUT pozuna cevirir.

    RTMW skoru SIMCC kaynaklidir; olasilik degildir ve 1'i asabilir (0031).
    `guven` alani 0..1'e kirpilir, ham skor `ek["ham_skor"]`da kalir.
    """
    noktalar, skorlar = np.asarray(noktalar, float), np.asarray(skorlar, float)
    if noktalar.shape != (133, 2) or skorlar.shape != (133,):
        raise ValueError("COCO-WholeBody icin (133,2) nokta ve (133,) skor gerekli")
    xy = noktalar[COCO_WB_INDEKS].copy()
    ham = skorlar[COCO_WB_INDEKS]
    gorunur = np.isfinite(xy).all(axis=1) & np.isfinite(ham) & (ham >= esik)
    xy[~gorunur] = np.nan
    return Poz2B(TAM_VUCUT, xy, np.clip(np.nan_to_num(ham), 0, 1), gorunur, True, boyut,
                 model=model, ek={"kaynak_iskelet": "coco-wholebody-133", "guven_esigi": esik,
                                  "ham_skor": ham.tolist(), **(ek or {})})


def el_ata(bilekler: np.ndarray, el_kokleri: list[np.ndarray], *,
           en_fazla: float) -> dict[int, str]:
    """Bulunan elleri govdenin bileklerine en yakin eslesmeyle atar.

    MediaPipe'in sag/sol etiketi aynalanmis goruntu varsayar ve aynalanmamis
    karede ters doner; etikete guvenmek yerine el kokunu pozun bilegine
    (0: sol, 1: sag) yakinlikla eslestiririz. `en_fazla` pikselden uzak el
    atanmaz. Doner: {el sirasi: "sol" | "sag"}.
    """
    adaylar = []
    for i, kok in enumerate(el_kokleri):
        for j, taraf in enumerate(("sol", "sag")):
            if np.isfinite(bilekler[j]).all() and np.isfinite(kok).all():
                d = float(np.linalg.norm(bilekler[j] - kok))
                if d <= en_fazla:
                    adaylar.append((d, i, taraf))
    atama: dict[int, str] = {}
    for _, i, taraf in sorted(adaylar):
        if i not in atama and taraf not in atama.values():
            atama[i] = taraf
    return atama


def mediapipe_tam_vucut(poz_px, poz_guven, eller_px: list, boyut, *, model: str,
                        esik: float = 0.5, el_mesafe_orani: float = 0.6) -> Poz2B:
    """MediaPipe poz (33 nokta) + el (her biri 21 nokta) -> TAM_VUCUT.

    `poz_px`: (33,2) piksel, `poz_guven`: (33,) min(visibility, presence).
    `eller_px`: bulunan her el icin (21,2) piksel. Eller bileklere `el_ata` ile
    atanir; esik omuz genisliginin `el_mesafe_orani` katidir. Bilegi gorunmeyen
    tarafa el atanmaz.
    """
    poz_px, poz_guven = np.asarray(poz_px, float), np.asarray(poz_guven, float)
    if poz_px.shape != (33, 2) or poz_guven.shape != (33,):
        raise ValueError("MediaPipe poz icin (33,2) nokta ve (33,) guven gerekli")
    n = len(TAM_VUCUT)
    xy, guven, gorunur = np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool)
    var = MP_POZ_INDEKS >= 0
    xy[:23][var] = poz_px[MP_POZ_INDEKS[var]]
    guven[:23][var] = poz_guven[MP_POZ_INDEKS[var]]
    gorunur[:23] = var & (guven[:23] >= esik) & np.isfinite(xy[:23]).all(axis=1)
    xy[~gorunur] = np.nan
    omuz = float(np.linalg.norm(xy[5] - xy[6]))
    en_fazla = el_mesafe_orani * omuz if np.isfinite(omuz) and omuz > 0 else 0.0
    eller = [np.asarray(e, float) for e in eller_px]
    for el in eller:
        if el.shape != (21, 2):
            raise ValueError("el icin (21,2) nokta gerekli")
    # Cokmus tespitleri ele: avuc boyu (el kok -> orta parmak kok) omuz
    # genisliginin makul bir kesri olmali. REHAB24'te 2 ve 8 piksellik "eller"
    # (omzun %2-5'i) goruldu; gercek avuc ~%25-40'tir.
    elenen = 0
    if en_fazla > 0:
        makul = []
        for el in eller:
            avuc = float(np.linalg.norm(el[0] - el[9]))
            ok = np.isfinite(avuc) and AVUC_ORANI[0] * omuz <= avuc <= AVUC_ORANI[1] * omuz
            elenen += not ok
            if ok:
                makul.append(el)
        eller = makul
    atama = el_ata(xy[[9, 10]], [e[0] for e in eller], en_fazla=en_fazla)
    for i, taraf in atama.items():
        dilim = GRUPLAR[f"{taraf}_el"]
        ok = np.isfinite(eller[i]).all(axis=1)
        xy[list(dilim)] = np.where(ok[:, None], eller[i], np.nan)
        # El modeli nokta basina guven vermez; el tespit edildiyse noktalar gorunur.
        guven[list(dilim)] = ok.astype(float)
        gorunur[list(dilim)] = ok
    return Poz2B(TAM_VUCUT, xy, guven, gorunur, True, boyut, model=model,
                 ek={"kaynak_iskelet": "mediapipe-pose33+hand21", "guven_esigi": esik,
                     "bulunan_el": len(eller) + elenen, "elenen_el": elenen,
                     "atanan_el": sorted(atama.values())})


_REF_KAYNAK = {"boyun": ("sol_omuz", "sag_omuz")}


def referansa_indir(poz: Poz2B) -> Poz2B:
    """TAM_VUCUT -> bitirme-13; mevcut form/valgus olcumleri degismeden calisir.

    Boyun, iki omuzun ortasidir (RTMPose adaptoruyle ayni kural).
    """
    n = len(REFERANS_ISKELET)
    xy, guven, gorunur = np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool)
    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        kaynak = [TAM_VUCUT.indeks(k) for k in _REF_KAYNAK.get(ad, (ad,))]
        if poz.gorunur[kaynak].all():
            xy[i] = poz.noktalar[kaynak].mean(axis=0)
            guven[i] = poz.guven[kaynak].min()
            gorunur[i] = True
    return Poz2B(REFERANS_ISKELET, xy, guven, gorunur, poz.tespit, poz.goruntu_boyutu,
                 model=poz.model, kamera=poz.kamera, kare=poz.kare, uzay=poz.uzay,
                 ek={**poz.ek, "turetilmis": ["boyun"], "kaynak": TAM_VUCUT.ad})
