"""EC3D (Exercise Correction in 3D) verisini proje sozlesmesine ceviren adaptor.

Bicim (dosyanin kendisinden ve yazarlarin kodundan dogrulandi, 25 Eylul):

- `data_3D.pickle`: `poses` (N, 3, 25) ve `labels` (N, 5) =
  [egzersiz, kisi, talimat etiketi, tekrar, kare].
- Iskelet OpenPose BODY_25 duzeninde (z yukari, sag taraf -x, orta kalca
  orijinde; dikey siralama ile dogrulandi). Koordinatlar **normalize**
  (±0,34), metrik degil: acilar gecerli, milimetre hatasi olculemez.
- Talimat etiketi **1 = dogru yapilmis** (yazarlarin `dataset.py`'si bunlari
  "correct" diye ayiriyor). Squat'ta 1 dogru, 2 ayaklar cok genis, 3 dizler
  ice, 4 yeterince inmiyor, 5 govde one egik (Zhao ve ark., ACCV 2022, Tablo 1;
  "dizler ice" 23 dizi, sayim tutuyor); 10 makalede adlandirilmiyor.

Guvenlik notu: pickle ucuncu taraf dosyadir (yazarlarin Google Drive'i);
yalnizca KAYNAK.json'daki SHA-256 ile eslesen dosya yuklenmeli.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B

# REFERANS eklemi -> BODY_25 indeksi. Boyun, iki hatta oldugu gibi iki omzun
# ortalamasindan turetilir (BODY_25'in 1 numarali boyun noktasi degil).
BODY25 = {
    "sag_omuz": 2, "sag_dirsek": 3, "sag_bilek": 4, "sol_omuz": 5, "sol_dirsek": 6,
    "sol_bilek": 7, "sag_kalca": 9, "sag_diz": 10, "sag_ayak_bilegi": 11,
    "sol_kalca": 12, "sol_diz": 13, "sol_ayak_bilegi": 14,
}
_TURETILMIS = {"boyun": (2, 5)}

# BODY_25 indeksi -> TAM_VUCUT (pose3d.tam_vucut) indeksi; -1: karsiligi yok
# (1 boyun, 8 kalca ortasi turetilmis noktalar). BODY_25: 0 burun, 1 boyun,
# 2-4 sag omuz/dirsek/bilek, 5-7 sol, 8 kalca ortasi, 9-11 sag kalca/diz/ayak
# bilegi, 12-14 sol, 15/16 sag/sol goz, 17/18 sag/sol kulak, 19-21 sol bas
# parmak/serce parmak/topuk, 22-24 sag.
BODY25_TAM_VUCUT = (0, -1, 6, 8, 10, 5, 7, 9, -1, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3,
                    17, 18, 19, 20, 21, 22)


def kareyi_cevir(poz_3x25: np.ndarray) -> Iskelet3B:
    """(3, 25) BODY_25 -> REFERANS_ISKELET sirasinda Iskelet3B (normalize birim)."""
    P25 = np.asarray(poz_3x25, float).T
    n = len(REFERANS_ISKELET)
    P = np.full((n, 3), np.nan)
    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        idx = _TURETILMIS.get(ad, (BODY25.get(ad),))
        P[i] = P25[list(idx)].mean(axis=0)
    gorunur = np.isfinite(P).all(axis=1)
    P[~gorunur] = np.nan
    return Iskelet3B(tanim=REFERANS_ISKELET, noktalar=P, gorunur=gorunur,
                     goren_kamera=np.zeros(n, int), artik_px=np.full(n, np.nan),
                     birim="normalize", cerceve="ec3d_kalca_merkezi", kaynak="EC3D",
                     ek={"gorulen_kamera_sayisi_biliniyor": False,
                         "etiket_sozlesmesi": "1=dogru; diger kusur turleri dogrulanmadi"})


def tekrarlar(yol: str | Path, egzersiz: str | None = None) -> dict[tuple, np.ndarray]:
    """(egzersiz, kisi, etiket, tekrar) -> (kare, 3, 25), kare sirasina gore."""
    with open(yol, "rb") as f:
        veri = pickle.load(f)
    etiket, poz = veri["labels"], veri["poses"]
    gruplar: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for i, (eg, kisi, lab, tek, kare) in enumerate(etiket):
        if egzersiz is None or eg == egzersiz:
            gruplar[(str(eg), str(kisi), int(lab), int(tek))].append((int(kare), i))
    return {k: poz[[i for _, i in sorted(v)]] for k, v in gruplar.items()}
