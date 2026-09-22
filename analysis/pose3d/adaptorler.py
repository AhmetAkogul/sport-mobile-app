"""Hazir poz modellerinin ciktisini proje sozlesmesine ceviren adaptorler.

`docs/kararlar/0007` sozlesmeyi model secilmeden sabitledi; burasi o kararin
karsiligini oduyor: model degisince **yalnizca bu dosya** degisir, hat kodu
degismez.

Bu modul `mediapipe` paketine **bagimli degildir**. Sonucu ordek-tipleme ile
alir (landmarks / world_landmarks listeleri), boylece adaptor modelin kurulu
olmadigi bir ortamda da yazilabilir ve sinanabilir. Kalibrasyon ortamini
bozmamak icin bu onemli (HAZIR-TEKNOLOJILER.md madde 5).
"""
from __future__ import annotations

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET, IskeletTanimi
from pose3d.pose2d import Poz2B

# MediaPipe PoseLandmarker'in 33 eklemi, kendi sirasiyla (mediapipe 1.0.1).
MEDIAPIPE_EKLEMLER = (
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
)

MEDIAPIPE_ISKELET = IskeletTanimi(ad="mediapipe-33", eklemler=MEDIAPIPE_EKLEMLER)

# Dogrudan karsiligi olan eklemler: bizim isim -> MediaPipe ismi.
MEDIAPIPE_ESLEME = {
    "sag_omuz": "RIGHT_SHOULDER", "sag_dirsek": "RIGHT_ELBOW", "sag_bilek": "RIGHT_WRIST",
    "sol_omuz": "LEFT_SHOULDER", "sol_dirsek": "LEFT_ELBOW", "sol_bilek": "LEFT_WRIST",
    "sag_kalca": "RIGHT_HIP", "sag_diz": "RIGHT_KNEE", "sag_ayak_bilegi": "RIGHT_ANKLE",
    "sol_kalca": "LEFT_HIP", "sol_diz": "LEFT_KNEE", "sol_ayak_bilegi": "LEFT_ANKLE",
}

# MediaPipe'da boyun eklemi YOK. Iki omuzun orta noktasindan turetilir.
# Bu bir modelleme karari: turetilmis eklem, olculmus eklemle ayni degildir --
# guveni bilesenlerinin **en dusugu** alinir, ortalamasi degil, cunku bir omuz
# guvenilmezse orta nokta da guvenilmezdir.
TUREVLER = {"boyun": ("LEFT_SHOULDER", "RIGHT_SHOULDER")}


def _mp_dizileri(landmarks, alan_sayisi: int):
    """MediaPipe landmark listesi -> (koordinat, gorunurluk, varlik) dizileri."""
    if len(landmarks) != len(MEDIAPIPE_EKLEMLER):
        raise ValueError(
            f"MediaPipe {len(landmarks)} eklem dondurdu, {len(MEDIAPIPE_EKLEMLER)} "
            "bekleniyor -- surum degismis olabilir, esleme dogrulanmali")
    koord = np.array([[getattr(l, "x"), getattr(l, "y")] + 
                      ([getattr(l, "z")] if alan_sayisi == 3 else [])
                      for l in landmarks], dtype=np.float64)
    gorunurluk = np.array([getattr(l, "visibility", 1.0) or 0.0 for l in landmarks])
    varlik = np.array([getattr(l, "presence", 1.0) or 0.0 for l in landmarks])
    return koord, gorunurluk, varlik


def _referansa_tasi(koord, guven, gorunur):
    """MediaPipe sirasindaki dizileri REFERANS_ISKELET sirasina cevir."""
    boyut = koord.shape[1]
    n = len(REFERANS_ISKELET)
    yeni_koord = np.full((n, boyut), np.nan)
    yeni_guven = np.zeros(n)
    yeni_gorunur = np.zeros(n, dtype=bool)

    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        if ad in MEDIAPIPE_ESLEME:
            k = MEDIAPIPE_EKLEMLER.index(MEDIAPIPE_ESLEME[ad])
            yeni_koord[i] = koord[k]
            yeni_guven[i] = guven[k]
            yeni_gorunur[i] = gorunur[k]
        elif ad in TUREVLER:
            indeksler = [MEDIAPIPE_EKLEMLER.index(x) for x in TUREVLER[ad]]
            yeni_koord[i] = koord[indeksler].mean(axis=0)
            yeni_guven[i] = float(guven[indeksler].min())
            yeni_gorunur[i] = bool(gorunur[indeksler].all())
        # eslesmeyen eklem: NaN ve gorunmez kalir -- uydurulmaz
    return yeni_koord, yeni_guven, yeni_gorunur


def mediapipe_poz2b(
    landmarks,
    goruntu_boyutu: tuple[int, int],
    guven_esigi: float = 0.5,
    kamera: int | None = None,
    kare: int | None = None,
    model: str = "mediapipe-1.0.1/pose_landmarker",
) -> Poz2B:
    """MediaPipe `PoseLandmarkerResult.pose_landmarks[0]` -> `Poz2B`.

    MediaPipe **normalize** koordinat dondurur (0..1). Kalibrasyon piksel
    uzayinda calistigi icin burada goruntu boyutuyla carpilir; atlanirsa
    triangulation sessizce sacma sonuc verir.
    """
    g, y = goruntu_boyutu
    if min(g, y) <= 0:
        raise ValueError("goruntu boyutlari pozitif olmali")
    koord, gorunurluk, varlik = _mp_dizileri(landmarks, 2)
    piksel = koord * np.array([g, y])
    guven = np.minimum(gorunurluk, varlik)
    gorunur = guven >= guven_esigi

    yeni_koord, yeni_guven, yeni_gorunur = _referansa_tasi(piksel, guven, gorunur)
    return Poz2B(
        iskelet=REFERANS_ISKELET,
        noktalar=np.nan_to_num(yeni_koord, nan=0.0),
        guven=yeni_guven, gorunur=yeni_gorunur,
        goruntu_boyutu=goruntu_boyutu, model=model, kamera=kamera, kare=kare,
        uzay="ozgun",
        ek={"kaynak_iskelet": MEDIAPIPE_ISKELET.ad, "guven_esigi": guven_esigi,
            "turetilmis": tuple(TUREVLER)},
    )


def mediapipe_dunya_noktalari(world_landmarks, guven_esigi: float = 0.5):
    """MediaPipe `world_landmarks` -> (noktalar, gorunur) referans iskelet sirasinda.

    **Bunlar telefonun kendi 3B kestirimidir** ve tezin olctugu sey tam olarak
    budur (`docs/kararlar/0006`). Metre biriminde ve kalca merkezlidir: mutlak
    konum degil, govdeye gore sekildir. Duzenek ciktisiyla karsilastirirken
    dogrudan konum farki alinamaz; rijit hizalama gerekir
    (`pose3d.hizalama.rijit_hizala`).
    """
    koord, gorunurluk, varlik = _mp_dizileri(world_landmarks, 3)
    guven = np.minimum(gorunurluk, varlik)
    yeni_koord, _, yeni_gorunur = _referansa_tasi(koord, guven, guven >= guven_esigi)
    return yeni_koord, yeni_gorunur
