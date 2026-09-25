"""Ucgenlemeden eklem basina 3B konum belirsizligi.

Eksik halka buydu. `eval/form.py` belirsizligi karara cevirebiliyordu ama
belirsizligin **kendisi** disaridan elle veriliyordu; gercek hat onu
uretemiyordu. Bu modul `Iskelet3B`'yi ve onu ureten gozlemleri alip eklem
basina 1-sigma dizisi cikarir, yani `form_degerlendir`'in
`konum_belirsizligi_m` girdisini gercek veriden besler.

Yontem birinci derece yayilim: bir eklemi N kamera goruyorsa, izdusumun 3B
noktaya gore Jacobian'i J (2N x 3) kurulur ve `Cov = sigma_px^2 (J^T J)^-1`
alinir. Gozlem gurultusu kameralar arasi bagimsiz ve her eksende `sigma_px`
kabul edilir.

**Onemli sinirlilik:** bu yalnizca **tespit gurultusunun** katkisidir.
Kalibrasyon belirsizligi dahil degildir ve hata butcesi deneyi
(`2026-09-22-triangulasyon-butcesi.md`) 0,1 px tespit gurultusunde kalibrasyonun
payini %79 olcmustu. Yani dusuk tespit gurultusu rejiminde buradan cikan sayi
gercek belirsizligi **kucuk gosterir**. Gerekce ve secenekler:
`docs/kararlar/0013-eklem-belirsizligi.md`.
"""
from __future__ import annotations

import numpy as np

from calib.multiview import Kalibrasyon
from pose3d.iskelet import Iskelet3B, eslestir
from pose3d.triangulate import kalibrasyondan_projeksiyonlar, nokta_kovaryansi


def eklem_kovaryanslari(
    kalib: Kalibrasyon,
    iskelet: Iskelet3B,
    gozlemler: dict[int, object],
    sigma_px: float = 1.0,
) -> list[np.ndarray | None]:
    """Eklem basina (3,3) kovaryans; hesaplanamayan eklem icin None.

    `gozlemler`: {kamera_indeksi: Poz2B} -- `iskelet_ucgenle`'ye verilenin
    aynisi. Gorunurluk karari **yeniden verilmez**: yalnizca iskelette zaten
    gorunur isaretli eklemler icin hesap yapilir, boylece bu modul ile
    `pose3d/iskelet.py` arasinda ikinci bir gorunurluk mantigi olusmaz.
    """
    iskelet.metrik_gerekli()
    if iskelet.cerceve != "kamera0":
        raise ValueError("kalibrasyon kovaryansi kamera0 cercevesi gerektirir")
    tanim = iskelet.tanim
    P_hepsi = kalibrasyondan_projeksiyonlar(kalib)
    haritalar = {k: (poz, eslestir(poz.iskelet, tanim))
                 for k, poz in gozlemler.items()}

    cikti: list[np.ndarray | None] = []
    for j in range(len(tanim)):
        if not iskelet.gorunur[j] or not np.isfinite(iskelet.noktalar[j]).all():
            cikti.append(None)
            continue
        projeksiyonlar = []
        for kamera, (poz, harita) in haritalar.items():
            kullanilan = iskelet.ek.get("kullanilan_kameralar")
            if kullanilan is not None and kamera not in kullanilan.get(tanim.eklemler[j], []):
                continue
            k = harita[j]
            if k >= 0 and poz.gorunur[k] and kamera < len(P_hepsi):
                projeksiyonlar.append(P_hepsi[kamera])
        cikti.append(
            nokta_kovaryansi(projeksiyonlar, iskelet.noktalar[j], sigma_px)
            if len(projeksiyonlar) >= 2 else None
        )
    return cikti


def eklem_belirsizligi(
    kalib: Kalibrasyon,
    iskelet: Iskelet3B,
    gozlemler: dict[int, object],
    sigma_px: float = 1.0,
) -> np.ndarray:
    """Eklem basina izotropik 1-sigma (metre); hesaplanamayan eklem NaN.

    Dogrudan `form_degerlendir(..., konum_belirsizligi_m=...)` icine verilir.
    Skalar karsilik `sqrt(iz/3)`: ayni toplam varyansi tasiyan izotropik
    dagilimin sigmasi. Ucgenleme hatasi gercekte anizotropiktir; yone duyarli
    hesap gerekiyorsa `eklem_kovaryanslari` kullanilmali.

    NaN dondurmek kasitli: `form_degerlendir` NaN belirsizligi "bilinmiyor"
    sayip BELIRSIZ verir. Sifir dondurulseydi hesaplanamayan
    eklem **kusursuz olcum** gibi gorunurdu.
    """
    return np.array([
        np.sqrt(np.trace(kov) / 3.0) if kov is not None else np.nan
        for kov in eklem_kovaryanslari(kalib, iskelet, gozlemler, sigma_px)
    ], dtype=float)


def belirsizlik_ozeti(sigmalar: np.ndarray, iskelet: Iskelet3B) -> dict:
    """Rapora giden ozet: kac eklem kestirilebildi, dagilim nasil."""
    gecerli = np.isfinite(sigmalar)
    mm = sigmalar[gecerli] * 1000.0
    return {
        "n_eklem": int(len(sigmalar)),
        "n_gorunur": int(iskelet.gorunur.sum()),
        "n_belirsizlik_kestirildi": int(gecerli.sum()),
        "medyan_mm": None if not gecerli.any() else round(float(np.median(mm)), 3),
        "en_buyuk_mm": None if not gecerli.any() else round(float(mm.max()), 3),
        "en_kotu_eklem": (
            None if not gecerli.any()
            else iskelet.tanim.eklemler[int(np.nanargmax(sigmalar))]
        ),
    }
