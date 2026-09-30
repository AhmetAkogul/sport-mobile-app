"""Hareket tanima: kisi ne yapiyor? (tam vucut 2B iskelet dizisinden; 0037)

Iskeletten hareket tanima literaturde cozulmus bir alandir (ST-GCN, PoseC3D).
Burada yeni bir mimari onerilmez; elimizdeki kucuk etiketli veride (REHAB24-6,
6 hareket) kisi-disarida dogrulanabilen yalin bir taban kurulur:

1. Kare ozellikleri (`kare_ozellikleri`): kalca merkezine tasinmis, govde
   boyuyla olceklenmis 2B eklem konumlari ve 2B eklem acilari (dirsek, omuz,
   kalca, diz, govde egimi). Olcek ve konumdan bagimsiz; bakis acisina bagli.
2. Pencere ozellikleri (`pencere_ozellikleri`): 2 s penceredeki her kare
   ozelliginin ortalama, standart sapma, %10 ve %90'lik degeri.
3. Siniflandirici: scikit-learn `HistGradientBoostingClassifier`; egitimde
   sol/sag aynalama ile cogaltma (hareketin hangi tarafla yapildigi etiketi
   degistirmez).

Sinif 0 "yok": tekrar disindaki kareler (bekleme, gecis). Canli kullanim
`HareketTanima` ile: kisinin son 2 s'lik gecmisinden etiket ve olasilik.
"""

from __future__ import annotations

import warnings

import numpy as np

from pose3d.tam_vucut import TAM_VUCUT

HAREKETLER = {0: "yok", 1: "kol_abduksiyonu", 2: "kol_vw", 3: "sinav",
              4: "bacak_abduksiyonu", 5: "lunge", 6: "squat"}

_I = TAM_VUCUT.indeks
# Konum ozelligi alinan noktalar: bas (burun), govde 12 eklem, ayak (topuk, bas parmak).
KONUM = [_I(a) for a in ("burun", "sol_omuz", "sag_omuz", "sol_dirsek", "sag_dirsek",
                         "sol_bilek", "sag_bilek", "sol_kalca", "sag_kalca", "sol_diz",
                         "sag_diz", "sol_ayak_bilegi", "sag_ayak_bilegi", "sol_topuk",
                         "sag_topuk", "sol_bas_parmak", "sag_bas_parmak")]
# (a, b, c): b'deki aci
ACILAR = [
    ("sol_omuz", "sol_dirsek", "sol_bilek"), ("sag_omuz", "sag_dirsek", "sag_bilek"),
    ("sol_kalca", "sol_omuz", "sol_dirsek"), ("sag_kalca", "sag_omuz", "sag_dirsek"),
    ("sol_omuz", "sol_kalca", "sol_diz"), ("sag_omuz", "sag_kalca", "sag_diz"),
    ("sol_kalca", "sol_diz", "sol_ayak_bilegi"), ("sag_kalca", "sag_diz", "sag_ayak_bilegi"),
]
_ACI_IX = np.array([[_I(a), _I(b), _I(c)] for a, b, c in ACILAR])

# Sol <-> sag degisimi (aynalama cogaltmasi icin)
_TAKAS = np.arange(len(TAM_VUCUT))
for _ad in TAM_VUCUT.eklemler:
    if _ad.startswith("sol_"):
        _TAKAS[_I(_ad)] = _I("sag_" + _ad[4:])
        _TAKAS[_I("sag_" + _ad[4:])] = _I(_ad)

PENCERE_S = 2.0


def aynala(P: np.ndarray, G: np.ndarray, genislik: float) -> tuple[np.ndarray, np.ndarray]:
    """(T, 65, 2) diziyi yatay aynala ve sol/sag adlarini degistir."""
    Q = P[:, _TAKAS].copy()
    Q[..., 0] = genislik - 1 - Q[..., 0]
    return Q, G[:, _TAKAS]


def _aci(a, b, c):
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    with np.errstate(all="ignore"):
        cos = np.sum(u * v, axis=-1) / (nu * nv)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def kare_ozellikleri(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """(T, 65, 2) piksel + (T, 65) gorunurluk -> (T, F); eksik NaN.

    Olcek: pencerenin medyan govde boyu (omuz ortasi - kalca ortasi); boylece
    squat'ta govde egilince olcek bozulmaz.
    """
    P = np.where(G[..., None], P, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        kalca = np.nanmean(P[:, [_I("sol_kalca"), _I("sag_kalca")]], axis=1)
        omuz = np.nanmean(P[:, [_I("sol_omuz"), _I("sag_omuz")]], axis=1)
        govde = np.linalg.norm(omuz - kalca, axis=1)
        olcek = np.nanmedian(govde) if np.isfinite(govde).any() else np.nan
        konum = (P[:, KONUM] - kalca[:, None]) / olcek                    # (T, K, 2)
        acilar = _aci(P[:, _ACI_IX[:, 0]], P[:, _ACI_IX[:, 1]], P[:, _ACI_IX[:, 2]])
        d = omuz - kalca
        # govde egimi: 0 dik (bas yukarda), 90 yatay, 180 bas asagida
        egim = np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))
        oran = govde / olcek
    # Not (0037): acik iki-taraf asimetri ozellikleri (diz/kalca/dirsek farki, ayaklar
    # arasi mesafe) denendi; REHAB24'te fark yok, EC3D'de lunge %79,5 -> %52,2 kotulesti.
    return np.concatenate([konum.reshape(len(P), -1), acilar, egim[:, None], oran[:, None]], 1)


def pencere_ozellikleri(F: np.ndarray) -> np.ndarray:
    """(T, F) -> (4F + 1,): ortalama, std, %10, %90 ve eksik oran."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        ort = np.nanmean(F, 0)
        std = np.nanstd(F, 0)
        p10, p90 = np.nanpercentile(F, [10, 90], axis=0)
    eksik = np.array([np.mean(~np.isfinite(F))])
    return np.concatenate([ort, std, p10, p90, eksik])


def pencereler(n_kare: int, kare_hizi: float, pencere_s: float = PENCERE_S,
               kaydir_s: float = 0.5) -> list[tuple[int, int]]:
    """(bas, son) kare dilimleri; hepsi tam pencere."""
    n = max(int(round(pencere_s * kare_hizi)), 2)
    k = max(int(round(kaydir_s * kare_hizi)), 1)
    return [(i, i + n) for i in range(0, n_kare - n + 1, k)]


def siniflandirici(tohum: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          l2_regularization=1.0, random_state=tohum)


class HareketTanima:
    """Canli: kisinin son `pencere_s` saniyelik iskeletinden hareket etiketi.

    Olasiliklar ustel ortalamayla yumusatilir (`yumusatma`); en olasi sinifin
    olasiligi `esik`in altindaysa "belirsiz" doner. Karar degil tahmin: kullanan
    taraf olasiligi gostermeli.
    """

    def __init__(self, model, *, pencere_s: float = PENCERE_S, yumusatma: float = 0.5,
                 esik: float = 0.5):
        self.model = model
        self.pencere_s = pencere_s
        self.yumusatma = yumusatma
        self.esik = esik
        self._p: dict[int, np.ndarray] = {}

    def tahmin(self, kimlik: int, zaman: np.ndarray, P: np.ndarray,
               G: np.ndarray) -> tuple[str, float] | None:
        if len(zaman) < 2 or zaman[-1] - zaman[0] < self.pencere_s * 0.9:
            return None
        sec = zaman >= zaman[-1] - self.pencere_s
        x = pencere_ozellikleri(kare_ozellikleri(P[sec], G[sec]))[None]
        p = self.model.predict_proba(x)[0]
        onceki = self._p.get(kimlik)
        p = p if onceki is None else self.yumusatma * onceki + (1 - self.yumusatma) * p
        self._p[kimlik] = p
        i = int(np.argmax(p))
        sinif = int(self.model.classes_[i])
        if p[i] < self.esik:
            return "belirsiz", float(p[i])
        return HAREKETLER.get(sinif, str(sinif)), float(p[i])

    def unut(self, kimlik: int) -> None:
        self._p.pop(kimlik, None)
