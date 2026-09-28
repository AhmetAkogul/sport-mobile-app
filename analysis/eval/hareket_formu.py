"""Hareket basina dogru/yanlis olculeri: kol abduksiyonu, kol VW, sinav, bacak abduksiyonu.

Squat (0014, sagital olculer) ve lunge'taki (diz ayak ucunu geciyor) yontemin
devami (0038): her hareket icin literaturdeki yaygin hatalari karsilayan,
yorumlanabilir aday olculer; once mocap'ta (yer gercegi) fizyoterapist
etiketine karsi sinanir, sonra ayni fonksiyon telefon iskeletiyle olculur.

Olculer referans iskelet (13 eklem, `pose3d.iskelet.REFERANS_ISKELET`) uzerinde
tanimlidir; boylece mocap (REHAB24 26 eklemden cevrilen) ve MediaPipe dunya
noktalari ayni koddan gecer. `yukari` yercekimine ters birim vektordur
(mocap +y, MediaPipe dunya -y).

Her olcu **"buyukse yanlis"** yonunde tanimlidir (hipotez onceden sabit:
AUC sonradan yon secilerek sisirilmez). Tekrar skoru, 0,2 s kesintisiz
tutulan en buyuk degerdir (`surekli_en_buyuk`).
"""

from __future__ import annotations

import warnings

import numpy as np

from pose3d.iskelet import REFERANS_ISKELET

_I = REFERANS_ISKELET.indeks
BOYUN = _I("boyun")


def _j(ad: str, taraf: str | None = None) -> int:
    return _I(f"{taraf}_{ad}" if taraf else ad)


def _birim(v):
    with np.errstate(all="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _aci(u, v):
    with np.errstate(all="ignore"):
        c = np.sum(_birim(u) * _birim(v), axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _eksenler(X: np.ndarray, yukari: np.ndarray):
    """Kare basina (yukari u, kisinin solu l, one f) birim eksenleri.

    Sag el koordinatinda (mocap y-yukari, MediaPipe dunya): kuzeye bakan,
    basi yukarda kisinin solu batidir ve bati x yukari = kuzey, yani one = l x u.
    (Bir ara ters cevrildi; squat'ta "diz onde" AUC'si 0,89 yerine 0,34 cikinca
    fark edildi, 0038.)
    """
    u = np.broadcast_to(_birim(np.asarray(yukari, float)), X[:, 0].shape)
    kalca = X[:, _j("kalca", "sol")] - X[:, _j("kalca", "sag")]
    l_ = _birim(kalca - np.sum(kalca * u, -1, keepdims=True) * u)
    f = np.cross(l_, u)
    return u, l_, f


def _pelvis(X):
    return X[:, [_j("kalca", "sol"), _j("kalca", "sag")]].mean(axis=1)


def kol_acisi(X, yukari, taraf):
    """Ust kol ile asagi yon arasindaki aci (0 kol yanda asagi, 90 yatay)."""
    return _aci(X[:, _j("dirsek", taraf)] - X[:, _j("omuz", taraf)], -np.asarray(yukari, float))


def dirsek_bukulme(X, taraf):
    """180 - dirsek acisi (0 duz kol)."""
    return 180 - _aci(X[:, _j("omuz", taraf)] - X[:, _j("dirsek", taraf)],
                      X[:, _j("bilek", taraf)] - X[:, _j("dirsek", taraf)])


def diz_bukulme(X, taraf):
    return 180 - _aci(X[:, _j("kalca", taraf)] - X[:, _j("diz", taraf)],
                      X[:, _j("ayak_bilegi", taraf)] - X[:, _j("diz", taraf)])


def bacak_acisi(X, yukari, taraf):
    """Uyluk ile asagi yon arasindaki aci (0 ayakta)."""
    return _aci(X[:, _j("diz", taraf)] - X[:, _j("kalca", taraf)], -np.asarray(yukari, float))


def govde_yana(X, yukari):
    """Govdenin frontal duzlemde yana egimi (derece, mutlak)."""
    u, l_, _ = _eksenler(X, yukari)
    g = X[:, BOYUN] - _pelvis(X)
    return np.abs(np.degrees(np.arctan2(np.sum(g * l_, -1), np.sum(g * u, -1))))


def govde_egimi(X, yukari):
    """Govdenin dikeyle acisi (derece)."""
    return _aci(X[:, BOYUN] - _pelvis(X), np.asarray(yukari, float))


def duzlem_disi(X, yukari, bas: int, son: int, esik_aci: np.ndarray):
    """Segmentin (bas->son) frontal duzlemden one/arkaya sapmasi (derece);
    segment `esik_aci` > 30 derece kalkmadiysa NaN (asagidaki kolun yonu anlamsiz)."""
    _, _, f = _eksenler(X, yukari)
    s = _birim(X[:, son] - X[:, bas])
    with np.errstate(all="ignore"):
        d = np.degrees(np.arcsin(np.clip(np.abs(np.sum(s * f, -1)), 0, 1)))
    return np.where(esik_aci > 30, d, np.nan)


def omuz_kalkma(X, yukari, taraf):
    """Omzun pelvise gore yuksekliginin tekrar basina gore artisi / govde boyu."""
    u = np.asarray(yukari, float)
    h = np.sum((X[:, _j("omuz", taraf)] - _pelvis(X)) * u, -1)
    govde = np.nanmedian(np.linalg.norm(X[:, BOYUN] - _pelvis(X), axis=-1))
    return (h - np.nanmedian(h[: max(len(h) // 10, 1)])) / govde


def kalca_cizgiden(X, yukari):
    """Sinav: pelvisin omuz ortasi -> ayak bilegi ortasi cizgisinden yukari sapmasi
    / vucut boyu. Pozitif: kalca yukarda (pike), negatif: sarkma."""
    u = np.asarray(yukari, float)
    a = X[:, BOYUN]
    b = X[:, [_j("ayak_bilegi", "sol"), _j("ayak_bilegi", "sag")]].mean(axis=1)
    p = _pelvis(X)
    boy = np.linalg.norm(b - a, axis=-1)
    with np.errstate(all="ignore"):
        t = np.sum((p - a) * (b - a), -1) / boy ** 2
    yakin = a + t[:, None] * (b - a)
    return np.sum((p - yakin) * u, -1) / boy


def dirsek_acilmasi(X, taraf):
    """Sinav: ust kol ile govde ekseni (boyun -> pelvis) arasindaki aci."""
    return _aci(X[:, _j("dirsek", taraf)] - X[:, _j("omuz", taraf)], _pelvis(X) - X[:, BOYUN])


def pelvis_egimi(X, yukari):
    """Kalca hattinin yatayla acisi (derece, mutlak): kalca kaldirma telafisi."""
    h = _birim(X[:, _j("kalca", "sol")] - X[:, _j("kalca", "sag")])
    u = _birim(np.asarray(yukari, float))
    with np.errstate(all="ignore"):
        return np.abs(np.degrees(np.arcsin(np.clip(np.sum(h * u, -1), -1, 1))))


def diz_onde(X, yukari, taraf):
    """Dizin ayak bileginin onune gecmesi (one eksende) / kaval boyu."""
    _, _, f = _eksenler(X, yukari)
    d = X[:, _j("diz", taraf)] - X[:, _j("ayak_bilegi", taraf)]
    kaval = np.nanmedian(np.linalg.norm(d, axis=-1))
    return np.sum(d * f, -1) / kaval


def diz_ice(X, yukari, taraf):
    """Frontal duzlemde dizin kalca-ayak bilegi cizgisinden ice (orta hatta) sapmasi / uyluk."""
    _, l_, _ = _eksenler(X, yukari)
    k, d, a = X[:, _j("kalca", taraf)], X[:, _j("diz", taraf)], X[:, _j("ayak_bilegi", taraf)]
    orta = (k + a) / 2
    ice = l_ if taraf == "sag" else -l_          # sag bacakta ice = kisinin solu
    uyluk = np.nanmedian(np.linalg.norm(d - k, axis=-1))
    return np.sum((d - orta) * ice, -1) / uyluk


def surekli_en_buyuk(v: np.ndarray, n: int) -> float:
    """`n` kare kesintisiz tutulan en buyuk deger (tek karelik sivri hatayi yok sayar)."""
    v = np.asarray(v, float)
    if len(v) < n or not np.isfinite(v).any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w = np.lib.stride_tricks.sliding_window_view(v, n)
        m = np.where(np.isfinite(w).all(1), w.min(1), np.nan)
    return float(np.nanmax(m)) if np.isfinite(m).any() else float("nan")


# `ayak` dizisinin sirasi: sag topuk, sag ayak ucu, sol topuk, sol ayak ucu
# (MediaPipe 30, 32, 29, 31 ile ayni; mocap'ta topuk yok -> NaN).
AYAK_UCU = {"sag": 1, "sol": 3}


def diz_parmak_onde(X, ayak, yukari, taraf):
    """Dizin ayak ucunun onune gecmesi, **ayak yonunde** (bilek -> ayak ucu, yatay)
    / kaval boyu. Lunge'ta dogrulanmis tanim (`docs/deney/2026-09-28-ec3d-lunge.md`);
    ayak yonu kullanildigi icin koordinat el kuralindan bagimsizdir."""
    u = _birim(np.asarray(yukari, float))
    uc = ayak[:, AYAK_UCU[taraf]]
    yon = uc - X[:, _j("ayak_bilegi", taraf)]
    yon = _birim(yon - np.sum(yon * u, -1, keepdims=True) * u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = np.nanmedian(yon, axis=0)          # tekrar boyunca ayak yonu sabit kabul
    f = _birim(f)
    d = X[:, _j("diz", taraf)] - uc
    kaval = np.nanmedian(np.linalg.norm(X[:, _j("diz", taraf)] - X[:, _j("ayak_bilegi", taraf)],
                                        axis=-1))
    return np.sum(d * f, -1) / kaval


def diz_parmak_onde_2b(P2: np.ndarray, taraf: str) -> np.ndarray:
    """`diz_parmak_onde`'nin goruntu duzlemi karsiligi: (T, 65, 2) TAM_VUCUT piksel.

    Kamera yatay kabul edilir; ayak yonu = tekrar boyunca (bas parmak - ayak
    bilegi) x bileseninin isareti. Dizin bas parmagi o yonde gecmesi / 2B kaval.
    Yandan telefon lunge'ta fizyoterapisti mocap kadar ayiriyordu (AUC 0,85,
    `docs/deney/2026-09-28-ec3d-lunge.md`); onden ayak yonu belirsiz -> anlamsiz.
    """
    from pose3d.tam_vucut import TAM_VUCUT
    k = P2[:, TAM_VUCUT.indeks(f"{taraf}_diz")]
    a = P2[:, TAM_VUCUT.indeks(f"{taraf}_ayak_bilegi")]
    uc = P2[:, TAM_VUCUT.indeks(f"{taraf}_bas_parmak")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yon = np.sign(np.nanmedian(uc[:, 0] - a[:, 0]))
        kaval = np.nanmedian(np.linalg.norm(k - a, axis=-1))
    if not (np.isfinite(yon) and yon != 0 and np.isfinite(kaval) and kaval > 0):
        return np.full(len(P2), np.nan)
    return (k[:, 0] - uc[:, 0]) * yon / kaval


def ondeki_bacak_2b(P2: np.ndarray) -> str | None:
    """Lunge'ta ondeki bacak: ayak yonunde bilegi daha onde olan (goruntu x'i).

    Dizi daha cok bukulen bacak degil: lunge'ta arka diz de ~90 derece bukulur.
    """
    from pose3d.tam_vucut import TAM_VUCUT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yon = np.sign(np.nanmedian(np.concatenate([
            P2[:, TAM_VUCUT.indeks(f"{t}_bas_parmak"), 0]
            - P2[:, TAM_VUCUT.indeks(f"{t}_ayak_bilegi"), 0] for t in ("sag", "sol")])))
        if not np.isfinite(yon) or yon == 0:
            return None
        ileri = {t: np.nanmedian(P2[:, TAM_VUCUT.indeks(f"{t}_ayak_bilegi"), 0]) * yon
                 for t in ("sag", "sol")}
    if not all(np.isfinite(v) for v in ileri.values()):
        return None
    return max(ileri, key=ileri.get)


def tekrar_olculeri(X: np.ndarray, yukari, egzersiz: int, taraf: str | None,
                    kare_hizi: float, ayak: np.ndarray | None = None,
                    P2: np.ndarray | None = None) -> dict[str, float]:
    """(T, 13, 3) tekrar -> {olcu: skor}; her skor "buyukse yanlis" yonunde.

    `egzersiz`: REHAB24 1 kol abduksiyonu, 2 kol VW, 3 sinav, 4 bacak abduksiyonu,
    5 lunge, 6 squat. `taraf`: aktif kol/bacak ("sag" | "sol"); VW, sinav ve
    squat'ta kullanilmaz, lunge'ta verilmezse dizi daha cok bukulen bacak.
    `P2`: istege bagli (T, 65, 2) goruntu noktalari (TAM_VUCUT); lunge/squat'ta
    `diz_parmak_onde_2b` icin. Yoksa o olcu NaN.
    """
    n = max(int(round(0.2 * kare_hizi)), 1)

    def s(v):
        return surekli_en_buyuk(v, n)
    u = np.asarray(yukari, float)
    if taraf is None and egzersiz in (1, 4):
        taraf = aktif_taraf(X, u, egzersiz)
    if egzersiz == 1:
        a = kol_acisi(X, u, taraf)
        return {"kol_yetersiz": -s(a), "kol_asiri": s(a),
                "dirsek_bukulme": s(dirsek_bukulme(X, taraf)),
                "govde_yana": s(govde_yana(X, u)), "govde_egimi": s(govde_egimi(X, u)),
                "kol_one": s(duzlem_disi(X, u, _j("omuz", taraf), _j("dirsek", taraf), a)),
                "omuz_kalkma": s(omuz_kalkma(X, u, taraf))}
    if egzersiz == 2:
        sag, sol = kol_acisi(X, u, "sag"), kol_acisi(X, u, "sol")
        return {"kol_yetersiz": -s(np.fmin(sag, sol)), "kol_asiri": s(np.fmax(sag, sol)),
                "asimetri": s(np.abs(sag - sol)),
                "dirsek_bukulme": s(np.fmax(dirsek_bukulme(X, "sag"), dirsek_bukulme(X, "sol"))),
                "govde_egimi": s(govde_egimi(X, u)), "govde_yana": s(govde_yana(X, u)),
                "kol_one": s(np.fmax(duzlem_disi(X, u, _j("omuz", "sag"), _j("dirsek", "sag"), sag),
                                     duzlem_disi(X, u, _j("omuz", "sol"), _j("dirsek", "sol"), sol))),
                "omuz_kalkma": s(np.fmax(omuz_kalkma(X, u, "sag"), omuz_kalkma(X, u, "sol")))}
    if egzersiz == 3:
        k = kalca_cizgiden(X, u)
        derin = np.fmax(dirsek_bukulme(X, "sag"), dirsek_bukulme(X, "sol"))
        return {"kalca_sarkma": s(-k), "kalca_pike": s(k), "sig": -s(derin),
                "dirsek_acilma": s(np.fmax(dirsek_acilmasi(X, "sag"), dirsek_acilmasi(X, "sol")))}
    if egzersiz == 4:
        diger = "sol" if taraf == "sag" else "sag"
        a = bacak_acisi(X, u, taraf)
        return {"bacak_yetersiz": -s(a), "bacak_asiri": s(a), "govde_yana": s(govde_yana(X, u)),
                "govde_egimi": s(govde_egimi(X, u)),
                "bacak_one": s(duzlem_disi(X, u, _j("kalca", taraf), _j("diz", taraf), a)),
                "diz_bukulme": s(diz_bukulme(X, taraf)),
                "destek_diz": s(diz_bukulme(X, diger)), "pelvis_egimi": s(pelvis_egimi(X, u))}
    if egzersiz in (5, 6):
        # squat (6): iki bacak; lunge (5): ondeki bacak = dizi daha cok bukulen
        if egzersiz == 5:
            taraf = taraf or max(("sag", "sol"), key=lambda t: np.nanmax(diz_bukulme(X, t)))
            bacaklar = (taraf,)
        else:
            bacaklar = ("sag", "sol")
        with warnings.catch_warnings():         # tum-NaN sutun (ayak yok) beklenen durum
            warnings.simplefilter("ignore", RuntimeWarning)
            bukulme = np.nanmax([diz_bukulme(X, t) for t in bacaklar], axis=0)
            parmak = (np.full(len(X), np.nan) if ayak is None else
                      np.nanmax([diz_parmak_onde(X, ayak, u, t) for t in bacaklar], axis=0))
            onde = np.nanmax([diz_onde(X, u, t) for t in bacaklar], axis=0)
            ice = np.nanmax([diz_ice(X, u, t) for t in bacaklar], axis=0)
            parmak2 = (np.full(len(X), np.nan) if P2 is None else
                       np.nanmax([diz_parmak_onde_2b(P2, t) for t in
                                  ((ondeki_bacak_2b(P2) or taraf,) if egzersiz == 5
                                   else bacaklar)], axis=0))
        return {"sig": -s(bukulme), "derin": s(bukulme), "govde_egimi": s(govde_egimi(X, u)),
                "diz_onde": s(onde), "diz_parmak_onde": s(parmak), "diz_ice": s(ice),
                "diz_parmak_onde_2b": s(parmak2)}
    raise ValueError(f"desteklenmeyen egzersiz: {egzersiz}")


def auc(pozitif, negatif) -> float | None:
    """P(pozitif > negatif) + 0,5 P(esit); NaN'lar atilir."""
    p = np.asarray([x for x in pozitif if np.isfinite(x)])
    q = np.asarray([x for x in negatif if np.isfinite(x)])
    if not len(p) or not len(q):
        return None
    return float((np.sum(p[:, None] > q[None]) + 0.5 * np.sum(p[:, None] == q[None]))
                 / (len(p) * len(q)))


def bakis_acisi(P: np.ndarray) -> float:
    """Kamera cercevesinde govdenin bakis acisi, derece: 0 onden, 90 profil.

    Sol-sag kalca ve omuz vektorlerinin ortalamasinin X (sag) - Z (derinlik)
    duzlemindeki yonu. Kameraya donuk ile sirti donuk ayrilmaz (ikisi de 0).
    `P`: (13, 3) REFERANS sirasinda, kamera eksenleriyle hizali (X sag, Z ileri;
    MediaPipe dunya noktalari boyledir).
    """
    v = [P[_j("kalca", "sol")] - P[_j("kalca", "sag")], P[_j("omuz", "sol")] - P[_j("omuz", "sag")]]
    v = [x for x in v if np.isfinite(x).all()]
    if not v:
        return float("nan")
    x, z = np.mean(v, axis=0)[[0, 2]]
    if x == 0 and z == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(z), abs(x))))


def aci_grubu(aci: float) -> str:
    """0-30 onden, 30-60 yarim profil, 60-90 profil (REHAB24 kamera yonleriyle ayni ad)."""
    if not np.isfinite(aci):
        return "bilinmiyor"
    return "front" if aci < 30 else ("half-profile" if aci < 60 else "profile")


def aktif_taraf(X: np.ndarray, yukari, egzersiz: int) -> str:
    """Tek tarafli harekette aktif kol/bacak: aci araligi buyuk olan taraf.

    Etikete (ya da modelin sol/sag adina) guvenilmez: arkadan goren kamerada
    MediaPipe sol/sagi karistirabilir; REHAB24 c18'de etiketle secilen kolda kol
    abduksiyonu AUC'si 0,25'e (ters) dustu (0038).
    """
    f = kol_acisi if egzersiz == 1 else bacak_acisi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        aralik = {t: np.nanpercentile(f(X, yukari, t), 95) - np.nanpercentile(f(X, yukari, t), 5)
                  for t in ("sag", "sol")}
    return max(aralik, key=lambda t: aralik[t] if np.isfinite(aralik[t]) else -1)


def ana_sinyal(X: np.ndarray, yukari, egzersiz: int) -> np.ndarray:
    """Tekrar sayimi icin hareketin ana sinyali (derece, kare basina).

    1 kol abduksiyonu: kollarin en buyuk kaldirma acisi; 2 VW: iki kolun
    ortalamasi; 3 sinav: dirsek bukulmesi; 4 bacak abduksiyonu: uyluk acisi;
    5 lunge, 6 squat: diz bukulmesi (en cok bukulen bacak).
    """
    u = np.asarray(yukari, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if egzersiz == 1:
            return np.fmax(kol_acisi(X, u, "sag"), kol_acisi(X, u, "sol"))
        if egzersiz == 2:
            return (kol_acisi(X, u, "sag") + kol_acisi(X, u, "sol")) / 2
        if egzersiz == 3:
            return np.fmax(dirsek_bukulme(X, "sag"), dirsek_bukulme(X, "sol"))
        if egzersiz == 4:
            return np.fmax(bacak_acisi(X, u, "sag"), bacak_acisi(X, u, "sol"))
        if egzersiz in (5, 6):
            return np.fmax(diz_bukulme(X, "sag"), diz_bukulme(X, "sol"))
    raise ValueError(f"desteklenmeyen egzersiz: {egzersiz}")


class TekrarSayaci:
    """Nedensel tepe sayaci: canlida gelecegi gormeden tekrar bulur (0038).

    Bir tepe, sinyal tepeden `belirginlik` kadar dustugunde onaylanir; tepe
    `yukseklik`i gecmeli ve onceki tepeden en az `en_az_aralik_s` sonra olmali.
    Tekrar araligi: tepeden onceki en dusuk noktadan onay anina kadar.
    Ilk denemedeki histerezisli sayac REHAB24'te kol VW'yi cift sayiyordu
    (video basina ort. 5 tekrar hata); tepe tabanli sayac 2'ye indirdi.

    `ekle(t, deger)` biten tekrari (bas_s, son_s) dondurur, yoksa None.
    """

    def __init__(self, yukseklik: float, belirginlik: float, en_az_aralik_s: float):
        if belirginlik <= 0:
            raise ValueError("belirginlik pozitif olmali")
        self.yukseklik, self.belirginlik, self.en_az_aralik_s = yukseklik, belirginlik, en_az_aralik_s
        self._dip = (None, np.inf)            # (zaman, deger): son onaydan beri en dusuk
        self._tepe = (None, -np.inf)          # dipten sonraki en yuksek
        self._son_tepe_t = -np.inf

    def ekle(self, t: float, deger: float) -> tuple[float, float] | None:
        if not np.isfinite(deger):
            return None
        if self._tepe[0] is None:
            if deger < self._dip[1]:
                self._dip = (t, deger)
            if deger - self._dip[1] >= self.belirginlik:
                self._tepe = (t, deger)
            return None
        if deger > self._tepe[1]:
            self._tepe = (t, deger)
            return None
        if self._tepe[1] - deger >= self.belirginlik:
            (tt, tv), bas = self._tepe, self._dip[0]
            self._tepe, self._dip = (None, -np.inf), (t, deger)
            if tv >= self.yukseklik and tt - self._son_tepe_t >= self.en_az_aralik_s:
                self._son_tepe_t = tt
                return (bas if bas is not None else tt), t
        return None


def esikler_ogren(dinlenme: np.ndarray, tepeler: np.ndarray,
                  sureler_s: np.ndarray) -> tuple[float, float, float]:
    """Sayac parametreleri: (yukseklik, belirginlik, en az aralik s).

    Dinlenme medyani r, tepe medyani p: yukseklik r + 0,4 (p - r), belirginlik
    0,3 (p - r), aralik tipik tekrar suresinin yarisi (REHAB24'te kisi-disarida
    taramayla secildi; 0038).
    """
    r, p = float(np.nanmedian(dinlenme)), float(np.nanmedian(tepeler))
    return r + 0.4 * (p - r), 0.3 * (p - r), 0.5 * float(np.median(sureler_s))


# --- Tekrar karari ve sayac egitimi (0038; eskiden scripts/deney_hareket_formu.py) ---

# Bir (hareket, aci) grubu, kisi-disarida AUC bu degerin altindaysa "olculemez".
OLCULEBILIR_AUC = 0.70


def esik_kurali(egitim_x, egitim_y):
    """Youden J'yi en buyuten esik (x > esik -> yanlis)."""
    x, y = np.asarray(egitim_x, float), np.asarray(egitim_y, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if not len(x) or y.all() or not y.any():
        return np.inf
    aday = np.unique(x)
    j = [np.mean(x[y] > t) - np.mean(x[~y] > t) for t in aday]
    return float(aday[int(np.argmax(j))])

def kisi_disarida_dogruluk(satirlar, olcu):
    """Her kisi icin esik diger kisilerden; dogruluk, duyarlilik, ozgulluk."""
    tahmin, gercek = [], []
    for k in sorted({s["kisi"] for s in satirlar}):
        eg = [s for s in satirlar if s["kisi"] != k]
        te = [s for s in satirlar if s["kisi"] == k and np.isfinite(s[olcu])]
        t = esik_kurali([s[olcu] for s in eg], [s["yanlis"] for s in eg])
        tahmin += [s[olcu] > t for s in te]
        gercek += [s["yanlis"] for s in te]
    tahmin, gercek = np.array(tahmin, bool), np.array(gercek, bool)
    if not len(gercek):
        return None
    return {"dogruluk": round(float(np.mean(tahmin == gercek)), 3), "n": int(len(gercek)),
            "duyarlilik": round(float(np.mean(tahmin[gercek])), 3) if gercek.any() else None,
            "ozgulluk": round(float(np.mean(~tahmin[~gercek])), 3) if (~gercek).any() else None}

def birlesik_model():
    """Butun olculer: eksik -> medyan, olcekleme, dengeli lojistik regresyon."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced",
                                                              max_iter=1000))

def birlesik_kisi_disarida(satirlar, olculer):
    """Butun olculeri birlestiren model, kisi-disarida: AUC ve 0,5 esikte dogruluk."""
    X = np.array([[s[o] for o in olculer] for s in satirlar], float)
    y = np.array([s["yanlis"] for s in satirlar], bool)
    kisi = np.array([s["kisi"] for s in satirlar])
    p = np.full(len(y), np.nan)
    for k in sorted(set(kisi)):
        te = kisi == k
        if y[~te].all() or not y[~te].any():
            continue
        p[te] = birlesik_model().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    ok = np.isfinite(p)
    if not ok.any():
        return None
    a = auc(p[ok & y], p[ok & ~y])
    t = p[ok] > 0.5
    return {"auc": None if a is None else round(a, 3),
            "dogruluk": round(float(np.mean(t == y[ok])), 3), "n": int(ok.sum()),
            "duyarlilik": round(float(np.mean(t[y[ok]])), 3),
            "ozgulluk": round(float(np.mean(~t[~y[ok]])), 3)}

def nedensel_medyan(x, n: int = 3):
    """Nedensel medyan (son n kare): canlida da ayni sekilde uygulanir."""
    out = np.array(x, float)
    for i in range(len(x)):
        w = x[max(0, i - n + 1):i + 1]
        w = w[np.isfinite(w)]
        out[i] = np.median(w) if len(w) else np.nan
    return out

def sayac_esikleri(veri, kare_hizi: float = 30.0):
    """Tekrar sayaci esikleri, etiketli videolardan: her oge {"kare", "gercek": [(ilk, son)],
    "sinyal"}; gercek tekrar disi dinlenme, tekrar ici tepe ve sure `esikler_ogren`'e."""
    dinlenme, tepe, sure = [], [], []
    for v in veri:
        ic = np.zeros(len(v["kare"]), bool)
        for a, b in v["gercek"]:
            sec = (v["kare"] >= a) & (v["kare"] <= b)
            ic |= sec
            sure.append((b - a) / kare_hizi)
            if sec.any() and np.isfinite(v["sinyal"][sec]).any():
                tepe.append(np.nanmax(v["sinyal"][sec]))
        dinlenme += v["sinyal"][~ic].tolist()
    return esikler_ogren(np.array(dinlenme), np.array(tepe), np.array(sure))


def risk_kapsama(p, y, bantlar=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)) -> list[dict]:
    """Secici karar: |p - 0,5| < bant ise karar yok. Bant basina kapsama ve dogruluk.

    `p`: "yanlis" olasiligi, `y`: gercek "yanlis" etiketi. Kapsama, karar
    verilen oran; dogruluk yalniz karar verilenlerde. Copilot incelemesi (0040).
    """
    p, y = np.asarray(p, float), np.asarray(y, bool)
    out = []
    for b in bantlar:
        karar = np.abs(p - 0.5) >= b
        n = int(karar.sum())
        out.append({"bant": b, "kapsama": round(float(karar.mean()), 3) if len(p) else None,
                    "dogruluk": round(float(np.mean((p[karar] > 0.5) == y[karar])), 3)
                    if n else None, "n": n})
    return out
