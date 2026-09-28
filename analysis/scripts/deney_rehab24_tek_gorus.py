"""REHAB24-6: gercek harekette tek gorus form karari, bakis yonune gore.

Tezin manset sorusunun (form karari dogrulugu vs kamera acisi) ilk gercek
hareket verisiyle olcumu. Ayni squat tekrari iki senkron kameradan gorulur;
kamera 17'ye gore "front" olan tekrar kamera 18'den "profile"dir.

**Katman 1 -- kahin 2B (bu betik).** Telefonun 2B'si, mocap iskeletinin o
kameraya izdusumudur (veri setinin kendi 2B'si): dedektor hatasi yok. Tek
gorus 3B `mono.phone.tek_gorus_3b` ile kurulur; kemik uzunlugu onculeri kisi
basina mocap medyanidir ("anatomi bir kez olculmus"). Olculen hata yalnizca
**tek gorus geometrisinin** hatasidir; gercek dedektorle (Katman 2, video)
hata bundan buyuk olacaktir -- telefon hatasinin alt siniri.

Referans karar, ayni kuralin mocap iskeletine uygulanmasidir: soru "telefon
referansla ayni karari veriyor mu" (0070 iddia 2).

Tekrar karari (referans ve telefon icin ayni, profil `eval.egzersiz.SQUAT`):
valgus (sag ya da sol) en az `kusur_suresi_s` kesintisiz esigi asarsa KUSURLU;
yoksa karelerin en az `min_kapsama`'sinda kesin karar varsa DOGRU; yoksa
BELIRSIZ.

    python scripts/deney_rehab24_tek_gorus.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.egzersiz import SQUAT, diz_fleksiyonu  # noqa: E402
from eval.form import VARSAYILAN_ESIKLER, Karar, form_degerlendir  # noqa: E402
from eval.protokol import Oge, Sayim, kisi_agirlikli, kisi_bootstrap  # noqa: E402
from mono.canli import govde_acisi  # noqa: E402,F401  (acinin tek tanimi)
from mono.phone import tek_gorus_3b  # noqa: E402
from mono.taylor import taylor_3b  # noqa: E402
from pose3d.hizalama import rijit_hizala  # noqa: E402
from pose3d.iskelet import REFERANS_ISKELET, Iskelet3B  # noqa: E402
from pose3d.pose2d import Poz2B  # noqa: E402
from veri.rehab24 import (  # noqa: E402
    EKLEMLER, _TURETILMIS, kamera_kestir, kareyi_cevir, tekrar_kareleri, tekrarlar_oku,
)

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data/dis/rehab24_6"
FPS = 30.0
KAMERALAR = {"c17": (1920, 1080), "c18": (1080, 1920)}
# Segmentation.txt: kamera 17 yonu -> kamera 18 yonu (kameralar dik).
C18_YONU = {"front": "profile", "half-profile": "half-profile", "profile": "front"}
VALGUS = ("diz_valgusu_sag", "diz_valgusu_sol")


def _referans_2b(x26: np.ndarray) -> np.ndarray:
    """(26, 2) -> REFERANS sirasinda (13, 2); boyun iki omuz ortalamasi."""
    P = np.full((len(REFERANS_ISKELET), 2), np.nan)
    for i, ad in enumerate(REFERANS_ISKELET.eklemler):
        idx = _TURETILMIS.get(ad, (EKLEMLER.get(ad),))
        P[i] = x26[list(idx)].mean(axis=0)
    return P


def poz_kur(x26: np.ndarray, boyut: tuple[int, int], kamera: int, kare: int) -> Poz2B:
    """Kahin 2B: kadraj disindaki eklem gorunmez (gercek dedektor de goremez)."""
    P = _referans_2b(x26)
    w, h = boyut
    icinde = (np.isfinite(P).all(axis=1) & (P[:, 0] >= 0) & (P[:, 0] < w)
              & (P[:, 1] >= 0) & (P[:, 1] < h))
    P[~icinde] = np.nan
    return Poz2B(iskelet=REFERANS_ISKELET, noktalar=P, guven=icinde.astype(float),
                 gorunur=icinde, tespit=bool(icinde.any()), goruntu_boyutu=boyut,
                 model="rehab24_kahin_2b", kamera=kamera, kare=kare)


def kisi_onculeri(tekrarlar, kareler_al) -> dict[str, dict]:
    """Kisi basina kemik uzunlugu medyani (metre), butun squat karelerinden."""
    uzunluk: dict = defaultdict(lambda: defaultdict(list))
    for t in tekrarlar:
        for q in kareler_al(t)[::5]:
            isk = kareyi_cevir(q)
            for a, b in REFERANS_ISKELET.baglantilar:
                u = isk.uzunluk(a, b)
                if np.isfinite(u) and u > 0:
                    uzunluk[t.kisi][(a, b)].append(u)
    return {k: {bag: float(np.median(v)) for bag, v in d.items()} for k, d in uzunluk.items()}


def tekrar_karari(kare_kararlari: list[tuple[Karar, Karar]]) -> str:
    """Kare basina (sag, sol) valgus kararlarindan tekrar karari (profil SQUAT)."""
    gerekli = int(np.ceil(SQUAT.kusur_suresi_s * FPS))
    for taraf in (0, 1):
        seri = 0
        for kk in kare_kararlari:
            seri = seri + 1 if kk[taraf] is Karar.KUSURLU else 0
            if seri >= gerekli:
                return "kusurlu"
    kesin = np.mean([all(k is not Karar.BELIRSIZ for k in kk) for kk in kare_kararlari])
    return "dogru" if kesin >= SQUAT.min_kapsama else "belirsiz"


def kaydirilmis_kararlar(kk: list[tuple[Karar, Karar]], deg: np.ndarray,
                         kayma: float) -> list[tuple[Karar, Karar]]:
    """Kare kararlarini `deg - kayma` ile yeniden verir (yalin esik, 0014 tek yonlu).

    Orijinali BELIRSIZ olan kare (gorunmez eklem, imkansiz geometri) BELIRSIZ
    kalir: duzeltme yalnizca olculmus degeri kaydirir, karar verilemeyen kareye
    karar uydurmaz. `kayma=0` orijinal kararlari aynen uretmelidir (denetlenir).
    """
    esik = VARSAYILAN_ESIKLER[VALGUS[0]]
    assert all(VARSAYILAN_ESIKLER[a] == esik for a in VALGUS) and esik.tek_yonlu
    cikti = []
    for kare_kk, kare_deg in zip(kk, deg):
        cikti.append(tuple(
            Karar.BELIRSIZ if k is Karar.BELIRSIZ or not np.isfinite(d)
            else (Karar.KUSURLU if d - kayma > esik.deger else Karar.DOGRU)
            for k, d in zip(kare_kk, kare_deg)))
    return cikti


def kayma_ayristir(ogeler: list[tuple[str, np.ndarray]]) -> dict | None:
    """Isaretli valgus farkinin (telefon - referans) ayrisimi, derece.

    `ogeler`: (kisi, fark) ciftleri; fark bir tekrarin (T, 2) kare dizisi
    (sutunlar sag, sol). Her taraf ayri seri sayilir: iki dizin kaymasi farkli
    olabilir, birlestirmek tekrar ici yayilimi sisirir.
    kayma = butun karelerin ortalamasi (sistematik, duzeltilebilir kisim);
    tekrar_ici_sd = seri icindeki kare gurultusu (zaman filtresinin hedefi);
    tekrarlar_arasi_sd / kisi_arasi_sd = seri ve kisi ortalamalarinin yayilimi.
    """
    temiz = [(k, f[:, s][np.isfinite(f[:, s])]) for k, f in ogeler for s in range(f.shape[1])]
    temiz = [(k, f) for k, f in temiz if f.size]
    if not temiz:
        return None
    hepsi = np.concatenate([f for _, f in temiz])
    kisiler = sorted({k for k, _ in temiz})
    kisi_ort = [np.concatenate([f for k, f in temiz if k == kk]).mean() for kk in kisiler]
    ic = [f.var() for _, f in temiz if f.size > 1]
    taraf = [np.concatenate([f[:, s] for _, f in ogeler]) for s in range(ogeler[0][1].shape[1])]
    return {"n_kare": int(hepsi.size), "kayma": round(float(hepsi.mean()), 2),
            "kayma_taraf": [round(float(np.nanmean(t)), 2) if np.isfinite(t).any() else None
                            for t in taraf],
            "medyan": round(float(np.median(hepsi)), 2),
            "tekrar_ici_sd": round(float(np.sqrt(np.mean(ic))), 2) if ic else None,
            "tekrarlar_arasi_sd": round(float(np.std([f.mean() for _, f in temiz])), 2),
            "kisi_arasi_sd": round(float(np.std(kisi_ort)), 2)}


def lopo_kayma(ogeler: list[tuple[str, np.ndarray]]) -> list[float]:
    """Her oge icin, o kisi DISINDAKI kisilerin kare-agirlikli ortalama farki.

    Test kisisi kendi duzeltmesine girmez (0070 §3). Baska kisi yoksa (ya da
    hic sonlu fark yoksa) kayma 0: duzeltme uygulanmaz.
    """
    cikti = []
    for kisi, _ in ogeler:
        diger = [f[np.isfinite(f)] for k, f in ogeler if k != kisi]
        diger = np.concatenate(diger) if diger else np.empty(0)
        cikti.append(float(diger.mean()) if diger.size else 0.0)
    return cikti


def aci_kayma_dogrusu(ogeler: list[tuple[str, np.ndarray, float]]) -> dict | None:
    """(aci, tekrar ortalama farki) ciftlerine en kucuk kareler dogrusu.

    Donus {"b0", "b1", "ortalama", "n_tekrar"}; kayma(aci) = b0 + b1 * aci.
    Iki farkli sonlu aci yoksa b1 = 0 ve b0 = ortalama; sonlu fark yoksa None.
    """
    e = [(a, float(np.nanmean(f))) for _, f, a in ogeler if np.isfinite(f).any()]
    if not e:
        return None
    a_, m_ = np.array(e, dtype=float).T
    sonlu = np.isfinite(a_)
    b1, b0 = (np.polyfit(a_[sonlu], m_[sonlu], 1) if np.unique(a_[sonlu]).size >= 2
              else (0.0, m_.mean()))
    return {"b0": float(b0), "b1": float(b1), "ortalama": float(m_.mean()),
            "n_tekrar": int(m_.size)}


def lopo_aci_kayma(ogeler: list[tuple[str, np.ndarray, float]]) -> list[float]:
    """Kaymayi govde acisinin dogrusal fonksiyonu olarak, kisi-disari ogrenir.

    `ogeler`: (kisi, fark (T, 2), tekrarin medyan govde acisi). Egitim, test
    kisisi disindaki tekrarlarin (aci, ortalama fark) ciftlerine en kucuk
    kareler dogrusu; yon etiketi kullanilmaz. Egitimde iki farkli aci yoksa
    ortalama farka, o da yoksa 0'a duser. Acisi bilinmeyen tekrar ortalama alir.
    """
    cikti = []
    for kisi, _, aci in ogeler:
        d = aci_kayma_dogrusu([o for o in ogeler if o[0] != kisi])
        if d is None:
            cikti.append(0.0)
        elif not np.isfinite(aci):
            cikti.append(d["ortalama"])
        else:
            cikti.append(d["b0"] + d["b1"] * aci)
    return cikti


def _valgus_kararlari(isk) -> tuple[tuple[Karar, Karar], tuple[float, float]]:
    r = form_degerlendir(isk)
    return (tuple(r.olcumler[a].karar for a in VALGUS),
            tuple(float(r.olcumler[a].deger) for a in VALGUS))


def _ozet(v) -> dict | None:
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "medyan": round(float(np.median(v)), 2),
            "p95": round(float(np.percentile(v, 95)), 2)}


def _temiz(o):
    """NaN -> None (JSON null), ic ice."""
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _temiz(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_temiz(v) for v in o]
    return o


def _tespit_yukle(backend: str, video: str, cam: str) -> dict | None:
    yol = KOK / f"out/rehab24_tespit/{backend}/{video}-{cam}.npz"
    if not yol.exists():
        return None
    d = np.load(yol)
    return {"indeks": {int(k): i for i, k in enumerate(d["kare"])},
            **{k: d[k] for k in ("noktalar", "guven", "gorunur", "tespit", "dunya",
                                 "dunya_gorunur", "boyut", "dunya_tam", "dunya_guven")
               if k in d.files}}


def _tespit_pozu(d: dict, i: int, ci: int, kare: int) -> Poz2B:
    P = d["noktalar"][i].copy()
    g = d["gorunur"][i].astype(bool) & np.isfinite(P).all(axis=1)
    P[~g] = np.nan
    return Poz2B(iskelet=REFERANS_ISKELET, noktalar=P, guven=np.where(g, d["guven"][i], 0.0),
                 gorunur=g, tespit=bool(d["tespit"][i]),
                 goruntu_boyutu=tuple(int(x) for x in d["boyut"]),
                 model="rehab24_tespit", kamera=ci, kare=kare)


def _dunya_tam_iskeleti(d: dict, i: int) -> Iskelet3B:
    """Gorunurluge bakmadan MediaPipe dunya tahmini (gorunmez eklem dahil).

    Profilde arkadaki bacak 2B'de gorunmez sayilir ama model 3B'de yine de bir
    tahmin verir; bu kol o tahminin karara yetip yetmedigini olcer.
    """
    n = len(REFERANS_ISKELET)
    P = np.asarray(d["dunya_tam"][i], float).copy()
    g = np.isfinite(P).all(axis=1)
    return Iskelet3B(REFERANS_ISKELET, P, g, g.astype(int), np.full(n, np.nan),
                     cerceve="mediapipe_kalca_merkezi", kaynak="mediapipe_world_tam")


def _dunya_iskeleti(d: dict, i: int) -> Iskelet3B:
    n = len(REFERANS_ISKELET)
    g = d["dunya_gorunur"][i].astype(bool) & d["gorunur"][i].astype(bool)
    P = np.asarray(d["dunya"][i], float).copy()
    g &= np.isfinite(P).all(axis=1)
    P[~g] = np.nan
    return Iskelet3B(REFERANS_ISKELET, P, g, g.astype(int), np.full(n, np.nan),
                     cerceve="mediapipe_kalca_merkezi", kaynak="mediapipe_world")


def main() -> None:
    tekrarlar = [t for t in tekrarlar_oku(VERI / "Segmentation.csv") if not t.mocap_hatali]
    onbellek: dict[str, np.ndarray] = {}

    def iki_b(video: str, cam: str) -> np.ndarray:
        anahtar = f"{video}-{cam}"
        if anahtar not in onbellek:
            onbellek[anahtar] = np.load(VERI / f"2d_joints/Ex6/{anahtar}-30fps.npy")
        return onbellek[anahtar]

    # Kameralar butun kayitlarda sabit (DLT: rms 0,000 px, 25 Eylul); tek bir
    # video uzerinden kamera basina bir kez kestirilir.
    ornek = tekrarlar[0].video
    X_ornek = np.load(VERI / f"3d_joints/Ex6/{ornek}-30fps.npy")[::10, :, :3]
    kameralar = {c: kamera_kestir(X_ornek.reshape(-1, 3), iki_b(ornek, c)[::10].reshape(-1, 2))
                 for c in KAMERALAR}
    onculer = kisi_onculeri(tekrarlar, lambda t: tekrar_kareleri(VERI, t))

    # Yontemler: kahin 2B her zaman; tespit dosyasi olan her backend icin
    # geometri tabani, MediaPipe icin ayrica pose_world tabani.
    backendler = sorted(p.name for p in (KOK / "out/rehab24_tespit").glob("*") if p.is_dir())
    # Taylor (2000): isaret kaynagi mocap (ust sinir) ya da MediaPipe pose_world.
    yontem_adlari = ["geometri_kahin", "taylor_kahin"] + [
        f"geometri_{b}" for b in backendler] + (
        ["mediapipe_world", "mediapipe_world_tam", "taylor_kahin_mpisaret", "taylor_mediapipe"]
        if "mediapipe" in backendler else [])

    satirlar, diziler = [], []
    for t in tekrarlar:
        kareler = tekrar_kareleri(VERI, t)
        ref_iskelet = [kareyi_cevir(q) for q in kareler]
        ref = [_valgus_kararlari(r) for r in ref_iskelet]
        ref_karar = tekrar_karari([r[0] for r in ref])
        ref_deg = np.array([r[1] for r in ref], dtype=float)
        for ci, (cam, boyut) in enumerate(KAMERALAR.items()):
            kam = kameralar[cam]
            x = iki_b(t.video, cam)[t.ilk:t.son + 1]
            tespitler = {b: _tespit_yukle(b, t.video, cam) for b in backendler}
            for yontem in yontem_adlari:
                b = yontem.split("_", 1)[1] if yontem.startswith("geometri_") else "mediapipe"
                d = tespitler.get(b)
                if yontem not in ("geometri_kahin", "taylor_kahin") and d is None:
                    continue
                tel, hata, sekil, gorunur_oran, valgus_fark = [], [], [], [], []
                seri_deg: list[tuple[float, float]] = []
                seri_aci: list[float] = []
                seri_fleks: list[float] = []
                for j, q in enumerate(kareler):
                    kare = t.ilk + j
                    P_ref = (kam.R @ ref_iskelet[j].noktalar.T + kam.t[:, None]).T
                    if yontem == "geometri_kahin":
                        isk = tek_gorus_3b(poz_kur(x[j], boyut, ci, kare), kam.K, onculer[t.kisi])
                    elif yontem == "taylor_kahin":
                        isk = taylor_3b(poz_kur(x[j], boyut, ci, kare), kam.K, onculer[t.kisi],
                                        isaret_kaynagi=P_ref)
                    elif kare not in d["indeks"]:
                        tel.append((Karar.BELIRSIZ, Karar.BELIRSIZ))
                        seri_deg.append((np.nan, np.nan))
                        seri_aci.append(np.nan)
                        seri_fleks.append(np.nan)
                        gorunur_oran.append(0.0)
                        continue
                    elif yontem == "mediapipe_world":
                        isk = _dunya_iskeleti(d, d["indeks"][kare])
                    elif yontem == "mediapipe_world_tam":
                        if "dunya_tam" not in d:
                            raise RuntimeError("tespit dosyasi eski: dunya_tam yok (28 Eylul "
                                               "oncesi); rehab24_tespit.py'yi yeniden kosun")
                        isk = _dunya_tam_iskeleti(d, d["indeks"][kare])
                    elif yontem == "taylor_kahin_mpisaret":
                        isk = taylor_3b(poz_kur(x[j], boyut, ci, kare), kam.K, onculer[t.kisi],
                                        isaret_kaynagi=_dunya_iskeleti(d, d["indeks"][kare]).noktalar)
                    elif yontem == "taylor_mediapipe":
                        i_ = d["indeks"][kare]
                        isk = taylor_3b(_tespit_pozu(d, i_, ci, kare), kam.K, onculer[t.kisi],
                                        isaret_kaynagi=_dunya_iskeleti(d, i_).noktalar)
                    else:
                        isk = tek_gorus_3b(_tespit_pozu(d, d["indeks"][kare], ci, kare),
                                           kam.K, onculer[t.kisi])
                    kk, deg = _valgus_kararlari(isk)
                    tel.append(kk)
                    seri_deg.append(deg)
                    seri_aci.append(govde_acisi(isk.noktalar))
                    fl = diz_fleksiyonu(isk)
                    seri_fleks.append(float("nan") if fl is None else fl)
                    valgus_fark.extend(abs(a - b_) for a, b_ in zip(deg, ref[j][1])
                                       if np.isfinite(a) and np.isfinite(b_))
                    m = isk.gorunur & np.isfinite(P_ref).all(axis=1)
                    gorunur_oran.append(float(m.mean()))
                    if not m.any():
                        continue
                    # Birincil: mutlak hata, kamera cercevesinde (donusum kalibrasyondan).
                    # MediaPipe world mutlak konum vermez: yalnizca ikincil sekil hatasi.
                    if not yontem.startswith("mediapipe_world"):
                        hata.append(float(np.mean(np.linalg.norm(
                            isk.noktalar[m] - P_ref[m], axis=1)) * 1000))
                    # Ikincil (0070: tani, basari olcutu degil): kare basina rijit hizali sekil.
                    if m.sum() >= 3:
                        h = rijit_hizala(isk.noktalar[m], P_ref[m])
                        sekil.append(float(np.mean(np.linalg.norm(
                            h.uygula(isk.noktalar[m]) - P_ref[m], axis=1)) * 1000))
                satirlar.append({
                    "yontem": yontem, "kisi": t.kisi, "video": t.video, "tekrar": t.tekrar_no,
                    "kamera": cam, "yon": t.yon if cam == "c17" else C18_YONU[t.yon],
                    "fizyoterapist_dogru": t.dogru, "referans": ref_karar,
                    "telefon": tekrar_karari(tel),
                    "mpjpe_mm": float(np.mean(hata)) if hata else float("nan"),
                    "sekil_mm_hizali": float(np.mean(sekil)) if sekil else float("nan"),
                    "valgus_mutlak_fark_derece": float(np.median(valgus_fark)) if valgus_fark
                    else float("nan"),
                    "gorunur_eklem_orani": float(np.mean(gorunur_oran)),
                })
                aci = np.asarray(seri_aci, dtype=float)
                satirlar[-1]["govde_acisi_derece"] = (float(np.median(aci[np.isfinite(aci)]))
                                                      if np.isfinite(aci).any() else float("nan"))
                diziler.append({"satir": satirlar[-1], "kk": tel,
                                "deg": np.array(seri_deg, dtype=float), "ref_deg": ref_deg,
                                "aci": aci, "fleks": np.asarray(seri_fleks, dtype=float)})

    # Isaretli valgus farki ve yone bagli kayma duzeltmesi (28 Eylul). Kayma her
    # (yontem, yon) icin test kisisi DISINDAKI kisilerden ogrenilir (LOPO). Yon
    # veri setinin etiketidir (kahin): telefonda pozdan kestirilmesi gerekir.
    kayma_analizi: dict = defaultdict(dict)
    uyusmazlik = 0
    duzeltilmis = [f"{y}+kayma" for y in yontem_adlari]
    for yontem in yontem_adlari:
        for yon in ("front", "half-profile", "profile"):
            grup = [d for d in diziler
                    if d["satir"]["yontem"] == yontem and d["satir"]["yon"] == yon]
            if not grup:
                continue
            ogeler = [(d["satir"]["kisi"], d["deg"] - d["ref_deg"]) for d in grup]
            kayma_analizi[yontem][yon] = kayma_ayristir(ogeler)
            for d, kayma in zip(grup, lopo_kayma(ogeler)):
                if tekrar_karari(kaydirilmis_kararlar(d["kk"], d["deg"], 0.0)) != \
                        d["satir"]["telefon"]:
                    uyusmazlik += 1
                satirlar.append({**d["satir"], "yontem": f"{yontem}+kayma",
                                 "lopo_kayma_derece": kayma,
                                 "telefon": tekrar_karari(
                                     kaydirilmis_kararlar(d["kk"], d["deg"], kayma))})
    # Yon etiketi olmadan: kayma, tekrarin kestirilen govde acisinin fonksiyonu
    # (butun yonler birlikte, kisi-disari). Yon yalnizca raporlamada kullanilir.
    duzeltilmis += [f"{y}+aci" for y in yontem_adlari]
    aci_dogrulari = {}
    for yontem in yontem_adlari:
        grup = [d for d in diziler if d["satir"]["yontem"] == yontem]
        ogeler = [(d["satir"]["kisi"], d["deg"] - d["ref_deg"], d["satir"]["govde_acisi_derece"])
                  for d in grup]
        # Dis veride (Fitness-AQA) kullanilmak uzere 9 kisinin tamamiyla dondurulan dogru.
        aci_dogrulari[yontem] = aci_kayma_dogrusu(ogeler)
        for d, kayma in zip(grup, lopo_aci_kayma(ogeler)):
            satirlar.append({**d["satir"], "yontem": f"{yontem}+aci", "lopo_kayma_derece": kayma,
                             "telefon": tekrar_karari(
                                 kaydirilmis_kararlar(d["kk"], d["deg"], kayma))})
    # kayma=0 ile yeniden verilen karar orijinalle ayni olmali: yeniden
    # uygulanan esik kuralinin form_degerlendir ile esdeger oldugunun kaniti.
    if uyusmazlik:
        raise RuntimeError(f"yeniden karar {uyusmazlik} tekrarda orijinalden farkli")

    yontemler: dict = {}
    for yontem in yontem_adlari + duzeltilmis:
        yonler = {}
        for yon in ("front", "half-profile", "profile"):
            s = [r for r in satirlar if r["yontem"] == yontem and r["yon"] == yon]
            if not s:
                continue
            og = [Oge(r["kisi"], f'{r["video"]}-{r["kamera"]}', r["referans"], r["telefon"])
                  for r in s]
            yonler[yon] = {
                "n_tekrar": len(s), "n_kisi": len({r["kisi"] for r in s}),
                "protokol_0070": Sayim.ciftlerden((o.referans, o.telefon) for o in og).oranlar(),
                "kisi_agirlikli": kisi_agirlikli(og),
                "kisi_bootstrap": kisi_bootstrap(og),
                "mpjpe_mm": _ozet([r["mpjpe_mm"] for r in s]),
                "sekil_mm_hizali": _ozet([r["sekil_mm_hizali"] for r in s]),
                "valgus_mutlak_fark_derece": _ozet([r["valgus_mutlak_fark_derece"] for r in s]),
                "gorunur_eklem_orani": round(float(np.mean(
                    [r["gorunur_eklem_orani"] for r in s])), 4),
                "referans_kusurlu": sum(r["referans"] == "kusurlu" for r in s),
                "karisiklik": {f"{a}->{b}": sum(r["referans"] == a and r["telefon"] == b
                                                 for r in s)
                               for a in ("dogru", "kusurlu", "belirsiz")
                               for b in ("dogru", "kusurlu", "belirsiz")},
                "kayma_analizi": kayma_analizi.get(yontem, {}).get(yon),
                "govde_acisi_derece": _ozet([r["govde_acisi_derece"] for r in s]),
            }
        yontemler[yontem] = yonler

    sonuc = {
        "deney": "rehab24_tek_gorus",
        "katmanlar": {"geometri_kahin": "kahin 2B (mocap izdusumu) + kemik onculu geometri",
                      "taylor_kahin": "kahin 2B + Taylor (2000), isaret mocap'ten (ust sinir)",
                      "taylor_kahin_mpisaret": "kahin 2B + Taylor, isaret MediaPipe world'den",
                      "taylor_mediapipe": "MediaPipe 2B + Taylor, isaret MediaPipe world'den",
                      "geometri_<backend>": "gercek dedektor 2B + kemik onculu geometri",
                      "mediapipe_world": "MediaPipe pose_world (ogrenilmis 3B, mutlak konum yok)",
                      "mediapipe_world_tam": "pose_world, gorunurluge bakmadan (gorunmez eklem "
                                             "tahmini dahil)",
                      "<yontem>+kayma": "yone bagli valgus kaymasi, kisi-disari ogrenilip "
                                        "cikarilmis (yon kahin)"},
        "kaynak": "REHAB24-6 Ex6; 3d_joints + 2d_joints 30 fps; videolar (tespit)",
        "kameralar": {c: {"fx": round(float(k.K[0, 0]), 1), "fy": round(float(k.K[1, 1]), 1),
                          "rms_px": round(k.rms_px, 6),
                          "merkez_m": np.round(-k.R.T @ k.t, 3).tolist()}
                      for c, k in kameralar.items()},
        "tekrar_kurali": {"kusur_suresi_s": SQUAT.kusur_suresi_s,
                          "min_kapsama": SQUAT.min_kapsama, "fps": FPS},
        "onculer": "kisi basina mocap kemik uzunlugu medyani",
        "yontemler": yontemler,
        "aci_kayma_dogrusu": aci_dogrulari,
        "not": ("9 kisi < 10: on kanit. Referans karar ayni kuralin mocap'e uygulanmasi; "
                "0014 valgus tanimi EC3D'de secildi. sekil_mm_hizali kare basina rijit "
                "hizalamadir: 0070'e gore yalnizca ikincil tani."),
    }
    (KOK / "out").mkdir(exist_ok=True)
    # Kare dizileri: belirsizlik modeli (deney_rehab24_belirsizlik) bunlardan ogrenir.
    with open(KOK / "out/rehab24_tek_gorus_diziler.pkl", "wb") as f:
        pickle.dump([{k: v for k, v in d.items() if k != "kk"} | {
            "kk": [tuple(str(x) for x in k) for k in d["kk"]]} for d in diziler], f)
    (KOK / "out/rehab24_tek_gorus.json").write_text(
        json.dumps(_temiz(sonuc), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (KOK / "out/rehab24_tek_gorus_satirlar.json").write_text(
        json.dumps(_temiz(satirlar), ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8")
    for yontem, yonler in yontemler.items():
        for yon, y in yonler.items():
            o = y["protokol_0070"]
            print(f"{yontem:18s} {yon:12s}", {k: o[k] for k in ("N", "D", "C", "A", "K", "Y")},
                  "mpjpe", (y["mpjpe_mm"] or {}).get("medyan"),
                  "sekil", (y["sekil_mm_hizali"] or {}).get("medyan"),
                  "valgus_fark", (y["valgus_mutlak_fark_derece"] or {}).get("medyan"))


if __name__ == "__main__":
    main()
