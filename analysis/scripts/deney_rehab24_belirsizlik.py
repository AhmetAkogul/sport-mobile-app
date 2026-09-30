"""REHAB24-6: telefona "olcemiyorum" demeyi ogretmek -- kalibre belirsizlikle tekrar karari.

28 Eylul'de profil kusurlara guvenle "dogru" dedi (0/6 yakalandi). Mukemmel
sistem emin olmadiginda karar vermemeli. Burada valgus hatasinin dagilimi,
bakis acisi ve diz bukulmesine gore, **kisi-disari** ogrenilir ve karar bu
belirsizlikle verilir:

1. Duzeltilmis valgus: `mediapipe_world_tam`, aci-kayma dogrusu kisi-disari
   (`lopo_aci_kayma`, yon etiketi yok).
2. sigma(aci, fleksiyon): egitim kisilerinin artigi (duzeltilmis - mocap),
   hucre basina dayanikli SD (1,4826 * MAD). Az ornekli hucre, egitim
   kisilerinin genel sigmasina duser.
3. Tekrar: her 0,2 s pencerenin degeri pencere minimumudur (kesintisiz
   tutulan), sigmasi pencerenin sigma medyanidir. Bir pencerede
   deger - z*sigma > esik -> kusurlu; butun pencerelerde deger + z*sigma < esik
   -> dogru; arasi belirsiz. z = 1,645 (tek yonlu %95).
4. Kalibrasyon: test kisisinin karelerinde |artik| < z*sigma orani (~%90 beklenir).

Yon etiketleri yalniz raporlamada kullanilir.

    python scripts/deney_rehab24_belirsizlik.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tg", KOK / "scripts/deney_rehab24_tek_gorus.py")
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)

YONTEM = "mediapipe_world_tam"
ACI_SINIR = (0, 20, 40, 60, 90.01)
FLEKS_SINIR = (0, 30, 60, 90, 180.01)
Z = 1.645
# Guven duzeyi ve "sigma su kadar kucuk olsaydi" taramasi: karar verebilmek icin
# belirsizligin ne kadar dusmesi gerektigini olcer (gereksinim, basari degil).
TARAMA_Z = (0.0, 0.5, 1.0, 1.645)
TARAMA_OLCEK = (1.0, 0.5, 0.25)
MIN_HUCRE = 200
GEREKLI = int(np.ceil(tg.SQUAT.kusur_suresi_s * tg.FPS))


def hucre(aci: np.ndarray, fleks: np.ndarray) -> np.ndarray:
    """Kare basina hucre indeksi; aci ya da fleksiyon yoksa ya da aralik disiysa -1."""
    ia = np.digitize(aci, ACI_SINIR) - 1
    if_ = np.digitize(fleks, FLEKS_SINIR) - 1
    ok = (np.isfinite(aci) & np.isfinite(fleks) & (ia >= 0) & (ia < len(ACI_SINIR) - 1)
          & (if_ >= 0) & (if_ < len(FLEKS_SINIR) - 1))
    return np.where(ok, ia * (len(FLEKS_SINIR) - 1) + if_, -1)


def dayanikli_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def sigma_ogren(artik: np.ndarray, hucreler: np.ndarray) -> tuple[dict[int, float], float]:
    """Hucre basina dayanikli SD; MIN_HUCRE'den az ornekli hucre genel SD'yi alir."""
    genel = dayanikli_sd(artik)
    tablo = {}
    for h in np.unique(hucreler[hucreler >= 0]):
        a = artik[hucreler == h]
        a = a[np.isfinite(a)]
        tablo[int(h)] = dayanikli_sd(a) if a.size >= MIN_HUCRE else genel
    return tablo, genel


def belirsiz_karar(v: np.ndarray, gecerli: np.ndarray, sig: np.ndarray, esik: float,
                   z: float = Z, g: int = GEREKLI) -> str:
    """v: (T, 2) duzeltilmis valgus; gecerli: (T, 2) kare karari belirsiz degil;
    sig: (T,) kare sigmasi."""
    en_alt, tum_ust_alti, pencere = -np.inf, True, 0
    for s in range(v.shape[1]):
        for i in range(len(v) - g + 1):
            w, ok, sw = v[i:i + g, s], gecerli[i:i + g, s], sig[i:i + g]
            if not (ok.all() and np.isfinite(w).all() and np.isfinite(sw).all()):
                continue
            pencere += 1
            deger, sm = float(w.min()), float(np.median(sw))
            en_alt = max(en_alt, deger - z * sm)
            if deger + z * sm >= esik:
                tum_ust_alti = False
    if en_alt > esik:
        return "kusurlu"
    if pencere and tum_ust_alti:
        return "dogru"
    return "belirsiz"


def _karar_ozeti(s: list[dict], kol: str) -> dict:
    karar = [x for x in s if x[kol] != "belirsiz"]
    kus = [x for x in s if x["referans"] == "kusurlu"]
    return {"n": len(s), "karar_verilen": len(karar),
            "dogru_karar": sum(x[kol] == x["referans"] for x in karar),
            "n_kusurlu": len(kus),
            "kusur_yakalanan": sum(x[kol] == "kusurlu" for x in kus),
            "kusur_dogru_denen": sum(x[kol] == "dogru" for x in kus),
            "kusur_belirsiz": sum(x[kol] == "belirsiz" for x in kus),
            "yanlis_alarm": sum(x[kol] == "kusurlu" for x in s if x["referans"] == "dogru")}


def main() -> None:
    with open(KOK / "out/rehab24_tek_gorus_diziler.pkl", "rb") as f:
        D = [d for d in pickle.load(f) if d["satir"]["yontem"] == YONTEM]
    esik = tg.VARSAYILAN_ESIKLER[tg.VALGUS[0]].deger
    ogeler = [(d["satir"]["kisi"], d["deg"] - d["ref_deg"], d["satir"]["govde_acisi_derece"])
              for d in D]
    for d, kayma in zip(D, tg.lopo_aci_kayma(ogeler)):
        d["kayma"] = kayma
        d["duz"] = d["deg"] - kayma
        d["artik"] = d["duz"] - d["ref_deg"]
        d["hucre"] = hucre(d["aci"], d["fleks"])
        d["karar"] = [tuple(tg.Karar(x) for x in kk) for kk in d["kk"]]
        d["gecerli"] = np.array([[k is not tg.Karar.BELIRSIZ for k in kk]
                                 for kk in d["karar"]], bool)

    kisiler = sorted({d["satir"]["kisi"] for d in D}, key=int)
    kalibrasyon, satirlar = [], []
    for kisi in kisiler:
        egitim = [d for d in D if d["satir"]["kisi"] != kisi]
        tablo, genel = sigma_ogren(np.concatenate([d["artik"].ravel() for d in egitim]),
                                   np.concatenate([np.repeat(d["hucre"], 2) for d in egitim]))
        for d in (x for x in D if x["satir"]["kisi"] == kisi):
            sig = np.array([tablo.get(int(h), genel) if h >= 0 else genel for h in d["hucre"]])
            m = np.isfinite(d["artik"])
            kalibrasyon.extend((np.abs(d["artik"]) < Z * sig[:, None])[m].tolist())
            tarama = {f"z{z:g}_olcek{o:g}": belirsiz_karar(d["duz"], d["gecerli"], o * sig, esik, z)
                      for z in TARAMA_Z for o in TARAMA_OLCEK}
            satirlar.append({**tarama,
                "kisi": kisi, "yon": d["satir"]["yon"], "referans": d["satir"]["referans"],
                "belirsizliksiz": tg.tekrar_karari(
                    tg.kaydirilmis_kararlar(d["karar"], d["deg"], d["kayma"])),
                "belirsizlikli": belirsiz_karar(d["duz"], d["gecerli"], sig, esik),
                "sigma_medyan": float(np.median(sig))})

    sonuc = {"deney": "rehab24_belirsizlik", "yontem": YONTEM, "z": Z,
             "kalibrasyon_kapsama": round(float(np.mean(kalibrasyon)), 3),
             "kalibrasyon_beklenen": 0.90, "yonler": {}}
    for yon in ("front", "half-profile", "profile"):
        s = [x for x in satirlar if x["yon"] == yon]
        sonuc["yonler"][yon] = {
            "sigma_medyan": round(float(np.median([x["sigma_medyan"] for x in s])), 2),
            **{kol: _karar_ozeti(s, kol) for kol in ("belirsizliksiz", "belirsizlikli")}}
    for yon in ("front", "half-profile", "profile"):
        s = [x for x in satirlar if x["yon"] == yon]
        sonuc["yonler"][yon]["tarama"] = {
            f"z{z:g}_olcek{o:g}": _karar_ozeti(s, f"z{z:g}_olcek{o:g}")
            for z in TARAMA_Z for o in TARAMA_OLCEK}
    (KOK / "out/rehab24_belirsizlik.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("kalibrasyon:", sonuc["kalibrasyon_kapsama"], "(beklenen 0,90)")
    for yon, v in sonuc["yonler"].items():
        print(yon, "sigma", v["sigma_medyan"])
        for ad, o in v["tarama"].items():
            print(f"   {ad:16s} karar {o['karar_verilen']:3d}/{o['n']} dogru {o['dogru_karar']:3d} "
                  f"kusur yakalanan {o['kusur_yakalanan']}/{o['n_kusurlu']} "
                  f"kusura dogru {o['kusur_dogru_denen']} yanlis alarm {o['yanlis_alarm']}")
        for kol in ("belirsizliksiz", "belirsizlikli"):
            print("  ", f"{kol:15s}", v[kol])


if __name__ == "__main__":
    main()
