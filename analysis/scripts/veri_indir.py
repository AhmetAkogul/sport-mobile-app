"""Dis veri seti ve model katalogu + indirici.

Katalog tek yerde: her kaynagin adresi, lisansi, erisim turu ve projedeki rolu.
Erisim turleri:

- `dogrudan`  -- hesapsiz indirilir; bu betik indirir.
- `hesap`     -- sitede ucretsiz hesapla indirilir; tarayicidan indirilip
                 `data/dis/<ad>/` altina konur (betik yalnizca yeri gosterir).
- `basvuru`   -- form/lisans onayi gerekir; onay e-postasindaki yolla indirilir.

Indirilen her dosyanin yanina `KAYNAK.json` yazilir: adres, lisans, tarih,
SHA-256. Veri git'e girmez (`data/` .gitignore'da).

    python scripts/veri_indir.py --liste              # katalog
    python scripts/veri_indir.py --indir aistpp_kamera  # tek kaynak
    python scripts/veri_indir.py --indir-hepsi-dogrudan --en-cok-mb 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
HEDEF = KOK / "data" / "dis"
ERISIMLER = ("dogrudan", "hesap", "basvuru")
DENEME = 8          # kopan buyuk indirmeler icin surdurme denemesi

# Rol kodlari: R=duzenek/referans dogrulama, T=telefon-referans eslesmesi (kopru),
# F=form/kalite etiketi, B=biyomekanik/buyuk mocap, M=hazir model.
KATALOG: dict[str, dict] = {
    # --- hemen alinabilir (dogrudan) --------------------------------------------
    "aistpp_kamera": {
        "url": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip",
        "lisans": "CC BY 4.0 (aciklamalar)", "erisim": "dogrudan", "mb": 0.02, "rol": "R",
        "not": "AIST++ 9 kamerali kalibrasyon ayarlari; coklu kamera formatini sinamak icin"},
    "aistpp_keypoints3d": {
        "url": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/keypoints3d.zip",
        "lisans": "CC BY 4.0 (aciklamalar)", "erisim": "dogrudan", "mb": 834, "rol": "R",
        "not": "30 kisi, 9 gorus, 3B anahtar nokta; videolar AIST Dance DB kosullariyla ayri"},
    "aistpp_hareket": {
        "url": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip",
        "lisans": "CC BY 4.0 (aciklamalar)", "erisim": "dogrudan", "mb": 79, "rol": "R",
        "not": "AIST++ SMPL hareket dizileri"},
    "rtmw_x_l_384": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "lisans": "Apache 2.0 (MMPose)", "erisim": "dogrudan", "mb": 250, "rol": "M",
        "not": "RTMW tum vucut 2B (133 nokta), en buyuk varyant; RTMPose-m ile kiyas"},
    "vitpose_base": {
        "url": "https://huggingface.co/usyd-community/vitpose-base-simple",
        "dosyalar": [f"https://huggingface.co/usyd-community/vitpose-base-simple/resolve/main/{d}"
                     for d in ("config.json", "preprocessor_config.json", "model.safetensors",
                               "README.md")],
        "lisans": "Apache 2.0 (model karti)", "erisim": "dogrudan", "mb": 360, "rol": "M",
        "not": "ViTPose-B, 2B dogruluk referansi adayi (transformers)"},
    "mmfit": {
        "url": "https://zenodo.org/records/7672767", "zenodo": "7672767",
        "lisans": "CC BY 4.0", "erisim": "dogrudan", "mb": 39100, "rol": "F",
        "not": "20 antrenman RGB videosu; IMU/iskelet verisi mmfit.github.io'da"},
    "mediapipe_pose_lite": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "lisans": "Apache 2.0", "erisim": "dogrudan", "mb": 6, "rol": "M",
        "not": "MediaPipe varyant karsilastirmasi (lite)"},
    "mediapipe_pose_heavy": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "lisans": "Apache 2.0", "erisim": "dogrudan", "mb": 30, "rol": "M",
        "not": "MediaPipe varyant karsilastirmasi (heavy)"},
    "addbiomechanics": {
        "url": "http://archive.simtk.org/addbiomechanics/addbiomechanics.zip",
        "lisans": "CC BY 4.0", "erisim": "dogrudan", "mb": 417666, "rol": "B",
        "not": "273 kisi, marker + kuvvet plakasi; TAM PAKET 418 GB (diskten buyuk), "
               "alt kume secilerek indirilmeli"},
    "ec3d": {
        "url": "https://drive.google.com/drive/folders/1Y00Qw6QyAnhrxUWelyFcTm2a7tnZB93y",
        "lisans": "belirtilmemis (akademik kullan, atif ver)", "erisim": "dogrudan", "mb": 982,
        "rol": "F",
        "not": "4 GoPro, dogru/yanlis squat-lunge-plank, 3B + kamera; 4 kisi (0070'e gore az). "
               "Google Drive klasoru: gdown gerekir"},
    # --- hesapla (tarayici) -----------------------------------------------------
    "opencap_labvalidation": {
        "url": "https://simtk.org/frs/?group_id=2385",
        "lisans": "Apache 2.0", "erisim": "hesap", "mb": 19051, "rol": "R,T",
        "not": "10 kisi, 5 kamera (iPhone), marker mocap, squat; BAGIMSIZ REFERANS. "
               "LabValidation_withVideos.zip -> data/dis/opencap_labvalidation/"},
    "cmu_panoptic_2": {
        "url": "https://simtk.org/projects/cmupanopticdata",
        "lisans": "ticari olmayan arastirma", "erisim": "hesap", "mb": None, "rol": "R",
        "not": "86 kisi, 31 HD kamera + IMU, eklem hareket acikligi; SimTK hesabiyla"},
    "ui_prmd": {
        "url": "https://webpages.uidaho.edu/ui-prmd/",
        "lisans": "ODC Public Domain (PDDL 1.0)", "erisim": "hesap", "mb": None, "rol": "R,F",
        "not": "10 kisi, 10 egzersiz (derin squat dahil) DOGRU + YANLIS, Vicon + Kinect. "
               "Sunucu otomatik indirmeyi engelliyor: tarayicidan"},
    # --- basvuru ----------------------------------------------------------------
    "flex": {"url": "https://github.com/HaoYin116/FLEX", "lisans": "CC BY-NC-SA 4.0",
             "erisim": "basvuru", "mb": None, "rol": "T,F",
             "not": "38 kisi, 4 kalibre kamera + telefon, uzman hata etiketleri, 20 egzersiz"},
    "m3gym": {"url": "https://finalyou.github.io/M3GYM/", "lisans": "ticari olmayan akademik",
              "erisim": "basvuru", "mb": None, "rol": "R,F",
              "not": "50+ kisi, 8 kamera, gercek spor salonu, 3B + uzman degerlendirmesi"},
    "fit3d": {"url": "https://fit3d.imar.ro/", "lisans": "ticari olmayan arastirma",
              "erisim": "basvuru", "mb": None, "rol": "R",
              "not": "13 kisi, 37 egzersiz, 4 RGB + Vicon"},
    "fitness_aqa": {"url": "https://github.com/ParitoshParmar/Fitness-AQA",
                    "lisans": "ticari olmayan", "erisim": "basvuru", "mb": None, "rol": "F",
                    "not": "gercek salon videolari; squat 'diz ice' (valgus) etiketi"},
    "human36m": {"url": "http://vision.imar.ro/human3.6m/", "lisans": "akademik lisans",
                 "erisim": "basvuru", "mb": None, "rol": "R",
                 "not": "11 kisi, 4 kamera, Vicon; standart kiyas"},
    "3dpw": {"url": "https://virtualhumans.mpi-inf.mpg.de/3DPW/", "lisans": "ticari olmayan",
             "erisim": "basvuru", "mb": None, "rol": "T",
             "not": "telefonla cekilmis, IMU'dan SMPL yer gercegi"},
    "emdb": {"url": "https://eth-ait.github.io/emdb/", "lisans": "ticari olmayan",
             "erisim": "basvuru", "mb": None, "rol": "T",
             "not": "iPhone, elektromanyetik sensorle yer gercegi"},
    "kimore": {"url": "https://vrai.dii.univpm.it/content/kimore-dataset", "lisans": "akademik",
               "erisim": "basvuru", "mb": None, "rol": "F",
               "not": "78 kisi (34 hasta), 5 egzersiz (squat dahil), klinik skor, Kinect"},
    "sportspose": {"url": "https://github.com/ChristianIngwersen/SportsPose",
                   "lisans": "arastirma", "erisim": "basvuru", "mb": None, "rol": "R",
                   "not": "24 kisi, 7 kamera + mocap, dinamik spor hareketleri"},
    "qevd": {"url": "https://github.com/Qualcomm-AI-research/FitCoach", "lisans": "arastirma",
             "erisim": "basvuru", "mb": None, "rol": "F",
             "not": "300K kisa klip, hata geri bildirimi (VLM kocluk); tek kamera"},
}


def _sha256(yol: Path) -> str:
    h = hashlib.sha256()
    with yol.open("rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def katalog_denetle(katalog: dict = KATALOG) -> list[str]:
    hatalar = []
    for ad, k in katalog.items():
        for alan in ("url", "lisans", "erisim", "rol", "not"):
            if not k.get(alan):
                hatalar.append(f"{ad}: '{alan}' eksik")
        if k.get("erisim") not in ERISIMLER:
            hatalar.append(f"{ad}: erisim {ERISIMLER} olmali")
        if not str(k.get("url", "")).startswith(("http://", "https://")):
            hatalar.append(f"{ad}: url http(s) olmali")
    return hatalar


def _dosya_adresi_mi(url: str) -> bool:
    return "drive.google.com" not in url and "zenodo.org/records" not in url


def _dosya_listesi(k: dict) -> list[str]:
    """Kaynagin indirilecek dosya adresleri: tek url, acik liste ya da Zenodo kaydi."""
    if k.get("dosyalar"):
        return list(k["dosyalar"])
    if k.get("zenodo"):
        sonuc = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error",
             f"https://zenodo.org/api/records/{k['zenodo']}"],
            capture_output=True, text=True)
        if sonuc.returncode != 0:
            raise ValueError(f"Zenodo listesi alinamadi: {sonuc.stderr.strip()}")
        return [f["links"]["self"] for f in json.loads(sonuc.stdout)["files"]]
    if not _dosya_adresi_mi(k["url"]):
        raise ValueError(f"klasor/sayfa adresi: dosya secerek indirilir ({k['url']})")
    return [k["url"]]


def _curl(url: str, hedef: Path) -> None:
    # Sistem curl'u: macOS'ta python.org Python'unun sertifika deposu bos olabilir
    # (CERTIFICATE_VERIFY_FAILED); dogrulamayi kapatmak yerine isletim sisteminin
    # deposunu kullanan curl'e birakilir. --fail: HTTP hatasi dosya yazdirmaz.
    # Yarim dosya (.kismi) silinmez: sunucu baglantiyi koparirsa (curl 18) bir
    # sonraki deneme kaldigi yerden surer (--continue-at -). Buyuk dosyalarda
    # (MM-Fit, 1-3 GB) Zenodo baglantiyi yarida kapatabiliyor.
    gecici = hedef.with_name(hedef.name + ".kismi")
    sonuc = None
    for _ in range(DENEME):
        sonuc = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--retry", "3", "--retry-all-errors", "--continue-at", "-",
             "--output", str(gecici), url], capture_output=True, text=True)
        if sonuc.returncode == 0:
            gecici.replace(hedef)
            return
    raise ValueError(f"indirme basarisiz ({sonuc.returncode}) {url}: {sonuc.stderr.strip()} "
                     f"-- yarim dosya {gecici.name} saklandi, tekrar calistirinca surer")


def _dosya_adi(url: str) -> str:
    """.../files/w12_rgb.mp4/content ve .../resolve/main/x.json icin dosya adi."""
    parcalar = [p for p in url.split("?")[0].split("/") if p]
    return parcalar[-2] if parcalar[-1] == "content" else parcalar[-1]


def indir(ad: str, hedef_kok: Path = HEDEF) -> Path:
    """Kaynagi indir; var olan dosyalar atlanir (yarida kalan indirme surdurulur)."""
    k = KATALOG[ad]
    if k["erisim"] != "dogrudan":
        raise ValueError(f"{ad} '{k['erisim']}' erisimli: tarayicidan indirip "
                         f"{hedef_kok / ad}/ altina koyun ({k['url']})")
    adresler = _dosya_listesi(k)
    dizin = hedef_kok / ad
    dizin.mkdir(parents=True, exist_ok=True)
    kayit_yolu = dizin / "KAYNAK.json"
    kayit = (json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists()
             else {"ad": ad, "url": k["url"], "lisans": k["lisans"], "rol": k["rol"],
                   "dosyalar": {}})
    if "dosyalar" not in kayit and "dosya" in kayit:
        # Eski tek dosyali bicim (ilk tur): dosya kaydini listeye tasi.
        kayit["dosyalar"] = {kayit.pop("dosya"): {
            "url": kayit["url"], "sha256": kayit.pop("sha256"), "bayt": kayit.pop("bayt"),
            "indirme_utc": kayit.pop("indirme_utc")}}
    for url in adresler:
        dosya = dizin / _dosya_adi(url)
        if dosya.name in kayit["dosyalar"] and dosya.exists():
            continue
        _curl(url, dosya)
        kayit["dosyalar"][dosya.name] = {
            "url": url, "sha256": _sha256(dosya), "bayt": dosya.stat().st_size,
            "indirme_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        # Her dosyadan sonra yazilir: yarida kesilirse biten dosyalar kayitli kalir.
        kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    # Hic dosya inmese de yaz: eski bicimden donusum diske islensin.
    kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
    return dizin


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dis veri seti katalogu ve indirici")
    p.add_argument("--liste", action="store_true")
    p.add_argument("--indir", metavar="AD")
    p.add_argument("--indir-hepsi-dogrudan", action="store_true")
    p.add_argument("--en-cok-mb", type=float, default=1000.0)
    a = p.parse_args(argv)
    if hatalar := katalog_denetle():
        print("\n".join(hatalar), file=sys.stderr)
        return 1
    if a.liste or not (a.indir or a.indir_hepsi_dogrudan):
        for ad, k in sorted(KATALOG.items(), key=lambda kv: (kv[1]["erisim"], kv[0])):
            mb = "?" if k["mb"] is None else f"{k['mb']:.0f}"
            print(f"{k['erisim']:9s} {ad:24s} {k['rol']:5s} {mb:>7s} MB  {k['lisans']}")
        return 0
    adlar = [a.indir] if a.indir else [
        ad for ad, k in KATALOG.items()
        if k["erisim"] == "dogrudan" and k["mb"] is not None and k["mb"] <= a.en_cok_mb
        and (_dosya_adresi_mi(k["url"]) or k.get("dosyalar") or k.get("zenodo"))]
    for ad in adlar:
        try:
            print(f"indiriliyor: {ad} -> {indir(ad)}")
        except ValueError as exc:
            print(f"atlandi: {ad}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
