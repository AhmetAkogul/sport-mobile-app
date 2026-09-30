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
import time
from datetime import datetime, timezone
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
HEDEF = KOK / "data" / "dis"
ERISIMLER = ("dogrudan", "hesap", "basvuru")
DENEME = 60         # her deneme kaldigi yerden surer; kopma basina bir deneme

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
    "movi_f": {
        "url": "https://www.biomotionlab.ca/movi/",
        "dataverse": "doi:10.5683/SP2/JRHDRN",
        # F turu: tam vucut mocap + 2 telefon (CP) + 2 sabit kamera (PG); ilk 8 kisi.
        "secim": r"^(README\.pdf|Camera Parameters\.tar|F_AMASS\.tar|F_Subjects_1_45\.tar"
                 r"|F_(CP1|CP2|PG1|PG2)_Subject_[1-8](_.*)?\.(mp4|avi))$",
        "lisans": "ticari olmayan bilimsel arastirma (Dataverse kullanim kosullari)",
        "erisim": "dogrudan", "mb": 12000, "rol": "R,T",
        "not": "90 kisi, optik mocap; sabit kameralar kalibre + senkron (duzenek dogrulamasi). "
               "Telefonlar elde, kalibresiz ve mocap ile SENKRON DEGIL; squat yok. "
               "OpenCap'in yerini tutmaz. Tam veri 290 GB: F turu, ilk 8 kisi"},
    "rehab24_6": {
        "url": "https://zenodo.org/records/13305826", "zenodo": "13305826",
        "lisans": "CC BY-NC 4.0", "erisim": "dogrudan", "mb": 5670, "rol": "R,T,F",
        "not": "6 rehabilitasyon egzersizi (Ex6 = SQUAT), dogru + yanlis tekrar etiketi, "
               "2 senkron RGB kamera + 41 marker optik mocap (26 eklem 3B + 2B izdusum), "
               "tekrar bolutleme. Tek kamera + bagimsiz referans + form etiketi ayni kayitta"},
    "ucophyrehab": {
        "url": "https://zenodo.org/records/17935737",
        "dosyalar": [f"https://zenodo.org/api/records/17935737/files/{d}/content"
                     for d in ("dataset_3d_with_angles.json", "ucophyrehab2_data.jsonl",
                               "samples.zip")],
        "lisans": "CC BY 4.0", "erisim": "dogrudan", "mb": 557, "rol": "F",
        "not": "cok gorunuslu rehabilitasyon; HAM RGB VIDEO YOK (gizlilik). Alinan: aciyla 3B "
               "iskelet + ornekler; siluet/optik akis/segmentasyon (~9 GB) alinmadi"},
    "knee_pad": {
        "url": "https://zenodo.org/records/12112951", "zenodo": "12112951",
        "lisans": "CC BY 4.0", "erisim": "dogrudan", "mb": 289, "rol": "F",
        "not": "31 diz hastasi; squat, bacak uzatma, yurume -- dogru + 2 yanlis varyasyon; "
               "EMG + IMU odakli (video yayimlanip yayimlanmadigi arsivden anlasilacak)"},
    "ul_red": {
        "url": "https://datacat.liverpool.ac.uk/2729/",
        "dosyalar": [f"https://datacat.liverpool.ac.uk/2729/{i}/S{i:02d}.zip" for i in range(1, 11)]
                    + ["https://datacat.liverpool.ac.uk/2729/11/code.zip"],
        "lisans": "CC BY 4.0", "erisim": "dogrudan", "mb": 5600, "rol": "R",
        "not": "10 kisi, 22 egzersiz (mini squat dahil), OptiTrack marker + markersiz iskelet + "
               "derinlik; RGB video yok. Markerli vs markersiz karsilastirmasi"},
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


def _dataverse_listesi(k: dict) -> list[tuple[str, str]]:
    """Dataverse kaydindan `secim` desenine uyan (adres, dosya adi) ciftleri."""
    import re
    sonuc = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error",
         f"https://borealisdata.ca/api/datasets/:persistentId/?persistentId={k['dataverse']}"],
        capture_output=True, text=True)
    if sonuc.returncode != 0:
        raise ValueError(f"Dataverse listesi alinamadi: {sonuc.stderr.strip()}")
    desen = re.compile(k["secim"])
    dosyalar = json.loads(sonuc.stdout)["data"]["latestVersion"]["files"]
    return [(f"https://borealisdata.ca/api/access/datafile/{f['dataFile']['id']}",
             f["dataFile"]["filename"]) for f in dosyalar
            if desen.match(f["dataFile"]["filename"]) and not f.get("restricted")]


def _dosya_listesi(k: dict) -> list:
    """Kaynagin indirilecek dosya adresleri: tek url, acik liste, Zenodo ya da Dataverse.

    Oge ya adres ya da (adres, dosya adi) ciftidir (Dataverse adresi kimlik tasir).
    """
    if k.get("dataverse"):
        return _dataverse_listesi(k)
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
    # curl'un kendi --retry'i KULLANILMAZ: yeniden denerken dosyayi o curl
    # cagrisinin basladigi boyuta geri kirpiyor, yani oturumda inen her sey
    # kopmada kayboluyordu (REHAB24-6'da .kismi dosyalari kuculdu, 25 Eylul).
    # Her deneme yeni bir curl'dur; --continue-at - ofseti dosyadan yeniden okur.
    gecici = hedef.with_name(hedef.name + ".kismi")
    sonuc = None
    for deneme in range(DENEME):
        sonuc = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--continue-at", "-", "--output", str(gecici), url],
            capture_output=True, text=True)
        if sonuc.returncode == 0:
            gecici.replace(hedef)
            return
        time.sleep(min(60, 2 ** deneme))
    raise ValueError(f"indirme basarisiz ({sonuc.returncode}) {url}: {sonuc.stderr.strip()} "
                     f"-- yarim dosya {gecici.name} saklandi, tekrar calistirinca surer")


def _dosya_adi(url: str) -> str:
    """.../files/w12_rgb.mp4/content ve .../resolve/main/x.json icin dosya adi."""
    parcalar = [p for p in url.split("?")[0].split("/") if p]
    return parcalar[-2] if parcalar[-1] == "content" else parcalar[-1]


def indir(ad: str, hedef_kok: Path = HEDEF, paralel: int = 1) -> Path:
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
    import threading
    from concurrent.futures import ThreadPoolExecutor
    kilit = threading.Lock()

    def tek(oge) -> None:
        url, ad_ = oge if isinstance(oge, tuple) else (oge, _dosya_adi(oge))
        dosya = dizin / ad_
        if dosya.name in kayit["dosyalar"] and dosya.exists():
            return
        _curl(url, dosya)
        bilgi = {"url": url, "sha256": _sha256(dosya), "bayt": dosya.stat().st_size,
                 "indirme_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with kilit:
            kayit["dosyalar"][dosya.name] = bilgi
            # Her dosyadan sonra yazilir: yarida kesilirse biten dosyalar kayitli kalir.
            kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # paralel > 1: tek sunucudan baglanti basina hiz sinirli oldugunda (Zenodo)
    # dosyalar ayni anda iner; her biri yine kaldigi yerden surer.
    with ThreadPoolExecutor(max_workers=max(1, paralel)) as havuz:
        for sonuc in havuz.map(tek, adresler):
            pass
    # Hic dosya inmese de yaz: eski bicimden donusum diske islensin.
    kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8")
    return dizin


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dis veri seti katalogu ve indirici")
    p.add_argument("--liste", action="store_true")
    p.add_argument("--indir", metavar="AD")
    p.add_argument("--indir-hepsi-dogrudan", action="store_true")
    p.add_argument("--en-cok-mb", type=float, default=1000.0)
    p.add_argument("--paralel", type=int, default=1, help="kaynak icinde es zamanli dosya")
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
        and (_dosya_adresi_mi(k["url"]) or k.get("dosyalar") or k.get("zenodo")
             or k.get("dataverse"))]
    for ad in adlar:
        try:
            print(f"indiriliyor: {ad} -> {indir(ad, paralel=a.paralel)}")
        except ValueError as exc:
            print(f"atlandi: {ad}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
