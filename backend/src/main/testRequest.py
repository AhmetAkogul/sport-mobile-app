# -*- coding: utf-8 -*-
r"""
Asama 7B - Istatistik endpointleri testi.

Calistirmadan once backend ayakta olmali:  .\mvnw.cmd spring-boot:run

NOT: 3. bolumde 65 saniye beklenir. Sebep: sure istatistigi DAKIKA bazinda
ve Java Duration.toMinutes() asagi yuvarlar; 4 saniyelik seans 0 dakika eder.
Gercek bir "1 dakika" kaniti icin gercekten 1 dakikadan fazla beklemek gerekir.
"""

import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TABAN = "http://localhost:8080"
GECEN = 0
KALAN = 0

EPOSTA = "istatistik@test.local"
SIFRE = "parola1234"
EPOSTA_2 = "istatistik2@test.local"
KILO = 80.0

# Java MetValues tablosunun bagimsiz kopyasi. Bilerek ayri yazildi:
# test, sunucunun tablosunu okumaz, kendi hesabini yapar.
MET_VARSAYILAN = {
    "CHEST": 5.0,
    "BACK": 5.0,
    "LEGS": 6.0,
    "SHOULDERS": 4.5,
    "ARMS": 3.5,
    "CORE": 3.5,
    "CARDIO": 8.0,
    "FULL_BODY": 6.5,
}


# --------------------------------------------------------------------------
# Yardimcilar
# --------------------------------------------------------------------------

def kontrol(ad, beklenen, gelen):
    global GECEN, KALAN
    if beklenen == gelen:
        GECEN += 1
        print("  [OK]   " + ad)
    else:
        KALAN += 1
        print("  [HATA] " + ad)
        print("         beklenen: " + repr(beklenen))
        print("         gelen   : " + repr(gelen))


def kontrol_yakin(ad, beklenen, gelen, tolerans=0.01):
    global GECEN, KALAN
    if gelen is not None and abs(float(gelen) - float(beklenen)) <= tolerans:
        GECEN += 1
        print("  [OK]   " + ad + " (" + str(gelen) + ")")
    else:
        KALAN += 1
        print("  [HATA] " + ad)
        print("         beklenen: " + repr(beklenen))
        print("         gelen   : " + repr(gelen))


def istek(metot, yol, govde=None, token=None):
    basliklar = {}
    veri = None
    if govde is not None:
        veri = json.dumps(govde).encode("utf-8")
        basliklar["Content-Type"] = "application/json"
    if token:
        basliklar["Authorization"] = "Bearer " + token

    istek_nesnesi = urllib.request.Request(TABAN + yol, data=veri,
                                           headers=basliklar, method=metot)
    try:
        with urllib.request.urlopen(istek_nesnesi) as cevap:
            ham = cevap.read().decode("utf-8")
            return cevap.status, (json.loads(ham) if ham else None)
    except urllib.error.HTTPError as hata:
        ham = hata.read().decode("utf-8")
        try:
            return hata.code, (json.loads(ham) if ham else None)
        except json.JSONDecodeError:
            return hata.code, ham


_KESIR = re.compile(r"\.(\d+)")


def ana_cevir(metin):
    """ISO-8601 metni datetime'a cevirir. Java 9 haneye kadar nanosaniye
    yazabilir; Python 3.7-3.10 fromisoformat en fazla 6 hane kabul eder."""
    temiz = metin.replace("Z", "+00:00")
    temiz = _KESIR.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), temiz, count=1)
    return datetime.datetime.fromisoformat(temiz)


def saniye_farki(baslangic, bitis):
    """Java Duration.getSeconds() ile ayni: asagi yuvarlanmis tam saniye."""
    fark = ana_cevir(bitis) - ana_cevir(baslangic)
    return fark.days * 86400 + fark.seconds


def kontrol_an(ad, beklenen, gelen, tolerans_ms=1):
    """timestamptz mikrosaniye saklar, Java Instant nanosaniye uretir.
    Ayni an, farkli metin. Toleransla karsilastirilir."""
    global GECEN, KALAN
    if beklenen is None or gelen is None:
        kontrol(ad, beklenen, gelen)
        return
    fark = abs((ana_cevir(beklenen) - ana_cevir(gelen)).total_seconds())
    if fark < tolerans_ms / 1000.0:
        GECEN += 1
        print("  [OK]   " + ad)
    else:
        KALAN += 1
        print("  [HATA] " + ad)
        print("         beklenen: " + repr(beklenen))
        print("         gelen   : " + repr(gelen))


def beklenen_kcal(kas_gruplari, baslangic, bitis, kilo):
    """CalorieCalculator.calculate()'in bagimsiz tekrar yazimi."""
    saniye = saniye_farki(baslangic, bitis)
    if kilo is None or saniye <= 0 or not kas_gruplari:
        return None
    ortalama_met = sum(MET_VARSAYILAN[g] for g in kas_gruplari) / len(kas_gruplari)
    kcal = ortalama_met * 3.5 * kilo / 200.0 * (saniye / 60.0)
    return Decimal(repr(kcal)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def ondalik(deger):
    """JSON'dan gelen sayiyi karsilastirilabilir Decimal'e cevirir."""
    return None if deger is None else Decimal(str(deger))


def token_al(eposta, sifre, ad="Istatistik Test"):
    istek("POST", "/api/auth/register", {
        "name": ad,
        "email": eposta,
        "password": sifre,
        "weight": KILO,
    })
    durum, govde = istek("POST", "/api/auth/login", {"email": eposta, "password": sifre})
    if durum != 200:
        raise SystemExit("Giris yapilamadi (" + str(durum) + "): " + str(govde))
    token = govde["token"]
    # Onceki kosudan kalan kilo farkli olabilir; testin sabit olmasi icin esitle.
    istek("PUT", "/api/users/me", {"name": ad, "weight": KILO}, token)
    return token


def seanslari_sil(token):
    durum, liste = istek("GET", "/api/workout-sessions", None, token)
    if durum != 200 or not liste:
        return
    for seans in liste:
        istek("DELETE", "/api/workout-sessions/" + str(seans["id"]), None, token)


# --------------------------------------------------------------------------
# 0. Hazirlik
# --------------------------------------------------------------------------

print("\n=== 0. HAZIRLIK ===")
token = token_al(EPOSTA, SIFRE)
print("  token alindi, kullanici hazir (kilo=" + str(KILO) + ")")

seanslari_sil(token)
print("  eski seanslar silindi")

durum, egzersizler = istek("GET", "/api/exercises", None, token)
if durum != 200:
    raise SystemExit("Egzersiz katalogu okunamadi: " + str(durum))
grup_egzersiz = {}
for egzersiz in egzersizler:
    grup_egzersiz.setdefault(egzersiz["muscleGroup"], egzersiz["id"])
eksik = [g for g in ("CHEST", "LEGS", "CORE") if g not in grup_egzersiz]
if eksik:
    raise SystemExit("Katalogda su kas gruplarinda egzersiz yok: " + ", ".join(eksik))
print("  egzersiz katalogu hazir: " + str(grup_egzersiz))


# --------------------------------------------------------------------------
# 1. Bos kullanici: her sey sifir
# --------------------------------------------------------------------------

print("\n=== 1. BOS KULLANICI: TUM ISTATISTIKLER SIFIR ===")

durum, genel = istek("GET", "/api/statistics/overview", None, token)
kontrol("overview 200", 200, durum)
kontrol("finishedSessions", 0, genel["finishedSessions"])
kontrol("totalDurationMinutes", 0, genel["totalDurationMinutes"])
kontrol("totalCalories", Decimal("0"), ondalik(genel["totalCalories"]))
kontrol("totalSets", 0, genel["totalSets"])
kontrol("totalReps", 0, genel["totalReps"])
kontrol("totalVolumeKg", Decimal("0"), ondalik(genel["totalVolumeKg"]))
kontrol("firstWorkoutAt null", None, genel["firstWorkoutAt"])
kontrol("lastWorkoutAt null", None, genel["lastWorkoutAt"])

durum, kaslar = istek("GET", "/api/statistics/muscle-groups", None, token)
kontrol("muscle-groups 200", 200, durum)
kontrol("muscle-groups bos liste", [], kaslar)

durum, dogruluk = istek("GET", "/api/statistics/accuracy", None, token)
kontrol("accuracy 200", 200, durum)
kontrol("accuracy totalReps", 0, dogruluk["totalReps"])
kontrol("accuracy trackedReps", 0, dogruluk["trackedReps"])
kontrol("accuracy correctReps", 0, dogruluk["correctReps"])
kontrol("accuracy incorrectReps", 0, dogruluk["incorrectReps"])
# KRITIK: AI verisi yokken 0 degil null. "%0 dogru" demek yaniltirdi.
kontrol("accuracyPercent null (0 degil)", None, dogruluk["accuracyPercent"])


# --------------------------------------------------------------------------
# 2. Bitmemis seans istatistige girmez
# --------------------------------------------------------------------------

print("\n=== 2. ACILAN AMA BITIRILMEYEN SEANS ISTATISTIGE GIRMEZ ===")

durum, seans_a = istek("POST", "/api/workout-sessions", {}, token)
kontrol("seans A acildi (201)", 201, durum)
sid_a = seans_a["id"]

durum, govde = istek("POST", "/api/workout-sessions/" + str(sid_a) + "/exercises",
                     {"exerciseId": grup_egzersiz["CHEST"], "position": 1,
                      "targetSets": 3, "targetReps": 10, "targetRestSeconds": 60}, token)
kontrol("CHEST eklendi (201)", 201, durum)

durum, govde = istek("POST", "/api/workout-sessions/" + str(sid_a) + "/exercises",
                     {"exerciseId": grup_egzersiz["LEGS"], "position": 2,
                      "targetSets": 3, "targetReps": 12, "targetRestSeconds": 90}, token)
kontrol("LEGS eklendi (201)", 201, durum)

se_id = {}
for e in govde["exercises"]:
    se_id[e["muscleGroup"]] = e["id"]

durum, genel = istek("GET", "/api/statistics/overview", None, token)
kontrol("overview hala 0 seans", 0, genel["finishedSessions"])
kontrol("overview hala 0 set", 0, genel["totalSets"])

durum, kaslar = istek("GET", "/api/statistics/muscle-groups", None, token)
kontrol("kas grubu hala bos", [], kaslar)


# --------------------------------------------------------------------------
# 3. Seans A bitirilir (65 saniye beklenir)
# --------------------------------------------------------------------------

print("\n=== 3. SEANS A BITIRILIR ===")
print("  ... 65 saniye bekleniyor (sure istatistigi dakika bazinda)")

time.sleep(65)

sonuc_a = {
    "note": "istatistik testi A",
    "exercises": [
        {"sessionExerciseId": se_id["CHEST"], "sets": [
            {"setNumber": 1, "reps": 10, "weight": 60.0, "incorrectReps": 1},
            {"setNumber": 2, "reps": 10, "weight": 60.0, "incorrectReps": 0},
            {"setNumber": 3, "reps": 8, "weight": 70.0, "incorrectReps": 2},
        ]},
        {"sessionExerciseId": se_id["LEGS"], "sets": [
            {"setNumber": 1, "reps": 12, "weight": 40.0},
            {"setNumber": 2, "reps": 12, "weight": 40.0},
            {"setNumber": 3, "reps": 10, "weight": 50.0, "incorrectReps": 1},
        ]},
    ],
}

durum, bitmis_a = istek("PUT", "/api/workout-sessions/" + str(sid_a) + "/finish", sonuc_a, token)
kontrol("seans A finish 200", 200, durum)
kontrol("seans A bitis saati dolu", True, bitmis_a["finishedAt"] is not None)

# Beklenen degerler (elle hesaplandi):
#   set       : 3 + 3                       = 6
#   tekrar    : 10+10+8 + 12+12+10          = 62
#   hacim     : 10*60+10*60+8*70+12*40+12*40+10*50 = 3220
#   AI'li     : 10+10+8 + 10 (incorrectReps dolu olan setler) = 38
#   yanlis    : 1+0+2+1                     = 4
#   dogru     : 38-4                        = 34  ->  %89.5
A_SETS = 6
A_REPS = 62
A_HACIM = 3220.0
A_TRACKED = 38
A_INCORRECT = 4
A_DOGRULUK = Decimal("89.5")

kontrol("seans A suresi (sn)", True, saniye_farki(bitmis_a["startedAt"], bitmis_a["finishedAt"]) >= 65)
kontrol("seans A kalori (bagimsiz formul)",
        beklenen_kcal(["CHEST", "LEGS"], bitmis_a["startedAt"], bitmis_a["finishedAt"], KILO),
        ondalik(bitmis_a["calories"]))


# --------------------------------------------------------------------------
# 4. Overview
# --------------------------------------------------------------------------

print("\n=== 4. OVERVIEW ===")

durum, genel = istek("GET", "/api/statistics/overview", None, token)
kontrol("overview 200", 200, durum)
kontrol("finishedSessions", 1, genel["finishedSessions"])
kontrol("totalDurationMinutes (65 sn -> 1 dk)", 1, genel["totalDurationMinutes"])
kontrol("totalSets", A_SETS, genel["totalSets"])
kontrol("totalReps", A_REPS, genel["totalReps"])
kontrol_yakin("totalVolumeKg", A_HACIM, genel["totalVolumeKg"])
kontrol("totalCalories = seans A kalorisi",
        ondalik(bitmis_a["calories"]), ondalik(genel["totalCalories"]))
kontrol_an("firstWorkoutAt = seans A baslangici", bitmis_a["startedAt"], genel["firstWorkoutAt"])
kontrol_an("lastWorkoutAt = seans A bitisi", bitmis_a["finishedAt"], genel["lastWorkoutAt"])


# --------------------------------------------------------------------------
# 5. Kas grubu dagilimi
# --------------------------------------------------------------------------

print("\n=== 5. KAS GRUBU DAGILIMI ===")

durum, kaslar = istek("GET", "/api/statistics/muscle-groups", None, token)
kontrol("muscle-groups 200", 200, durum)
# Siralama COUNT esitliginde tanimsiz; testte kendimiz siraliyoruz.
kaslar = sorted(kaslar, key=lambda k: k["muscleGroup"])
kontrol("kas grubu sayisi", 2, len(kaslar))
kontrol("kas gruplari", ["CHEST", "LEGS"], [k["muscleGroup"] for k in kaslar])

kontrol("CHEST sets", 3, kaslar[0]["sets"])
kontrol("CHEST reps", 28, kaslar[0]["reps"])
kontrol_yakin("CHEST volumeKg (600+600+560)", 1760.0, kaslar[0]["volumeKg"])

kontrol("LEGS sets", 3, kaslar[1]["sets"])
kontrol("LEGS reps", 34, kaslar[1]["reps"])
kontrol_yakin("LEGS volumeKg (480+480+500)", 1460.0, kaslar[1]["volumeKg"])


# --------------------------------------------------------------------------
# 6. AI dogrulugu
# --------------------------------------------------------------------------

print("\n=== 6. AI DOGRULUGU ===")

durum, dogruluk = istek("GET", "/api/statistics/accuracy", None, token)
kontrol("accuracy 200", 200, durum)
kontrol("totalReps", A_REPS, dogruluk["totalReps"])
kontrol("trackedReps", A_TRACKED, dogruluk["trackedReps"])
kontrol("correctReps", A_TRACKED - A_INCORRECT, dogruluk["correctReps"])
kontrol("incorrectReps", A_INCORRECT, dogruluk["incorrectReps"])
kontrol("accuracyPercent", A_DOGRULUK, ondalik(dogruluk["accuracyPercent"]))
# Squat'in ilk 2 seti AI'siz: 24 tekrar orana katilmamali.
kontrol("AI'siz tekrar orana girmiyor", A_REPS - A_TRACKED,
        dogruluk["totalReps"] - dogruluk["trackedReps"])
# incorrectReps=0 olan set AI'li sayilir (null ile karistirilmamali).
kontrol("incorrectReps=0 AI'li sayilir", True, dogruluk["trackedReps"] > 24)


# --------------------------------------------------------------------------
# 7. Kamerasiz seans: oran ve hacim bozulmaz
# --------------------------------------------------------------------------

print("\n=== 7. KAMERASIZ 2. SEANS: DOGRULUK ORANI VE HACIM BOZULMAZ ===")

durum, seans_b = istek("POST", "/api/workout-sessions", {}, token)
kontrol("seans B acildi (201)", 201, durum)
sid_b = seans_b["id"]

durum, govde = istek("POST", "/api/workout-sessions/" + str(sid_b) + "/exercises",
                     {"exerciseId": grup_egzersiz["CORE"], "position": 1,
                      "targetSets": 2, "targetReps": 30, "targetRestSeconds": 45}, token)
kontrol("CORE eklendi (201)", 201, durum)
se_core = govde["exercises"][0]["id"]

time.sleep(3)

sonuc_b = {
    "note": "istatistik testi B",
    "exercises": [
        {"sessionExerciseId": se_core, "sets": [
            {"setNumber": 1, "reps": 30},
            {"setNumber": 2, "reps": 30},
        ]},
    ],
}

durum, bitmis_b = istek("PUT", "/api/workout-sessions/" + str(sid_b) + "/finish", sonuc_b, token)
kontrol("seans B finish 200", 200, durum)
# Plank vucut agirligi: weight gonderilmedi, DB'de NULL.
kontrol("vucut agirligi setlerde weight null",
        [None, None], [s["weight"] for s in bitmis_b["exercises"][0]["sets"]])

durum, dogruluk2 = istek("GET", "/api/statistics/accuracy", None, token)
kontrol("totalReps 62+60", A_REPS + 60, dogruluk2["totalReps"])
kontrol("trackedReps degismedi (AI'siz seans katilmaz)", A_TRACKED, dogruluk2["trackedReps"])
kontrol("accuracyPercent degismedi", A_DOGRULUK, ondalik(dogruluk2["accuracyPercent"]))

durum, genel2 = istek("GET", "/api/statistics/overview", None, token)
kontrol("finishedSessions 2", 2, genel2["finishedSessions"])
kontrol("totalSets 6+2", 8, genel2["totalSets"])
kontrol("totalReps 62+60", A_REPS + 60, genel2["totalReps"])
# NULL * reps = NULL, SUM NULL'i atlar: hacim sadece agirlikli setlerden gelir.
kontrol_yakin("totalVolumeKg degismedi (vucut agirligi hacme girmez)",
              A_HACIM, genel2["totalVolumeKg"])
kontrol("totalDurationMinutes 1+0", 1, genel2["totalDurationMinutes"])
kontrol("toplam kalori = A + B",
        ondalik(bitmis_a["calories"]) + ondalik(bitmis_b["calories"]),
        ondalik(genel2["totalCalories"]))
kontrol_an("firstWorkoutAt hala seans A", bitmis_a["startedAt"], genel2["firstWorkoutAt"])
kontrol_an("lastWorkoutAt artik seans B", bitmis_b["finishedAt"], genel2["lastWorkoutAt"])

durum, kaslar2 = istek("GET", "/api/statistics/muscle-groups", None, token)
kaslar2 = sorted(kaslar2, key=lambda k: k["muscleGroup"])
kontrol("kas grubu sayisi 3", 3, len(kaslar2))
kontrol("kas gruplari", ["CHEST", "CORE", "LEGS"], [k["muscleGroup"] for k in kaslar2])
kontrol("CORE sets", 2, kaslar2[1]["sets"])
kontrol("CORE reps", 60, kaslar2[1]["reps"])
kontrol_yakin("CORE volumeKg 0 (vucut agirligi)", 0.0, kaslar2[1]["volumeKg"])


# --------------------------------------------------------------------------
# 8. Haftalik
# --------------------------------------------------------------------------

print("\n=== 8. HAFTALIK KOVALAR ===")

durum, haftalik = istek("GET", "/api/statistics/weekly?weeks=4", None, token)
kontrol("weekly 200", 200, durum)
kontrol("4 kova dondu", 4, len(haftalik))

tarihler = [datetime.date.fromisoformat(k["weekStart"]) for k in haftalik]
kontrol("hafta baslangiclari 7 gun arayla artiyor",
        [7, 7, 7], [(tarihler[i + 1] - tarihler[i]).days for i in range(3)])
kontrol("hepsi pazartesi", [0, 0, 0, 0], [t.weekday() for t in tarihler])

bugun_utc = datetime.datetime.now(datetime.timezone.utc).date()
bu_hafta = bugun_utc - datetime.timedelta(days=bugun_utc.weekday())
kontrol("son kova = bu haftanin pazartesisi", bu_hafta.isoformat(), haftalik[3]["weekStart"])

# Bos haftalar sifirla donmeli, grafikte bosluk olmamali.
kontrol("bos hafta 1: 0 seans", 0, haftalik[0]["sessions"])
kontrol("bos hafta 2: 0 seans", 0, haftalik[1]["sessions"])
kontrol("bos hafta 3: 0 seans", 0, haftalik[2]["sessions"])
kontrol("bos haftalar kalori 0", [Decimal("0"), Decimal("0"), Decimal("0")],
        [ondalik(haftalik[i]["calories"]) for i in range(3)])
kontrol("bu hafta 2 seans", 2, haftalik[3]["sessions"])
kontrol("bu hafta 1 dakika", 1, haftalik[3]["durationMinutes"])
kontrol("bu hafta kalori = A + B",
        ondalik(bitmis_a["calories"]) + ondalik(bitmis_b["calories"]),
        ondalik(haftalik[3]["calories"]))

durum, tek = istek("GET", "/api/statistics/weekly?weeks=1", None, token)
kontrol("weeks=1 tek kova", 1, len(tek))
kontrol("weeks=1 seans sayisi 2", 2, tek[0]["sessions"])

durum, haftalik8 = istek("GET", "/api/statistics/weekly?weeks=8", None, token)
kontrol("weeks=8 sekiz kova", 8, len(haftalik8))
kontrol("weeks=8 ilk 7 kova bos", [0] * 7, [k["sessions"] for k in haftalik8[:7]])
kontrol("weeks=8 son kova 2 seans", 2, haftalik8[7]["sessions"])

durum, _ = istek("GET", "/api/statistics/weekly?weeks=0", None, token)
kontrol("weeks=0 -> 400", 400, durum)
durum, _ = istek("GET", "/api/statistics/weekly?weeks=53", None, token)
kontrol("weeks=53 -> 400", 400, durum)
durum, _ = istek("GET", "/api/statistics/weekly?weeks=abc", None, token)
kontrol("weeks=abc -> 400", 400, durum)
durum, _ = istek("GET", "/api/statistics/weekly?zone=Yok/Boyle", None, token)
kontrol("gecersiz zone -> 400", 400, durum)

durum, istanbul = istek("GET", "/api/statistics/weekly?weeks=4&zone=Europe/Istanbul", None, token)
kontrol("Europe/Istanbul 200", 200, durum)
kontrol("Europe/Istanbul 4 kova", 4, len(istanbul))
kontrol("Europe/Istanbul son kova 2 seans", 2, istanbul[3]["sessions"])


# --------------------------------------------------------------------------
# 9. Kullanici izolasyonu
# --------------------------------------------------------------------------

print("\n=== 9. KULLANICI IZOLASYONU ===")

token2 = token_al(EPOSTA_2, SIFRE, ad="Istatistik Test 2")
durum, genel_baska = istek("GET", "/api/statistics/overview", None, token2)
kontrol("2. kullanici overview 200", 200, durum)
kontrol("2. kullanicida 0 seans", 0, genel_baska["finishedSessions"])
kontrol("2. kullanicida 0 set", 0, genel_baska["totalSets"])
kontrol("2. kullanicida 0 kalori", Decimal("0"), ondalik(genel_baska["totalCalories"]))

durum, kaslar_baska = istek("GET", "/api/statistics/muscle-groups", None, token2)
kontrol("2. kullanicida kas grubu bos", [], kaslar_baska)

durum, dogruluk_baska = istek("GET", "/api/statistics/accuracy", None, token2)
kontrol("2. kullanicida accuracy 0", 0, dogruluk_baska["totalReps"])
kontrol("2. kullanicida accuracyPercent null", None, dogruluk_baska["accuracyPercent"])


# --------------------------------------------------------------------------
# 10. Seans silinince istatistikten duser
# --------------------------------------------------------------------------

print("\n=== 10. SEANS SILININCE ISTATISTIKTEN DUSER ===")

durum, _ = istek("DELETE", "/api/workout-sessions/" + str(sid_a), None, token)
kontrol("seans A silindi (204)", 204, durum)

durum, genel3 = istek("GET", "/api/statistics/overview", None, token)
kontrol("finishedSessions 1", 1, genel3["finishedSessions"])
kontrol("totalSets 2", 2, genel3["totalSets"])
kontrol("totalReps 60", 60, genel3["totalReps"])
kontrol_yakin("totalVolumeKg 0 (sadece vucut agirligi kaldi)", 0.0, genel3["totalVolumeKg"])
kontrol("totalCalories = sadece B", ondalik(bitmis_b["calories"]), ondalik(genel3["totalCalories"]))
kontrol_an("firstWorkoutAt artik seans B", bitmis_b["startedAt"], genel3["firstWorkoutAt"])

durum, dogruluk3 = istek("GET", "/api/statistics/accuracy", None, token)
kontrol("totalReps 60", 60, dogruluk3["totalReps"])
kontrol("trackedReps 0", 0, dogruluk3["trackedReps"])
kontrol("correctReps 0", 0, dogruluk3["correctReps"])
kontrol("incorrectReps 0", 0, dogruluk3["incorrectReps"])
# KRITIK: AI'li seans silindi -> oran null. "%0 dogru" DEGIL.
kontrol("accuracyPercent null (AI verisi kalmadi)", None, dogruluk3["accuracyPercent"])

durum, kaslar3 = istek("GET", "/api/statistics/muscle-groups", None, token)
kontrol("tek kas grubu kaldi", ["CORE"], [k["muscleGroup"] for k in kaslar3])


# --------------------------------------------------------------------------
# Sonuc
# --------------------------------------------------------------------------

print("\n=== SONUC: " + str(GECEN) + " GECEN / " + str(KALAN) + " KALAN ===")
