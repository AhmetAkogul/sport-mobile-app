import json
import urllib.request
import urllib.error
import time

BASE = "http://localhost:8080"


def istek(method, path, body=None, token=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def kontrol(aciklama, beklenen, method, path, body=None, token=None):
    kod, govde = istek(method, path, body, token)
    durum = "GECTI" if kod == beklenen else "BASARISIZ"
    print(f"[{durum}] {aciklama}")
    print(f"         {method} {path}  ->  {kod}   (beklenen {beklenen})")
    print(f"         {govde[:250]}")
    print()


email = "test" + str(int(time.time())) + "@example.com"
sifre = "parola123"

print("=" * 70)
print(" KURULUM: test kullanicisi olusturuluyor")
print("=" * 70)
istek("POST", "/api/auth/register",
      {"name": "Test", "email": email, "password": sifre, "gender": "MALE"})
kod, govde = istek("POST", "/api/auth/login", {"email": email, "password": sifre})
token = json.loads(govde)["token"]
print(f"kullanici: {email}")
print(f"token alindi (uzunluk {len(token)})\n")

print("=" * 70)
print(" A) EGZERSIZ KATALOGU")
print("=" * 70)
kontrol("Token'siz istek reddedilmeli", 401, "GET", "/api/exercises")
kontrol("Bozuk token reddedilmeli", 401, "GET", "/api/exercises", token="bozuktoken")
kontrol("Liste (seed yok -> bos dizi)", 200, "GET", "/api/exercises", token=token)
kontrol("Kas grubu filtresi", 200, "GET", "/api/exercises?muscleGroup=CHEST", token=token)
kontrol("Ekipman filtresi", 200, "GET", "/api/exercises?equipment=DUMBBELL", token=token)
kontrol("Iki filtre birlikte", 200, "GET",
        "/api/exercises?muscleGroup=CHEST&equipment=DUMBBELL", token=token)
kontrol("Gecersiz enum degeri", 400, "GET", "/api/exercises?muscleGroup=YOKBOYLE", token=token)
kontrol("Olmayan id", 404, "GET", "/api/exercises/99999", token=token)
kontrol("id sayi degil", 400, "GET", "/api/exercises/abc", token=token)

print("=" * 70)
print(" B) PROFIL (dun yaptigimiz)")
print("=" * 70)
kontrol("Kendi profilim", 200, "GET", "/api/users/me", token=token)
kontrol("Profil guncelle", 200, "PUT", "/api/users/me",
        {"name": "Yeni Ad", "age": 30, "gender": "FEMALE",
         "height": 165.25, "weight": 55.5}, token=token)
kontrol("Bos ad reddedilmeli", 400, "PUT", "/api/users/me",
        {"name": ""}, token=token)
kontrol("Yanlis mevcut sifre", 400, "PUT", "/api/users/me/password",
        {"currentPassword": "yanlis", "newPassword": "yeniparola1"}, token=token)
kontrol("Sifre degistir", 204, "PUT", "/api/users/me/password",
        {"currentPassword": sifre, "newPassword": "yeniparola1"}, token=token)
kontrol("Yeni sifreyle giris", 200, "POST", "/api/auth/login",
        {"email": email, "password": "yeniparola1"})
kontrol("Eski sifreyle giris reddedilmeli", 401, "POST", "/api/auth/login",
        {"email": email, "password": sifre})

print("=" * 70)
print(f" Test kullanicisi: {email}")
print("=" * 70)