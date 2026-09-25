import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8080"


def request(method, path, body=None):
    print("=" * 55)
    print(f"{method} {path}")
    print("GONDERILEN:", json.dumps(body, ensure_ascii=False) if body else "-")

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")

    try:
        r = urllib.request.urlopen(req)
        print("GELEN:", r.status, r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        print("GELEN:", e.code, e.read().decode("utf-8", "replace"))
    except urllib.error.URLError as e:
        print("BAGLANTI YOK:", e.reason)
        sys.exit(1)


request("POST", "/api/auth/register", {"name": "Ali", "email": "ali@test.com",
                                     "password": "parola1234", "age": 25, "gender": "MALE"})
request("POST", "/api/auth/register", {"name": "Ali", "email": "ali@test.com",
                                     "password": "parola1234"})
request("POST", "/api/auth/register", {"name": "", "email": "abc", "password": "123"})
request("GET", "/api/yok")