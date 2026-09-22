#!/usr/bin/env python3
"""Ortam dogrulama: projenin calismasi icin gereken her sey yerinde mi?

Faz 0 kontrol listesi maddesi. Yeni katilan herkes once bunu calistirir.
Cikis kodu 0 = her sey tamam, 1 = eksik var.
"""
import sys

SORUNLAR = []


def kontrol(ad, kosul, cozum):
    isaret = "OK " if kosul else "!! "
    print(f"{isaret}{ad}")
    if not kosul:
        SORUNLAR.append((ad, cozum))
    return kosul


def main():
    print("=" * 60)
    print("ORTAM KONTROLU")
    print("=" * 60)

    kontrol(
        f"Python {sys.version_info.major}.{sys.version_info.minor}",
        sys.version_info >= (3, 10),
        "Python 3.10 veya uzeri gerekli",
    )

    try:
        import cv2
        surum = cv2.__version__
        print(f"   OpenCV surumu: {surum}")
        kontrol(
            "OpenCV 5.x",
            surum.startswith("5."),
            "pip install opencv-python==5.0.0.93",
        )
        kontrol(
            "calibrateMultiview erisilebilir",
            hasattr(cv2, "calibrateMultiview"),
            "OpenCV 5 gerekli; 4.x bu fonksiyonu icermez",
        )
        kontrol(
            "ArUco/ChArUco modulu",
            hasattr(cv2, "aruco"),
            "opencv-python paketinde aruco bulunmali",
        )
    except ImportError:
        kontrol("OpenCV kurulu", False, "make kurulum")

    for paket in ("numpy", "scipy", "yaml"):
        try:
            __import__(paket)
            kontrol(f"{paket}", True, "")
        except ImportError:
            kontrol(f"{paket}", False, "make kurulum")

    print("=" * 60)
    if SORUNLAR:
        print(f"{len(SORUNLAR)} SORUN VAR:\n")
        for ad, cozum in SORUNLAR:
            print(f"  - {ad}\n      cozum: {cozum}")
        return 1
    print("Her sey yerinde. Calismaya baslayabilirsiniz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
