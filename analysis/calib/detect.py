"""ChArUco kose tespiti ve calibrateMultiview icin detectionMask uretimi.

Bu modulun asil isi kose bulmak degil, **detectionMask'i dogru uretmek.**
`calibrateMultiview(objPoints, imagePoints, imageSize, detectionMask, ...)` cagrisinda
mask, hangi kameranin hangi karede board'u gordugunu bildirir. Yanlis mask hata
vermez; sessizce bozuk kalibrasyon uretir. Bu yuzden:

- Bir tespit ancak `min_kose` esigini gecerse "gorulmus" sayilir. Uc kose goren bir
  kamerayi gormus saymak, gormemis saymaktan daha zararlidir.
- Mask ile imagePoints her zaman birlikte uretilir, ayri ayri degil (`hazirla`).

Kullanim:
    python -m calib.detect --dizin data/isik-testi
    python -m calib.detect --dizin data/oturum-01 --min-kose 8 --rapor out/007.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calib.board import BoardSpec, board_kur

# Bir karenin kalibrasyona katkida bulunmasi icin gereken en az ic kose sayisi.
# Homografi 4 nokta ile cikar ama 4 noktali kare gurultuye asiri duyarli olur;
# 6 pratik bir alt sinir. Isik testinde bu esigin etkisi olculebilir.
VARSAYILAN_MIN_KOSE = 6

RESIM_UZANTILARI = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class Tespit:
    """Tek bir karedeki tespit sonucu.

    `koseler` her zaman (N, 1, 2) float32 duzenindedir. OpenCV 5'in detectBoard'u
    koseleri (N, 2) olarak donduruyor, OpenCV 4 ise (N, 1, 2) donduruyordu; bu
    modul sinirda normalize eder ki ust katmanlar surum farkina bagli kalmasin.
    """

    koseler: np.ndarray   # (N, 1, 2) float32 -- goruntu koordinati
    idler: np.ndarray     # (N,) int32 -- board uzerindeki kose kimligi
    marker_sayisi: int

    @property
    def kose_sayisi(self) -> int:
        return len(self.idler)


def dedektor_kur(spec: BoardSpec) -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(board_kur(spec))


def kose_bul(
    goruntu: np.ndarray,
    dedektor: cv2.aruco.CharucoDetector,
    min_kose: int = VARSAYILAN_MIN_KOSE,
) -> Tespit | None:
    """Karede board kosesi ara. Esigi gecmezse None -- "gormedi" demektir."""
    # Poz icin en az 4 kose (homografi) gerekir; 0 ya da negatif esik her
    # tespiti gecirirdi (dis inceleme B.2.1).
    if type(min_kose) is not int or min_kose < 4:
        raise ValueError(f"min_kose en az 4 olan tamsayi olmali, {min_kose!r} geldi")
    if goruntu.ndim == 3:
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    koseler, idler, _, marker_idler = dedektor.detectBoard(goruntu)
    if koseler is None or idler is None or len(idler) < min_kose:
        return None
    return Tespit(
        # OpenCV surumune gore (N,2) veya (N,1,2) gelebilir -> tek duzene indir
        koseler=np.asarray(koseler, dtype=np.float32).reshape(-1, 1, 2),
        idler=np.asarray(idler, dtype=np.int32).ravel(),
        marker_sayisi=0 if marker_idler is None else len(marker_idler),
    )


def hazirla(
    tespitler: list[list[Tespit | None]],
    spec: BoardSpec,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], np.ndarray]:
    """Kamera x kare tespitlerini calibrateMultiview girdilerine cevir.

    Girdi: `tespitler[kamera][kare]` -- gormediyse None.
    Cikti: (objPoints, imagePoints, detectionMask)

    objPoints board'un 3B model noktalari; her kare icin **o karede gorulen kose
    kimliklerine** gore suzulur, boylece imagePoints ile birebir eslesir. Kamera
    sayisi ve kare sayisi tum kameralarda ayni olmak zorunda -- degilse mask
    anlamsizlasir.
    """
    if not tespitler:
        raise ValueError("tespit listesi bos")
    kare_sayilari = {len(k) for k in tespitler}
    if len(kare_sayilari) != 1:
        raise ValueError(f"kameralarda kare sayisi farkli: {sorted(kare_sayilari)}")

    n_kamera = len(tespitler)
    n_kare = kare_sayilari.pop()
    if n_kare == 0:
        # Bos sahne tum denetimlerden tutarli gecip OpenCV'de anlasilmaz bir
        # hataya donusuyordu (dis inceleme B.2.3).
        raise ValueError("hic kare yok: kalibrasyon icin en az bir kare gerekli")
    model = board_kur(spec).getChessboardCorners()  # (K, 3) float32, metre

    # OpenCV detectionMask: (kamera x kare), CV_8UC1. 1 = o kamera o karede gordu.
    mask = np.zeros((n_kamera, n_kare), dtype=np.uint8)
    obj_noktalari: list[np.ndarray] = []
    img_noktalari: list[list[np.ndarray]] = [[] for _ in range(n_kamera)]

    for kare in range(n_kare):
        # Bir kare ancak en az iki kamera gorduyse coklu-gorus kalibrasyona katkida
        # bulunur; tek kamerali kare sadece o kameranin ic parametrelerini besler.
        gorenler = [k for k in range(n_kamera) if tespitler[k][kare] is not None]
        if not gorenler:
            continue
        # Kareyi tum gorenlerde ortak olan kose kimliklerine indir: objPoints tek
        # bir liste oldugu icin kare basina tek bir nokta kumesi olmak zorunda.
        ortak = set(tespitler[gorenler[0]][kare].idler.tolist())
        for k in gorenler[1:]:
            ortak &= set(tespitler[k][kare].idler.tolist())
        ortak_sirali = np.array(sorted(ortak), dtype=np.int32)
        if len(ortak_sirali) < VARSAYILAN_MIN_KOSE:
            continue

        obj_noktalari.append(model[ortak_sirali].astype(np.float32))
        for k in range(n_kamera):
            t = tespitler[k][kare]
            if t is None:
                img_noktalari[k].append(np.empty((0, 1, 2), dtype=np.float32))
                continue
            sira = {int(i): j for j, i in enumerate(t.idler)}
            secim = [sira[int(i)] for i in ortak_sirali]
            img_noktalari[k].append(t.koseler[secim].astype(np.float32))
            mask[k, len(obj_noktalari) - 1] = 1

    return obj_noktalari, img_noktalari, mask[:, : len(obj_noktalari)]


def dizin_tara(
    dizin: Path,
    spec: BoardSpec,
    min_kose: int = VARSAYILAN_MIN_KOSE,
) -> dict:
    """Bir dizindeki tum resimlerde tespit orani olc. Isik testi raporunun kaynagi."""
    dedektor = dedektor_kur(spec)
    dosyalar = sorted(p for p in dizin.iterdir() if p.suffix.lower() in RESIM_UZANTILARI)
    if not dosyalar:
        raise FileNotFoundError(f"{dizin} icinde resim yok")

    satirlar, bulunan = [], 0
    for p in dosyalar:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            satirlar.append({"dosya": p.name, "hata": "okunamadi"})
            continue
        t = kose_bul(g, dedektor, min_kose)
        bulunan += t is not None
        satirlar.append({
            "dosya": p.name,
            "kose": 0 if t is None else t.kose_sayisi,
            "marker": 0 if t is None else t.marker_sayisi,
            "gordu": t is not None,
        })

    return {
        "dizin": str(dizin),
        "spec": {"kare": [spec.squares_x, spec.squares_y], "kenar_mm": spec.square_mm,
                 "sozluk": spec.dictionary, "ic_kose": spec.inner_corners},
        "min_kose": min_kose,
        "kare_sayisi": len(dosyalar),
        "tespit_edilen": bulunan,
        "tespit_orani": round(bulunan / len(dosyalar), 4),
        "kareler": satirlar,
    }


def _cli() -> int:
    ap = argparse.ArgumentParser(description="ChArUco tespit orani olc")
    ap.add_argument("--dizin", type=Path, required=True)
    ap.add_argument("--kare", nargs=2, type=int, metavar=("X", "Y"), default=[5, 7])
    ap.add_argument("--kenar", type=float, default=60.0)
    ap.add_argument("--marker", type=float, default=45.0)
    ap.add_argument("--min-kose", type=int, default=VARSAYILAN_MIN_KOSE)
    ap.add_argument("--rapor", type=Path, help="sonucu JSON olarak yaz")
    a = ap.parse_args()

    spec = BoardSpec(a.kare[0], a.kare[1], a.kenar, a.marker)
    sonuc = dizin_tara(a.dizin, spec, a.min_kose)

    print(f"dizin        : {sonuc['dizin']}")
    print(f"kare sayisi  : {sonuc['kare_sayisi']}")
    print(f"tespit edilen: {sonuc['tespit_edilen']}")
    print(f"tespit orani : {sonuc['tespit_orani']:.1%}")
    print(f"beklenen kose: {spec.inner_corners}  (esik: {a.min_kose})")
    if a.rapor:
        a.rapor.parent.mkdir(parents=True, exist_ok=True)
        a.rapor.write_text(json.dumps(sonuc, indent=2, ensure_ascii=False))
        print(f"rapor        : {a.rapor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
