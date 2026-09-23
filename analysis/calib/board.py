"""ChArUco board uretimi ve baskiya hazir PDF cikisi.

Kalibrasyonun fiziksel referansi bu board. Iki sey kritik:

1. **Olcek.** Board kagida yazdirilirken yaziciya "sigdir/scale to fit" denirse kare
   kenari degisir ve kalibrasyon sessizce bozulur -- hata mesaji almazsiniz, sadece
   yanlis mm sonuclari alirsiniz. Uretilen PDF tam fiziksel olcekte cizilir ve altina
   olculmesi gereken kare kenari yazilir. Basilan board'un kare kenarini cetvelle
   dogrulamak zorunludur.
2. **Duzluk.** PROJE-PLANI.md: kagida basilip duvara bantlanan board kalibrasyonu
   bozar. Baski aluminyum kompozit gibi duz bir yuzeye monte edilmelidir.

Kullanim:
    python -m calib.board --cikti out/board.pdf
    python -m calib.board --kare 5 7 --kenar 60 --marker 45 --cikti out/board.pdf
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

# Varsayilan sozluk: DICT_4X4_50.
# 4x4, 5x5/6x6'ya gore daha az bit tasir; bu da uzak mesafede ve dusuk cozunurlukte
# daha guvenilir tespit demek. Coklu kamera odasinda kameralar metrelerce uzakta
# oldugu icin mesafe dayanikliligi, sozluk buyuklugunden onemli.
VARSAYILAN_SOZLUK = "DICT_4X4_50"


@dataclass(frozen=True)
class BoardSpec:
    """Board'un fiziksel tanimi. Kalibrasyon sonuclarinin birimi buradan gelir."""

    squares_x: int = 5           # kolon sayisi
    squares_y: int = 7           # satir sayisi
    square_mm: float = 60.0      # satranc karesi kenari (mm)
    marker_mm: float = 45.0      # ArUco marker kenari (mm)
    dictionary: str = VARSAYILAN_SOZLUK

    def __post_init__(self) -> None:
        # Pozitiflik once: square_mm=0, marker_mm=-5 onceden gecip goruntu
        # uretiminde negatif piksel boyutuna donusuyordu (dis inceleme B.1.1).
        for ad, deger in (("square_mm", self.square_mm), ("marker_mm", self.marker_mm)):
            if not np.isfinite(deger) or deger <= 0:
                raise ValueError(f"{ad} sonlu ve pozitif olmali, {deger} geldi")
        if self.marker_mm >= self.square_mm:
            raise ValueError("marker kenari kare kenarindan kucuk olmali")
        if min(self.squares_x, self.squares_y) < 3:
            raise ValueError("her iki yonde en az 3 kare gerekli")

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def inner_corners(self) -> int:
        """calibrateMultiview'e gidecek nokta sayisi."""
        return (self.squares_x - 1) * (self.squares_y - 1)


def sozluk_al(ad: str) -> cv2.aruco.Dictionary:
    # `hasattr` tek basina yetmez: "CharucoBoard" gibi bir sinif adi da gecerdi
    # ve OpenCV'den anlasilmaz bir hata gelirdi (dis inceleme B.1.2).
    if not (isinstance(ad, str) and ad.startswith("DICT_")
            and isinstance(getattr(cv2.aruco, ad, None), int)):
        raise ValueError(f"bilinmeyen ArUco sozlugu: {ad}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ad))


def board_kur(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    """BoardSpec -> OpenCV CharucoBoard.

    OpenCV metre bekliyor, biz mm ile konusuyoruz; donusum tek yerde, burada.
    """
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_mm / 1000.0,
        spec.marker_mm / 1000.0,
        sozluk_al(spec.dictionary),
    )


def _dpi_dogrula(dpi) -> None:
    if type(dpi) is not int or dpi <= 0:
        raise ValueError(f"dpi pozitif tamsayi olmali, {dpi!r} geldi")


def goruntu_uret(spec: BoardSpec, dpi: int = 300) -> np.ndarray:
    """Board'u verilen dpi'da, tam fiziksel oranda gri tonlu goruntu olarak uret.

    marginSize=0 kullaniyoruz: kenar boslugu PDF yerlesiminde veriliyor, boylece
    goruntunun piksel/mm oranı tam olarak korunuyor.
    """
    _dpi_dogrula(dpi)
    px_per_mm = dpi / 25.4
    genislik = int(round(spec.width_mm * px_per_mm))
    yukseklik = int(round(spec.height_mm * px_per_mm))
    return board_kur(spec).generateImage((genislik, yukseklik), marginSize=0)


def pdf_yaz(spec: BoardSpec, cikti: Path, dpi: int = 300, kenar_mm: float = 12.0) -> Path:  # noqa: E501
    """Tam fiziksel olcekte, baskiya hazir PDF yaz.

    Yaziciya **"gercek boyut / 100%"** denmeli. Altbilgideki kare kenarini
    cetvelle dogrulamadan board kullanilmaz.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    goruntu = goruntu_uret(spec, dpi)
    mm_to_in = 1 / 25.4
    altbilgi_mm = 16.0
    sayfa_g = (spec.width_mm + 2 * kenar_mm) * mm_to_in
    sayfa_y = (spec.height_mm + 2 * kenar_mm + altbilgi_mm) * mm_to_in

    fig = plt.figure(figsize=(sayfa_g, sayfa_y), dpi=dpi)
    # Eksen dikdortgenini sayfa oraninda ver -> board tam olcekte basilir.
    sol = kenar_mm * mm_to_in / sayfa_g
    alt = (kenar_mm + altbilgi_mm) * mm_to_in / sayfa_y
    ax = fig.add_axes([sol, alt,
                       spec.width_mm * mm_to_in / sayfa_g,
                       spec.height_mm * mm_to_in / sayfa_y])
    ax.imshow(goruntu, cmap="gray", aspect="auto", interpolation="none")
    ax.set_axis_off()

    fig.text(
        0.5, (kenar_mm * 0.5 * mm_to_in) / sayfa_y,
        f"{spec.squares_x}x{spec.squares_y} ChArUco · kare {spec.square_mm:g} mm · "
        f"marker {spec.marker_mm:g} mm · {spec.dictionary} · {spec.inner_corners} ic kose\n"
        f"%100 olcekte basin. Basildiktan sonra bir kare kenarini cetvelle olcun: "
        f"{spec.square_mm:g} mm olmali. Degilse board kullanilmaz.",
        ha="center", va="center", fontsize=7, family="monospace",
    )

    _dpi_dogrula(dpi)
    if not np.isfinite(kenar_mm) or kenar_mm < 0:
        raise ValueError(f"kenar_mm sonlu ve negatif olmayan olmali, {kenar_mm} geldi")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, format="pdf")
    plt.close(fig)
    sayfa_mm = (sayfa_g / mm_to_in, sayfa_y / mm_to_in)

    # Spec'i PDF'in yanina yaz: hangi board'la kalibre edildigi sonradan sorulacak.
    # Atomik yazim (calib/io.py ile ayni disiplin): yarim kalan yan dosya,
    # hangi board'la kalibre edildigi sorusuna yanlis cevap verir (B.1.3).
    yan = cikti.with_suffix(".json")
    gecici = yan.with_name(yan.name + ".tmp")
    gecici.write_text(json.dumps(
        {**asdict(spec),
         "sayfa_mm": [round(sayfa_mm[0], 1), round(sayfa_mm[1], 1)],
         "kagit": kagit_oner(*sayfa_mm),
         "dpi": dpi},
        indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    gecici.replace(yan)
    return cikti


# Yaygin kagit boyutlari (mm). Board %100 olcekte basilmak zorunda oldugu icin
# sayfanin sigacagi kagidi bilmek gerekir; kucuk kagida "sigdir" demek board'u
# kucultur ve kalibrasyonu sessizce bozar.
KAGITLAR = (("A4", 210, 297), ("A3", 297, 420), ("A2", 420, 594),
            ("A1", 594, 841), ("A0", 841, 1189))


def kagit_oner(genislik_mm: float, yukseklik_mm: float) -> str:
    """Sayfanin %100 olcekte sigacagi en kucuk standart kagit."""
    for ad, kg, ky in KAGITLAR:
        if (genislik_mm <= kg and yukseklik_mm <= ky) or (genislik_mm <= ky and yukseklik_mm <= kg):
            return ad
    return "A0 ustu -- plot/afis baskisi gerekir"


def _cli() -> int:
    ap = argparse.ArgumentParser(description="ChArUco board uret")
    ap.add_argument("--kare", nargs=2, type=int, metavar=("X", "Y"), default=[5, 7])
    ap.add_argument("--kenar", type=float, default=60.0, help="kare kenari (mm)")
    ap.add_argument("--marker", type=float, default=45.0, help="marker kenari (mm)")
    ap.add_argument("--sozluk", default=VARSAYILAN_SOZLUK)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cikti", type=Path, default=Path("out/board.pdf"))
    a = ap.parse_args()

    spec = BoardSpec(a.kare[0], a.kare[1], a.kenar, a.marker, a.sozluk)
    yol = pdf_yaz(spec, a.cikti, a.dpi)
    yan = json.loads(yol.with_suffix(".json").read_text())
    sg, sy = yan["sayfa_mm"]
    print(f"yazildi  : {yol}")
    print(f"board    : {spec.width_mm:g} x {spec.height_mm:g} mm, {spec.inner_corners} ic kose")
    print(f"sayfa    : {sg:g} x {sy:g} mm  ->  en az {yan['kagit']} kagit")
    print(f"spec     : {yol.with_suffix('.json')}")
    print("\nUYARI: %100 (gercek boyut) olcekte basin -- 'sigdir/scale to fit' board'u")
    print("bozar. Bastiktan sonra bir kare kenarini cetvelle olcun:"
          f" {spec.square_mm:g} mm olmali.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
