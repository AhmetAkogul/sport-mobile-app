"""Kalibrasyonu tarih ve surum damgali JSON'a yazan-okuyan modul.

Neden var: kalici kurulumin tez iddiasi **zaman icindeki suruklenme**. Suruklenme
egrisi ancak her olcumun tarih, surum ve baglam bilgisiyle birlikte saklanmasiyle
cizilebilir; ad-hoc .npy dosyalari bu baglami tasimaz. Ayrica Kapi 1'in
"tekrarlanabilirlik" olcutu (ayni veriyle iki kosu ayni sonuc) ancak sonucun
diskte birebir saklanabilmesiyle kanitlanir.

Tasarim kararlari:

- **JSON metni** -- ikili bicimler (pickle, .npz) surum degisiminde sessizce
  kirilir; JSON + kayittaki bicim surumu uyumsuzlugun okuma aninda yaklanmasini
  saglar.
- **Sayilar metin cinsinden** (`repr(float)` -> `float`) tasinir. Bu, kayan
  nokta degerlerin yaz-oku turunde birebir korunmasinin en basit ve standart
  yolu; numpy ikili serilestirmesi yerine bilerek secildi.
- **Atomik yazim** -- once `.tmp` sonra `replace`; yarim kalmis dosya okuyucuyu
  hicbir zaman bulamaz.
- **Yalniz sonucu yazar** -- `obj_noktalari` / `img_noktalari` gibi buyuk
  giris dizileri yazilmaz; meta icinde ozetleri (kare sayisi, gorunurluk) tasinir.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from calib.multiview import Kalibrasyon

# Kayit biciminin surumu -- okuyucu uyumsuz bicimi reddeder, sessizce cozmeye calismaz.
BICIM_SURUMU = 1


@dataclass
class Meta:
    """Kalibrasyon oturumunun baglami. Her alan suruklenme kaydinin parcasidir."""

    tarih_iso: str                       # UTC, ISO 8601
    bicim_surumu: int = BICIM_SURUMU
    surum: str = ""                      # git commit / surum etiketi
    notlar: str = ""
    n_kamera: int = 0
    n_kare: int = 0                      # kaynaktaki kare sayisi (yazan tasiyir)
    gorunurluk: float | None = None      # kamera x kare gozlem orani (0-1)
    rms_px: float | None = None          # kopya; tek bakisda karsilastirma icin
    # Kamera basina "pinhole" / "fisheye". Bos ise pinhole varsayilir (eski
    # kayitlar). Bu alan olmadan 4 elemanli fisheye bozulmasi 5 elemanli
    # pinhole'a tamamlanip modelin ne oldugu kayboluyordu -- dis inceleme B.3.2.
    modeller: list[str] = field(default_factory=list)
    etiketler: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Dis inceleme B.3.5: suruklenme egrisi tarihe gore siralanir; ayrisamayan
        # bir tarih haftalari sessizce karistirir. Dilimsiz tarih UTC kabul
        # edilir (uncertainty/haftalik.py; karisik dilim testi).
        try:
            datetime.fromisoformat(str(self.tarih_iso))
        except ValueError as exc:
            raise ValueError(f"tarih_iso ISO 8601 olmali: {self.tarih_iso!r}") from exc


@dataclass
class Kabin:
    """Diskteki kalibrasyon kaydi: meta + parametreler.

    `to_sozluk` / `from_sozluk` cifti JSON katmanindan ayridir; boylece yaz-oku
    turu (KOD-PLANI kabul testi) dosya olmadan da sinanabilir.
    """

    meta: Meta
    Ks: list[np.ndarray]
    bozulmalar: list[np.ndarray]
    Rs: list[np.ndarray]
    Ts: list[np.ndarray]

    def to_sozluk(self) -> dict:
        return {
            "bicim_surumu": BICIM_SURUMU,
            "meta": asdict(self.meta),
            "Ks": [_matris_yaz(M) for M in self.Ks],
            "bozulmalar": [_vektor_yaz(d) for d in self.bozulmalar],
            "Rs": [_matris_yaz(R) for R in self.Rs],
            "Ts": [_matris_yaz(T) for T in self.Ts],
        }

    @classmethod
    def from_sozluk(cls, s: dict) -> Kabin:
        surum = s.get("bicim_surumu")
        if surum != BICIM_SURUMU:
            raise ValueError(
                f"bicim_surumu {surum!r}, bu okuyucu {BICIM_SURUMU} bekliyor -- "
                "dosyayi elle duzeltmeyin; dosyayi ureten kodla yeniden yazin")
        m = s.get("meta") or {}
        meta_surum = int(m.get("bicim_surumu", BICIM_SURUMU))
        if meta_surum != BICIM_SURUMU:
            raise ValueError(f"meta bicim_surumu {meta_surum} taninmiyor")
        meta = Meta(
            tarih_iso=str(m["tarih_iso"]),
            bicim_surumu=meta_surum,
            surum=str(m.get("surum", "")),
            notlar=str(m.get("notlar", "")),
            n_kamera=int(m.get("n_kamera", 0)),
            n_kare=int(m.get("n_kare", 0)),
            gorunurluk=(float(m["gorunurluk"]) if m.get("gorunurluk") is not None else None),
            rms_px=(float(m["rms_px"]) if m.get("rms_px") is not None else None),
            modeller=[str(x) for x in (m.get("modeller") or [])],
            etiketler={str(k): str(v) for k, v in (m.get("etiketler") or {}).items()},
        )
        kabin = cls(
            meta=meta,
            Ks=[_matris_oku(a) for a in s["Ks"]],
            bozulmalar=[_vektor_oku(a) for a in s["bozulmalar"]],
            Rs=[_matris_oku(a) for a in s["Rs"]],
            Ts=[_matris_oku(a) for a in s["Ts"]],
        )
        kabin._dogrula()
        return kabin

    def _dogrula(self) -> None:
        n = len(self.Ks)
        if not (len(self.bozulmalar) == len(self.Rs) == len(self.Ts) == n):
            raise ValueError(
                f"kamera listeleri uyusmuyor: {n} K, {len(self.bozulmalar)} bozulma, "
                f"{len(self.Rs)} R, {len(self.Ts)} T")
        if self.meta.n_kamera and self.meta.n_kamera != n:
            raise ValueError(
                f"meta {self.meta.n_kamera} kamera diyor, kayitta {n} var")
        for i, R in enumerate(self.Rs):
            if R.shape != (3, 3):
                raise ValueError(f"Rs[{i}] (3,3) olmali, {R.shape} geldi")
        for i, T in enumerate(self.Ts):
            if T.shape != (3, 1):
                raise ValueError(f"Ts[{i}] (3,1) olmali, {T.shape} geldi")
        # Ks ve bozulmalar da denetlenmeli: bozuk bir kaydi sessizce kabul etmek,
        # yanlis kalibrasyonla ucgenleme yapmak demek (B.3.1).
        for i, K in enumerate(self.Ks):
            if np.asarray(K).shape != (3, 3):
                raise ValueError(f"Ks[{i}] (3,3) olmali, {np.asarray(K).shape} geldi")
        for i, d in enumerate(self.bozulmalar):
            a = np.asarray(d)
            if a.ndim != 1 or a.size not in (4, 5, 8, 12, 14):
                raise ValueError(
                    f"bozulmalar[{i}] tek boyutlu ve 4/5/8/12/14 elemanli olmali, "
                    f"{a.shape} geldi")
        if self.meta.modeller and len(self.meta.modeller) != n:
            raise ValueError(
                f"meta.modeller {len(self.meta.modeller)} eleman, {n} kamera var")

    def __post_init__(self) -> None:
        """Yazma aninda dogrula.

        Onceden yalnizca `from_sozluk` dogruluyordu; `kalibrasyondan` dogrudan
        Kabin kurdugu icin tutarsizlik ancak **okuma** aninda patliyordu ve
        hatanin kaynagi gizleniyordu (B.3.4).
        """
        self._dogrula()


# ---------------------------------------------------------------------------
# Donusumler: Kalibrasyon <-> Kabin
# ---------------------------------------------------------------------------

def kalibrasyondan(
    kalib: Kalibrasyon,
    *,
    n_kare: int = 0,
    gorunurluk: float | None = None,
    surum: str = "",
    notlar: str = "",
    etiketler: dict[str, str] | None = None,
    tarih_iso: str | None = None,
) -> Kabin:
    """Kalibrasyon sonucunu kayda cevir; baglam alanlarini yazan saglar."""
    if tarih_iso is None:
        tarih_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Kabin(
        meta=Meta(
            tarih_iso=tarih_iso,
            surum=surum,
            notlar=notlar,
            n_kamera=kalib.n_kamera,
            n_kare=n_kare,
            gorunurluk=gorunurluk,
            rms_px=kalib.rms_px,
            modeller=list(kalib.modeller),
            etiketler=dict(etiketler or {}),
        ),
        Ks=[np.asarray(K, dtype=np.float64) for K in kalib.Ks],
        bozulmalar=[np.asarray(d, dtype=np.float64).ravel() for d in kalib.bozulmalar],
        Rs=[np.asarray(R, dtype=np.float64) for R in kalib.Rs],
        Ts=[np.asarray(T, dtype=np.float64).reshape(3, 1) for T in kalib.Ts],
    )


def kalibrasyona(kabin: Kabin) -> Kalibrasyon:
    """Kaydi geri Kalibrasyon'a cevir. rms yoksa 0.0 konur (parametreler etkilenmez).

    Model bilgisi varsa fisheye bozulmasi 4 elemana geri kirpilir: `_vektor_oku`
    okurken pinhole sozlesmesine tamamliyor, model kaydi olmadan bu tamamlama
    geri alinamazdi (B.3.2).
    """
    modeller = tuple(kabin.meta.modeller)
    bozulmalar = []
    for i, d in enumerate(kabin.bozulmalar):
        a = np.asarray(d, dtype=np.float64)
        if i < len(modeller) and modeller[i] == "fisheye" and a.size == 5:
            a = a[:4]
        bozulmalar.append(a)
    return Kalibrasyon(
        rms_px=float(kabin.meta.rms_px) if kabin.meta.rms_px is not None else 0.0,
        modeller=modeller,
        Ks=[np.asarray(K, dtype=np.float64) for K in kabin.Ks],
        bozulmalar=bozulmalar,
        Rs=[np.asarray(R, dtype=np.float64) for R in kabin.Rs],
        Ts=[np.asarray(T, dtype=np.float64).reshape(3, 1) for T in kabin.Ts],
    )


# ---------------------------------------------------------------------------
# Disk katmani
# ---------------------------------------------------------------------------

def yaz(kabin: Kabin, yol: str | Path) -> Path:
    """Kaydi JSON olarak yazar (atomik). Donen deger yazilan dosyanin yolu."""
    p = Path(yol)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    metin = json.dumps(kabin.to_sozluk(), ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(metin + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def oku(yol: str | Path) -> Kabin:
    """JSON kaydini okur. Bicim surumu uyusmazsa ValueError -- asla sessiz cozum yok."""
    metin = Path(yol).read_text(encoding="utf-8")
    return Kabin.from_sozluk(json.loads(metin))


# ---------------------------------------------------------------------------
# Metin cinsinden sayi yardimcilari
# ---------------------------------------------------------------------------

def _matris_yaz(M: np.ndarray) -> list[list[str]]:
    """(3,3) veya (3,1) diziyi metin cinsinden ic ice listeye indirger."""
    a = np.asarray(M, dtype=np.float64)
    if a.size == 9:
        a = a.reshape(3, 3)
    elif a.size == 3:
        a = a.reshape(3, 1)
    else:
        raise ValueError(f"beklenmeyen boyut {a.shape}; 9 veya 3 eleman gerekli")
    # Sayilar metin olarak yazildigi icin allow_nan onlari yakalamaz: "nan"
    # sessizce dolasir ve okuyan float("nan") yapar (dis inceleme B.3.3).
    if not np.isfinite(a).all():
        raise ValueError("kalibrasyon matrisi sonlu olmayan deger iceriyor")
    return [[repr(float(x)) for x in satir] for satir in a]


def _matris_oku(a: list) -> np.ndarray:
    x = np.array([[float(v) for v in satir] for satir in a], dtype=np.float64)
    if x.shape not in ((3, 3), (3, 1)):
        raise ValueError(f"beklenmeyen matris boyutu {x.shape}")
    return x


def _vektor_yaz(v: np.ndarray) -> list[str]:
    """Bozulma vektoru: (4,) fisheye, (5,) pinhole sozlesmesi."""
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size not in (4, 5):
        raise ValueError(f"bozulma vektoru 4 veya 5 elemanli olmali, {a.size} geldi")
    if not np.isfinite(a).all():
        raise ValueError("bozulma vektoru sonlu olmayan deger iceriyor")
    return [repr(float(x)) for x in a]


def _vektor_oku(a: list) -> np.ndarray:
    x = np.array([float(v) for v in a], dtype=np.float64)
    if x.size == 4:
        return np.append(x, 0.0)   # 4 parametreli fisheye kaydi pinhole sozlesmesine tamamlanir
    if x.size == 5:
        return x
    raise ValueError(f"bozulma vektoru 4 veya 5 elemanli olmali, {x.size} geldi")
