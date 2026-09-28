"""MediaPipe Tasks modelini mevcut Poz2B sözleşmesine bağlar."""

from dataclasses import replace
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from capture.alignment import file_sha256
from pose3d.adaptorler import mediapipe_poz2b, mediapipe_dunya_noktalari
from pose3d.iskelet import REFERANS_ISKELET
from pose3d.pose2d import Poz2B


# El kirpintisinin model girisine buyutuldugu kenar (piksel).
EL_GIRIS = 256


def _sayisal_surum(metin: str) -> tuple[int, ...]:
    """'1.0.0.post1' -> (1, 0, 0); sayisal olmayan ilk parcada durur."""
    parcalar: list[int] = []
    for parca in str(metin).split("."):
        if not parca.isdigit():
            break
        parcalar.append(int(parca))
    return tuple(parcalar)


def result_to_pose(result, size, *, model, threshold=0.5):
    """Tespit yoksa görünmez poz döner; eski karenin pozunu taşımaz.

    "Tespit var mı" bilgisi artık `ek["tespit"]` sözlüğünde değil, `Poz2B.tespit`
    alanında taşınır: yeni bir adaptör yazarken unutulması sessiz bir hataya
    dönüşemez, çünkü alan zorunludur.
    """
    if not result.pose_landmarks:
        # Tespit yoksa koordinat da yoktur: NaN yazilir, 0.0 (goruntunun sol ust
        # kosesi) gibi gecerli gorunen bir deger yazilmaz (tek eksik veri politikasi).
        n = len(REFERANS_ISKELET)
        return Poz2B(REFERANS_ISKELET, np.full((n, 2), np.nan), np.zeros(n),
                     np.zeros(n, bool), False, size, model=model)
    if len(result.pose_landmarks) != 1:
        raise ValueError("Tek kişi sözleşmesinde birden fazla poz döndü.")
    return mediapipe_poz2b(result.pose_landmarks[0], size, guven_esigi=threshold,
                           model=model, tespit=True)


class MediaPipeEstimator:
    """BGR uint8 -> özgün piksellerde Poz2B; her çağrı bağımsız IMAGE modu.

    Durumsuz mod, aynı kareyi farklı video/kameralar arasında izleme geçmişi
    taşımadan işler. VIDEO takip optimizasyonu bu ilk baseline'a dahil değildir.
    """

    def __init__(self, model_path, *, threshold=0.5, el_modeli=None, kisi_sayisi=1):
        """`el_modeli`: hand_landmarker.task; verilirse `tam_vucut` parmaklari da doner.

        `kisi_sayisi` > 1 yalniz `kisiler` icindir; `__call__` tek kisi
        sozlesmesini korur ve birden fazla poz gelirse hata verir.
        """
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Güven eşiği 0..1 aralığında olmalı.")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        import mediapipe as mp
        surum = _sayisal_surum(mp.__version__)
        # Koruma "1.0.0 ve 1.0.1" ile sinirli degil: sikayet edilen davranis 1.0
        # hattinin tamami icin gecerli (0030). '1.0.2' cikarsa koruma kalkmaz;
        # 1.1+ dogrulanmadigi icin engellenmez, denenip karar kaydina yazilmali.
        if sys.platform == "darwin" and surum[:2] == (1, 0):
            raise RuntimeError(
                f"Bu Mac için MediaPipe 0.10.35 kullanın; {mp.__version__} yerel süreç "
                "çökmesine yol açabilir (docs/kararlar/0030).")
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        # capture.alignment'taki tek sha256 uygulamasi kullanilir (tum dosyayi
        # belleğe alan ikinci bir kopya yok).
        self.sha256 = file_sha256(path)
        # Varyant (lite/full/heavy) model kimliginde gorunur; aksi halde üç ay
        # sonra "hangi MediaPipe" sorusu karar kaydina bakmadan cevaplanamaz.
        self.model_id = (f"mediapipe-{mp.__version__}/{path.stem}"
                         f"/sha256:{self.sha256}")
        self.threshold = threshold
        self._mp = mp
        self._closed = False
        # CPU delegate bilincli: donanim bagimsiz ve iki backend ayni kaynak
        # kosullarinda olculsun (GPU delegate olsaydi B2 tablosu donanima bagli
        # olurdu). GPU yolu denenirse karar kaydinda ayrica belirtilmeli.
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path.resolve()), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.IMAGE, num_poses=int(kisi_sayisi),
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            output_segmentation_masks=False)
        self._detector = PoseLandmarker.create_from_options(options)
        self._el = None
        self.el_sha256 = None
        if el_modeli is not None:
            el_yolu = Path(el_modeli)
            if not el_yolu.is_file():
                raise FileNotFoundError(el_yolu)
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            self.el_sha256 = file_sha256(el_yolu)
            self._el = HandLandmarker.create_from_options(HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(el_yolu.resolve()),
                                         delegate=BaseOptions.Delegate.CPU),
                running_mode=RunningMode.IMAGE, num_hands=1,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5))

    def _goruntu(self, image_bgr):
        if self._closed:
            raise RuntimeError("Model kapalı.")
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0):
            raise ValueError("Girdi boş olmayan BGR uint8 görüntü olmalı.")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), rgb

    def _eller(self, rgb, lm, width, height):
        """Her kol icin el bolgesini kirpip el modelini orada calistirir.

        Tam boy karede el ~50 piksel kalir ve MediaPipe'in avuc dedektoru onu
        kacirir (REHAB24'te 0/3 karede el). Holistic'in yaptigi gibi el
        bolgesi pozdan kestirilir: dirsek->bilek yonunde bilegin otesi, kenar
        onkol boyunun ~1,6 kati; kirpinti `EL_GIRIS` piksele buyutulur.
        """
        if getattr(self, "_el", None) is None:
            return []
        eller = []
        for dirsek, bilek in ((13, 15), (14, 16)):
            d = np.array([lm[dirsek].x * width, lm[dirsek].y * height])
            b = np.array([lm[bilek].x * width, lm[bilek].y * height])
            onkol = float(np.linalg.norm(b - d))
            if not np.isfinite(onkol) or onkol < 4:
                continue
            merkez = b + 0.45 * (b - d)
            kenar = 1.6 * onkol
            x0, y0 = merkez - kenar / 2
            ix0, iy0 = int(round(x0)), int(round(y0))
            ik = int(round(kenar))
            # Kare disina tasan kisim siyahla doldurulur (olcek bozulmasin).
            kir = np.zeros((ik, ik, 3), np.uint8)
            sx0, sy0 = max(ix0, 0), max(iy0, 0)
            sx1, sy1 = min(ix0 + ik, width), min(iy0 + ik, height)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            kir[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = rgb[sy0:sy1, sx0:sx1]
            kir = cv2.resize(kir, (EL_GIRIS, EL_GIRIS), interpolation=cv2.INTER_LINEAR)
            sonuc = self._el.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                   data=np.ascontiguousarray(kir)))
            olcek = ik / EL_GIRIS
            for el in sonuc.hand_landmarks:
                eller.append(np.array([[ix0 + p.x * EL_GIRIS * olcek, iy0 + p.y * EL_GIRIS * olcek]
                                       for p in el]))
        return eller

    def _tam(self, lm, eller, width, height):
        from pose3d.tam_vucut import mediapipe_tam_vucut
        px = np.array([[p.x * width, p.y * height] for p in lm])
        guven = np.array([min(getattr(p, "visibility", 0.0) or 0.0, getattr(p, "presence", 1.0) or 0.0)
                          for p in lm])
        el_sha = getattr(self, "el_sha256", None)
        model = self.model_id + (f"+hand/sha256:{el_sha}" if el_sha else "")
        return mediapipe_tam_vucut(px, guven, eller, (width, height), model=model,
                                   esik=self.threshold)

    def tam_vucut(self, image_bgr):
        """(referans Poz2B, TAM_VUCUT Poz2B): govde, bas, ayaklar ve -- el modeli
        verildiyse -- parmak eklemleri. Referans poz `__call__` ile aynidir."""
        from pose3d.tam_vucut import bos_poz
        mp_image, rgb = self._goruntu(image_bgr)
        height, width = image_bgr.shape[:2]
        result = self._detector.detect(mp_image)
        pose = self._sonuc_pozu(result, width, height)
        if not result.pose_landmarks:
            return pose, bos_poz((width, height), self.model_id)
        lm = result.pose_landmarks[0]
        return pose, self._tam(lm, self._eller(rgb, lm, width, height), width, height)

    def kisiler(self, image_bgr):
        """Karedeki her kisi icin TAM_VUCUT Poz2B listesi (sira: MediaPipe'inki,
        kimlik tasimaz; kimlik icin `mono.coklu_kisi.KisiTakip`)."""
        mp_image, rgb = self._goruntu(image_bgr)
        height, width = image_bgr.shape[:2]
        result = self._detector.detect(mp_image)
        return [self._tam(lm, self._eller(rgb, lm, width, height), width, height)
                for lm in result.pose_landmarks]

    def __call__(self, image_bgr):
        mp_image, _ = self._goruntu(image_bgr)
        height, width = image_bgr.shape[:2]
        return self._sonuc_pozu(self._detector.detect(mp_image), width, height)

    def _sonuc_pozu(self, result, width, height):
        pose = result_to_pose(result, (width, height), model=self.model_id, threshold=self.threshold)
        world = getattr(result, "pose_world_landmarks", [])
        extra = {}
        if pose.tespit and len(world) == 1:
            xyz, conf, visible = mediapipe_dunya_noktalari(world[0], guven_esigi=self.threshold)
            # Maskelenmemis kopya: modelin gorunmez (or. profilde arkadaki bacak)
            # eklem icin de verdigi tahmin. Kullanan taraf guveni kendisi tartar;
            # `world_points_m` sozlesmesi (yalniz gorunur) degismez.
            extra = {"world_points_all_m": xyz.tolist(), "world_confidence": conf.tolist()}
            visible = visible & pose.gorunur
            xyz = xyz.copy()
            xyz[~visible] = np.nan
            extra |= {"world_points_m": xyz.tolist(), "world_visible": visible.tolist()}
            # Yuz gorunurlugu (burun, gozler): Poz2B yuzu tasimaz; sirti donuk kisiyi
            # (onden bakisla ayni govde acisi) ayirmak icin kullanilir.
            yuz = [result.pose_landmarks[0][i] for i in (0, 2, 5)]
            extra["yuz_guveni"] = float(max(min(getattr(p, "visibility", 0.0) or 0.0,
                                                getattr(p, "presence", 1.0) or 0.0) for p in yuz))
            # Ayak noktalari (REFERANS_ISKELET tasimaz): sag topuk, sag ayak ucu, sol
            # topuk, sol ayak ucu -- MediaPipe 30, 32, 29, 31. Lunge'da "diz ayak
            # ucunu geciyor" icin (docs/deney/2026-09-28-ec3d-lunge.md).
            ayak = (30, 32, 29, 31)
            lm2, lm3 = result.pose_landmarks[0], world[0]
            extra["ayak_px"] = [[lm2[i].x * width, lm2[i].y * height] for i in ayak]
            extra["ayak_dunya_m"] = [[lm3[i].x, lm3[i].y, lm3[i].z] for i in ayak]
            extra["ayak_guveni"] = [float(min(getattr(lm2[i], "visibility", 0.0) or 0.0,
                                              getattr(lm2[i], "presence", 1.0) or 0.0))
                                    for i in ayak]
        # `replace`: Poz2B frozen; `ek.update` yerine yeni bir kopya uretilir.
        return replace(pose, ek={**pose.ek, **extra, "model_sha256": self.sha256,
                                 "running_mode": "IMAGE", "device": "CPU"})

    def close(self):
        if not self._closed:
            self._closed = True
            self._detector.close()
            if getattr(self, "_el", None) is not None:
                self._el.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
