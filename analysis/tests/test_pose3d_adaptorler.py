"""pose3d.adaptorler kabul testleri.

mediapipe paketi kurulu OLMADAN calisir: sonuc ordek-tipleme ile alindigi icin
sahte landmark nesneleriyle sinanir. Adaptorun tum isi bir cevirim oldugu icin
bu yeterli ve calisma ortamini bozmuyor.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from pose3d.adaptorler import (
    MEDIAPIPE_EKLEMLER, MEDIAPIPE_ESLEME, TUREVLER,
    mediapipe_dunya_noktalari, mediapipe_poz2b,
)
from pose3d.iskelet import REFERANS_ISKELET as ISK

BOYUT = (1280, 720)


@dataclass
class SahteLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


def sahte_sonuc(**ayar):
    """33 eklemli sahte MediaPipe ciktisi; eklem adiyla deger ayarlanabilir."""
    lm = []
    for i, ad in enumerate(MEDIAPIPE_EKLEMLER):
        varsayilan = dict(x=(i + 1) / 100.0, y=(i + 1) / 200.0, z=(i + 1) / 300.0)
        varsayilan.update(ayar.get(ad, {}))
        lm.append(SahteLandmark(**varsayilan))
    return lm


def test_eklem_sayisi_degisirse_hata():
    """MediaPipe surumu eklem sayisini degistirirse esleme sessizce kaymamali."""
    with pytest.raises(ValueError, match="surum degismis olabilir"):
        mediapipe_poz2b(sahte_sonuc()[:30], BOYUT)


def test_normalize_koordinat_piksele_cevrilir():
    lm = sahte_sonuc(RIGHT_KNEE={"x": 0.25, "y": 0.5})
    poz = mediapipe_poz2b(lm, BOYUT)
    j = ISK.indeks("sag_diz")
    assert poz.noktalar[j] == pytest.approx([0.25 * 1280, 0.5 * 720])


def test_boyun_omuz_ortasindan_turetilir():
    lm = sahte_sonuc(LEFT_SHOULDER={"x": 0.4, "y": 0.3},
                     RIGHT_SHOULDER={"x": 0.6, "y": 0.3})
    poz = mediapipe_poz2b(lm, BOYUT)
    boyun = poz.noktalar[ISK.indeks("boyun")]
    assert boyun == pytest.approx([0.5 * 1280, 0.3 * 720])
    assert "boyun" in TUREVLER


def test_turetilmis_eklemin_guveni_en_dusuk_bilesendir():
    """Bir omuz guvenilmezse orta nokta da guvenilmez; ortalama alinmaz."""
    lm = sahte_sonuc(LEFT_SHOULDER={"visibility": 0.9},
                     RIGHT_SHOULDER={"visibility": 0.2})
    poz = mediapipe_poz2b(lm, BOYUT, guven_esigi=0.5)
    j = ISK.indeks("boyun")
    assert poz.guven[j] == pytest.approx(0.2)
    assert not poz.gorunur[j]


def test_guven_gorunurluk_ve_varligin_kucugu():
    lm = sahte_sonuc(RIGHT_KNEE={"visibility": 0.9, "presence": 0.3})
    poz = mediapipe_poz2b(lm, BOYUT)
    assert poz.guven[ISK.indeks("sag_diz")] == pytest.approx(0.3)


def test_esik_altindaki_eklem_gorunmez():
    lm = sahte_sonuc(RIGHT_ANKLE={"visibility": 0.1})
    poz = mediapipe_poz2b(lm, BOYUT, guven_esigi=0.5)
    assert not poz.gorunur[ISK.indeks("sag_ayak_bilegi")]
    assert poz.gorunur[ISK.indeks("sag_diz")]


def test_sag_sol_karismiyor():
    """MediaPipe'in LEFT/RIGHT'i ayna degil; esleme dogrudan olmali."""
    lm = sahte_sonuc(RIGHT_KNEE={"x": 0.1, "y": 0.1},
                     LEFT_KNEE={"x": 0.9, "y": 0.9})
    poz = mediapipe_poz2b(lm, BOYUT)
    assert poz.noktalar[ISK.indeks("sag_diz")][0] == pytest.approx(0.1 * 1280)
    assert poz.noktalar[ISK.indeks("sol_diz")][0] == pytest.approx(0.9 * 1280)


def test_yuz_eklemleri_atiliyor():
    """KVKK: yuz verisi tasinmaz. Referans iskelette yuz eklemi olmamali."""
    tasinanlar = set(MEDIAPIPE_ESLEME.values()) | {
        x for cift in TUREVLER.values() for x in cift}
    yuz = {"NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
           "MOUTH_LEFT", "MOUTH_RIGHT"}
    assert not (tasinanlar & yuz)


def test_ciktinin_sozlesmesi_dogru():
    poz = mediapipe_poz2b(sahte_sonuc(), BOYUT, kamera=2, kare=7)
    assert poz.iskelet is ISK
    assert len(poz.noktalar) == len(ISK)
    assert poz.uzay == "ozgun"
    assert poz.kamera == 2 and poz.kare == 7
    # Model adi varsayilan olarak uydurulmaz: cagiran gercek kimligi gecirir (A.7).
    assert poz.model == "bilinmiyor"
    assert poz.tespit is True
    assert poz.ek["kaynak_iskelet"] == "mediapipe-33"


def test_nan_koordinat_sifira_cevrilmez():
    """Eksik nokta NaN kalir: 0.0 gecerli bir piksel koordinatidir (A.1/X.4)."""
    poz = mediapipe_poz2b(sahte_sonuc(RIGHT_KNEE={"x": float("nan"), "y": float("nan")}), BOYUT)
    j = ISK.indeks("sag_diz")
    assert np.isnan(poz.noktalar[j]).all()
    assert not poz.gorunur[j]
    assert poz.gorunur[ISK.indeks("sag_kalca")]


def test_nan_guven_sifir_sayilir():
    """`getattr(...) or 0.0` NaN'i temizlemez: NaN truthy'dir (A.2)."""
    poz = mediapipe_poz2b(sahte_sonuc(RIGHT_KNEE={"visibility": float("nan")}), BOYUT)
    j = ISK.indeks("sag_diz")
    assert poz.guven[j] == 0.0
    assert not poz.gorunur[j]


def test_alan_sayisi_gecersizse_hata():
    from pose3d.adaptorler import _mp_dizileri
    with pytest.raises(ValueError, match="alan_sayisi"):
        _mp_dizileri(sahte_sonuc(), 4)


def test_verilen_model_kimligi_tasinir():
    poz = mediapipe_poz2b(sahte_sonuc(), BOYUT, model="mediapipe-0.10.35/pose_landmarker_full")
    assert poz.model == "mediapipe-0.10.35/pose_landmarker_full"


def test_gecersiz_goruntu_boyutu_reddedilir():
    with pytest.raises(ValueError, match="pozitif"):
        mediapipe_poz2b(sahte_sonuc(), (0, 720))


def test_dunya_noktalari_uc_boyutlu():
    """world_landmarks telefonun kendi 3B kestirimi -- tezin olctugu sey."""
    lm = sahte_sonuc(RIGHT_KNEE={"x": 0.1, "y": -0.4, "z": 0.05})
    noktalar, guven, gorunur = mediapipe_dunya_noktalari(lm)
    assert noktalar.shape == (len(ISK), 3)
    assert noktalar[ISK.indeks("sag_diz")] == pytest.approx([0.1, -0.4, 0.05])
    assert gorunur.all()
    # Guven de doner: duzeltme katmani ham guveni isteyecek (A.5).
    assert guven.shape == (len(ISK),) and (guven > 0).all()


def test_dunya_noktalarinda_boyun_da_turetilir():
    lm = sahte_sonuc(LEFT_SHOULDER={"x": 0.0, "y": 0.5, "z": 0.0},
                     RIGHT_SHOULDER={"x": 0.4, "y": 0.5, "z": 0.2})
    noktalar, _, _ = mediapipe_dunya_noktalari(lm)
    assert noktalar[ISK.indeks("boyun")] == pytest.approx([0.2, 0.5, 0.1])
