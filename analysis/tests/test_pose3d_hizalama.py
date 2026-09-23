"""pose3d.hizalama kabul testleri.

Rijit hizalama, "mutlak konum hatasi" ile "sekil bozulmasi"ni birbirinden ayirir.
Bu ayrim olmadan cerceve farki hata gibi gorunur -- deneyin ilk surumunde
208 mm'lik sahte bir hata tam olarak bu yuzden olcuklmustu.
"""
import numpy as np
import pytest

from pose3d.hizalama import rijit_hizala


def _donme(eksen, derece):
    import cv2
    e = np.asarray(eksen, float)
    e = e / np.linalg.norm(e)
    return cv2.Rodrigues((e * np.deg2rad(derece)).reshape(3, 1))[0]


NOKTALAR = np.array([[0.0, 0.0, 2.0], [0.5, 0.1, 2.2], [-0.4, 0.3, 2.5],
                     [0.2, -0.3, 1.8], [-0.1, 0.4, 2.1]])


def test_ayni_kume_birim_donusum():
    h = rijit_hizala(NOKTALAR, NOKTALAR)
    assert h.rms_mm == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(h.R, np.eye(3), atol=1e-9)
    assert h.olcek == pytest.approx(1.0)


def test_saf_oteleme_tam_kaldirilir():
    kaydirma = np.array([0.2, -0.15, 0.3])
    h = rijit_hizala(NOKTALAR, NOKTALAR + kaydirma)
    assert h.rms_mm < 1e-6
    assert np.allclose(h.t, kaydirma, atol=1e-9)


def test_saf_donme_tam_kaldirilir():
    R = _donme([0.3, 1.0, 0.2], 17.0)
    h = rijit_hizala(NOKTALAR, (R @ NOKTALAR.T).T)
    assert h.rms_mm < 1e-6
    assert np.allclose(h.R, R, atol=1e-8)


def test_cerceve_farki_hata_olarak_gorunmez():
    """Asil amac: donme + oteleme uygulanmis kume sifir hata vermeli."""
    R = _donme([1.0, 0.4, -0.2], 25.0)
    donusmus = (R @ NOKTALAR.T).T + np.array([1.5, -0.8, 0.4])
    ham_fark = float(np.linalg.norm(donusmus - NOKTALAR, axis=1).mean()) * 1000
    h = rijit_hizala(NOKTALAR, donusmus)
    assert ham_fark > 500          # hizalamadan once yuzlerce mm "hata"
    assert h.rms_mm < 1e-6         # hizalamadan sonra hata yok


def test_olcek_serbest_olcegi_bulur():
    h = rijit_hizala(NOKTALAR, NOKTALAR * 1.02, olcek_serbest=True)
    assert h.olcek == pytest.approx(1.02, rel=1e-6)
    assert h.rms_mm < 1e-6


def test_olcek_sabitken_olcek_hatasi_artikta_kalir():
    h = rijit_hizala(NOKTALAR, NOKTALAR * 1.02, olcek_serbest=False)
    assert h.olcek == 1.0
    assert h.rms_mm > 1.0          # %2 olcek hatasi mm mertebesinde artik birakir


def test_sekil_bozulmasi_yakalanir():
    """Rijit olmayan bozulma hizalamayla kaldirilamaz -- kalmasi gerekir."""
    bozuk = NOKTALAR.copy()
    bozuk[2] += np.array([0.01, 0.0, 0.0])     # tek noktayi 10 mm kaydir
    h = rijit_hizala(bozuk, NOKTALAR, olcek_serbest=True)
    assert h.rms_mm > 1.0
    assert h.en_buyuk_mm > 3.0


def test_yansima_uretilmez():
    """Aynalanmis kume icin gecerli bir donme uretilmeli, yansima degil."""
    aynali = NOKTALAR * np.array([1.0, 1.0, -1.0])
    h = rijit_hizala(aynali, NOKTALAR)
    assert np.linalg.det(h.R) == pytest.approx(1.0, abs=1e-9)


def test_uygula_hizalamayi_tekrarlar():
    R = _donme([0.2, 0.5, 1.0], 12.0)
    hedef = (R @ NOKTALAR.T).T + np.array([0.1, 0.2, 0.3])
    h = rijit_hizala(NOKTALAR, hedef)
    assert np.allclose(h.uygula(NOKTALAR), hedef, atol=1e-9)


@pytest.mark.parametrize("kaynak,hedef", [
    (NOKTALAR[:2], NOKTALAR[:2]),                 # 3'ten az nokta
])
def test_yetersiz_nokta_reddedilir(kaynak, hedef):
    with pytest.raises(ValueError, match="en az 3"):
        rijit_hizala(kaynak, hedef)


def test_uyusmayan_kume_reddedilir():
    with pytest.raises(ValueError, match="ayni sekilde"):
        rijit_hizala(NOKTALAR, NOKTALAR[:3])


# --- dis inceleme H.1 / H.2 / H.3 --------------------------------------------

def test_dejenere_kaynak_reddediliyor():
    """Butun noktalar ayniysa rotasyon tanimsiz; keyfi bir R dondurulmemeli (H.1).

    SVD bos bir matriste keyfi bir donme uretir ve olcek 1.0'a duser; sonuc
    anlamsiz ama gecerli gorunur.
    """
    import numpy as np
    import pytest
    from pose3d.hizalama import rijit_hizala

    ayni = np.tile(np.array([0.1, 0.2, 0.3]), (4, 1))
    with pytest.raises(ValueError, match="dejenere"):
        rijit_hizala(ayni, ayni + 0.5)


def test_hizalama_alanlari_dogrulaniyor():
    """Hizalama dogrulamasiz bir sinifti (H.2)."""
    import numpy as np
    import pytest
    from pose3d.hizalama import Hizalama

    with pytest.raises(ValueError, match="R"):
        Hizalama(R=np.eye(2), t=np.zeros(3), olcek=1.0, rms_mm=0.0, en_buyuk_mm=0.0)
    with pytest.raises(ValueError, match="olcek"):
        Hizalama(R=np.eye(3), t=np.zeros(3), olcek=-1.0, rms_mm=0.0, en_buyuk_mm=0.0)


def test_uygula_yanlis_sekli_reddediyor():
    """reshape(-1, 3) boyut uydurur; (N,4) girdi sessizce saclanirdi (H.3)."""
    import numpy as np
    import pytest
    from pose3d.hizalama import Hizalama

    h = Hizalama(R=np.eye(3), t=np.zeros(3), olcek=1.0, rms_mm=0.0, en_buyuk_mm=0.0)
    with pytest.raises(ValueError):
        h.uygula(np.zeros((5, 4)))
