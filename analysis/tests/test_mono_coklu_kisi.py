"""Tek kamerada cok kisi: kutu, guven, takip sarmali ve kisi gecmisi."""

import numpy as np
import pytest

from mono.coklu_kisi import CokKisiHatti, KisiGecmisi, KisiTakip, kisi_rengi, poz_guveni, poz_kutusu
from pose3d.tam_vucut import GRUPLAR, coco_wholebody_poz


def _poz(x0, kutu=True, el=True):
    P = np.stack([np.linspace(x0, x0 + 50, 133), np.linspace(10, 200, 133)], 1)
    s = np.full(133, 5.0)
    if not el:
        s[91:] = 0.0
    return coco_wholebody_poz(P, s, (640, 480), model="t", esik=1.0,
                              ek={"kutu": [x0, 10, x0 + 50, 200]} if kutu else None)


def test_kutu_ekten_ya_da_noktalardan():
    np.testing.assert_allclose(poz_kutusu(_poz(100)), [100, 10, 150, 200])
    k = poz_kutusu(_poz(100, kutu=False), pay=0.0)
    np.testing.assert_allclose(k, [100, 10, 150, 200])


def test_guven_elleri_saymaz():
    # El bulunamasa da govde tam gorunurse guven 1 olmali (ByteTrack iz esigi 0,35).
    assert poz_guveni(_poz(0, el=False)) == pytest.approx(1.0)
    assert not _poz(0, el=False).gorunur[list(GRUPLAR["sol_el"])].any()


class _SahteTakip:
    """Sirayla kimlik verir; ikinci karede sirayi tersler (kimlik sirayla degil kutuyla)."""

    def __init__(self):
        self.cagri = 0

    def __call__(self, kutular, guven, sira):
        self.cagri += 1
        kimlik = np.arange(1, len(sira) + 1)
        if self.cagri == 2:
            return kimlik[::-1], sira
        return kimlik, sira


def test_takip_kimligi_dogru_poza_baglar_ve_kutusuzu_atlar():
    tk = KisiTakip(tracker=_SahteTakip())
    a, b = _poz(0), _poz(300)
    bos = coco_wholebody_poz(np.full((133, 2), np.nan), np.zeros(133), (640, 480), model="t", esik=1.0)
    r = tk.guncelle([a, bos, b])
    assert [(k, p is a or p is b) for k, p in r] == [(1, True), (2, True)]
    assert r[1][1] is b                             # kutusuz poz takipciye gitmedi
    r2 = tk.guncelle([a, b])
    assert r2[0] == (2, a) and r2[1] == (1, b)


def test_gecmis_sinirli_ve_dizi_sekli():
    g = KisiGecmisi(1, uzunluk=3)
    for t in range(5):
        g.ekle(t, _poz(t))
    zaman, P, gor = g.dizi()
    assert zaman.tolist() == [2, 3, 4] and P.shape == (3, 65, 2) and gor.shape == (3, 65)
    assert KisiGecmisi(2).dizi()[1].shape == (0, 0, 2)


def test_hat_gorulmeyen_kimligi_unutur():
    h = CokKisiHatti(KisiTakip(tracker=lambda k, g, s: (np.arange(1, len(s) + 1), s)), unut_s=1.0)
    h.adim(0.0, [_poz(0), _poz(300)])
    assert set(h.kisiler) == {1, 2}
    h.adim(0.5, [_poz(0)])
    h.adim(2.0, [_poz(0)])
    assert set(h.kisiler) == {1} and len(h.kisiler[1].zaman) == 3


def test_kisi_rengi_dolanir():
    assert kisi_rengi(1) == kisi_rengi(9) and kisi_rengi(1) != kisi_rengi(2)


def test_bytetrack_gercek_kimlik_surekliligi():
    sv = pytest.importorskip("supervision")
    assert sv
    tk = KisiTakip(kare_hizi=15, kayip_s=2.0)
    kimlikler = []
    for t in range(10):
        r = tk.guncelle([_poz(100 + 3 * t), _poz(400 - 3 * t)])
        kimlikler.append([k for k, _ in r])
    assert all(k == kimlikler[0] for k in kimlikler)
    # 2 s tampon 15 fps'te 30 kare demek (supervision 30 fps birimiyle olcekler)
    assert tk._tracker._t.max_time_lost == 30


def test_poz_cihazi_secimi():
    from mono.rtmw_model import poz_cihazi
    assert poz_cihazi("cpu") == "cpu" and poz_cihazi("mps") == "mps"
    assert poz_cihazi("auto") in ("cpu", "mps")
    with pytest.raises(ValueError):
        poz_cihazi("gpu")
