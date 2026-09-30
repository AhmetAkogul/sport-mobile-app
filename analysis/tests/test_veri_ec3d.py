"""veri.ec3d: BODY_25 -> REFERANS eslemesi (sentetik kare; gercek veri gerekmez)."""
import numpy as np

from pose3d.iskelet import REFERANS_ISKELET as R
from veri.ec3d import BODY25, kareyi_cevir


def test_body25_eslemesi_ve_boyun_omuz_ortalamasi():
    poz = np.arange(75, dtype=float).reshape(3, 25)
    isk = kareyi_cevir(poz)
    np.testing.assert_allclose(isk.noktalar[R.indeks("sag_diz")], poz[:, BODY25["sag_diz"]])
    np.testing.assert_allclose(isk.noktalar[R.indeks("boyun")], poz[:, [2, 5]].mean(axis=1))
    assert isk.gorunur.all()


def test_eksik_eklem_nan_ve_gorunmez():
    poz = np.ones((3, 25))
    poz[:, 11] = np.nan                          # sag ayak bilegi
    isk = kareyi_cevir(poz)
    assert not isk.gorunur[R.indeks("sag_ayak_bilegi")]
    assert np.isnan(isk.noktalar[R.indeks("sag_ayak_bilegi")]).all()
