"""Antrenor: kilit, tekrar karari, olculemez, ipucu sinirlama, set ozeti (sahte modellerle)."""

import numpy as np
import pytest

from mono import antrenor as an
from pose3d.iskelet import REFERANS_ISKELET

J = REFERANS_ISKELET.indeks
UP = np.array([0.0, -1.0, 0.0])           # MediaPipe: y asagi


class _Tanima:
    def __init__(self):
        self.etiket = "squat"

    def tahmin(self, kimlik, zaman, P, G):
        return (self.etiket, 0.9)

    def unut(self, kimlik):
        pass


def _iskelet(bukulme_derece, profil=False):
    """Kameraya donuk (ya da profil) ayakta iskelet; dizler verilen acida bukuk."""
    X = np.zeros((13, 3))
    yan = (np.array([0, 0, 0.1]) if profil else np.array([0.1, 0, 0]))
    X[J("boyun")] = [0, -0.5, 0]
    for t, s in (("sol", 1), ("sag", -1)):
        X[J(f"{t}_omuz")] = X[J("boyun")] + 2 * s * yan
        X[J(f"{t}_dirsek")] = X[J(f"{t}_omuz")] + [0, 0.3, 0]
        X[J(f"{t}_bilek")] = X[J(f"{t}_dirsek")] + [0, 0.25, 0]
        X[J(f"{t}_kalca")] = s * yan
        a = np.radians(bukulme_derece)
        X[J(f"{t}_diz")] = X[J(f"{t}_kalca")] + [0, 0.45, 0]
        X[J(f"{t}_ayak_bilegi")] = X[J(f"{t}_diz")] + 0.45 * np.array([0, np.cos(a), -np.sin(a)])
    return X


class _Model:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]])


def _modeller(p=0.9, gruplar=None):
    return {6: {"ad": "squat", "olculer": ["sig", "derin", "govde_egimi", "diz_onde",
                                            "diz_parmak_onde", "diz_ice"],
                "sayac": (40.0, 20.0, 1.0),
                "gruplar": gruplar or {"front": {"auc": 0.9, "model": _Model(p)},
                                       "profile": {"auc": 0.8}}}}


def _akis(a, sure=12.0, profil=False, hiz=10.0):
    out = []
    for i in range(int(sure * hiz)):
        t = i / hiz
        buk = 0 if t < 2 else 45 * (1 - np.cos(2 * np.pi * (t - 2) / 2.5))    # 2,5 s periyot
        out += a.adim(1, t, np.zeros((65, 2)), np.ones(65, bool), _iskelet(buk, profil))
    return out


def test_kilit_tekrar_ve_duzelt():
    a = an.Antrenor(_Tanima(), _modeller(p=0.9))
    m = _akis(a)
    assert [x for x in m if x["tur"] == "hareket"][0]["hareket"] == "squat"
    tekrar = [x for x in m if x["tur"] == "tekrar"]
    assert len(tekrar) >= 3 and all(x["karar"] == "yanlis" for x in tekrar)
    assert all(x["grup"] == "front" for x in tekrar)


def test_profilde_model_yoksa_olculemez_ve_aci_onerisi():
    a = an.Antrenor(_Tanima(), _modeller(gruplar={"front": {"auc": 0.9, "model": _Model(0.1)},
                                                 "profile": {"auc": 0.6}}))
    m = _akis(a, profil=True)
    ol = [x for x in m if x["tur"] == "olculemez"]
    assert ol and "karsidan dur" in ol[0]["metin"]
    assert not [x for x in m if x["tur"] == "tekrar"]


def test_set_ozeti_yok_gelince():
    tn = _Tanima()
    a = an.Antrenor(tn, _modeller(p=0.1))
    m = _akis(a)
    tn.etiket = "yok"
    for i in range(40):
        m += a.adim(1, 12 + i / 10, np.zeros((65, 2)), np.ones(65, bool), _iskelet(0))
    s = [x for x in m if x["tur"] == "set"]
    assert len(s) == 1 and s[0]["tekrar"] >= 3 and s[0]["dogru"] == s[0]["tekrar"]
    assert a.bitir() == []


def test_en_buyuk_katki_ve_ipucu_metinleri():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 1] > 0.3
    m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(X, y)
    assert an.en_buyuk_katki(m, np.array([0.0, 2.0]), ["a", "diz_onde"]) == "diz_onde"
    assert an.en_buyuk_katki(object(), np.zeros(2), ["a", "b"]) is None
    assert set(_modeller()[6]["olculer"]) <= set(an.IPUCLARI)


def test_tanima_belirsizse_kilitlenmez():
    tn = _Tanima()
    tn.tahmin = lambda *a: ("squat", 0.4)
    a = an.Antrenor(tn, _modeller())
    assert not [x for x in _akis(a, sure=5) if x["tur"] == "hareket"]


@pytest.mark.parametrize("olcu", ["kol_one", "kalca_pike", "govde_yana", "diz_parmak_onde"])
def test_ipucu_ascii(olcu):
    assert an.IPUCLARI[olcu].isascii()


def test_gec_kilitte_ilk_tekrar_kacmaz():
    tn = _Tanima()
    tn.tahmin = lambda k, z, P, G: ("squat", 0.9) if z[-1] >= 5.0 else ("yok", 0.9)
    a = an.Antrenor(tn, _modeller(p=0.1))
    m = _akis(a, sure=7.5)
    kilit = [x for x in m if x["tur"] == "hareket"][0]
    tekrar = [x for x in m if x["tur"] == "tekrar"]
    assert kilit["t"] >= 6.4
    assert tekrar and tekrar[0]["bas"] < 3.0          # kilitten once biten tekrar da sayildi


def test_risk_kapsama_bant_buyudukce_kapsama_azalir():
    from eval.hareket_formu import risk_kapsama
    p = [0.9, 0.8, 0.55, 0.45, 0.2, 0.1]
    y = [True, True, False, True, False, False]
    r = {x["bant"]: x for x in risk_kapsama(p, y, bantlar=(0.0, 0.1))}
    assert r[0.0]["kapsama"] == 1.0 and r[0.0]["dogruluk"] == round(4 / 6, 3)
    assert r[0.1]["kapsama"] == round(4 / 6, 3) and r[0.1]["dogruluk"] == 1.0


def test_kararsiz_bantta_emin_degilim():
    a = an.Antrenor(_Tanima(), _modeller(p=0.55), kararsiz_bant=0.1)
    m = _akis(a)
    bel = [x for x in m if x["tur"] == "belirsiz"]
    assert len(bel) >= 3 and not [x for x in m if x["tur"] == "tekrar"]
    assert all(x["karar"] == "belirsiz" and "emin degilim" in x["metin"] for x in bel)
    a2 = an.Antrenor(_Tanima(), _modeller(p=0.55))          # varsayilan: bant yok
    assert all(x["karar"] == "yanlis" for x in _akis(a2) if x["tur"] == "tekrar")
