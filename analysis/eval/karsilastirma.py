"""Ayni kilitli JSON manifestte yontem karsilastirmasi (0070).

CLI: python -m eval.karsilastirma manifest.json --output rapor.json
Poz dosyalari mono.run_phone results.jsonl bicimindedir; referans da ayni
sozlesmededir. Diger cercevedeki pozlar once BAGIMSIZ sabit kalibrasyonla
donusturulmelidir. Bu arac test pozuna donusum uydurmaz.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from capture.alignment import file_sha256
from eval.protokol import Oge, Sayim, kisi_agirlikli, kisi_bootstrap, manifest_kilidi
from mono.demo import satirdan_iskelet


def konum_hatasi_mm(tahmin, referans, maske=None):
    """Birim/cerceve uyusmazligi acik hata; hizalama ve sifirla doldurma yok."""
    for p in (tahmin, referans):
        p.metrik_gerekli()
    if tahmin.cerceve != referans.cerceve or tahmin.tanim != referans.tanim:
        raise ValueError("konum karsilastirmasi ayni cerceve ve eklem tanimi gerektirir")
    valid = tahmin.gorunur & referans.gorunur
    if maske is not None:
        valid &= maske
    return float(np.linalg.norm(tahmin.noktalar[valid]-referans.noktalar[valid], axis=1).mean()*1000) if valid.any() else None


def karsilastir(manifest, *, kok=Path(".")):
    """Planli her oge her yontemde korunur; eksik tahmin U'ya girer.

    Manifest: {split: test|validation, participants: {id: split}, methods: [ad], faults: [kusur],
samples: [{id, person, session, frame_index, reference: {path, sha256},
reference_labels: {kusur: etiket}, predictions: {ad: {path, sha256}},
prediction_labels: {ad: {kusur: etiket}}}]}.
Etiketler onceden kilitlenmis AYNI hedef tanimina ait olmalidir.
"""
    split = manifest["split"]
    if split not in ("test", "validation"):
        raise ValueError("rapor split test veya validation olmali")
    methods = manifest["methods"]
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("en az iki benzersiz yontem gerekli")
    if not manifest.get("label_definition") or not manifest.get("locked_before_evaluation"):
        raise ValueError("etiket tanimi ve onceden kilitlenmis manifest gerekli")
    faults = manifest["faults"]
    if not faults or len(set(faults)) != len(faults) or not manifest["samples"]:
        raise ValueError("benzersiz kusurlar ve bos olmayan manifest gerekli")
    cache, counts, errors, rows, seen = {}, defaultdict(list), defaultdict(list), [], set()

    def oku(spec, frame):
        if spec is None:
            return None
        path = (kok / spec["path"]).resolve()
        key = (str(path), spec["sha256"])
        if key not in cache:
            if file_sha256(path) != spec["sha256"]:
                raise ValueError(f"dosya hash uyusmazligi: {path.name}")
            content = [json.loads(x) for x in path.read_text().splitlines()]
            if len({x["frame_index"] for x in content}) != len(content):
                raise ValueError("tekrarli frame_index")
            cache[key] = {x["frame_index"]: x for x in content}
        row = cache[key].get(frame)
        if row is None:
            return None
        for field in ("birim_3d", "cerceve_3d", "kaynak_3d"):
            if not row.get(field):
                raise ValueError(f"zorunlu 3B sozlesmesi eksik: {field}")
        return satirdan_iskelet(row)

    for s in manifest["samples"]:
        if s["id"] in seen:
            raise ValueError("tekrarli manifest oge kimligi")
        seen.add(s["id"])
        if manifest["participants"].get(s["person"]) != split:
            raise ValueError("kisi bolmesi sizintisi")
        ref = oku(s["reference"], s["frame_index"])
        preds = {m: oku(s["predictions"].get(m), s["frame_index"]) for m in methods}
        common = ref.gorunur.copy() if ref is not None else np.zeros(13, bool)
        konum_uygun = ref is not None and ref.birim == "metre" and all(
            p is not None and p.birim == "metre" and p.cerceve == ref.cerceve for p in preds.values())
        for p in preds.values():
            common &= p.gorunur if p is not None else False
        metrics = {}
        for m, p in preds.items():
            # Farkli cerceve/normalize veri konum iddiasina giremez;
            # form etiketleri yine ayri hedef olarak karsilastirilir.
            uygun = (ref is not None and p is not None and p.birim == ref.birim == "metre"
                     and p.cerceve == ref.cerceve)
            value = konum_hatasi_mm(p, ref, common) if konum_uygun else None
            metrics[m] = {"mpjpe_mm": value, "ortak_eklem": int(common.sum()) if konum_uygun else 0,
                          "konum_durumu": "hesaplandi" if value is not None else "dogrulanmadi",
                          "referans_eklem": int(ref.gorunur.sum()) if ref is not None else 0,
                          "yontem_gecerli_eklem": int((p.gorunur & ref.gorunur).sum()) if uygun else 0}
            if value is not None:
                errors[(m, s["person"], s["session"])].append(value)
            for fault in faults:
                r = s["reference_labels"].get(fault) if ref is not None else None
                t = s.get("prediction_labels", {}).get(m, {}).get(fault) if p is not None else None
                counts[(m, fault)].append(Oge(s["person"], s["session"], r, t))
        rows.append({"id": s["id"], "yontemler": metrics})
    summaries = {}
    for m in methods:
        persons = defaultdict(list)
        for (method, person, _), vals in errors.items():
            if method == m:
                persons[person].append(float(np.mean(vals)))
        summaries[m] = {
            "mpjpe_mm": float(np.mean([np.mean(v) for v in persons.values()])) if persons else None,
            "konum_kisi_sayisi": len(persons),
            "form": {f: {"sayim": Sayim.ciftlerden((o.referans, o.telefon) for o in os).oranlar(),
                         "kisi_ortalama": kisi_agirlikli(os), "aralik": kisi_bootstrap(os)}
                     for (method, f), os in counts.items() if method == m}}
    return {"manifest_sha256": manifest_kilidi(manifest), "split": split,
            "samples": rows, "methods": summaries, "ustunluk_kabulu": "dogrulanmadi",
            "not": "Betimsel karsilastirma; eslestirilmis fark ve referans belirsizligi olmadan ustunluk yok."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = karsilastir(json.loads(a.manifest.read_text()), kok=a.manifest.parent)
    with a.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
