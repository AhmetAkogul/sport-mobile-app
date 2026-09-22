"""Kesilen kaydın doğrulanmış tam gruplarını yeni oturuma kurtarır."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from capture.reader import iter_batches
from capture.record import _utc, _write_json


def recover_session(source_dir, output_dir):
    """Kaynağı değiştirmez; bozuk tam grup varsa sessizce atlamadan durur.

    Çağıran önce kaydı durdurmalıdır. Başlangıçta görülen grup kümesi alınır;
    işlem sonunda meta veri veya grup kümesi değişmişse çıktı yayımlanmaz.
    .pending kopyalanmaz/silinmez. Bu işlem kayda devam etmez.
    """
    source = Path(source_dir).resolve()
    output = Path(output_dir).absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError("Kurtarma çıktısı kaynak oturumun içinde olamaz.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    session_path = source / "session.json"
    batches = source / "batches"
    if session_path.is_symlink() or batches.is_symlink():
        raise ValueError("Kaynak meta veri ve grup kökü bağlantı olamaz.")
    original_bytes = session_path.read_bytes()
    metadata = json.loads(original_bytes)
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("Desteklenmeyen oturum sürümü.")
    if metadata.get("status") not in {"recording", "completed", "source_stopped", "failed", "interrupted", "recovered"}:
        raise ValueError("Bilinmeyen kaynak oturum durumu.")
    groups = sorted(batches.iterdir())
    names = [g.name for g in groups]
    metadata = dict(metadata)
    metadata.update(status="recovered", completed_batches=len(groups), recovered_at_utc=_utc())
    # Kaynak kaydın bitiş zamanı bilinmiyorsa bir bitiş zamanı uydurulmaz.
    metadata["recovery"] = {
        "source_session": str(source), "original_status": json.loads(original_bytes)["status"],
        "pending_ignored": (source / ".pending").exists(),
        "original_metadata": "recovery-original-session.json",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capture-recovery-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "batches").mkdir()
        for group in groups:
            if group.is_symlink() or not group.is_dir():
                raise ValueError("Kaynak grup dizini geçersiz.")
            # Bağlantılar takip edilmez; okuyucu ilgili dosyalardakileri reddeder.
            shutil.copytree(group, staging / "batches" / group.name, symlinks=True)
        _write_json(staging / "session.json", metadata)
        with (staging / "frames.jsonl").open("x", encoding="utf-8") as index:
            for frames in iter_batches(staging):
                group = staging / "batches" / f"{frames[0].batch_id:06d}"
                batch = json.loads((group / "frames.json").read_text(encoding="utf-8"))
                index.write(json.dumps(batch, ensure_ascii=False) + "\n")
        (staging / "recovery-original-session.json").write_bytes(original_bytes)
        if session_path.read_bytes() != original_bytes or sorted(p.name for p in batches.iterdir()) != names:
            raise RuntimeError("Kaynak kayıt değişti; önce kayıt işlemini durdurun.")
        # mkdir yarışan veya mevcut hedefi değiştirmeden reddeder.
        output.mkdir(exist_ok=False)
        try:
            for name in ("batches", "frames.jsonl", "recovery-original-session.json", "session.json"):
                (staging / name).rename(output / name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Kesilen kaydı yeni dizine kurtar")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = recover_session(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
