import hashlib, os, zipfile
from pathlib import Path
root = Path("/content/outputs/artifact")
out = Path("/content/outputs/tabdpt_regressor_artifact.zip")
files = sorted(p for p in root.rglob("*") if p.is_file())
tmp = out.with_name(out.name + ".tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files:
        info = zipfile.ZipInfo(p.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        with p.open("rb") as src, zf.open(info, "w", force_zip64=True) as dst:
            dst.write(src.read())
os.replace(tmp, out)
print({"members": [p.relative_to(root).as_posix() for p in files], "member_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, "zip_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size})
