"""Download part of the DCASE 2023 Task 2 development set (CC BY 4.0).

    python -m acoustic.fetch_dcase2023 --machines fan valve bearing --out data/dcase2023

Source: Zenodo record 7882613 (https://zenodo.org/records/7882613).
Each machine is about 270-390 MB zipped. Audio is not committed to this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

RECORD = "https://zenodo.org/records/7882613/files/dev_{m}.zip?download=1"
MACHINES = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]

# Published MD5 digests from https://zenodo.org/api/records/7882613
ZENODO_MD5: Dict[str, str] = {
    "bearing": "8a813bc8d8f156b5395bfccdfac7673c",
    "fan": "9348591e96fb0ad499a1e33b082562fc",
    "gearbox": "b6e55f6a31faa0fc8569ec0afdd53ccf",
    "slider": "b3f8dee36b4718c36d659a4fd1c4afe0",
    "ToyCar": "4e3bf15f4101ed4ed4f1fecde2e2b2a3",
    "ToyTrain": "6b02a6c65eebb3b8b1ae59a6b25bb897",
    "valve": "b2051a2022eadb53cd97581120811cae",
}
DEFAULT_TIMEOUT_S = 60.0
_CHUNK_BYTES = 1024 * 1024


def compute_digest(path: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_digest(path: Path, expected: str) -> None:
    """Verify file digest against 'md5:<hex>', 'sha256:<hex>', or raw hex digest."""
    spec = expected.strip().lower()
    if ":" in spec:
        algo, hex_expected = spec.split(":", 1)
    elif len(spec) == 32:
        algo, hex_expected = "md5", spec
    elif len(spec) == 64:
        algo, hex_expected = "sha256", spec
    else:
        raise ValueError(f"Unrecognized digest specification: {expected!r}")
    actual = compute_digest(path, algo)
    if actual.lower() != hex_expected.lower():
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {algo}:{hex_expected}, got {algo}:{actual}"
        )


def safe_extract_zip(zpath: Path, out: Path) -> None:
    """Extract `zpath` into `out` after checking every member against Zip-Slip traversal."""
    root = out.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ":" in Path(name).drive:
                raise ValueError(f"Unsafe absolute zip member path: {name!r}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"Refusing to extract symlink zip member: {name!r}")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Zip-Slip path traversal blocked: {name!r} escapes {root}")
        zf.extractall(root)


def download_atomic(url: str, dest: Path, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
    """Stream `url` to `dest.part` with explicit socket timeout, then atomically rename."""
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, part.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=_CHUNK_BYTES)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def fetch(
    machine: str,
    out: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    expected_digest: Optional[str] = None,
    url_template: str = RECORD,
) -> Path:
    if machine not in MACHINES and expected_digest is None:
        raise ValueError(f"Unknown machine {machine!r}; expected one of {MACHINES}")
    out.mkdir(parents=True, exist_ok=True)
    train_dir = out / machine / "train"
    if train_dir.is_dir() and any(train_dir.glob("*.wav")):
        print(f"{machine}: already there")
        return out / machine

    digest = expected_digest if expected_digest is not None else ZENODO_MD5.get(machine)
    zpath = out / f"dev_{machine}.zip"

    if zpath.exists():
        try:
            if not zipfile.is_zipfile(zpath):
                raise zipfile.BadZipFile(f"{zpath} is not a valid zip archive")
            if digest:
                verify_digest(zpath, digest)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"{machine}: removing corrupt or incomplete archive {zpath.name} ({exc})")
            zpath.unlink(missing_ok=True)

    if not zpath.exists():
        url = url_template.format(m=machine)
        print(f"{machine}: downloading {url}")
        download_atomic(url, zpath, timeout=timeout)

    try:
        if digest:
            verify_digest(zpath, digest)
        safe_extract_zip(zpath, out)
    except (ValueError, zipfile.BadZipFile):
        zpath.unlink(missing_ok=True)
        raise

    zpath.unlink(missing_ok=True)
    return out / machine


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", nargs="+", default=["fan", "valve", "bearing"], choices=MACHINES)
    ap.add_argument("--out", type=Path, default=Path("data/dcase2023"))
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="HTTP socket timeout in seconds")
    args = ap.parse_args(argv)
    for m in args.machines:
        fetch(m, args.out, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
