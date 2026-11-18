"""Download part of the DCASE 2023 Task 2 development set (CC BY 4.0).

    python -m acoustic.fetch_dcase2023 --machines fan valve bearing --out data/dcase2023

Source: Zenodo record 7882613 (https://zenodo.org/records/7882613).
Each machine is about 270-390 MB zipped. Audio is not committed to this repo.
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/records/7882613/files/dev_{m}.zip?download=1"
MACHINES = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]


def fetch(machine: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if (out / machine / "train").is_dir():
        print(f"{machine}: already there")
        return out / machine
    zpath = out / f"dev_{machine}.zip"
    if not zpath.exists():
        print(f"{machine}: downloading {RECORD.format(m=machine)}")
        urllib.request.urlretrieve(RECORD.format(m=machine), zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    zpath.unlink()
    return out / machine


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", nargs="+", default=["fan", "valve", "bearing"], choices=MACHINES)
    ap.add_argument("--out", type=Path, default=Path("data/dcase2023"))
    args = ap.parse_args(argv)
    for m in args.machines:
        fetch(m, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
