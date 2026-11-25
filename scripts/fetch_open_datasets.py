"""On-demand downloader and URL verifier for `lab-ears` open sensory datasets (2024-2026+).

Downloads external machine-audio WAV archives and wet-lab protocol text corpora into
gitignored `data/<dataset_id>/` with atomic `.part` transfers, MD5 checksum verification,
Zip-Slip extraction guards, and commercial-license gates. No audio or external dataset
payloads are ever committed to git.

Usage:
    python scripts/fetch_open_datasets.py --list
    python scripts/fetch_open_datasets.py --verify-urls
    python scripts/fetch_open_datasets.py --dataset lab_bench_protocolqa
    python scripts/fetch_open_datasets.py --dataset bioprobench --subset ERR
    python scripts/fetch_open_datasets.py --dataset dcase2024_task2_dev --machines fan valve
    python scripts/fetch_open_datasets.py --dataset mimii --machines pump --snr-db 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_TIMEOUT_S = 60.0
_CHUNK_BYTES = 1024 * 1024
_USER_AGENT = "lab-ears-open-datasets/1.0"


def _load_open_datasets_module():
    """Load `acoustic/open_datasets.py` directly without triggering top-level numpy imports."""
    mod_name = "_lab_ears_open_datasets_standalone"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    mod_path = REPO_ROOT / "acoustic" / "open_datasets.py"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_OPEN_DS = _load_open_datasets_module()
OPEN_SENSORY_DATASETS = _OPEN_DS.OPEN_SENSORY_DATASETS
OpenSensoryDataset = _OPEN_DS.OpenSensoryDataset
get_open_dataset = _OPEN_DS.get_open_dataset
list_open_datasets = _OPEN_DS.list_open_datasets
normalize_protocol_corpus_to_jsonl = _OPEN_DS.normalize_protocol_corpus_to_jsonl

ZENODO_FILE_TEMPLATES: Dict[str, str] = {
    "dcase2024_task2_dev": "https://zenodo.org/records/10902294/files/dev_{machine}.zip?download=1",
    "dcase2025_task2_dev": "https://zenodo.org/records/15097779/files/dev_{machine}.zip?download=1",
    "dcase2026_task2_dev": "https://zenodo.org/records/19336329/files/dev_{machine}.zip?download=1",
    "mimii_dg": "https://zenodo.org/records/6529888/files/{machine}.zip?download=1",
    "mimii": "https://zenodo.org/records/3384388/files/{snr_key}_{machine}.zip?download=1",
}

PROTOCOL_RAW_URLS: Dict[str, Dict[str, str]] = {
    "lab_bench_protocolqa": {
        "public": "https://raw.githubusercontent.com/Future-House/LAB-Bench/main/ProtocolQA/protocolqa-v1-public.jsonl",
        "openanswer": "https://raw.githubusercontent.com/Future-House/LAB-Bench/main/ProtocolQA/protocolqa-v1-openanswer.jsonl",
    },
    "bioprobench": {
        "ERR": "https://raw.githubusercontent.com/YuyangSunshine/bioprobench/main/Data/ERR_test.json",
        "PQA": "https://raw.githubusercontent.com/YuyangSunshine/bioprobench/main/Data/PQA_test.json",
        "ORD": "https://raw.githubusercontent.com/YuyangSunshine/bioprobench/main/Data/ORD_test.json",
        "GEN": "https://raw.githubusercontent.com/YuyangSunshine/bioprobench/main/Data/GEN_test.json",
    },
}

BIOPLANNER_SAMPLE_IDS: Tuple[str, ...] = (
    "10051",
    "1007",
    "10157",
    "10176",
    "10238",
    "10256",
    "10279",
    "10606",
)


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
    """Verify file digest against `md5:<hex>`, `sha256:<hex>`, or bare 32/64-char hex."""
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
    """Extract `zpath` into `out` after checking every member against Zip-Slip and symlinks."""
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=_CHUNK_BYTES)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def check_url_live(url: str, *, timeout: float = 15.0) -> int:
    """Perform an HTTP HEAD (with GET range fallback) live check on `url` and return status code."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 405):
            get_req = urllib.request.Request(
                url,
                headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-63"},
            )
            with urllib.request.urlopen(get_req, timeout=timeout) as resp:
                return int(resp.status)
        raise


def _check_commercial_gate(ds: OpenSensoryDataset, *, require_commercial_ok: bool) -> None:
    if require_commercial_ok and not ds.commercial_use_ok:
        raise PermissionError(
            f"Dataset {ds.dataset_id!r} carries restricted/non-commercial license {ds.spdx_license!r} "
            f"(commercial_use_ok=False); blocked by --require-commercial-ok."
        )


def fetch_machine_audio_dataset(
    dataset_id: str,
    out_dir: Path,
    *,
    machines: Optional[Sequence[str]] = None,
    snr_db: int = 0,
    timeout: float = DEFAULT_TIMEOUT_S,
    require_commercial_ok: bool = False,
    url_template_override: Optional[str] = None,
    expected_digest_override: Optional[str] = None,
) -> Path:
    """Fetch machine-audio ZIP archives into `out_dir` with MD5 and Zip-Slip checks."""
    ds = get_open_dataset(dataset_id)
    if ds.channel != "machine_audio":
        raise ValueError(f"{dataset_id!r} is channel={ds.channel!r}, not 'machine_audio'")
    _check_commercial_gate(ds, require_commercial_ok=require_commercial_ok)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset_id not in ZENODO_FILE_TEMPLATES and url_template_override is None:
        manifest_path = out_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(ds.to_dict(), indent=2) + "\n", encoding="utf-8")
        return manifest_path

    selected = list(machines) if machines else [ds.machines_or_domains[0]]
    for m in selected:
        if m not in ds.machines_or_domains and expected_digest_override is None and url_template_override is None:
            raise ValueError(f"Unknown machine {m!r} for {dataset_id}; valid: {ds.machines_or_domains}")

    template = url_template_override or ZENODO_FILE_TEMPLATES[dataset_id]
    snr_key = f"{int(snr_db)}_dB"

    for m in selected:
        if dataset_id == "mimii":
            if snr_db not in (-6, 0, 6) and expected_digest_override is None:
                raise ValueError(f"MIMII snr_db must be -6, 0, or 6, got {snr_db}")
            archive_key = f"{snr_key}_{m}"
            zname = f"{archive_key}.zip"
            target_subdir = out_dir / snr_key
        else:
            archive_key = m
            zname = f"dev_{m}.zip" if dataset_id.startswith("dcase") else f"{m}.zip"
            target_subdir = out_dir

        target_subdir.mkdir(parents=True, exist_ok=True)
        zpath = out_dir / zname
        digest = expected_digest_override if expected_digest_override is not None else ds.archive_md5.get(archive_key)
        url = template.format(machine=m, snr_key=snr_key)

        if zpath.exists():
            try:
                if not zipfile.is_zipfile(zpath):
                    raise zipfile.BadZipFile(f"{zpath} is not a valid zip archive")
                if digest:
                    verify_digest(zpath, digest)
            except (OSError, ValueError, zipfile.BadZipFile):
                zpath.unlink(missing_ok=True)

        if not zpath.exists():
            download_atomic(url, zpath, timeout=timeout)

        try:
            if digest:
                verify_digest(zpath, digest)
            safe_extract_zip(zpath, target_subdir)
        except (ValueError, zipfile.BadZipFile):
            zpath.unlink(missing_ok=True)
            raise
        zpath.unlink(missing_ok=True)

    return out_dir


def fetch_protocol_text_dataset(
    dataset_id: str,
    out_dir: Path,
    *,
    subset: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    require_commercial_ok: bool = False,
    source_url_override: Optional[str] = None,
) -> Path:
    """Fetch an open wet-lab protocol corpus and normalize into `out_dir/protocols.jsonl`."""
    ds = get_open_dataset(dataset_id)
    if ds.channel != "protocol_text":
        raise ValueError(f"{dataset_id!r} is channel={ds.channel!r}, not 'protocol_text'")
    _check_commercial_gate(ds, require_commercial_ok=require_commercial_ok)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "protocols.jsonl"

    if source_url_override is not None:
        ext = ".jsonl" if source_url_override.lower().endswith(".jsonl") else ".json"
        raw_dest = out_dir / f"raw_source{ext}"
        download_atomic(source_url_override, raw_dest, timeout=timeout)
        normalize_protocol_corpus_to_jsonl(raw_dest, out_jsonl, dataset_id=dataset_id)
        return out_jsonl

    if dataset_id == "lab_bench_protocolqa":
        sub = subset or "public"
        urls = PROTOCOL_RAW_URLS["lab_bench_protocolqa"]
        if sub not in urls:
            raise ValueError(f"Unknown lab_bench_protocolqa subset={sub!r}; expected {list(urls)}")
        raw_dest = out_dir / f"protocolqa_{sub}.jsonl"
        download_atomic(urls[sub], raw_dest, timeout=timeout)
        normalize_protocol_corpus_to_jsonl(
            raw_dest,
            out_jsonl,
            dataset_id=dataset_id,
            default_domain="protocol_troubleshooting",
        )
        return out_jsonl

    if dataset_id == "bioprobench":
        sub = (subset or "ERR").upper()
        urls = PROTOCOL_RAW_URLS["bioprobench"]
        if sub not in urls:
            raise ValueError(f"Unknown bioprobench subset={sub!r}; expected {list(urls)}")
        raw_dest = out_dir / f"{sub}_test.json"
        download_atomic(urls[sub], raw_dest, timeout=timeout)
        normalize_protocol_corpus_to_jsonl(
            raw_dest,
            out_jsonl,
            dataset_id=dataset_id,
            default_domain=sub.lower(),
        )
        return out_jsonl

    if dataset_id == "bioplanner_bioprot":
        raw_dir = out_dir / "raw_bioprot"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for pid in BIOPLANNER_SAMPLE_IDS:
            url = f"https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/{pid}.json"
            download_atomic(url, raw_dir / f"{pid}.json", timeout=timeout)
        normalize_protocol_corpus_to_jsonl(
            raw_dir,
            out_jsonl,
            dataset_id=dataset_id,
            default_domain="wetlab_planning",
        )
        return out_jsonl

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(ds.to_dict(), indent=2) + "\n", encoding="utf-8")
    return manifest_path


def format_registry_table(datasets: Sequence[OpenSensoryDataset]) -> str:
    header = (
        f"{'dataset_id':<24}{'channel':<16}{'year':>6}{'sr_hz':>8}  "
        f"{'spdx_license':<25}{'comm_ok':<9}{'url'}"
    )
    lines = [header, "-" * len(header)]
    for ds in datasets:
        sr_str = str(ds.sample_rate_hz) if ds.sample_rate_hz is not None else "-"
        comm_str = "yes" if ds.commercial_use_ok else "no"
        lines.append(
            f"{ds.dataset_id:<24}{ds.channel:<16}{ds.release_year:>6}{sr_str:>8}  "
            f"{ds.spdx_license:<25}{comm_str:<9}{ds.url}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="List registered open sensory datasets")
    ap.add_argument("--json", action="store_true", help="Emit JSON output for --list or --verify-urls")
    ap.add_argument("--channel", choices=["protocol_text", "machine_audio"], default=None)
    ap.add_argument("--commercial-only", action="store_true", help="Filter --list to commercial_use_ok=True")
    ap.add_argument("--min-year", type=int, default=None, help="Filter --list by minimum release_year")
    ap.add_argument("--verify-urls", action="store_true", help="Verify HTTP reachability of registered URLs")
    ap.add_argument("--dataset", default=None, help="Dataset ID to fetch on-demand into gitignored data/")
    ap.add_argument("--machines", nargs="+", default=None, help="Machine subset for audio datasets")
    ap.add_argument("--snr-db", type=int, default=0, choices=[-6, 0, 6], help="SNR level in dB for MIMII")
    ap.add_argument("--subset", default=None, help="Protocol corpus subset (e.g. public, ERR, PQA, ORD, GEN)")
    ap.add_argument("--out", type=Path, default=None, help="Destination folder (defaults to data/<dataset_id>)")
    ap.add_argument("--require-commercial-ok", action="store_true", help="Block non-commercial/unlicensed datasets")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="HTTP socket timeout in seconds")
    args = ap.parse_args(argv)

    try:
        filtered = list_open_datasets(
            channel=args.channel,
            commercial_only=args.commercial_only,
            min_year=args.min_year,
        )
        if args.list or (not args.verify_urls and args.dataset is None):
            if args.json:
                print(json.dumps([d.to_dict() for d in filtered], indent=2))
            else:
                print(format_registry_table(filtered))
            if not args.verify_urls and args.dataset is None:
                return 0

        if args.verify_urls:
            results: List[Dict[str, object]] = []
            all_ok = True
            for ds in filtered:
                try:
                    code = check_url_live(ds.url, timeout=args.timeout)
                    ok = 200 <= code < 400
                except Exception as exc:
                    code = 0
                    ok = False
                    if not args.json:
                        print(f"FAIL {ds.dataset_id}: {ds.url} ({exc})", file=sys.stderr)
                if not ok:
                    all_ok = False
                results.append({"dataset_id": ds.dataset_id, "url": ds.url, "status": code, "ok": ok})
                if not args.json and ok:
                    print(f"OK {code} {ds.dataset_id}: {ds.url}")
            if args.json:
                print(json.dumps(results, indent=2))
            if not all_ok:
                return 1

        if args.dataset is not None:
            ds = get_open_dataset(args.dataset)
            dest = args.out if args.out is not None else (DEFAULT_DATA_ROOT / ds.dataset_id)
            if ds.channel == "machine_audio":
                target = fetch_machine_audio_dataset(
                    ds.dataset_id,
                    dest,
                    machines=args.machines,
                    snr_db=args.snr_db,
                    timeout=args.timeout,
                    require_commercial_ok=args.require_commercial_ok,
                )
            else:
                target = fetch_protocol_text_dataset(
                    ds.dataset_id,
                    dest,
                    subset=args.subset,
                    timeout=args.timeout,
                    require_commercial_ok=args.require_commercial_ok,
                )
            print(f"Fetched {ds.dataset_id} -> {target}")
    except (KeyError, ValueError, PermissionError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
