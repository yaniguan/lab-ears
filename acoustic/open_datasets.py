"""Curated 2024-2026+ open-source sensory dataset registry and loaders for lab-ears (Layer L2).

Unifies two complementary sensory ingestion channels beyond WLP-2018 and DCASE-2023:
1. `machine_audio`: Unsupervised benchtop/industrial machine acoustic anomaly datasets
   (DCASE 2024/2025/2026 Task 2, IMAD-DS 2024, MIMII, MIMII DG, ToyADMOS2), loaded with
   mandatory `[300, 3400] Hz` speech-band excision (`raw_audio_kept=False`) and Nyquist checks.
2. `protocol_text`: Open wet-lab biological and chemical protocol text corpora
   (LAB-Bench ProtocolQA, ChemBench, BioProBench, BioPlanner BioProt, ProtoAct BioP2E),
   loaded from UTF-8 JSONL into normalized `ParsedQuantity` spans via `parse_quantities()`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

SensoryChannel = Literal["protocol_text", "machine_audio"]
VALID_CHANNELS: Tuple[str, str] = ("protocol_text", "machine_audio")

SPEECH_BAND_HZ: Tuple[float, float] = (300.0, 3400.0)
TARGET_SR_HZ: int = 16000
MIN_MACHINE_SR_HZ: int = 8000

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARSE_MODULE_KEY = "_lab_ears_open_datasets_parse_protocol"

_NONCOMMERCIAL_MARKERS = ("-NC", "_NC", "NONCOMMERCIAL", "OTHER-NC")
_UNLICENSED_SPDX = {"NOASSERTION", "NONE", "UNKNOWN", "CUSTOM"}

_DCASE_STEM_RE = re.compile(
    r"^section_(?P<section>\d+)"
    r"(?:_(?P<domain>source|target))?"
    r"_(?P<split>train|test|supplemental)"
    r"(?:_(?P<cond>normal|anomaly))?"
    r"_(?P<clip_id>\d+)"
    r"(?:_(?P<tail>.+))?$"
)
_SNR_DIR_RE = re.compile(r"^(?P<snr>[-+]?\d+(?:\.\d+)?)_?dB(?:_[A-Za-z0-9]+)?$", re.IGNORECASE)


def _load_parse_protocol_module():
    """Load `demo/parse_protocol.py` relative to repository root without cwd sensitivity."""
    cached = sys.modules.get(_PARSE_MODULE_KEY)
    if cached is not None:
        return cached
    mod_path = _REPO_ROOT / "demo" / "parse_protocol.py"
    spec = importlib.util.spec_from_file_location(_PARSE_MODULE_KEY, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load protocol parser at {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PARSE_MODULE_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_acoustic_features():
    """Import numpy/scipy acoustic frontend components for WAV loading and log-mel extraction."""
    from scipy.io import wavfile

    from .features import (
        N_FFT,
        N_MELS,
        SR,
        STFT_SKIRT_HZ,
        load_wav_info,
        log_mel,
        mel_filter_edges_hz,
        retained_mel_bands_hz,
        speech_band_mask,
    )

    return (
        wavfile,
        SR,
        N_FFT,
        N_MELS,
        STFT_SKIRT_HZ,
        load_wav_info,
        log_mel,
        mel_filter_edges_hz,
        retained_mel_bands_hz,
        speech_band_mask,
    )


@dataclass(frozen=True)
class OpenSensoryDataset:
    """Typed registry descriptor for an open-source protocol-text or machine-audio dataset."""

    dataset_id: str
    channel: SensoryChannel
    sample_rate_hz: Optional[int]
    machines_or_domains: Tuple[str, ...]
    url: str
    spdx_license: str
    commercial_use_ok: bool
    title: str = ""
    release_year: int = 2024
    notes: str = ""
    archive_md5: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "machines_or_domains", tuple(self.machines_or_domains))
        object.__setattr__(self, "archive_md5", dict(self.archive_md5))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.dataset_id, str) or not re.fullmatch(r"[a-z0-9_]+", self.dataset_id):
            raise ValueError(
                f"dataset_id must be a non-empty lowercase alphanumeric/underscore identifier, got {self.dataset_id!r}"
            )
        if self.channel not in VALID_CHANNELS:
            raise ValueError(f"channel must be one of {VALID_CHANNELS}, got {self.channel!r}")
        if self.channel == "machine_audio":
            if (
                not isinstance(self.sample_rate_hz, int)
                or isinstance(self.sample_rate_hz, bool)
                or self.sample_rate_hz < MIN_MACHINE_SR_HZ
            ):
                raise ValueError(
                    f"machine_audio dataset {self.dataset_id!r} requires integer sample_rate_hz >= {MIN_MACHINE_SR_HZ}, "
                    f"got {self.sample_rate_hz!r}"
                )
            if 0.5 * float(self.sample_rate_hz) <= SPEECH_BAND_HZ[1]:
                raise ValueError(
                    f"machine_audio dataset {self.dataset_id!r} Nyquist ({0.5 * self.sample_rate_hz} Hz) must exceed "
                    f"speech band upper bound {SPEECH_BAND_HZ[1]} Hz"
                )
        else:
            if self.sample_rate_hz is not None:
                raise ValueError(
                    f"protocol_text dataset {self.dataset_id!r} must have sample_rate_hz=None, got {self.sample_rate_hz!r}"
                )

        if not self.machines_or_domains or not all(
            isinstance(item, str) and item.strip() for item in self.machines_or_domains
        ):
            raise ValueError(f"{self.dataset_id!r}: machines_or_domains must be a non-empty tuple of strings")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ValueError(f"{self.dataset_id!r}: url must be an https:// URL, got {self.url!r}")
        if not isinstance(self.spdx_license, str) or not self.spdx_license.strip():
            raise ValueError(f"{self.dataset_id!r}: spdx_license must be non-empty")
        if not isinstance(self.commercial_use_ok, bool):
            raise ValueError(f"{self.dataset_id!r}: commercial_use_ok must be a bool")
        if not isinstance(self.release_year, int) or isinstance(self.release_year, bool) or self.release_year < 2018:
            raise ValueError(f"{self.dataset_id!r}: invalid release_year={self.release_year!r}")

        lic_upper = self.spdx_license.strip().upper()
        is_restricted = lic_upper in _UNLICENSED_SPDX or any(tok in lic_upper for tok in _NONCOMMERCIAL_MARKERS)
        if is_restricted and self.commercial_use_ok:
            raise ValueError(
                f"{self.dataset_id!r}: restricted/unlicensed spdx_license={self.spdx_license!r} "
                f"cannot set commercial_use_ok=True"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["machines_or_domains"] = list(self.machines_or_domains)
        return d


# Live-verified Zenodo / HuggingFace / GitHub open sensory datasets (verified 2026-11-25)
OPEN_SENSORY_DATASETS: Dict[str, OpenSensoryDataset] = {
    ds.dataset_id: ds
    for ds in (
        OpenSensoryDataset(
            dataset_id="dcase2024_task2_dev",
            channel="machine_audio",
            sample_rate_hz=16000,
            machines_or_domains=("bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"),
            url="https://zenodo.org/records/10902294",
            spdx_license="CC-BY-NC-SA-4.0",
            commercial_use_ok=False,
            title="DCASE 2024 Challenge Task 2 Development Dataset (First-Shot Unsupervised ASD)",
            release_year=2024,
            notes=(
                "16 kHz mono 10 s clips across 7 machines with hidden operational attributes on selected "
                "sections; Zenodo record 10902294 (Hitachi & NTT, CC BY-NC-SA 4.0)."
            ),
            archive_md5={
                "bearing": "b4b750604c096d50586520a2fe6359a7",
                "fan": "636a17ea9a130683f3a413724b8f5493",
                "gearbox": "cf146c49d78ced7532206ca0a83a1444",
                "slider": "55aca264168ab94d7bd2bc50542e8a3d",
                "ToyCar": "1ada4ccd2fa5b07020f25fa017321593",
                "ToyTrain": "7a27477369497ba3e027b530fc40ef09",
                "valve": "d03786b9502275226c6879f37a6cfe31",
            },
        ),
        OpenSensoryDataset(
            dataset_id="dcase2025_task2_dev",
            channel="machine_audio",
            sample_rate_hz=16000,
            machines_or_domains=("bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"),
            url="https://zenodo.org/records/15097779",
            spdx_license="CC-BY-NC-SA-4.0",
            commercial_use_ok=False,
            title="DCASE 2025 Challenge Task 2 Development Dataset (First-Shot Unsupervised ASD)",
            release_year=2025,
            notes=(
                "16 kHz mono 10-12 s clips across 7 machines with supplemental clean/noise adaptation "
                "clips; Zenodo record 15097779 (Hitachi & NTT, CC BY-NC-SA 4.0)."
            ),
            archive_md5={
                "bearing": "b3a13693b69b477ce814db8b40783955",
                "fan": "61b6aae6d5ebb8cb937faa86741fbf47",
                "gearbox": "9542948d980fd575426f60f2a41060b8",
                "slider": "6c1f2d2ef0bc1a960db5938adb8425d1",
                "ToyCar": "16bd2069065281ea7ac0e96894f8592c",
                "ToyTrain": "c2524ff503bc1631ca308dc840bc5010",
                "valve": "5640c0dc03982107ef354fe9480c9a83",
            },
        ),
        OpenSensoryDataset(
            dataset_id="dcase2026_task2_dev",
            channel="machine_audio",
            sample_rate_hz=16000,
            machines_or_domains=(
                "bearingEmu",
                "fan",
                "gearboxEmu",
                "sliderEmu",
                "ToyCar_r2",
                "ToyCarEmu",
                "valveEmu",
            ),
            url="https://zenodo.org/records/19336329",
            spdx_license="CC-BY-NC-SA-4.0",
            commercial_use_ok=False,
            title="DCASE 2026 Challenge Task 2 Development Dataset (Noise-Aware Two-Channel ASD)",
            release_year=2026,
            notes=(
                "16 kHz two-channel near/far-field 10-12 s clips for noise-aware machine condition "
                "monitoring; Zenodo record 19336329 (Hitachi & NTT, CC BY-NC-SA 4.0)."
            ),
            archive_md5={
                "bearingEmu": "82c2d04c5f48d09842152c34ef273667",
                "fan": "58b79bac4b05bf14a11e9d53e6cdfae9",
                "gearboxEmu": "21ef8b6dea66472bc4e7c91a175efdb7",
                "sliderEmu": "1cbc87a09883a1aa05a6b2a4308efb83",
                "ToyCar_r2": "353201db217e25377a7e7eccf9c1c8a7",
                "ToyCarEmu": "187b0b3f84d2452d8944d5c831305f15",
                "valveEmu": "f7f74e59773eefa512f24625d88d58f0",
            },
        ),
        OpenSensoryDataset(
            dataset_id="imad_ds_2024",
            channel="machine_audio",
            sample_rate_hz=16000,
            machines_or_domains=("RoboticArm", "BrushlessMotor"),
            url="https://zenodo.org/records/12665499",
            spdx_license="CC-BY-SA-4.0",
            commercial_use_ok=True,
            title="IMAD-DS: Industrial Multi-Sensor Anomaly Detection Under Domain Shift",
            release_year=2024,
            notes=(
                "16 kHz analog microphone + 6.7 kHz 3-axis accelerometer/gyroscope recordings on a "
                "wafer-handling robotic arm and brushless motor; Zenodo record 12665499 (CC BY-SA 4.0)."
            ),
            archive_md5={
                "RoboticArm.7z": "a64498b7dcc297946a7fb8366e38ba33",
                "BrushlessMotor.7z": "d5f1a49b928b1b0b01c3d6de930cd277",
            },
        ),
        OpenSensoryDataset(
            dataset_id="mimii",
            channel="machine_audio",
            sample_rate_hz=16000,
            machines_or_domains=("fan", "pump", "slider", "valve"),
            url="https://zenodo.org/records/3384388",
            spdx_license="CC-BY-SA-4.0",
            commercial_use_ok=True,
            title="MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation",
            release_year=2019,
            notes=(
                "8-channel 16 kHz 10 s WAV clips for industrial valves, pumps, fans, and slide rails "
                "across -6 dB, 0 dB, and +6 dB SNR; Zenodo record 3384388 (Hitachi, CC BY-SA 4.0)."
            ),
            archive_md5={
                "-6_dB_fan": "f02ae808a58d84b6815b7ec38ff30879",
                "-6_dB_pump": "d20b783a0ff9c93d58f452f98c37b112",
                "-6_dB_slider": "49913eda7d37f182cbf8ed5c984140e0",
                "-6_dB_valve": "fdfaf185fea61b21e11952a070a4ada7",
                "0_dB_fan": "6354d1cc2165c52168f9ef1bcd9c7c52",
                "0_dB_pump": "488748295c3f60b25de07b58fe75b049",
                "0_dB_slider": "4d674c21474f0646ecd75546db6c0c4e",
                "0_dB_valve": "178478eb0d11c79080a35562bfdeee71",
                "6_dB_fan": "0890f7d3c2fd8448634e69ff1d66dd47",
                "6_dB_pump": "a09ba6060c10fc09cd4c8770213b0b9f",
                "6_dB_slider": "838c2b3441858359c4704ef13a1b27ff",
                "6_dB_valve": "fe5fb7c337cd701b1d31dc641e621892",
            },
        ),
        OpenSensoryDataset(
            dataset_id="mimii_dg",
            channel="machine_audio",
            sample_rate_hz=16000,
            machines_or_domains=("bearing", "fan", "gearbox", "slider", "valve"),
            url="https://zenodo.org/records/6529888",
            spdx_license="CC-BY-NC-SA-4.0",
            commercial_use_ok=False,
            title="MIMII DG: Malfunctioning Industrial Machine Sound Dataset for Domain Generalization",
            release_year=2022,
            notes=(
                "16 kHz source/target domain-shift clips across 5 industrial machine classes; Zenodo API "
                "lists cc-by-4.0 while Conditions of Use text specifies CC BY-NC-SA 4.0 (fail-closed NC)."
            ),
            archive_md5={
                "bearing": "6381a00f9efc0ced779c8ad847e4ff59",
                "fan": "a1a9b488934a82426bacc933d87aacde",
                "gearbox": "c165dfef8c404256bd719c6fe1f7036f",
                "slider": "8c3a5466cf53e54872fd94998a67bfac",
                "valve": "1da37b2e82942dfba720984541e2ef60",
            },
        ),
        OpenSensoryDataset(
            dataset_id="toyadmos2",
            channel="machine_audio",
            sample_rate_hz=48000,
            machines_or_domains=("ToyCar", "ToyTrain"),
            url="https://zenodo.org/records/4580270",
            spdx_license="LicenseRef-ToyADMOS2-NC",
            commercial_use_ok=False,
            title="ToyADMOS2: Miniature-Machine Operating Sounds Under Domain Shift Conditions",
            release_year=2021,
            notes=(
                "48 kHz multi-microphone normal and damaged miniature-machine recordings; Zenodo record "
                "4580270 under custom non-commercial license (LICENSE.pdf)."
            ),
        ),
        OpenSensoryDataset(
            dataset_id="lab_bench_protocolqa",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=(
                "molecular_biology",
                "cell_culture",
                "biochemistry",
                "protocol_troubleshooting",
            ),
            url="https://huggingface.co/datasets/futurehouse/lab-bench",
            spdx_license="CC-BY-SA-4.0",
            commercial_use_ok=True,
            title="FutureHouse LAB-Bench ProtocolQA Wet-Lab Protocol Corpus",
            release_year=2024,
            notes=(
                "Published wet-lab protocols paired with experimental failure and parameter reasoning "
                "questions (arXiv:2407.10362, updated 2025-2026, CC BY-SA 4.0)."
            ),
        ),
        OpenSensoryDataset(
            dataset_id="chembench",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=(
                "organic_chemistry",
                "analytical_chemistry",
                "inorganic_chemistry",
                "chemical_safety",
            ),
            url="https://huggingface.co/datasets/jablonkagroup/ChemBench",
            spdx_license="MIT",
            commercial_use_ok=True,
            title="ChemBench: Chemical Synthesis, Analytical Procedures, and Lab Safety Corpus",
            release_year=2026,
            notes=(
                "Curated chemistry reasoning, stoichiometry, synthesis, and lab safety tasks "
                "(jablonkagroup/ChemBench, updated 2026-04, MIT License)."
            ),
        ),
        OpenSensoryDataset(
            dataset_id="bioprobench",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=(
                "protocol_qa",
                "step_ordering",
                "error_correction",
                "protocol_generation",
            ),
            url="https://huggingface.co/datasets/BioProBench/BioProBench",
            spdx_license="CC-BY-NC-4.0",
            commercial_use_ok=False,
            title="BioProBench: Biological Protocol Reasoning Benchmark (ICML 2026)",
            release_year=2026,
            notes=(
                "22,413 human-written wet-lab protocols and 523,784 structured PQA/ORD/ERR/GEN/REA "
                "instances (arXiv:2505.07889, ICML 2026, CC BY-NC 4.0 non-commercial)."
            ),
        ),
        OpenSensoryDataset(
            dataset_id="bioplanner_bioprot",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=("pcr_sequencing", "molecular_cloning", "wetlab_planning"),
            url="https://github.com/bioplanner/bioplanner",
            spdx_license="NOASSERTION",
            commercial_use_ok=False,
            title="BioPlanner (BioProt): Wet-Lab Protocols and Executable Pseudocode Corpus",
            release_year=2024,
            notes=(
                "protocols.io-derived wet-lab biology procedures paired with planning pseudocode "
                "(arXiv:2310.10632, repo updated 2024-07; no explicit repository license, fail-closed)."
            ),
        ),
        OpenSensoryDataset(
            dataset_id="protoact_biop2e",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=("cell_culture", "embodied_pipetting", "robot_actions"),
            url="https://github.com/gjm112233/ProtoAct",
            spdx_license="NOASSERTION",
            commercial_use_ok=False,
            title="ProtoAct (BioP2E): Wet-Lab Cell-Culture Protocols Aligned to Embodied Robot Actions",
            release_year=2026,
            notes=(
                "Free-form wet-lab SOPs mapped to state-aware robot action sequences "
                "(arXiv:2608.01690, 2026-08; no explicit repository license, fail-closed)."
            ),
        ),
    )
}


def get_open_dataset(dataset_id: str) -> OpenSensoryDataset:
    """Return a registered `OpenSensoryDataset` by `dataset_id`."""
    if dataset_id not in OPEN_SENSORY_DATASETS:
        known = ", ".join(sorted(OPEN_SENSORY_DATASETS))
        raise KeyError(f"Unknown dataset_id={dataset_id!r}; expected one of: {known}")
    return OPEN_SENSORY_DATASETS[dataset_id]


def list_open_datasets(
    *,
    channel: Optional[str] = None,
    commercial_only: bool = False,
    min_year: Optional[int] = None,
) -> List[OpenSensoryDataset]:
    """Filter registered open sensory datasets by channel, commercial permission, or release year."""
    if channel is not None and channel not in VALID_CHANNELS:
        raise ValueError(f"channel filter must be one of {VALID_CHANNELS}, got {channel!r}")
    out: List[OpenSensoryDataset] = []
    for ds in OPEN_SENSORY_DATASETS.values():
        if channel is not None and ds.channel != channel:
            continue
        if commercial_only and not ds.commercial_use_ok:
            continue
        if min_year is not None and ds.release_year < int(min_year):
            continue
        out.append(ds)
    return out


def verify_audio_privacy_and_nyquist(
    *,
    source_sr_hz: int,
    target_sr_hz: int = TARGET_SR_HZ,
    drop_speech_band: bool = True,
    allow_speech_band: bool = False,
    allow_upsample: bool = False,
) -> List[List[int]]:
    """Enforce `[300, 3400] Hz` Hann-skirt speech-band exclusion and source/target Nyquist bounds.

    Returns outward-rounded integer Hz retained mel bands `[lo_hz, hi_hz]` when valid, or
    raises `ValueError` if speech privacy or Nyquist realizability is violated.
    """
    (
        _wavfile,
        _sr_const,
        n_fft,
        n_mels,
        _skirt_const,
        _load_wav_info,
        _log_mel,
        mel_filter_edges_hz,
        retained_mel_bands_hz,
        speech_band_mask,
    ) = _require_acoustic_features()

    if isinstance(source_sr_hz, bool) or not isinstance(source_sr_hz, int) or source_sr_hz <= 0:
        raise ValueError(f"source_sr_hz must be a positive int, got {source_sr_hz!r}")
    if isinstance(target_sr_hz, bool) or not isinstance(target_sr_hz, int) or target_sr_hz < MIN_MACHINE_SR_HZ:
        raise ValueError(
            f"target_sr_hz={target_sr_hz!r} must be >= {MIN_MACHINE_SR_HZ} Hz for machine acoustic monitoring"
        )
    if source_sr_hz < MIN_MACHINE_SR_HZ:
        raise ValueError(
            f"Source WAV sample rate {source_sr_hz} Hz is below {MIN_MACHINE_SR_HZ} Hz machine acoustic floor"
        )
    if source_sr_hz < target_sr_hz and not allow_upsample:
        raise ValueError(
            f"Source WAV sample rate {source_sr_hz} Hz is below target {target_sr_hz} Hz "
            f"(Nyquist {0.5 * source_sr_hz:.1f} Hz < {0.5 * target_sr_hz:.1f} Hz; upsampling blocked)"
        )
    if not drop_speech_band and not allow_speech_band:
        raise ValueError(
            "Speech-band [300, 3400] Hz retention blocked: pass drop_speech_band=True "
            "(or allow_speech_band=True only for isolated offline ablation)"
        )

    eff_nyquist_hz = 0.5 * float(target_sr_hz if allow_upsample else min(source_sr_hz, target_sr_hz))
    if drop_speech_band and eff_nyquist_hz <= SPEECH_BAND_HZ[1]:
        raise ValueError(
            f"Effective Nyquist {eff_nyquist_hz:.1f} Hz does not exceed speech band upper bound "
            f"{SPEECH_BAND_HZ[1]:.1f} Hz; high-band cavitation/bearing energy is unrecoverable"
        )

    skirt_hz = 2.0 * float(target_sr_hz) / float(n_fft)
    edges, _ = mel_filter_edges_hz(sr=target_sr_hz, n_mels=n_mels)
    if drop_speech_band:
        keep = speech_band_mask(edges_hz=edges, skirt_hz=skirt_hz)
        if not bool(keep.any()):
            raise ValueError("No mel filters survived [300, 3400] Hz Hann-skirt speech-band drop")
        kept_edges = edges[keep]
        for lo_e, hi_e in kept_edges:
            if not (float(hi_e) + skirt_hz <= SPEECH_BAND_HZ[0] or float(lo_e) - skirt_hz >= SPEECH_BAND_HZ[1]):
                raise ValueError(
                    f"Retained mel filter [{lo_e:.2f}, {hi_e:.2f}] Hz (+/- {skirt_hz:.2f} Hz skirt) "
                    f"overlaps speech band {SPEECH_BAND_HZ} Hz"
                )

    retained = retained_mel_bands_hz(
        drop_speech_band=drop_speech_band,
        sr=target_sr_hz,
        n_fft=n_fft,
        n_mels=n_mels,
        skirt_hz=skirt_hz,
    )
    if not retained:
        raise ValueError("retained_mel_bands_hz produced an empty band set")

    target_nyquist_hz = 0.5 * float(target_sr_hz)
    for lo_i, hi_i in retained:
        if lo_i < 0 or lo_i > hi_i or float(hi_i) > target_nyquist_hz:
            raise ValueError(
                f"Retained band [{lo_i}, {hi_i}] Hz violates [0, Nyquist={target_nyquist_hz:.1f}] Hz"
            )
        if not allow_upsample and float(hi_i) > 0.5 * float(source_sr_hz):
            raise ValueError(
                f"Retained band [{lo_i}, {hi_i}] Hz exceeds source Nyquist {0.5 * source_sr_hz:.1f} Hz"
            )
        if drop_speech_band and not (float(hi_i) <= SPEECH_BAND_HZ[0] or float(lo_i) >= SPEECH_BAND_HZ[1]):
            raise ValueError(
                f"Outward-rounded band [{lo_i}, {hi_i}] Hz overlaps speech band {SPEECH_BAND_HZ} Hz"
            )
    return retained


@dataclass(frozen=True)
class AcousticClipDescriptor:
    """Lightweight filesystem/manifest metadata for a single machine-audio WAV clip."""

    dataset_id: str
    machine: str
    split: str
    label: Optional[str]
    is_anomaly: Optional[bool]
    domain: str
    section_or_model_id: str
    snr_db: Optional[float]
    attributes_hidden: bool
    wav_path: Path


@dataclass
class AcousticClipRecord:
    """Featurized machine-audio clip with speech-band privacy metadata and zero raw-audio storage."""

    dataset_id: str
    machine: str
    split: str
    label: Optional[str]
    is_anomaly: Optional[bool]
    domain: str
    section_or_model_id: str
    snr_db: Optional[float]
    attributes_hidden: bool
    wav_path: str
    sample_rate_hz: int
    source_sample_rate_hz: int
    n_channels_raw: int
    duration_s: float
    speech_band_dropped: bool
    raw_audio_kept: bool
    retained_bands_hz: List[List[int]]
    mel: Any  # np.ndarray of shape (n_frames, n_kept_mels), dtype float32

    def to_manifest_dict(self, *, relative_to: Optional[Path] = None) -> Dict[str, Any]:
        p = Path(self.wav_path)
        rel = str(p.relative_to(relative_to).as_posix()) if relative_to is not None else p.as_posix()
        return {
            "dataset_id": self.dataset_id,
            "machine": self.machine,
            "split": self.split,
            "label": self.label,
            "is_anomaly": self.is_anomaly,
            "domain": self.domain,
            "section_or_model_id": self.section_or_model_id,
            "snr_db": self.snr_db,
            "attributes_hidden": self.attributes_hidden,
            "wav_path": rel,
            "sample_rate_hz": self.sample_rate_hz,
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "n_channels_raw": self.n_channels_raw,
            "duration_s": self.duration_s,
            "speech_band_dropped": self.speech_band_dropped,
            "raw_audio_kept": self.raw_audio_kept,
            "retained_bands_hz": self.retained_bands_hz,
        }


def _featurize_descriptor(
    desc: AcousticClipDescriptor,
    *,
    drop_speech_band: bool = True,
    allow_speech_band: bool = False,
    resample: bool = False,
    allow_upsample: bool = False,
) -> AcousticClipRecord:
    (
        wavfile,
        sr_target,
        _n_fft,
        _n_mels,
        _skirt,
        load_wav_info,
        log_mel,
        _mel_edges,
        _ret_bands,
        _mask,
    ) = _require_acoustic_features()

    if not desc.wav_path.is_file():
        raise FileNotFoundError(f"WAV clip does not exist: {desc.wav_path}")

    raw_sr, raw_arr = wavfile.read(desc.wav_path)
    src_sr = int(raw_sr)
    n_channels = 1 if raw_arr.ndim == 1 else int(raw_arr.shape[1])

    retained_bands = verify_audio_privacy_and_nyquist(
        source_sr_hz=src_sr,
        target_sr_hz=sr_target,
        drop_speech_band=drop_speech_band,
        allow_speech_band=allow_speech_band,
        allow_upsample=allow_upsample,
    )
    min_src = MIN_MACHINE_SR_HZ if allow_upsample else sr_target
    waveform, verified_src_sr = load_wav_info(
        desc.wav_path,
        target_sr=sr_target,
        resample=resample,
        min_source_sr=min_src,
    )
    mel = log_mel(waveform, drop_speech_band=drop_speech_band)
    duration_s = round(float(waveform.size) / float(sr_target), 6)
    del waveform  # Never retain raw waveform samples on loaded dataset records

    return AcousticClipRecord(
        dataset_id=desc.dataset_id,
        machine=desc.machine,
        split=desc.split,
        label=desc.label,
        is_anomaly=desc.is_anomaly,
        domain=desc.domain,
        section_or_model_id=desc.section_or_model_id,
        snr_db=desc.snr_db,
        attributes_hidden=desc.attributes_hidden,
        wav_path=str(desc.wav_path),
        sample_rate_hz=sr_target,
        source_sample_rate_hz=verified_src_sr,
        n_channels_raw=n_channels,
        duration_s=duration_s,
        speech_band_dropped=bool(drop_speech_band),
        raw_audio_kept=False,
        retained_bands_hz=retained_bands,
        mel=mel,
    )


def parse_dcase_filename(filename: str, *, fallback_split: str = "test") -> Dict[str, Any]:
    """Parse DCASE 2023/2024/2025/2026 Task 2 clip filenames (including hidden-attribute clips)."""
    stem = Path(filename).stem
    m = _DCASE_STEM_RE.match(stem)
    if m is not None:
        section_id = f"section_{m.group('section')}"
        domain = m.group("domain") or "unknown"
        split = m.group("split") or fallback_split
        cond = m.group("cond")
        tail = m.group("tail")
        if cond is None and split == "train":
            cond = "normal"
        is_anom: Optional[bool] = None if cond is None else (cond == "anomaly")
        attrs_hidden = tail is None or tail.strip() == ""
        return {
            "section_or_model_id": section_id,
            "domain": domain,
            "split": split,
            "label": cond,
            "is_anomaly": is_anom,
            "attributes_hidden": attrs_hidden,
        }

    low = stem.lower()
    domain = "source" if "_source_" in low else ("target" if "_target_" in low else "unknown")
    if "_anomaly_" in low or low.endswith("_anomaly"):
        cond = "anomaly"
        is_anom = True
    elif "_normal_" in low or low.endswith("_normal") or fallback_split == "train":
        cond = "normal"
        is_anom = False
    else:
        cond = None
        is_anom = None
    sec_m = re.search(r"(section_\d+)", low)
    return {
        "section_or_model_id": sec_m.group(1) if sec_m else "section_00",
        "domain": domain,
        "split": fallback_split,
        "label": cond,
        "is_anomaly": is_anom,
        "attributes_hidden": True,
    }


def _resolve_dcase_machine_dir(root: Path, machine: str) -> Path:
    candidates = [
        root / machine,
        root / "dev_data" / machine,
        root / f"dev_{machine}" / machine,
        root,
    ]
    for cand in candidates:
        if cand.is_dir() and ((cand / "train").is_dir() or (cand / "test").is_dir()):
            if cand == root and root.name != machine and (root / machine).exists():
                continue
            return cand
    raise FileNotFoundError(
        f"Could not locate DCASE Task 2 directory for machine={machine!r} under {root} "
        f"(expected <root>/{machine}/train or <root>/{machine}/test)"
    )


def discover_dcase_task2_layout(
    root: Path | str,
    machine: str,
    *,
    dataset_id: str = "dcase2024_task2_dev",
    splits: Sequence[str] = ("train", "test"),
) -> List[AcousticClipDescriptor]:
    """Discover DCASE 2024/2025/2026 Task 2 `train/`, `test/`, and optional `supplemental/` WAVs."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"DCASE dataset root does not exist: {root_path}")
    mdir = _resolve_dcase_machine_dir(root_path, machine)

    descriptors: List[AcousticClipDescriptor] = []
    for sp in splits:
        sp_dir = mdir / sp
        if not sp_dir.is_dir():
            if sp == "supplemental":
                continue
            raise FileNotFoundError(f"Missing split directory '{sp_dir}' for machine={machine!r}")
        wavs = sorted(sp_dir.glob("*.wav"))
        if not wavs and sp in ("train", "test"):
            raise ValueError(f"No .wav files found in '{sp_dir}'")
        for w in wavs:
            info = parse_dcase_filename(w.name, fallback_split=sp)
            if sp == "train" and info["is_anomaly"] is True:
                raise ValueError(
                    f"Unsupervised one-class violation: anomalous clip {w.name!r} found inside train/ split"
                )
            descriptors.append(
                AcousticClipDescriptor(
                    dataset_id=dataset_id,
                    machine=machine,
                    split=sp,
                    label=info["label"],
                    is_anomaly=info["is_anomaly"],
                    domain=info["domain"],
                    section_or_model_id=info["section_or_model_id"],
                    snr_db=None,
                    attributes_hidden=bool(info["attributes_hidden"]),
                    wav_path=w,
                )
            )
    return descriptors


def load_dcase_task2_clips(
    root: Path | str,
    machine: str,
    *,
    dataset_id: str = "dcase2024_task2_dev",
    splits: Sequence[str] = ("train", "test"),
    drop_speech_band: bool = True,
    allow_speech_band: bool = False,
    resample: bool = False,
    allow_upsample: bool = False,
    max_clips_per_split: Optional[int] = None,
) -> List[AcousticClipRecord]:
    """Load DCASE 2024/2025/2026 Task 2 WAV clips into speech-stripped `AcousticClipRecord`s."""
    descriptors = discover_dcase_task2_layout(root, machine, dataset_id=dataset_id, splits=splits)
    if max_clips_per_split is not None:
        if max_clips_per_split < 1:
            raise ValueError(f"max_clips_per_split must be >= 1, got {max_clips_per_split}")
        counts: Dict[str, int] = {}
        filtered: List[AcousticClipDescriptor] = []
        for d in descriptors:
            c = counts.get(d.split, 0)
            if c < max_clips_per_split:
                filtered.append(d)
                counts[d.split] = c + 1
        descriptors = filtered

    return [
        _featurize_descriptor(
            d,
            drop_speech_band=drop_speech_band,
            allow_speech_band=allow_speech_band,
            resample=resample,
            allow_upsample=allow_upsample,
        )
        for d in descriptors
    ]


def _infer_snr_db_from_parts(parts: Sequence[str]) -> Optional[float]:
    for token in parts:
        m = _SNR_DIR_RE.match(token)
        if m is not None:
            return float(m.group("snr"))
    return None


def discover_mimii_layout(
    root: Path | str,
    machine: str,
    *,
    dataset_id: str = "mimii",
    model_ids: Optional[Sequence[str]] = None,
    snr_db: Optional[float] = None,
    normal_train_fraction: float = 0.8,
) -> List[AcousticClipDescriptor]:
    """Discover Hitachi MIMII (`<snr>/<machine>/id_XX/{normal,abnormal}/*.wav`) or MIMII DG layouts.

    In classic MIMII layouts, `normal/` clips are deterministically partitioned into one-class
    `train` (`normal_train_fraction`) and held-out `test` (`1 - normal_train_fraction`), while all
    `abnormal/` clips are assigned to `test` so unsupervised detectors never see anomalies during fit.
    """
    if not (0.0 < normal_train_fraction < 1.0):
        raise ValueError(f"normal_train_fraction must lie strictly in (0, 1), got {normal_train_fraction}")
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"MIMII root directory does not exist: {root_path}")

    # Support MIMII DG (`<root>/<machine>/{train,test}/*.wav`) transparently
    dg_cand = root_path / machine if (root_path / machine / "train").is_dir() else (
        root_path if root_path.name == machine and (root_path / "train").is_dir() else None
    )
    if dg_cand is not None:
        return discover_dcase_task2_layout(root_path, machine, dataset_id=dataset_id, splits=("train", "test"))

    # Locate `<machine>/id_XX` directories under `<root>`, `<root>/<machine>`, or `<root>/<snr_dB>/<machine>`
    id_dirs: List[Tuple[Path, Optional[float]]] = []
    search_bases: List[Tuple[Path, Optional[float]]] = []
    if root_path.name == machine:
        search_bases.append((root_path, _infer_snr_db_from_parts(root_path.parts)))
    if (root_path / machine).is_dir():
        search_bases.append((root_path / machine, _infer_snr_db_from_parts(root_path.parts)))
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        child_snr = _infer_snr_db_from_parts([child.name])
        if child_snr is not None:
            if (child / machine).is_dir():
                search_bases.append((child / machine, child_snr))
            elif any(p.is_dir() and p.name.startswith("id_") for p in child.iterdir()):
                if machine.lower() in child.name.lower():
                    search_bases.append((child, child_snr))

    seen_id_dirs: set[Path] = set()
    allowed_models = set(model_ids) if model_ids is not None else None
    for base_dir, detected_snr in search_bases:
        if snr_db is not None and (detected_snr is None or not math.isclose(detected_snr, float(snr_db), abs_tol=1e-6)):
            continue
        for sub in sorted(base_dir.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("id_"):
                continue
            if allowed_models is not None and sub.name not in allowed_models:
                continue
            resolved = sub.resolve()
            if resolved not in seen_id_dirs:
                seen_id_dirs.add(resolved)
                id_dirs.append((sub, detected_snr))

    if not id_dirs:
        raise FileNotFoundError(
            f"No MIMII model directories (`id_00`, `id_02`, ...) found for machine={machine!r} under {root_path}"
        )

    descriptors: List[AcousticClipDescriptor] = []
    for id_dir, det_snr in id_dirs:
        normal_dir = id_dir / "normal"
        abnormal_dir = id_dir / "abnormal"
        normal_wavs = sorted(normal_dir.glob("*.wav")) if normal_dir.is_dir() else []
        abnormal_wavs = sorted(abnormal_dir.glob("*.wav")) if abnormal_dir.is_dir() else []
        if not normal_wavs:
            raise ValueError(f"MIMII model directory '{id_dir}' contains zero normal/*.wav files")

        n_train = max(1, min(len(normal_wavs) - 1, int(round(len(normal_wavs) * normal_train_fraction)))) if len(normal_wavs) >= 2 else 1
        for idx, w in enumerate(normal_wavs):
            split = "train" if idx < n_train else "test"
            descriptors.append(
                AcousticClipDescriptor(
                    dataset_id=dataset_id,
                    machine=machine,
                    split=split,
                    label="normal",
                    is_anomaly=False,
                    domain="source",
                    section_or_model_id=id_dir.name,
                    snr_db=det_snr,
                    attributes_hidden=False,
                    wav_path=w,
                )
            )
        for w in abnormal_wavs:
            descriptors.append(
                AcousticClipDescriptor(
                    dataset_id=dataset_id,
                    machine=machine,
                    split="test",
                    label="anomaly",
                    is_anomaly=True,
                    domain="source",
                    section_or_model_id=id_dir.name,
                    snr_db=det_snr,
                    attributes_hidden=False,
                    wav_path=w,
                )
            )
    return descriptors


def load_mimii_clips(
    root: Path | str,
    machine: str,
    *,
    dataset_id: str = "mimii",
    model_ids: Optional[Sequence[str]] = None,
    snr_db: Optional[float] = None,
    normal_train_fraction: float = 0.8,
    drop_speech_band: bool = True,
    allow_speech_band: bool = False,
    resample: bool = False,
    allow_upsample: bool = False,
    max_clips_per_split: Optional[int] = None,
) -> List[AcousticClipRecord]:
    """Load MIMII (multi-channel or mono WAV) or MIMII DG clips with speech-band excision."""
    descriptors = discover_mimii_layout(
        root,
        machine,
        dataset_id=dataset_id,
        model_ids=model_ids,
        snr_db=snr_db,
        normal_train_fraction=normal_train_fraction,
    )
    if max_clips_per_split is not None:
        if max_clips_per_split < 1:
            raise ValueError(f"max_clips_per_split must be >= 1, got {max_clips_per_split}")
        counts: Dict[str, int] = {}
        filtered: List[AcousticClipDescriptor] = []
        for d in descriptors:
            c = counts.get(d.split, 0)
            if c < max_clips_per_split:
                filtered.append(d)
                counts[d.split] = c + 1
        descriptors = filtered

    return [
        _featurize_descriptor(
            d,
            drop_speech_band=drop_speech_band,
            allow_speech_band=allow_speech_band,
            resample=resample,
            allow_upsample=allow_upsample,
        )
        for d in descriptors
    ]


def write_acoustic_manifest_jsonl(
    items: Sequence[AcousticClipDescriptor | AcousticClipRecord],
    out_jsonl: Path | str,
    *,
    relative_to: Optional[Path | str] = None,
) -> int:
    """Serialize discovered clip descriptors or loaded clip records into a UTF-8 JSONL manifest."""
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rel_base = Path(relative_to) if relative_to is not None else None
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            if isinstance(item, AcousticClipRecord):
                row = item.to_manifest_dict(relative_to=rel_base)
            else:
                wp = item.wav_path.relative_to(rel_base).as_posix() if rel_base is not None else item.wav_path.as_posix()
                row = {
                    "dataset_id": item.dataset_id,
                    "machine": item.machine,
                    "split": item.split,
                    "label": item.label,
                    "is_anomaly": item.is_anomaly,
                    "domain": item.domain,
                    "section_or_model_id": item.section_or_model_id,
                    "snr_db": item.snr_db,
                    "attributes_hidden": item.attributes_hidden,
                    "wav_path": wp,
                }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _parse_manifest_label(raw_label: Any, raw_is_anom: Any) -> Tuple[Optional[str], Optional[bool]]:
    if isinstance(raw_is_anom, bool):
        return ("anomaly" if raw_is_anom else "normal"), raw_is_anom
    if isinstance(raw_is_anom, (int, float)) and not isinstance(raw_is_anom, bool):
        flag = bool(int(raw_is_anom) != 0)
        return ("anomaly" if flag else "normal"), flag
    if isinstance(raw_label, str) and raw_label.strip():
        low = raw_label.strip().lower()
        if low in ("anomaly", "abnormal", "1", "broken", "fault"):
            return "anomaly", True
        if low in ("normal", "good", "0", "nominal"):
            return "normal", False
    return None, None


def load_acoustic_manifest(
    manifest_path: Path | str,
    *,
    base_dir: Optional[Path | str] = None,
    dataset_id: str = "manifest_audio",
    drop_speech_band: bool = True,
    allow_speech_band: bool = False,
    resample: bool = False,
    allow_upsample: bool = False,
) -> List[AcousticClipRecord]:
    """Load WAV clips listed in a `.jsonl` or `.csv` manifest with speech-band & Nyquist checks."""
    mpath = Path(manifest_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"Acoustic manifest file does not exist: {mpath}")
    root_dir = Path(base_dir) if base_dir is not None else mpath.parent

    raw_rows: List[Dict[str, Any]] = []
    if mpath.suffix.lower() == ".csv":
        with mpath.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_rows.append(dict(row))
    else:
        for line_no, raw_line in enumerate(mpath.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{mpath}:{line_no}: expected JSON object per manifest line")
            raw_rows.append(obj)

    if not raw_rows:
        raise ValueError(f"Acoustic manifest {mpath} contains zero clip rows")

    records: List[AcousticClipRecord] = []
    for idx, row in enumerate(raw_rows):
        rel_wav = row.get("wav_path") or row.get("file_name") or row.get("path")
        if not isinstance(rel_wav, str) or not rel_wav.strip():
            raise ValueError(f"{mpath} row {idx}: missing 'wav_path' / 'file_name'")
        wpath = Path(rel_wav)
        if not wpath.is_absolute():
            wpath = root_dir / wpath

        split = str(row.get("split") or "test").strip()
        label, is_anom = _parse_manifest_label(row.get("label"), row.get("is_anomaly"))
        if split == "train" and is_anom is True:
            raise ValueError(f"{mpath} row {idx}: anomalous clip {rel_wav!r} is not allowed in 'train' split")

        snr_raw = row.get("snr_db")
        snr_val = float(snr_raw) if snr_raw not in (None, "") else None
        attrs_raw = row.get("attributes_hidden", False)
        attrs_hidden = attrs_raw if isinstance(attrs_raw, bool) else str(attrs_raw).lower() in ("true", "1", "yes")

        desc = AcousticClipDescriptor(
            dataset_id=str(row.get("dataset_id") or dataset_id),
            machine=str(row.get("machine") or "unknown"),
            split=split,
            label=label,
            is_anomaly=is_anom,
            domain=str(row.get("domain") or "source"),
            section_or_model_id=str(row.get("section_or_model_id") or row.get("model_id") or "section_00"),
            snr_db=snr_val,
            attributes_hidden=attrs_hidden,
            wav_path=wpath,
        )
        records.append(
            _featurize_descriptor(
                desc,
                drop_speech_band=drop_speech_band,
                allow_speech_band=allow_speech_band,
                resample=resample,
                allow_upsample=allow_upsample,
            )
        )
    return records


@dataclass(frozen=True)
class ProtocolSpanRecord:
    """Normalized wet-lab protocol text record parsed via `demo.parse_protocol.parse_quantities`."""

    dataset_id: str
    record_id: str
    domain: str
    text: str
    quantities: Tuple[Any, ...]

    def by_label(self) -> Dict[str, List[Any]]:
        grouped: Dict[str, List[Any]] = {}
        for q in self.quantities:
            grouped.setdefault(q.label, []).append(q)
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "record_id": self.record_id,
            "domain": self.domain,
            "text": self.text,
            "quantities": [asdict(q) for q in self.quantities],
        }


def _join_steps_value(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        parts: List[str] = []
        for item in val:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, Mapping):
                sub = item.get("text") or item.get("step") or item.get("instruction") or item.get("action")
                if isinstance(sub, str) and sub.strip():
                    parts.append(sub.strip())
        return "\n".join(parts)
    return ""


def extract_protocol_text(obj: Mapping[str, Any], *, fill_blank_answer: bool = True) -> str:
    """Extract canonical wet-lab protocol text across LAB-Bench, BioProBench, BioPlanner & JSONL schemas."""
    for key in (
        "text",
        "protocol_text",
        "protocol",
        "goal_text",
        "corrected_text",
        "steps",
        "correct_steps",
        "output",
        "corrupted_text",
        "original description",
    ):
        if key in obj and obj[key] is not None:
            candidate = _join_steps_value(obj[key])
            if candidate:
                return candidate

    question = obj.get("question")
    if isinstance(question, str) and question.strip():
        q_text = question.strip()
        ans = obj.get("answer")
        if fill_blank_answer and "____" in q_text and isinstance(ans, (str, int, float)) and not isinstance(ans, bool):
            ans_str = str(ans).strip()
            if ans_str:
                q_text = q_text.replace("____", ans_str, 1)
        return q_text
    return ""


def normalize_protocol_corpus_to_jsonl(
    src_path: Path | str,
    out_jsonl: Path | str,
    *,
    dataset_id: str = "protocol_jsonl",
    default_domain: str = "wet_lab",
    fill_blank_answer: bool = True,
) -> int:
    """Convert a protocol JSON array, directory of `.json` files, or `.jsonl` file into canonical JSONL."""
    spath = Path(src_path)
    if not spath.exists():
        raise FileNotFoundError(f"Protocol source path does not exist: {spath}")

    raw_items: List[Tuple[str, Mapping[str, Any]]] = []
    if spath.is_dir():
        for jfile in sorted(spath.glob("*.json")):
            loaded = json.loads(jfile.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw_items.append((jfile.stem, loaded))
            elif isinstance(loaded, list):
                for idx, elem in enumerate(loaded):
                    if isinstance(elem, dict):
                        raw_items.append((f"{jfile.stem}_{idx}", elem))
    elif spath.suffix.lower() == ".jsonl":
        for idx, line in enumerate(spath.read_text(encoding="utf-8").splitlines()):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"{spath}:{idx + 1}: expected JSON object per line")
            raw_items.append((f"{spath.stem}_{idx}", obj))
    else:
        loaded = json.loads(spath.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            for idx, elem in enumerate(loaded):
                if isinstance(elem, dict):
                    raw_items.append((f"{spath.stem}_{idx}", elem))
        elif isinstance(loaded, dict):
            raw_items.append((spath.stem, loaded))
        else:
            raise ValueError(f"Unsupported JSON root structure in {spath}")

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for fallback_id, item in raw_items:
            text = extract_protocol_text(item, fill_blank_answer=fill_blank_answer)
            if not text:
                continue
            rec_id = str(item.get("id") or item.get("protocol_id") or item.get("record_id") or fallback_id)
            domain = str(item.get("domain") or item.get("tag") or item.get("type") or default_domain)
            row = {
                "dataset_id": str(item.get("dataset_id") or dataset_id),
                "id": rec_id,
                "domain": domain,
                "text": text,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise ValueError(f"Zero non-empty protocol records extracted from {spath}")
    return written


def load_protocol_jsonl(
    jsonl_path: Path | str,
    *,
    dataset_id: str = "protocol_jsonl",
    default_domain: str = "wet_lab",
    include_concentration: bool = True,
    fill_blank_answer: bool = True,
    limit: Optional[int] = None,
) -> List[ProtocolSpanRecord]:
    """Load wet-lab protocol JSONL lines and extract normalized quantities via `parse_quantities()`."""
    jpath = Path(jsonl_path)
    if not jpath.is_file():
        raise FileNotFoundError(f"Protocol JSONL file does not exist: {jpath}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1 when provided, got {limit}")

    parser_mod = _load_parse_protocol_module()
    records: List[ProtocolSpanRecord] = []

    for line_no, raw_line in enumerate(jpath.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{jpath}:{line_no}: invalid JSON line: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{jpath}:{line_no}: expected JSON object, got {type(obj).__name__}")

        text = extract_protocol_text(obj, fill_blank_answer=fill_blank_answer)
        if not text:
            raise ValueError(f"{jpath}:{line_no}: protocol record contains no extractable text field")

        rec_id = str(obj.get("id") or obj.get("record_id") or obj.get("protocol_id") or f"line_{line_no}")
        domain = str(obj.get("domain") or obj.get("tag") or obj.get("type") or default_domain)
        ds_id = str(obj.get("dataset_id") or dataset_id)
        quantities = tuple(parser_mod.parse_quantities(text, include_concentration=include_concentration))
        records.append(
            ProtocolSpanRecord(
                dataset_id=ds_id,
                record_id=rec_id,
                domain=domain,
                text=text,
                quantities=quantities,
            )
        )
        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError(f"Protocol JSONL file {jpath} contained zero records")
    return records


def summarize_protocol_corpus(records: Iterable[ProtocolSpanRecord]) -> Dict[str, Any]:
    """Aggregate protocol count and extracted span counts per quantitative slot label."""
    rec_list = list(records)
    label_counts: Dict[str, int] = {
        "Amount": 0,
        "Temperature": 0,
        "Time": 0,
        "Speed": 0,
        "Concentration": 0,
    }
    total_spans = 0
    for rec in rec_list:
        for q in rec.quantities:
            label_counts[q.label] = label_counts.get(q.label, 0) + 1
            total_spans += 1
    return {
        "n_protocols": len(rec_list),
        "total_spans": total_spans,
        "spans_by_label": label_counts,
    }
