"""Offline test suite for 2024-2026+ open sensory datasets (`acoustic/open_datasets.py`
and `scripts/fetch_open_datasets.py`)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score

from acoustic import (
    OPEN_SENSORY_DATASETS,
    GaussianDetector,
    OpenSensoryDataset,
    SR,
    get_open_dataset,
    list_open_datasets,
    load_acoustic_manifest,
    load_dcase_task2_clips,
    load_mimii_clips,
    load_protocol_jsonl,
    normalize_protocol_corpus_to_jsonl,
    parse_dcase_filename,
    speech_band_mask,
    summarize_protocol_corpus,
    verify_audio_privacy_and_nyquist,
    write_acoustic_manifest_jsonl,
)
from autolab_wire import verify_speech_privacy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_script():
    mod_name = "_test_fetch_open_datasets_script"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    script_path = REPO_ROOT / "scripts" / "fetch_open_datasets.py"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synth_machine_wav(
    rng: np.random.Generator,
    *,
    sr: int = SR,
    seconds: float = 1.0,
    channels: int = 1,
    anomaly: bool = False,
    inject_speech_tone: bool = False,
) -> np.ndarray:
    t = np.arange(int(round(sr * seconds))) / float(sr)
    # Sub-speech 120 Hz rotor fundamental + 240 Hz second harmonic
    sig = 0.35 * np.sin(2.0 * np.pi * 120.0 * t) + 0.15 * np.sin(2.0 * np.pi * 240.0 * t)
    sig = sig + 0.015 * rng.standard_normal(t.size)
    if inject_speech_tone:
        # 1000 Hz human formant-band tone that must be excised by speech-band drop
        sig = sig + 0.45 * np.sin(2.0 * np.pi * 1000.0 * t)
    if anomaly and sr >= 12000:
        # High-band 5200 Hz cavitation/bearing burst above the 3400 Hz speech cutoff
        sig = sig + 0.40 * np.sin(2.0 * np.pi * 5200.0 * t)
    mono = np.clip(sig, -0.99, 0.99).astype(np.float32)
    if channels == 1:
        return mono
    cols = [
        np.clip(mono + 0.005 * rng.standard_normal(t.size), -0.99, 0.99).astype(np.float32)
        for _ in range(channels)
    ]
    return np.column_stack(cols)


def test_registry_metadata_licenses_and_2024_2026_coverage() -> None:
    assert len(OPEN_SENSORY_DATASETS) >= 10
    recent = list_open_datasets(min_year=2024)
    assert len(recent) >= 5

    audio_ds = list_open_datasets(channel="machine_audio")
    text_ds = list_open_datasets(channel="protocol_text")
    assert len(audio_ds) >= 5
    assert len(text_ds) >= 4

    expected_ids = {
        "dcase2024_task2_dev",
        "dcase2025_task2_dev",
        "dcase2026_task2_dev",
        "imad_ds_2024",
        "mimii",
        "mimii_dg",
        "toyadmos2",
        "lab_bench_protocolqa",
        "chembench",
        "bioprobench",
        "bioplanner_bioprot",
        "protoact_biop2e",
    }
    assert expected_ids.issubset(set(OPEN_SENSORY_DATASETS))

    for ds_id, ds in OPEN_SENSORY_DATASETS.items():
        assert ds.dataset_id == ds_id
        assert ds.url.startswith("https://")
        assert len(ds.machines_or_domains) >= 1
        if ds.channel == "machine_audio":
            assert isinstance(ds.sample_rate_hz, int) and ds.sample_rate_hz >= 16000
        else:
            assert ds.sample_rate_hz is None

    # Verify explicit permissive vs non-commercial/unlicensed flags
    assert get_open_dataset("mimii").commercial_use_ok is True
    assert get_open_dataset("imad_ds_2024").commercial_use_ok is True
    assert get_open_dataset("lab_bench_protocolqa").commercial_use_ok is True
    assert get_open_dataset("chembench").commercial_use_ok is True

    assert get_open_dataset("dcase2024_task2_dev").commercial_use_ok is False
    assert get_open_dataset("dcase2025_task2_dev").commercial_use_ok is False
    assert get_open_dataset("dcase2026_task2_dev").commercial_use_ok is False
    assert get_open_dataset("mimii_dg").commercial_use_ok is False
    assert get_open_dataset("toyadmos2").commercial_use_ok is False
    assert get_open_dataset("bioprobench").commercial_use_ok is False
    assert get_open_dataset("bioplanner_bioprot").commercial_use_ok is False
    assert get_open_dataset("protoact_biop2e").commercial_use_ok is False

    comm_only = list_open_datasets(commercial_only=True)
    assert all(d.commercial_use_ok for d in comm_only)
    assert {d.dataset_id for d in comm_only} >= {"mimii", "imad_ds_2024", "lab_bench_protocolqa", "chembench"}

    # Fail-closed validation against marking NC or NOASSERTION datasets commercial_use_ok=True
    with pytest.raises(ValueError, match="cannot set commercial_use_ok=True"):
        OpenSensoryDataset(
            dataset_id="bad_nc_flag",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=("wet_lab",),
            url="https://example.org/dataset",
            spdx_license="CC-BY-NC-4.0",
            commercial_use_ok=True,
        )
    with pytest.raises(ValueError, match="cannot set commercial_use_ok=True"):
        OpenSensoryDataset(
            dataset_id="bad_unlicensed_flag",
            channel="protocol_text",
            sample_rate_hz=None,
            machines_or_domains=("wet_lab",),
            url="https://example.org/dataset",
            spdx_license="NOASSERTION",
            commercial_use_ok=True,
        )
    with pytest.raises(KeyError, match="Unknown dataset_id"):
        get_open_dataset("nonexistent_dataset")


def test_speech_band_drop_and_nyquist_enforcement(tmp_path: Path) -> None:
    retained = verify_audio_privacy_and_nyquist(source_sr_hz=16000, target_sr_hz=16000, drop_speech_band=True)
    assert verify_speech_privacy(
        retained,
        False,
        raw_audio_kept=False,
        speech_band_dropped=True,
        sample_rate_hz=16000,
    )

    # Retaining [300, 3400] Hz without explicit ablation flag must fail closed
    with pytest.raises(ValueError, match="Speech-band \\[300, 3400\\] Hz retention blocked"):
        verify_audio_privacy_and_nyquist(source_sr_hz=16000, drop_speech_band=False)

    # Nyquist / sub-16kHz source sample rate checks
    with pytest.raises(ValueError, match="below 8000 Hz"):
        verify_audio_privacy_and_nyquist(source_sr_hz=6000, drop_speech_band=True, allow_upsample=True)
    with pytest.raises(ValueError, match="upsampling blocked"):
        verify_audio_privacy_and_nyquist(source_sr_hz=8000, target_sr_hz=16000, drop_speech_band=True)

    # Sub-16 kHz WAV on disk must be rejected by DCASE loader
    bad_dir = tmp_path / "dcase_8k" / "fan"
    (bad_dir / "train").mkdir(parents=True)
    (bad_dir / "test").mkdir(parents=True)
    wavfile.write(
        bad_dir / "train" / "section_00_source_train_normal_0000.wav",
        8000,
        np.zeros(8000, dtype=np.int16),
    )
    wavfile.write(
        bad_dir / "test" / "section_00_source_test_normal_0000.wav",
        8000,
        np.zeros(8000, dtype=np.int16),
    )
    with pytest.raises(ValueError, match="upsampling blocked"):
        load_dcase_task2_clips(tmp_path / "dcase_8k", "fan", resample=True, allow_upsample=False)


def test_dcase_2024_2025_2026_directory_loaders_and_detector_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)

    # Filename parser: DCASE 2024 hidden attributes vs explicit attributes vs unlabeled eval
    p_hidden = parse_dcase_filename("section_00_source_train_normal_0004.wav")
    assert p_hidden["attributes_hidden"] is True
    assert p_hidden["domain"] == "source"
    assert p_hidden["is_anomaly"] is False

    p_attr = parse_dcase_filename("section_00_target_test_anomaly_0012_vel_8_loc_B.wav")
    assert p_attr["attributes_hidden"] is False
    assert p_attr["domain"] == "target"
    assert p_attr["is_anomaly"] is True

    p_eval = parse_dcase_filename("section_00_test_0042.wav")
    assert p_eval["label"] is None and p_eval["is_anomaly"] is None

    # Build synthetic DCASE 2025 + 2026 hybrid fixture (hidden attrs + supplemental + 2-ch stereo)
    fan_dir = tmp_path / "dcase2025" / "fan"
    for sp in ("train", "test", "supplemental"):
        (fan_dir / sp).mkdir(parents=True)

    for i in range(6):
        dom = "source" if i < 4 else "target"
        tail = "" if i % 2 == 0 else "_spd_1200"
        wav = _synth_machine_wav(rng, channels=2, anomaly=False, inject_speech_tone=True)
        wavfile.write(fan_dir / "train" / f"section_00_{dom}_train_normal_{i:04d}{tail}.wav", SR, wav)

    for i in range(4):
        wav_norm = _synth_machine_wav(rng, channels=2, anomaly=False, inject_speech_tone=True)
        wav_anom = _synth_machine_wav(rng, channels=2, anomaly=True, inject_speech_tone=False)
        dom = "source" if i < 2 else "target"
        wavfile.write(fan_dir / "test" / f"section_00_{dom}_test_normal_{i:04d}.wav", SR, wav_norm)
        wavfile.write(fan_dir / "test" / f"section_00_{dom}_test_anomaly_{i:04d}.wav", SR, wav_anom)

    wavfile.write(
        fan_dir / "supplemental" / "section_00_source_supplemental_normal_0000.wav",
        SR,
        _synth_machine_wav(rng, channels=1, anomaly=False),
    )

    clips = load_dcase_task2_clips(
        tmp_path / "dcase2025",
        "fan",
        dataset_id="dcase2025_task2_dev",
        splits=("train", "test", "supplemental"),
    )
    assert len(clips) == 15
    expected_bins = int(speech_band_mask().sum())
    for rec in clips:
        assert rec.speech_band_dropped is True
        assert rec.raw_audio_kept is False
        assert rec.mel.shape[1] == expected_bins
        assert verify_speech_privacy(rec.retained_bands_hz, False, sample_rate_hz=rec.sample_rate_hz)

    train_mels = [r.mel for r in clips if r.split == "train"]
    test_recs = [r for r in clips if r.split == "test"]
    assert any(r.attributes_hidden for r in clips if r.split == "train")
    assert any(not r.attributes_hidden for r in clips if r.split == "train")
    assert all(r.n_channels_raw == 2 for r in test_recs)

    det = GaussianDetector().fit(train_mels)
    y_true = [1 if r.is_anomaly else 0 for r in test_recs]
    scores = [det.score(r.mel) for r in test_recs]
    assert roc_auc_score(y_true, scores) > 0.95

    # One-class guard: anomalous file inside train/ must raise ValueError
    wavfile.write(
        fan_dir / "train" / "section_00_source_train_anomaly_9999.wav",
        SR,
        _synth_machine_wav(rng, anomaly=True),
    )
    with pytest.raises(ValueError, match="Unsupervised one-class violation"):
        load_dcase_task2_clips(tmp_path / "dcase2025", "fan")


def test_mimii_8channel_layout_and_manifest_loaders(tmp_path: Path) -> None:
    rng = np.random.default_rng(3384388)
    id_dir = tmp_path / "mimii_root" / "0_dB" / "pump" / "id_00"
    (id_dir / "normal").mkdir(parents=True)
    (id_dir / "abnormal").mkdir(parents=True)

    for i in range(5):
        w_norm = _synth_machine_wav(rng, channels=8, anomaly=False, inject_speech_tone=True)
        wavfile.write(id_dir / "normal" / f"{i:08d}.wav", SR, w_norm)
    for i in range(3):
        w_abn = _synth_machine_wav(rng, channels=8, anomaly=True)
        wavfile.write(id_dir / "abnormal" / f"{i:08d}.wav", SR, w_abn)

    records = load_mimii_clips(tmp_path / "mimii_root", "pump", snr_db=0.0, normal_train_fraction=0.8)
    assert len(records) == 8
    train_recs = [r for r in records if r.split == "train"]
    test_recs = [r for r in records if r.split == "test"]
    assert len(train_recs) == 4
    assert len(test_recs) == 4  # 1 held-out normal + 3 abnormal
    assert all(r.is_anomaly is False for r in train_recs)
    assert all(r.n_channels_raw == 8 and r.snr_db == pytest.approx(0.0) for r in records)
    assert all(r.section_or_model_id == "id_00" for r in records)

    det = GaussianDetector().fit([r.mel for r in train_recs])
    test_scores = [det.score(r.mel) for r in test_recs]
    test_labels = [1 if r.is_anomaly else 0 for r in test_recs]
    assert roc_auc_score(test_labels, test_scores) == pytest.approx(1.0)

    # Round-trip through JSONL acoustic manifest
    manifest_jsonl = tmp_path / "mimii_root" / "pump_manifest.jsonl"
    n_written = write_acoustic_manifest_jsonl(records, manifest_jsonl, relative_to=tmp_path / "mimii_root")
    assert n_written == 8
    reloaded = load_acoustic_manifest(manifest_jsonl)
    assert len(reloaded) == 8
    assert [r.is_anomaly for r in reloaded] == [r.is_anomaly for r in records]

    # CSV manifest support + anomalous train row rejection
    csv_manifest = tmp_path / "mimii_root" / "bad_train.csv"
    csv_manifest.write_text(
        "wav_path,machine,split,label\n"
        "0_dB/pump/id_00/abnormal/00000000.wav,pump,train,abnormal\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not allowed in 'train' split"):
        load_acoustic_manifest(csv_manifest)


def test_protocol_jsonl_normalization_and_parse_quantities(tmp_path: Path) -> None:
    # 1. LAB-Bench ProtocolQA JSONL format
    labbench_raw = tmp_path / "protocolqa_sample.jsonl"
    labbench_raw.write_text(
        json.dumps(
            {
                "id": "pqa-001",
                "tag": "protocolqa",
                "protocol": (
                    "Dissolve 25 mg NaCl in 10 mL Tris (10 mM), incubate at 37 °C for 30 min, "
                    "and centrifuge at 12000 x g for 5 min."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recs_lb = load_protocol_jsonl(labbench_raw, dataset_id="lab_bench_protocolqa")
    assert len(recs_lb) == 1
    by_lbl = recs_lb[0].by_label()
    assert {q.canonical_unit for q in by_lbl["Amount"]} == {"mg", "uL"}
    assert by_lbl["Temperature"][0].canonical_value == pytest.approx(37.0)
    assert by_lbl["Speed"][0].canonical_value == pytest.approx(12000.0)
    assert by_lbl["Concentration"][0].canonical_value == pytest.approx(10.0)
    assert len(by_lbl["Time"]) == 2

    # 2. BioProBench PQA & ERR JSON array -> normalized JSONL -> parse_quantities
    bpb_json = tmp_path / "bpb_sample.json"
    bpb_json.write_text(
        json.dumps(
            [
                {
                    "id": "TEST-PQA-1",
                    "type": "parameter",
                    "question": "Create a ____ mg/mL stock and aliquot 50 uL on ice for 10 min.",
                    "answer": "25",
                },
                {
                    "id": "TEST-ERR-1",
                    "corrected_text": "Incubate 200 uL master mix at 95 °C for 2 min, then spin at 3000 rpm.",
                },
            ]
        ),
        encoding="utf-8",
    )
    norm_jsonl = tmp_path / "bpb_norm.jsonl"
    assert normalize_protocol_corpus_to_jsonl(bpb_json, norm_jsonl, dataset_id="bioprobench") == 2
    recs_bpb = load_protocol_jsonl(norm_jsonl, dataset_id="bioprobench")
    summary = summarize_protocol_corpus(recs_bpb)
    assert summary["n_protocols"] == 2
    assert summary["spans_by_label"]["Amount"] == 2
    assert summary["spans_by_label"]["Temperature"] == 2
    assert summary["spans_by_label"]["Time"] == 2
    assert summary["spans_by_label"]["Speed"] == 1
    assert summary["spans_by_label"]["Concentration"] == 1

    # 3. BioPlanner directory of JSON files
    bp_dir = tmp_path / "bioprot_dir"
    bp_dir.mkdir()
    (bp_dir / "10051.json").write_text(
        json.dumps(
            {
                "id": "10051",
                "steps": ["Add 12.5 uL PCR master mix.", "Denature at 98 °C for 30 s."],
            }
        ),
        encoding="utf-8",
    )
    bp_jsonl = tmp_path / "bioplanner.jsonl"
    assert normalize_protocol_corpus_to_jsonl(bp_dir, bp_jsonl, dataset_id="bioplanner_bioprot") == 1
    recs_bp = load_protocol_jsonl(bp_jsonl)
    assert recs_bp[0].record_id == "10051"
    assert len(recs_bp[0].quantities) == 3

    # Malformed JSONL guard
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON line"):
        load_protocol_jsonl(bad_jsonl)


def test_fetch_open_datasets_cli_security_and_commercial_gate(tmp_path: Path, capsys) -> None:
    fetch_mod = _load_fetch_script()

    assert fetch_mod.main(["--list", "--json", "--min-year", "2024"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) >= 5
    assert all(item["release_year"] >= 2024 for item in listed)

    # Zip-Slip traversal must be blocked
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../escaped.txt", "blocked")
    safe_out = tmp_path / "safe_out"
    safe_out.mkdir()
    with pytest.raises(ValueError, match="Zip-Slip"):
        fetch_mod.safe_extract_zip(evil_zip, safe_out)

    # Commercial gate blocks CC-BY-NC-SA-4.0 DCASE 2024 and CC-BY-NC-4.0 BioProBench
    with pytest.raises(PermissionError, match="commercial_use_ok=False"):
        fetch_mod.fetch_machine_audio_dataset(
            "dcase2024_task2_dev",
            tmp_path / "blocked_dcase",
            require_commercial_ok=True,
        )
    with pytest.raises(PermissionError, match="commercial_use_ok=False"):
        fetch_mod.fetch_protocol_text_dataset(
            "bioprobench",
            tmp_path / "blocked_bpb",
            require_commercial_ok=True,
        )

    # Offline local file:// fetch of MIMII archive (commercial_use_ok=True) with MD5 check
    wav_tmp = tmp_path / "pump0.wav"
    wavfile.write(wav_tmp, SR, np.zeros(SR // 2, dtype=np.int16))
    mimii_zip = tmp_path / "0_dB_pump.zip"
    with zipfile.ZipFile(mimii_zip, "w") as zf:
        zf.write(wav_tmp, arcname="pump/id_00/normal/00000000.wav")
    digest = hashlib.md5(mimii_zip.read_bytes()).hexdigest()

    extracted = fetch_mod.fetch_machine_audio_dataset(
        "mimii",
        tmp_path / "fetched_mimii",
        machines=["pump"],
        snr_db=0,
        require_commercial_ok=True,
        url_template_override=mimii_zip.as_uri(),
        expected_digest_override=digest,
    )
    assert (extracted / "0_dB" / "pump" / "id_00" / "normal" / "00000000.wav").is_file()

    # Offline local file:// fetch of protocol JSON corpus
    src_json = tmp_path / "local_pqa.json"
    src_json.write_text(
        json.dumps([{"id": "1", "protocol": "Add 10 uL enzyme at 4 °C for 5 min."}]),
        encoding="utf-8",
    )
    out_jsonl = fetch_mod.fetch_protocol_text_dataset(
        "lab_bench_protocolqa",
        tmp_path / "fetched_lb",
        require_commercial_ok=True,
        source_url_override=src_json.as_uri(),
    )
    loaded = load_protocol_jsonl(out_jsonl)
    assert len(loaded) == 1 and len(loaded[0].quantities) == 3


def test_pure_english_zero_cjk_in_new_files_and_docs() -> None:
    cjk_re = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
    checked_paths = [
        REPO_ROOT / "acoustic" / "open_datasets.py",
        REPO_ROOT / "scripts" / "fetch_open_datasets.py",
        REPO_ROOT / "tests" / "test_open_datasets_2026.py",
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "ARCHITECTURE_AND_HISTORY.md",
    ]
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert cjk_re.search(text) is None, f"CJK character detected in {path}"
