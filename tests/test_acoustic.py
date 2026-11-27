"""Comprehensive offline pytest suite for acoustic/ covering issues #1, #13..#23, #26."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score

from acoustic.detect import (
    AutoencoderDetector,
    GaussianDetector,
    KNNDetector,
    calibrate_threshold,
    calibrate_with_mode,
    load_detector_bundle,
    save_detector_bundle,
)
from acoustic.evaluate import main as evaluate_main, metrics as evaluate_metrics, run as evaluate_run
from acoustic.features import (
    HOP,
    MEL_CENTERS_HZ,
    MEL_EDGES_HZ,
    N_FFT,
    N_MELS,
    SR,
    _FB,
    load_wav,
    log_mel,
    outward_round_band_hz,
    retained_mel_bands_hz,
    speech_band_mask,
    stack_frames,
)
from acoustic.fetch_dcase2023 import ZENODO_MD5, fetch, safe_extract_zip, verify_digest
from acoustic.monitor import main as monitor_main, monitor_clip, monitor_stream
from acoustic.record import SCHEMA, AudioEvent, now_utc


def _clip(rng, knock=False, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    f0 = 120 * (1 + 0.01 * rng.standard_normal())
    x = sum(a * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) for k, a in [(1, 0.5), (2, 0.2), (3, 0.1)])
    x = x + 0.02 * rng.standard_normal(len(t))
    if knock:
        for c in np.arange(0.1, seconds, 0.25):
            i = int(c * SR)
            n_click = min(200, len(x) - i)
            if n_click > 0:
                x[i : i + n_click] += 0.6 * rng.standard_normal(n_click) * np.exp(-np.arange(n_click) / 40)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    train = [log_mel(_clip(rng)) for _ in range(30)]
    test = [log_mel(_clip(rng)) for _ in range(10)] + [log_mel(_clip(rng, knock=True)) for _ in range(10)]
    y = [0] * 10 + [1] * 10
    return train, test, y


def test_log_mel_shape_and_trailing_samples():
    m = log_mel(np.zeros(SR, dtype=np.float32))
    assert m.shape[1] == N_MELS and np.isfinite(m).all()

    # Issue #16: trailing sub-hop impulsive click in final samples must not be dropped
    n_samples = N_FFT + 2 * HOP + 300
    x_quiet = np.zeros(n_samples, dtype=np.float32)
    x_tail = np.zeros(n_samples, dtype=np.float32)
    x_tail[-100:] = 0.8
    mel_quiet = log_mel(x_quiet)
    mel_tail = log_mel(x_tail)
    assert mel_tail.shape[0] == 4  # 3 full hops + 1 padded tail frame
    assert mel_tail[-1].max() > mel_quiet[-1].max() + 40.0


# --- Issue #13: Deterministic dtype-aware WAV loading ---
def test_load_wav_dtypes_and_stereo(tmp_path: Path):
    # Silent / near-silent int16 (|x| <= 1) must still be scaled by 1/32768
    p_i16 = tmp_path / "quiet_i16.wav"
    wavfile.write(p_i16, SR, np.array([0, 1, -1, 1], dtype=np.int16))
    x_i16 = load_wav(p_i16)
    assert x_i16.dtype == np.float32
    assert np.max(np.abs(x_i16)) == pytest.approx(1.0 / 32768.0)

    # 32-bit signed PCM (int32)
    p_i32 = tmp_path / "pcm32.wav"
    half_fs_i32 = np.array([0, 1073741824, -1073741824], dtype=np.int32)
    wavfile.write(p_i32, SR, half_fs_i32)
    x_i32 = load_wav(p_i32)
    assert np.allclose(x_i32, [0.0, 0.5, -0.5], atol=1e-6)

    # 8-bit unsigned PCM (uint8): 128 -> 0.0, 255 -> ~+1.0, 0 -> -1.0
    p_u8 = tmp_path / "pcm8.wav"
    wavfile.write(p_u8, SR, np.array([128, 0, 255], dtype=np.uint8))
    x_u8 = load_wav(p_u8)
    assert x_u8[0] == pytest.approx(0.0)
    assert x_u8[1] == pytest.approx(-1.0)
    assert x_u8[2] == pytest.approx(127.0 / 128.0)

    # Stereo float32 and resampling
    p_f32_stereo = tmp_path / "stereo_f32.wav"
    stereo = np.column_stack([np.full(100, 0.2, dtype=np.float32), np.full(100, 0.6, dtype=np.float32)])
    wavfile.write(p_f32_stereo, SR, stereo)
    x_st = load_wav(p_f32_stereo)
    assert x_st.shape == (100,)
    assert np.allclose(x_st, 0.4, atol=1e-6)

    p_32k = tmp_path / "sr32k.wav"
    wavfile.write(p_32k, 32000, np.zeros(3200, dtype=np.float32))
    with pytest.raises(ValueError, match="expected 16000 Hz"):
        load_wav(p_32k, resample=False)
    x_res = load_wav(p_32k, resample=True)
    assert len(x_res) == 1600


# --- Issue #14: Short clips (< 5 STFT frames) in stack_frames and AutoencoderDetector ---
def test_short_clip_stack_frames_and_autoencoder(data):
    train, _, _ = data
    short_x = np.zeros(512, dtype=np.float32)  # 32 ms -> 1 STFT frame
    short_mel = log_mel(short_x)
    assert short_mel.shape[0] == 1
    stacked = stack_frames(short_mel, context=5)
    assert stacked.shape == (1, 5 * N_MELS)
    assert np.isfinite(stacked).all()

    ae = AutoencoderDetector(hidden=(16, 4, 16), epochs=10, seed=0).fit(train)
    s = ae.score(short_mel)
    assert np.isfinite(s) and s >= 0.0


# --- Issue #15: Speech-band filter skirt exclusion (300-3400 Hz) ---
def test_speech_band_mask_excludes_filter_skirts():
    keep = speech_band_mask()
    assert not keep[(MEL_CENTERS_HZ > 300) & (MEL_CENTERS_HZ < 3400)].any()
    # Every kept filter must have support strictly <= 300 Hz or >= 3400 Hz
    kept_edges = MEL_EDGES_HZ[keep]
    assert np.all((kept_edges[:, 1] <= 300.0) | (kept_edges[:, 0] >= 3400.0))

    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    speech_bins = (fft_freqs > 300.0) & (fft_freqs < 3400.0)
    assert np.all(_FB[keep][:, speech_bins] == 0.0)

    # Pure 305 Hz and 3395 Hz sine tones lose main-lobe energy when speech band is dropped
    t = np.arange(SR) / SR
    for f_hz in (305.0, 3395.0):
        tone = np.sin(2.0 * np.pi * f_hz * t).astype(np.float32)
        m_full = log_mel(tone, drop_speech_band=False)
        m_drop = log_mel(tone, drop_speech_band=True)
        assert m_drop.shape[1] == keep.sum()
        assert m_drop[:-1].max() < m_full[:-1].max() - 20.0
        assert m_drop.mean(axis=0).max() < m_full.mean(axis=0).max() - 20.0


# --- Issue #17, #18, #19: Detector numerics, few-shot clamping, and n_mels validation ---
@pytest.mark.parametrize(
    "det",
    [GaussianDetector(), KNNDetector(k=3), AutoencoderDetector(hidden=(32, 8, 32), epochs=20)],
)
def test_detectors_flag_knocks(data, det):
    train, test, y = data
    det.fit(train)
    assert det.n_mels_ == N_MELS
    assert det.speech_band_dropped_ is False
    assert roc_auc_score(y, [det.score(m) for m in test]) > 0.9


def test_gaussian_unsquared_mahalanobis(data):
    train, test, _ = data
    g_unsquared = GaussianDetector(squared=False).fit(train)
    g_squared = GaussianDetector(squared=True).fit(train)
    s = g_unsquared.score(test[0])
    s2 = g_squared.score(test[0])
    assert s == pytest.approx(np.sqrt(s2))


def test_few_shot_calibration_and_dimension_mismatch(data):
    train, _, _ = data
    with pytest.raises(ValueError, match="At least 2 normal clip"):
        GaussianDetector().fit(train[:1])

    with pytest.warns(UserWarning, match="clamped to n_neighbors=2"):
        knn = KNNDetector(k=5).fit(train[:2])
    assert knn.effective_k_ == 2
    assert np.isfinite(knn.score(train[0]))

    dropped_clip = log_mel(np.zeros(SR, dtype=np.float32), drop_speech_band=True)
    for det in [GaussianDetector().fit(train[:3]), knn, AutoencoderDetector(hidden=(8,), epochs=5).fit(train[:1])]:
        with pytest.raises(ValueError, match="dimension mismatch"):
            det.score(dropped_clip)


# --- Issue #20: Checksum verification, corrupt zip recovery, and Zip-Slip protection ---
def test_fetch_dcase2023_security_and_checksums(tmp_path: Path):
    assert set(ZENODO_MD5) == {"bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"}

    # Zip-Slip archive must be rejected before writing outside root
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    out_dir = tmp_path / "extracted"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Zip-Slip"):
        safe_extract_zip(evil_zip, out_dir)
    assert not (tmp_path / "escaped.txt").exists()

    # Valid local archive with corrupt leftover .zip recovery and MD5 check
    src_zip = tmp_path / "valid_fan.zip"
    wav_buf = tmp_path / "dummy.wav"
    wavfile.write(wav_buf, SR, np.zeros(SR // 4, dtype=np.int16))
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.write(wav_buf, arcname="fan/train/section_00_source_train_normal_0000.wav")
    digest = hashlib.md5(src_zip.read_bytes()).hexdigest()
    verify_digest(src_zip, f"md5:{digest}")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_digest(src_zip, "md5:" + "0" * 32)

    dest = tmp_path / "dcase"
    dest.mkdir()
    # Leave behind a truncated/corrupt zip to verify automatic recovery
    (dest / "dev_fan.zip").write_bytes(b"corrupt partial download")
    res_dir = fetch(
        "fan",
        dest,
        expected_digest=digest,
        url_template=src_zip.as_uri(),
        timeout=5.0,
    )
    assert (res_dir / "train" / "section_00_source_train_normal_0000.wav").is_file()


# --- Issue #21 & #22: evaluate.py validation & single-thread execution ---
def test_evaluate_validation_and_end_to_end(tmp_path: Path):
    assert evaluate_main(["--data", str(tmp_path / "missing")]) == 1

    with pytest.raises(ValueError, match="normal only"):
        evaluate_metrics(
            ["section_00_source_test_normal_0001.wav", "section_00_target_test_normal_0001.wav"],
            [0.1, 0.2],
        )

    # Synthetic mini DCASE directory structure
    rng = np.random.default_rng(7)
    m_dir = tmp_path / "dcase" / "valve"
    (m_dir / "train").mkdir(parents=True)
    (m_dir / "test").mkdir(parents=True)
    for i in range(4):
        wavfile.write(m_dir / "train" / f"s0_source_train_normal_{i:04d}.wav", SR, _clip(rng, seconds=0.5))
    test_specs = [
        ("s0_source_test_normal_0000.wav", False),
        ("s0_source_test_anomaly_0001.wav", True),
        ("s0_target_test_normal_0002.wav", False),
        ("s0_target_test_anomaly_0003.wav", True),
    ]
    for fname, is_anom in test_specs:
        wavfile.write(m_dir / "test" / fname, SR, _clip(rng, knock=is_anom, seconds=0.5))

    rows = evaluate_run(tmp_path / "dcase", ["valve"], ["gaussian", "knn"], drop_speech_band=True, latency_clips=2)
    assert len(rows) == 2
    assert all(0.0 <= r["pauc"] <= 1.0 for r in rows)


# --- Issue #23: AudioEvent validation and from_json() integrity ---
def test_audio_event_roundtrip_and_strict_validation():
    e = AudioEvent(
        protocol_id="pcr-setup-v2",
        step=4,
        instrument_id="centrifuge-01",
        start_utc=now_utc(),
        duration_s=10.0,
        detector="gaussian",
        score=12.5,
        threshold=10.0,
        mic="contact",
    )
    e.validate()
    line = e.to_json()
    assert json.loads(line)["flagged"] is True
    assert AudioEvent.from_json(line) == e

    bad_kwargs = [
        dict(raw_audio_kept=True, speech_band_dropped=True),
        dict(score=float("nan")),
        dict(score=float("inf")),
        dict(threshold=float("nan")),
        dict(duration_s=0.0),
        dict(duration_s=-1.0),
        dict(sample_rate_hz=0),
        dict(start_utc="2026-11-18T09:58:12"),  # timezone-naive
        dict(schema="wrong/schema/v0"),
    ]
    base = dict(
        protocol_id="p",
        step=0,
        instrument_id="i",
        start_utc=now_utc(),
        duration_s=1.0,
        detector="knn",
        score=0.5,
        threshold=1.0,
    )
    for override in bad_kwargs:
        kw = {**base, **override}
        ev = AudioEvent(**kw)
        with pytest.raises(ValueError):
            ev.validate()
        # from_json must also invoke validate()
        raw_obj = {**kw, "schema": kw.get("schema", SCHEMA)}
        with pytest.raises(ValueError):
            AudioEvent.from_json(json.dumps(raw_obj))


# --- Issue #26: calibrate_threshold and python -m acoustic.monitor CLI ---
def test_calibrate_threshold_and_monitor_cli(tmp_path: Path):
    rng = np.random.default_rng(11)
    norm_dir = tmp_path / "normal"
    norm_dir.mkdir()
    for i in range(6):
        wavfile.write(norm_dir / f"norm_{i}.wav", SR, _clip(rng, knock=False, seconds=0.6))

    ok_wav = tmp_path / "step_ok.wav"
    bad_wav = tmp_path / "step_knock.wav"
    wavfile.write(ok_wav, SR, _clip(rng, knock=False, seconds=0.6))
    wavfile.write(bad_wav, SR, _clip(rng, knock=True, seconds=0.6))

    train_mels = [log_mel(load_wav(p), drop_speech_band=True) for p in sorted(norm_dir.glob("*.wav"))]
    det = GaussianDetector().fit(train_mels)
    thr = calibrate_threshold(det, train_mels, quantile=0.99)
    assert np.isfinite(thr) and thr > 0.0

    ev_bad = monitor_clip(
        bad_wav,
        det,
        threshold=thr,
        protocol_id="spin-protocol",
        step=2,
        instrument_id="centrifuge-01",
    )
    assert ev_bad.flagged is True

    ledger = tmp_path / "ledger.jsonl"
    rc = monitor_main(
        [
            "--train-dir",
            str(norm_dir),
            "--wav",
            str(ok_wav),
            str(bad_wav),
            "--detector",
            "gaussian",
            "--quantile",
            "0.99",
            "--protocol-id",
            "spin-protocol",
            "--step",
            "1",
            "--instrument-id",
            "centrifuge-01",
            "--out-jsonl",
            str(ledger),
        ]
    )
    assert rc == 0
    lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [AudioEvent.from_json(line) for line in lines]
    assert [p.step for p in parsed] == [1, 2]
    assert parsed[1].flagged is True


# --- Issue #30: Pre-STFT speech notch >= 60 dB attenuation & outward frequency rounding ---
def test_issue30_hann_skirt_notch_60db_attenuation_and_outward_rounding():
    assert outward_round_band_hz(50.0, 300.4) == (50, 301)
    assert outward_round_band_hz(3399.6, 7800.0) == (3399, 7800)
    assert retained_mel_bands_hz(drop_speech_band=True) == [[0, 255], [3464, 8000]]

    t = np.arange(SR) / SR
    for f_hz in (310.0, 3390.0):
        speech_tone = (0.8 * np.sin(2.0 * np.pi * f_hz * t)).astype(np.float32)
        m_full = log_mel(speech_tone, drop_speech_band=False)
        m_drop = log_mel(speech_tone, drop_speech_band=True)
        assert (m_full.max() - m_drop.max()) >= 60.0


# --- Issue #33: Out-of-fold conformal calibration, causal sub-window pooling, bundles & streaming ---
def test_issue33_oof_conformal_calibration_prevents_false_alarms():
    rng = np.random.default_rng(0)
    train_mels = [log_mel(0.02 * rng.standard_normal(SR).astype(np.float32), drop_speech_band=True) for _ in range(10)]
    heldout_mels = [log_mel(0.02 * rng.standard_normal(SR).astype(np.float32), drop_speech_band=True) for _ in range(50)]

    for det in (
        GaussianDetector().fit(train_mels),
        KNNDetector(k=5).fit(train_mels),
        AutoencoderDetector(hidden=(16, 4, 16), epochs=15, seed=0).fit(train_mels),
    ):
        thr, mode = calibrate_with_mode(det, train_mels, quantile=0.99)
        assert mode == "oof_conformal"
        fpr = float(np.mean([det.score(m) > thr for m in heldout_mels]))
        assert fpr <= 0.05


def test_issue33_sliding_window_pooling_catches_brief_fault_in_60s_step():
    rng = np.random.default_rng(3)
    tr2 = [log_mel(0.03 * rng.standard_normal(2 * SR).astype(np.float32)) for _ in range(12)]
    va2 = [log_mel(0.03 * rng.standard_normal(2 * SR).astype(np.float32)) for _ in range(20)]

    x2 = 0.03 * rng.standard_normal(2 * SR).astype(np.float32)
    x2[1000 : 1000 + int(0.15 * SR)] += 0.5
    x60 = 0.03 * rng.standard_normal(60 * SR).astype(np.float32)
    x60[1000 : 1000 + int(0.15 * SR)] += 0.5
    m2, m60 = log_mel(x2), log_mel(x60)

    for det in (
        AutoencoderDetector(hidden=(32, 8, 32), epochs=20, seed=0).fit(tr2),
        GaussianDetector().fit(tr2),
        KNNDetector(k=5).fit(tr2),
    ):
        thr = calibrate_threshold(det, va2, quantile=0.99)
        assert det.score(m2) > thr
        assert det.score(m60) > thr


def test_issue33_non_pickle_bundle_upsample_guard_and_streaming(tmp_path: Path):
    rng = np.random.default_rng(21)
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir()
    for i in range(6):
        wavfile.write(norm_dir / f"n_{i}.wav", SR, _clip(rng, knock=False, seconds=0.6))

    knock_wav = tmp_path / "knock.wav"
    wavfile.write(knock_wav, SR, _clip(rng, knock=True, seconds=0.6))

    # Fit & persist each detector via CLI --save-model, then score via --load-model without retraining
    for det_name in ("gaussian", "knn", "autoencoder"):
        bundle_path = tmp_path / f"{det_name}_bundle.npz"
        assert (
            monitor_main(
                [
                    "--train-dir",
                    str(norm_dir),
                    "--detector",
                    det_name,
                    "--save-model",
                    str(bundle_path),
                ]
            )
            == 0
        )
        assert bundle_path.is_file() and bundle_path.with_suffix(".json").is_file()
        loaded_det, meta = load_detector_bundle(bundle_path)
        assert meta["calibration_mode"] == "oof_conformal"
        assert meta["drop_speech_band"] is True

        out_jsonl = tmp_path / f"{det_name}_events.jsonl"
        assert (
            monitor_main(
                [
                    "--load-model",
                    str(bundle_path),
                    "--wav",
                    str(knock_wav),
                    "--protocol-id",
                    "pcr-v3",
                    "--step",
                    "3",
                    "--instrument-id",
                    "pump-01",
                    "--out-jsonl",
                    str(out_jsonl),
                ]
            )
            == 0
        )
        ev = AudioEvent.from_json(out_jsonl.read_text(encoding="utf-8").strip())
        assert ev.flagged is True
        assert ev.score == pytest.approx(loaded_det.score(log_mel(load_wav(knock_wav), drop_speech_band=True)))

    # Reject silent upsampling from 8 kHz while recording source_sample_rate_hz on 32 kHz downsample
    low_sr_wav = tmp_path / "sr8k.wav"
    wavfile.write(low_sr_wav, 8000, np.zeros(4000, dtype=np.float32))
    g_det, g_meta = load_detector_bundle(tmp_path / "gaussian_bundle.npz")
    with pytest.raises(ValueError, match="below minimum required 16000 Hz"):
        monitor_clip(
            low_sr_wav,
            g_det,
            threshold=g_meta["threshold"],
            protocol_id="p",
            step=0,
            instrument_id="i",
            resample=True,
        )

    hi_sr_wav = tmp_path / "sr32k_ok.wav"
    wavfile.write(hi_sr_wav, 32000, _clip(rng, knock=False, seconds=1.2))
    ev_32k = monitor_clip(
        hi_sr_wav,
        g_det,
        threshold=g_meta["threshold"],
        protocol_id="p",
        step=0,
        instrument_id="i",
        resample=True,
    )
    assert ev_32k.sample_rate_hz == 16000
    assert ev_32k.source_sample_rate_hz == 32000

    # Causal PCM ring-buffer streaming monitor rejects mismatched calibration_window_s (0.6s vs 1.0s)
    # unless allow_window_mismatch=True is passed (#36)
    long_stream = np.concatenate([_clip(rng, knock=False, seconds=1.5), _clip(rng, knock=True, seconds=1.5)])
    chunks = np.array_split(long_stream, 12)
    with pytest.raises(ValueError, match="calibration_window_s"):
        list(
            monitor_stream(
                chunks,
                g_det,
                threshold=g_meta["threshold"],
                protocol_id="stream-proto",
                step=5,
                instrument_id="valve-01",
                window_s=1.0,
                hop_s=0.5,
            )
        )
    stream_events = list(
        monitor_stream(
            chunks,
            g_det,
            threshold=g_meta["threshold"],
            protocol_id="stream-proto",
            step=5,
            instrument_id="valve-01",
            window_s=1.0,
            hop_s=0.5,
            allow_window_mismatch=True,
        )
    )
    assert len(stream_events) >= 4
    assert any(e.flagged for e in stream_events)
    assert all(e.raw_audio_kept is False and e.raw_audio_saved is False for e in stream_events)

    # Verify evaluate_run streaming mode
    m_dir = tmp_path / "dcase_stream" / "valve"
    (m_dir / "train").mkdir(parents=True)
    (m_dir / "test").mkdir(parents=True)
    for i in range(6):
        wavfile.write(m_dir / "train" / f"s0_source_train_normal_{i:04d}.wav", SR, _clip(rng, seconds=1.0))
    for fname, is_anom in [
        ("s0_source_test_normal_0000.wav", False),
        ("s0_source_test_anomaly_0001.wav", True),
        ("s0_target_test_normal_0002.wav", False),
        ("s0_target_test_anomaly_0003.wav", True),
    ]:
        wavfile.write(m_dir / "test" / fname, SR, _clip(rng, knock=is_anom, seconds=1.0))

    s_rows = evaluate_run(
        tmp_path / "dcase_stream",
        ["valve"],
        ["gaussian"],
        drop_speech_band=True,
        latency_clips=2,
        streaming_window_s=0.5,
        streaming_hop_s=0.25,
    )
    assert s_rows[0]["streaming_window_s"] == pytest.approx(0.5)
    assert s_rows[0]["chunk_latency_ms_median"] >= 0.0
    assert s_rows[0]["false_alarms_per_hour"] >= 0.0
    assert "time_to_first_flag_s_median" in s_rows[0]
    assert "detection_delay_s_median" not in s_rows[0]
    assert "normal_chunk_fpr" in s_rows[0] and "target_chunk_fpr" in s_rows[0]


# --- Issue #36: Window-matched 2 s streaming calibration, bundle window guard & chunk FPR ---
def test_issue36_window_matched_2s_streaming_calibration_and_bundle_guard(tmp_path: Path):
    from acoustic.detect import calibrate_streaming_with_mode
    from acoustic.features import slice_wav_chunks

    rng = np.random.default_rng(36)
    quantile = 0.98
    target_fpr = 1.0 - quantile  # 0.02 -> 3x ceiling = 0.06

    # Train on 60 normal 6 s clips (5 chunks of 2 s at 1 s hop = 300 OOF normal chunks)
    train_wavs = [_clip(rng, knock=False, seconds=6.0) for _ in range(60)]
    train_mels = [log_mel(w, drop_speech_band=True) for w in train_wavs]
    train_chunk_mels = [
        [log_mel(ch, drop_speech_band=True) for ch in slice_wav_chunks(w, 2.0, 1.0, sr=SR)]
        for w in train_wavs
    ]

    # Fresh held-out normal 6 s streams (40 clips * 5 chunks = 200 fresh normal 2 s chunks)
    fresh_norm_wavs = [_clip(rng, knock=False, seconds=6.0) for _ in range(40)]
    fresh_knock_wav = _clip(rng, knock=True, seconds=6.0)

    for det in (
        GaussianDetector(pooling="mean").fit(train_mels),
        KNNDetector(k=5, pooling="mean").fit(train_mels),
    ):
        thr, mode = calibrate_streaming_with_mode(
            det,
            window_s=2.0,
            hop_s=1.0,
            quantile=quantile,
            drop_speech_band=True,
            train_mels=train_mels,
            train_chunk_mels=train_chunk_mels,
        )
        assert mode == "oof_chunk_conformal"
        assert det.calibration_window_s_ == pytest.approx(2.0)
        assert det.calibration_hop_s_ == pytest.approx(1.0)

        bundle_path = tmp_path / f"{det.name}_stream2s.npz"
        save_detector_bundle(bundle_path, det, threshold=thr, calibration_mode=mode, quantile=quantile)
        loaded_det, meta = load_detector_bundle(bundle_path)
        assert meta["calibration_window_s"] == pytest.approx(2.0)
        assert meta["calibration_hop_s"] == pytest.approx(1.0)

        # Reject mismatched window_s (1.0 s != 2.0 s) unless allow_window_mismatch=True
        with pytest.raises(ValueError, match="calibration_window_s"):
            monitor_stream(
                [fresh_norm_wavs[0]],
                loaded_det,
                threshold=meta["threshold"],
                protocol_id="p36",
                step=0,
                instrument_id="valve-01",
                window_s=1.0,
                hop_s=0.5,
            )

        # Score all 200 fresh normal 2 s chunks via monitor_stream(window_s=2.0, hop_s=1.0)
        norm_events = []
        for w in fresh_norm_wavs:
            norm_events.extend(
                monitor_stream(
                    [w],
                    loaded_det,
                    threshold=meta["threshold"],
                    protocol_id="p36",
                    step=0,
                    instrument_id="valve-01",
                    window_s=2.0,
                    hop_s=1.0,
                    flush_tail=False,
                )
            )
        assert len(norm_events) == 200
        chunk_fpr = float(np.mean([ev.flagged for ev in norm_events]))
        assert chunk_fpr <= 3.0 * target_fpr, f"{det.name} chunk_fpr={chunk_fpr:.4f} exceeded 3*(1-q)={3*target_fpr:.4f}"

        # Anomalous knock stream must still flag promptly
        knock_events = list(
            monitor_stream(
                [fresh_knock_wav],
                loaded_det,
                threshold=meta["threshold"],
                protocol_id="p36",
                step=1,
                instrument_id="valve-01",
                window_s=2.0,
                hop_s=1.0,
                flush_tail=False,
            )
        )
        assert any(ev.flagged for ev in knock_events)

    # CLI --window-s / --stream integration check
    norm_dir = tmp_path / "cli_norm"
    norm_dir.mkdir()
    for i in range(6):
        wavfile.write(norm_dir / f"n_{i}.wav", SR, train_wavs[i])
    knock_path = tmp_path / "cli_knock.wav"
    wavfile.write(knock_path, SR, fresh_knock_wav)
    cli_bundle = tmp_path / "cli_stream_bundle.npz"
    assert (
        monitor_main(
            [
                "--train-dir",
                str(norm_dir),
                "--detector",
                "gaussian",
                "--window-s",
                "2.0",
                "--hop-s",
                "1.0",
                "--save-model",
                str(cli_bundle),
            ]
        )
        == 0
    )
    _, cli_meta = load_detector_bundle(cli_bundle)
    assert cli_meta["calibration_window_s"] == pytest.approx(2.0)
    assert cli_meta["calibration_hop_s"] == pytest.approx(1.0)
    # Scoring with mismatched --window-s 1.0 via --stream fails closed (exit code 1)
    assert (
        monitor_main(
            [
                "--load-model",
                str(cli_bundle),
                "--wav",
                str(knock_path),
                "--stream",
                "--window-s",
                "1.0",
            ]
        )
        == 1
    )
    # Matching --window-s 2.0 succeeds
    assert (
        monitor_main(
            [
                "--load-model",
                str(cli_bundle),
                "--wav",
                str(knock_path),
                "--stream",
                "--window-s",
                "2.0",
            ]
        )
        == 0
    )


