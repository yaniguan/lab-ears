"""Unit and benchmark tests for lab-ears community_extensions (Idea #17 & Issue #37)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from community_extensions import (  # noqa: E402
    CrossModalInterlockDetector,
    PipettingCommandWindow,
    SpeechUnfilteredRMSDetector,
    extract_voice_stripped_band_energies,
    run_benchmark,
    synthesize_broadband_speech_distractor,
)
from community_extensions.cross_modal_interlock import (  # noqa: E402
    HIGH_CAVITATION_BAND_HZ,
    LOW_STALL_BAND_HZ,
    _synthesize_pipetting_clip,
    verify_privacy_bands,
)


def test_privacy_band_validator_rejects_speech_overlap() -> None:
    assert verify_privacy_bands((LOW_STALL_BAND_HZ, HIGH_CAVITATION_BAND_HZ)) is True
    assert verify_privacy_bands(((250.0, 500.0), HIGH_CAVITATION_BAND_HZ)) is False
    assert verify_privacy_bands(((1000.0, 2200.0),)) is False
    assert verify_privacy_bands(((3200.0, 3600.0),)) is False
    assert verify_privacy_bands(((200.0, 100.0),)) is False


def test_voice_stripping_eliminates_pure_speech_tones() -> None:
    sr = 16000
    t = np.arange(sr // 2, dtype=np.float64) / sr
    quiet = np.zeros_like(t)
    loud_voice = (
        0.5 * np.sin(2.0 * np.pi * 450.0 * t)
        + 0.4 * np.sin(2.0 * np.pi * 1200.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 2600.0 * t)
    )
    f_quiet = extract_voice_stripped_band_energies(quiet, sample_rate_hz=sr)
    f_voice = extract_voice_stripped_band_energies(loud_voice, sample_rate_hz=sr)

    assert f_voice.raw_audio_saved is False
    assert np.max(f_voice.lo_band_power) < 1e-20
    assert np.max(f_voice.hi_band_power) < 1e-20
    assert np.allclose(f_quiet.lo_band_power, f_voice.lo_band_power, atol=1e-20)


def test_rejects_low_sample_rate() -> None:
    with pytest.raises(ValueError, match="below 8000 Hz"):
        extract_voice_stripped_band_energies(np.zeros(1024), sample_rate_hz=4000)


def test_idle_transit_never_triggers_spurious_halt() -> None:
    sr = 16000
    rng = np.random.default_rng(7)
    t = np.arange(sr // 2, dtype=np.float64) / sr
    calib_cmd = PipettingCommandWindow(0, "aspirate", 0.1, 0.4, "aqueous", 50.0)
    calib_wavs = [
        (rng.normal(0.0, 0.004, size=t.size) + 0.01 * np.sin(2 * np.pi * 120 * t), calib_cmd)
        for _ in range(8)
    ]
    for mode in ("bandstop_only", "harmonic_onset_aware"):
        detector = CrossModalInterlockDetector(sample_rate_hz=sr, speech_rejection_mode=mode)
        detector.calibrate(calib_wavs)

        # Huge 6 kHz spike during idle transit should NOT trigger a pipetting run halt
        spike_wav = 0.25 * np.sin(2.0 * np.pi * 6000.0 * t)
        idle_cmd = PipettingCommandWindow(1, "idle_transit", 0.1, 0.4, "aqueous", 50.0)
        dec = detector.evaluate(spike_wav, idle_cmd)
        assert dec.halt_run is False
        assert dec.reason == "idle_transit_ignored"
        assert dec.raw_audio_saved is False


def test_dry_aspiration_and_stepper_stall_trigger_halt() -> None:
    sr = 16000
    rng = np.random.default_rng(19)
    t = np.arange(sr // 2, dtype=np.float64) / sr
    cmd = PipettingCommandWindow(2, "aspirate", 0.10, 0.38, "aqueous", 50.0)
    calib_wavs = [
        (rng.normal(0.0, 0.004, size=t.size) + 0.01 * np.sin(2 * np.pi * 120 * t), cmd)
        for _ in range(10)
    ]
    for mode in ("bandstop_only", "harmonic_onset_aware"):
        detector = CrossModalInterlockDetector(sample_rate_hz=sr, speech_rejection_mode=mode)
        detector.calibrate(calib_wavs)

        in_win = ((t >= 0.12) & (t <= 0.36)).astype(np.float64)
        cav_wav = calib_wavs[0][0] + in_win * 0.05 * np.sin(2.0 * np.pi * 5800.0 * t)
        cav_dec = detector.evaluate(cav_wav, cmd)
        assert cav_dec.halt_run is True
        assert "cavitation" in cav_dec.reason

        stall_wav = calib_wavs[0][0] + in_win * 0.06 * np.sin(2.0 * np.pi * 95.0 * t)
        stall_dec = detector.evaluate(stall_wav, cmd)
        assert stall_dec.halt_run is True
        assert "stall" in stall_dec.reason


def test_issue37_broadband_speech_false_halt_table_and_harmonic_rejection() -> None:
    report = run_benchmark()
    assert report["n_clips"] == 120
    assert report["sample_rate_hz"] == 16000
    assert report["raw_audio_saved"] is False
    assert report["speech_band_stripped_hz"] == [300, 3400]
    assert "inband_tones_380_3150hz_only" == report["cross_modal_voice_stripped_interlock"]["speech_signal"]

    unfiltered = report["speech_unfiltered_rms"]
    assert unfiltered["false_alarm_rate_under_speech"] >= 0.50

    # Pin the realistic broadband speech (pulse-train F0 100-220 Hz + 4-7.8 kHz fricatives) table
    broad = report["broadband_speech_evaluation"]
    vs_clip = {row["speech_level_db"]: row for row in broad["levels_vs_clip_rms_db"]}
    vs_conv = {row["speech_level_db"]: row for row in broad["levels_vs_conversation_rms_db"]}
    assert set(vs_clip) == {0.0, -10.0, -20.0, -30.0}
    assert set(vs_conv) == {0.0, -10.0, -20.0, -30.0}

    # 1. Raw bandstop_only DOES suffer substantial false halts under realistic broadband speech at 0 dB
    assert vs_clip[0.0]["bandstop_only"]["false_halt_rate_under_speech"] == pytest.approx(0.6731, abs=1e-3)
    assert vs_clip[0.0]["bandstop_only"]["false_halt_rate_active_speech"] == pytest.approx(0.7955, abs=1e-3)
    assert vs_conv[0.0]["bandstop_only"]["false_halt_rate_active_speech"] == pytest.approx(1.0000, abs=1e-3)
    assert vs_conv[-10.0]["bandstop_only"]["false_halt_rate_active_speech"] == pytest.approx(1.0000, abs=1e-3)
    assert vs_conv[-20.0]["bandstop_only"]["false_halt_rate_under_speech"] == pytest.approx(0.3654, abs=1e-3)
    assert vs_conv[-30.0]["bandstop_only"]["false_halt_rate_under_speech"] == pytest.approx(0.0000, abs=1e-3)

    # 2. Harmonic + command-onset aware mode cuts 0 dB clip-RMS false halts from 67.3% -> 0.0%
    #    and cuts -20 dB conversation-RMS false halts from 36.5% -> 0.0% with >= 95% fault recall
    assert vs_clip[0.0]["harmonic_onset_aware"]["false_halt_rate_under_speech"] == pytest.approx(0.0000, abs=1e-3)
    assert vs_clip[0.0]["harmonic_onset_aware"]["fault_recall"] == pytest.approx(0.9583, abs=1e-3)
    assert vs_clip[0.0]["harmonic_onset_aware"]["overall_fault_f1"] == pytest.approx(0.9787, abs=1e-3)
    for db in (-10.0, -20.0, -30.0):
        assert vs_clip[db]["harmonic_onset_aware"]["false_halt_rate_under_speech"] == pytest.approx(0.0, abs=1e-3)
        assert vs_clip[db]["harmonic_onset_aware"]["overall_fault_f1"] == pytest.approx(1.0, abs=1e-3)

    assert vs_conv[0.0]["harmonic_onset_aware"]["false_halt_rate_under_speech"] == pytest.approx(0.4808, abs=1e-3)
    assert vs_conv[-10.0]["harmonic_onset_aware"]["false_halt_rate_under_speech"] == pytest.approx(0.3846, abs=1e-3)
    assert vs_conv[-20.0]["harmonic_onset_aware"]["false_halt_rate_under_speech"] == pytest.approx(0.0000, abs=1e-3)
    assert vs_conv[-20.0]["harmonic_onset_aware"]["overall_fault_f1"] == pytest.approx(1.0000, abs=1e-3)
    assert vs_conv[-30.0]["harmonic_onset_aware"]["false_halt_rate_under_speech"] == pytest.approx(0.0000, abs=1e-3)

    # Ensure SpeechUnfilteredRMSDetector is also directly exercisable
    rms_det = SpeechUnfilteredRMSDetector()
    rms_det.calibrate([np.zeros(800, dtype=np.float64)])
    assert rms_det.evaluate(np.ones(800, dtype=np.float64), PipettingCommandWindow(0, "aspirate", 0.0, 0.05)).halt_run


def test_issue37_speech_dir_wav_mixing_hook_offline(tmp_path: Path) -> None:
    sr = 16000
    rng = np.random.default_rng(37)
    t = np.arange(sr, dtype=np.float64) / sr
    for idx in range(4):
        sp = synthesize_broadband_speech_distractor(t, rng, sample_rate_hz=sr)
        sp_norm = np.clip(0.3 * sp / (np.max(np.abs(sp)) + 1e-9), -0.99, 0.99).astype(np.float32)
        wavfile.write(tmp_path / f"utt_{idx}.wav", sr, sp_norm)

    report = run_benchmark(speech_dir=tmp_path)
    real_eval = report["real_speech_evaluation"]
    assert real_eval is not None
    assert real_eval["n_wav_files_loaded"] == 4
    clip_rows = real_eval["levels_vs_clip_rms_db"]
    assert [r["speech_level_db"] for r in clip_rows] == [0.0, -10.0, -20.0, -30.0]
    assert clip_rows[0]["harmonic_onset_aware"]["false_halt_rate_under_speech"] < (
        clip_rows[0]["bandstop_only"]["false_halt_rate_under_speech"]
    )


def test_issue37_real_librispeech_dir_when_present() -> None:
    env_dir = os.environ.get("LAB_EARS_SPEECH_DIR")
    speech_dir = Path(env_dir) if env_dir else (_ROOT / "data" / "librispeech")
    if not speech_dir.is_dir() or not any(speech_dir.rglob("*.wav")):
        pytest.skip(
            f"Optional local 16 kHz real speech corpus (e.g. LibriSpeech CC-BY-4.0) not found at {speech_dir}"
        )
    report = run_benchmark(speech_dir=speech_dir)
    assert report["real_speech_evaluation"] is not None
    assert report["real_speech_evaluation"]["n_wav_files_loaded"] >= 1


# --- Issue #29 & #30: High sample rates (44.1/48 kHz), NaN/Inf rejection, tail padding & Nyquist ---
@pytest.mark.parametrize("sr", [16000, 44100, 48000])
@pytest.mark.parametrize("mode", ["bandstop_only", "harmonic_onset_aware"])
def test_issue29_high_sample_rate_stepper_stall_and_cavitation_halts(sr: int, mode: str) -> None:
    rng = np.random.default_rng(42 + sr)
    calib = [_synthesize_pipetting_clip(rng, i, "nominal", False, "aqueous", sample_rate_hz=sr) for i in range(6)]
    det = CrossModalInterlockDetector(sample_rate_hz=sr, speech_rejection_mode=mode)
    det.calibrate(calib)
    assert np.isfinite(det.base_lo_power) and det.base_lo_power > 0.0
    assert np.isfinite(det.base_hi_power) and det.base_hi_power > 0.0

    stall_wav, stall_cmd = _synthesize_pipetting_clip(
        rng,
        99,
        "stepper_stall",
        True,
        "aqueous",
        sample_rate_hz=sr,
        speech_mode="broadband_pulse_fricative",
        speech_snr_db=-10.0,
    )
    stall_dec = det.evaluate(stall_wav, stall_cmd)
    assert np.isfinite(stall_dec.stall_ratio)
    assert stall_dec.halt_run is True
    assert "stall" in stall_dec.reason

    cav_wav, cav_cmd = _synthesize_pipetting_clip(
        rng,
        100,
        "dry_aspiration",
        True,
        "aqueous",
        sample_rate_hz=sr,
        speech_mode="broadband_pulse_fricative",
        speech_snr_db=-10.0,
    )
    cav_dec = det.evaluate(cav_wav, cav_cmd)
    assert np.isfinite(cav_dec.cavitation_ratio)
    assert cav_dec.halt_run is True
    assert "cavitation" in cav_dec.reason
    wire_d = cav_dec.to_wire_dict()
    assert wire_d["halt_run"] is True and wire_d["raw_audio_saved"] is False


def test_issue29_nan_inf_nyquist_and_trailing_subhop_padding() -> None:
    for bad_val in (np.nan, np.inf, -np.inf):
        bad = np.zeros(2048, dtype=np.float64)
        bad[50] = bad_val
        with pytest.raises(ValueError, match="NaN or Inf"):
            extract_voice_stripped_band_energies(bad, sample_rate_hz=16000)

    with pytest.raises(ValueError, match="Unsupported speech_rejection_mode"):
        CrossModalInterlockDetector(sample_rate_hz=16000, speech_rejection_mode="invalid")

    # 8 kHz cannot observe [3600, 7800] Hz cavitation above 4 kHz Nyquist
    with pytest.raises(ValueError, match="Nyquist|insufficient"):
        CrossModalInterlockDetector(sample_rate_hz=8000)
    with pytest.raises(ValueError, match="Nyquist"):
        extract_voice_stripped_band_energies(np.zeros(2048), sample_rate_hz=8000)

    # Explicit coarse n_fft leaving low band empty raises ValueError rather than NaN
    with pytest.raises(ValueError, match="Empty STFT band mask"):
        extract_voice_stripped_band_energies(np.zeros(2048), sample_rate_hz=48000, n_fft=512, hop_length=160)

    # Trailing sub-hop cavitation burst must be preserved via right-padding
    sr = 16000
    n_samples = 512 + 3 * 160 + 90
    t_tail = np.arange(90, dtype=np.float64) / sr
    x_quiet = np.zeros(n_samples, dtype=np.float64)
    x_burst = np.zeros(n_samples, dtype=np.float64)
    x_burst[-90:] = 0.5 * np.sin(2.0 * np.pi * 6000.0 * t_tail)
    f_q = extract_voice_stripped_band_energies(x_quiet, sample_rate_hz=sr)
    f_b = extract_voice_stripped_band_energies(x_burst, sample_rate_hz=sr)
    assert f_b.hi_band_power.shape[0] == 5
    assert f_b.hi_band_power[-1] > f_q.hi_band_power[-1] + 1e-4
