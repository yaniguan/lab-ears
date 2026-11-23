"""Unit and benchmark tests for lab-ears community_extensions (Idea #17)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from community_extensions import (  # noqa: E402
    CrossModalInterlockDetector,
    PipettingCommandWindow,
    SpeechUnfilteredRMSDetector,
    extract_voice_stripped_band_energies,
    run_benchmark,
)
from community_extensions.cross_modal_interlock import (  # noqa: E402
    HIGH_CAVITATION_BAND_HZ,
    LOW_STALL_BAND_HZ,
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
    detector = CrossModalInterlockDetector(sample_rate_hz=sr)
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
    detector = CrossModalInterlockDetector(sample_rate_hz=sr)
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


def test_benchmark_120_clips_speech_resilience_and_f1() -> None:
    report = run_benchmark()
    assert report["n_clips"] == 120
    assert report["sample_rate_hz"] == 16000
    assert report["raw_audio_saved"] is False
    assert report["speech_band_stripped_hz"] == [300, 3400]

    unfiltered = report["speech_unfiltered_rms"]
    interlock = report["cross_modal_voice_stripped_interlock"]

    # Unfiltered RMS suffers high false-alarm rate under human conversation
    assert unfiltered["false_alarm_rate_under_speech"] >= 0.50
    # Cross-modal voice-stripped interlock achieves near-zero false alarms under speech and high F1
    assert interlock["false_alarm_rate_under_speech"] <= 0.05
    assert interlock["dry_aspiration_f1"] >= 0.92
    assert interlock["overall_fault_f1"] >= 0.92
    assert interlock["dry_aspiration_f1"] > unfiltered["dry_aspiration_f1"] + 0.25

    # Ensure SpeechUnfilteredRMSDetector is also directly exercisable
    rms_det = SpeechUnfilteredRMSDetector()
    rms_det.calibrate([np.zeros(800, dtype=np.float64)])
    assert rms_det.evaluate(np.ones(800, dtype=np.float64), PipettingCommandWindow(0, "aspirate", 0.0, 0.05)).halt_run
