"""Idea #17: Acoustic-Command Cross-Modal Cavitation & Dry-Aspiration Interlock.

Inspired by zjunlp/LabVLA and wet-lab acoustic pipetting fault sensing threads.
Strips the [300, 3400] Hz human conversation frequency band (`raw_audio_saved = False`)
and cross-correlates step-indexed aspirate / dispense command windows plus liquid
rheology expectations against:
  - High-band (3.6-7.8 kHz) cavitation / dry-aspiration air-suck energy ratios, and
  - Low-band (60-260 Hz) stepper-motor stall cogging harmonics,
triggering an immediate run-halt before air bubbles are dispensed across an assay plate.

Important Scope & Speech-Realism Caveat (Issue #37):
  - The legacy `1.0` fault F1 (`0.0` false-alarm rate) held ONLY for synthetic pure tones
    confined strictly inside `[380, 3150] Hz`, which the `[300, 3400] Hz` bandstop removes
    by construction, until real-speech + physical pipette recordings are benchmarked.
  - Real human speech also contains voice-pitch fundamentals (`~85-255 Hz` with `1/h`
    glottal harmonics extending up to `7.8 kHz`) and unvoiced sibilant/fricative bursts
    (`/s/`, `/sh/` in `4-8 kHz`), both of which overlap `[60, 260] Hz` and `[3600, 7800] Hz`.
  - Under a realistic broadband glottal pulse-train (`F0 100-220 Hz`, `1/h` harmonics) +
    `4-7.8 kHz` fricative speech signal, raw `speech_rejection_mode="bandstop_only"`
    produces substantial false halts at high speech-to-clip levels (`0 dB` vs clip RMS and
    `0` to `-20 dB` vs room-conversation stand-in RMS).
  - `speech_rejection_mode="harmonic_onset_aware"` mitigates voiced harmonic and brief
    sibilant bleed strictly from the voice-stripped `[60, 260] Hz` and `[3600, 7800] Hz`
    bands via interharmonic comb-valley floor estimation, cross-band `F0` cogging-vs-voice
    disambiguation, and sustained command-window stroke persistence.
  - `RHEOLOGY_PRIORS` are heuristic synthetic simulation multipliers (no physical wet-lab
    liquid-handler dataset calibration yet); real benchtop deployment requires per-instrument
    contact-microphone calibration and optional `run_benchmark(speech_dir=...)` validation
    with real 16 kHz speech WAVs (e.g., LibriSpeech, CC-BY-4.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from scipy.signal import resample_poly

SPEECH_BAND_LO_HZ: float = 300.0
SPEECH_BAND_HI_HZ: float = 3400.0
LOW_STALL_BAND_HZ: tuple[float, float] = (60.0, 260.0)
HIGH_CAVITATION_BAND_HZ: tuple[float, float] = (3600.0, 7800.0)
SPEECH_REJECTION_MODES: tuple[str, str] = ("bandstop_only", "harmonic_onset_aware")
DEFAULT_SPEECH_SNR_LEVELS_DB: tuple[float, ...] = (0.0, -10.0, -20.0, -30.0)

# Heuristic synthetic spectral band power multipliers by liquid rheology class
# (hand-set simulation priors, not empirically fit to physical instrument recordings).
RHEOLOGY_PRIORS: dict[str, dict[str, float]] = {
    "aqueous": {
        "expected_lo_power": 1.00,
        "expected_hi_power": 1.00,
        "cavitation_ratio_limit": 2.45,
        "stall_ratio_limit": 2.55,
    },
    "viscous_glycerol": {
        "expected_lo_power": 1.45,
        "expected_hi_power": 0.85,
        "cavitation_ratio_limit": 2.55,
        "stall_ratio_limit": 2.50,
    },
    "volatile_solvent": {
        "expected_lo_power": 0.95,
        "expected_hi_power": 1.20,
        "cavitation_ratio_limit": 2.50,
        "stall_ratio_limit": 2.60,
    },
}


@dataclass(frozen=True)
class PipettingCommandWindow:
    """Step-indexed liquid-handler command window synchronized with benchtop audio."""

    step_index: int
    action: str  # "aspirate", "dispense", or "idle_transit"
    start_s: float
    end_s: float
    rheology: str = "aqueous"
    volume_ul: float = 50.0

    @property
    def is_active_transfer(self) -> bool:
        return self.action in ("aspirate", "dispense")


@dataclass(frozen=True)
class VoiceStrippedFeatures:
    """Privacy-preserving band power trajectory with zero raw-audio retention."""

    raw_audio_saved: bool
    sample_rate_hz: int
    retained_bands_hz: tuple[tuple[float, float], ...]
    frame_times_s: np.ndarray
    lo_band_power: np.ndarray
    hi_band_power: np.ndarray


@dataclass(frozen=True)
class InterlockDecision:
    """Result of evaluating a pipetting clip for cavitation or stepper stall."""

    step_index: int
    halt_run: bool
    reason: str
    raw_audio_saved: bool
    speech_band_dropped: bool
    cavitation_ratio: float
    stall_ratio: float
    score: float

    def to_wire_dict(self, *, instrument_id: str = "liquid-handler-01") -> dict[str, Any]:
        """Serialize interlock decision for `autolab-wire/v1` packet transport."""
        return {
            "step_index": int(self.step_index),
            "instrument_id": str(instrument_id),
            "halt_run": bool(self.halt_run),
            "reason": str(self.reason),
            "raw_audio_saved": bool(self.raw_audio_saved),
            "raw_audio_kept": bool(self.raw_audio_saved),
            "speech_band_dropped": bool(self.speech_band_dropped),
            "retained_bands_hz": (
                [list(LOW_STALL_BAND_HZ), list(HIGH_CAVITATION_BAND_HZ)]
                if self.speech_band_dropped
                else []
            ),
            "cavitation_ratio": float(self.cavitation_ratio),
            "stall_ratio": float(self.stall_ratio),
            "score": float(self.score),
        }


def verify_privacy_bands(
    bands_hz: tuple[tuple[float, float], ...],
    *,
    sample_rate_hz: int | None = None,
    skirt_hz: float = 0.0,
) -> bool:
    """Verify retained bands are non-empty, below Nyquist, and strictly outside (300, 3400) Hz."""
    if not bands_hz or skirt_hz < 0.0 or not math.isfinite(float(skirt_hz)):
        return False
    nyquist_hz = 0.5 * float(sample_rate_hz) if sample_rate_hz is not None else None
    for lo_hz, hi_hz in bands_hz:
        if not (math.isfinite(float(lo_hz)) and math.isfinite(float(hi_hz))):
            return False
        if lo_hz > hi_hz or lo_hz < 0.0:
            return False
        if nyquist_hz is not None and hi_hz > nyquist_hz:
            return False
        if not ((hi_hz + skirt_hz) <= SPEECH_BAND_LO_HZ or (lo_hz - skirt_hz) >= SPEECH_BAND_HI_HZ):
            return False
    return True


def _default_stft_params(sample_rate_hz: int, *, min_window_s: float = 0.032) -> tuple[int, int]:
    """Scale STFT window (power of two >= `min_window_s`) and hop (~10 ms) with sample rate."""
    target_n = max(512, int(math.ceil(float(sample_rate_hz) * float(min_window_s))))
    n_fft = 1 << (target_n - 1).bit_length()
    hop_length = max(1, int(round(float(sample_rate_hz) * 0.010)))
    return n_fft, hop_length


_COMB_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_comb_templates(
    sample_rate_hz: int,
    n_fft: int,
    lo_freqs: np.ndarray,
    hi_freqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute voiced glottal comb tooth/valley weights over `[3600, 7800] Hz` and `F0` masks."""
    key = (int(sample_rate_hz), int(n_fft))
    cached = _COMB_CACHE.get(key)
    if cached is not None:
        return cached
    f0_grid = np.linspace(96.0, 228.0, 67, dtype=np.float64)
    pmod = (hi_freqs[None, :] / f0_grid[:, None]) % 1.0
    tooth_w = ((pmod <= 0.18) | (pmod >= 0.82)).astype(np.float64)
    tooth_w /= np.maximum(np.sum(tooth_w, axis=1, keepdims=True), 1.0)
    valley_w = ((pmod >= 0.30) & (pmod <= 0.70)).astype(np.float64)
    valley_w /= np.maximum(np.sum(valley_w, axis=1, keepdims=True), 1.0)
    lo_keep_mat = ~(
        (np.abs(lo_freqs[None, :] - f0_grid[:, None]) <= 24.0)
        | (np.abs(lo_freqs[None, :] - 2.0 * f0_grid[:, None]) <= 24.0)
    )
    res = (f0_grid, tooth_w, valley_w, lo_keep_mat)
    _COMB_CACHE[key] = res
    return res


def extract_voice_stripped_band_energies(
    waveform: np.ndarray,
    sample_rate_hz: int = 16000,
    n_fft: int | None = None,
    hop_length: int | None = None,
    *,
    harmonic_comb_suppression: bool = False,
) -> VoiceStrippedFeatures:
    """Compute STFT band energies while hard-dropping every bin overlapping (300, 3400) Hz.

    A bin k with center f_k = k * fs / n_fft has main-lobe support roughly
    [f_k - df, f_k + df] under a Hann window (df = fs / n_fft). We retain ONLY bins
    whose full main-lobe skirt lies inside [60, 260] Hz or [3600, 7800] Hz, guaranteeing
    zero overlap with [300, 3400] Hz human conversation.

    When `harmonic_comb_suppression=True`, a >= 64 ms STFT (`df <= 15.625 Hz`) checks
    strictly within `[3600, 7800] Hz` for periodic glottal pitch-comb teeth `k * F0`
    (`F0 in [96, 228] Hz`). On voiced frames (`tooth / valley > 1.85`), high-band energy
    is estimated from the interharmonic valley floor `(k + 0.5) * F0` (preserving broadband
    cavitation turbulence while rejecting `1/h` voice harmonics) and low-band `[60, 260] Hz`
    notches out `F0` and `2 * F0` while keeping non-coincident stepper cogging tones.
    """
    if isinstance(sample_rate_hz, bool) or int(sample_rate_hz) < 8000:
        raise ValueError(f"sample_rate_hz={sample_rate_hz} below 8000 Hz machine floor")
    sample_rate_hz = int(sample_rate_hz)

    retained_bands = (LOW_STALL_BAND_HZ, HIGH_CAVITATION_BAND_HZ)
    nyquist_hz = 0.5 * float(sample_rate_hz)
    for lo_hz, hi_hz in retained_bands:
        if hi_hz > nyquist_hz:
            raise ValueError(
                f"Retained band [{lo_hz}, {hi_hz}] Hz exceeds Nyquist limit {nyquist_hz} Hz "
                f"for sample_rate_hz={sample_rate_hz} (>= {int(math.ceil(2.0 * hi_hz))} Hz required)"
            )

    min_win_s = 0.064 if harmonic_comb_suppression else 0.032
    def_n_fft, def_hop = _default_stft_params(sample_rate_hz, min_window_s=min_win_s)
    n_fft_val = def_n_fft if n_fft is None else int(n_fft)
    hop_val = def_hop if hop_length is None else int(hop_length)
    if n_fft_val < 16 or hop_val < 1:
        raise ValueError(f"Invalid STFT parameters n_fft={n_fft_val}, hop_length={hop_val}")

    raw_arr = np.asarray(waveform, dtype=np.float64)
    if raw_arr.size == 0:
        raise ValueError("Waveform must contain at least 1 sample")
    if not np.isfinite(raw_arr).all():
        raise ValueError("Waveform contains NaN or Inf")

    if not verify_privacy_bands(retained_bands, sample_rate_hz=sample_rate_hz):
        raise AssertionError("Retained bands violated [300, 3400] Hz speech-drop or Nyquist invariant")

    x = raw_arr.ravel()
    # Stage 1: Full-clip spectral notch removing [300, 3400] Hz BEFORE zero-padding to avoid step leakage
    clip_spec = np.fft.rfft(x)
    clip_freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    speech_notch = (clip_freqs >= SPEECH_BAND_LO_HZ) & (clip_freqs <= SPEECH_BAND_HI_HZ)
    clip_spec[speech_notch] = 0.0
    x = np.fft.irfft(clip_spec, n=x.size)

    if x.size < n_fft_val:
        x = np.pad(x, (0, n_fft_val - x.size))
    else:
        rem = (x.size - n_fft_val) % hop_val
        if rem != 0:
            x = np.pad(x, (0, hop_val - rem))
    x = np.ascontiguousarray(x, dtype=np.float64)

    window = np.hanning(n_fft_val)
    win_norm = float(np.sum(window * window)) + 1e-12

    freqs = np.fft.rfftfreq(n_fft_val, d=1.0 / sample_rate_hz)
    df = float(sample_rate_hz) / float(n_fft_val)

    # Full-skirt support [f - df, f + df] must stay within each target band and outside [300, 3400] Hz
    lo_mask = ((freqs - df) >= LOW_STALL_BAND_HZ[0]) & ((freqs + df) <= LOW_STALL_BAND_HZ[1])
    hi_mask = ((freqs - df) >= HIGH_CAVITATION_BAND_HZ[0]) & ((freqs + df) <= HIGH_CAVITATION_BAND_HZ[1])
    if not np.any(lo_mask) or not np.any(hi_mask):
        raise ValueError(
            f"Empty STFT band mask (lo_bins={int(np.sum(lo_mask))}, hi_bins={int(np.sum(hi_mask))}) "
            f"for sample_rate_hz={sample_rate_hz}, n_fft={n_fft_val} (df={df:.2f} Hz)"
        )
    assert not np.any(lo_mask & hi_mask)
    assert np.all((freqs[lo_mask] + df) <= SPEECH_BAND_LO_HZ)
    assert np.all((freqs[hi_mask] - df) >= SPEECH_BAND_HI_HZ)

    frames = sliding_window_view(x, n_fft_val)[::hop_val]
    n_frames = int(frames.shape[0])
    spec = np.fft.rfft(frames * window[None, :], axis=1)
    power = (np.abs(spec) ** 2) / win_norm

    hp = power[:, hi_mask]
    lp = power[:, lo_mask]
    if harmonic_comb_suppression:
        _, tooth_w, valley_w, lo_keep_mat = _get_comb_templates(
            sample_rate_hz,
            n_fft_val,
            freqs[lo_mask],
            freqs[hi_mask],
        )
        t_means = hp @ tooth_w.T
        v_means = hp @ valley_w.T
        ratios = (t_means + 1e-18) / (v_means + 1e-18)
        best_idx = np.argmax(ratios, axis=1)
        row_idx = np.arange(n_frames)
        best_r = ratios[row_idx, best_idx]
        n_top = min(3, hp.shape[1])
        top_share = np.sum(np.partition(hp, -n_top, axis=1)[:, -n_top:], axis=1) / (
            np.sum(hp, axis=1) + 1e-18
        )
        # Require multi-tooth comb occupancy (single narrowband whistle has top_share > 0.90)
        voiced = (best_r > 1.85) & (t_means[row_idx, best_idx] > 1e-12) & (top_share < 0.55)
        hi_power = np.where(voiced, v_means[row_idx, best_idx], np.mean(hp, axis=1))
        lo_power = np.mean(lp, axis=1)
        for frame_i in np.where(voiced)[0]:
            keep_bins = lo_keep_mat[best_idx[frame_i]]
            lo_power[frame_i] = float(np.mean(lp[frame_i, keep_bins])) if np.any(keep_bins) else 0.0
    else:
        lo_power = np.mean(lp, axis=1)
        hi_power = np.mean(hp, axis=1)

    if not (np.isfinite(lo_power).all() and np.isfinite(hi_power).all()):
        raise ValueError("Non-finite STFT band power computed")
    frame_times = (np.arange(n_frames, dtype=np.float64) * hop_val + 0.5 * n_fft_val) / sample_rate_hz

    return VoiceStrippedFeatures(
        raw_audio_saved=False,
        sample_rate_hz=sample_rate_hz,
        retained_bands_hz=retained_bands,
        frame_times_s=frame_times,
        lo_band_power=lo_power,
        hi_band_power=hi_power,
    )


class SpeechUnfilteredRMSDetector:
    """Baseline single-modal detector using broadband RMS without voice stripping or step gating."""

    def __init__(self, sigma_multiplier: float = 3.0) -> None:
        self.sigma_multiplier = float(sigma_multiplier)
        self.rms_threshold: float = 0.05

    def calibrate(self, normal_quiet_clips: list[np.ndarray]) -> None:
        if not normal_quiet_clips:
            raise ValueError("Calibration requires at least one quiet clip")
        vals: list[float] = []
        for c in normal_quiet_clips:
            arr = np.asarray(c, dtype=np.float64)
            if arr.size == 0 or not np.isfinite(arr).all():
                raise ValueError("Waveform must be non-empty and finite")
            vals.append(float(np.sqrt(np.mean(arr ** 2))))
        rms_vals = np.array(vals, dtype=np.float64)
        mu = float(np.mean(rms_vals))
        sd = float(np.std(rms_vals)) + 1e-9
        self.rms_threshold = mu + self.sigma_multiplier * sd

    def evaluate(
        self,
        waveform: np.ndarray,
        command: PipettingCommandWindow,
    ) -> InterlockDecision:
        arr = np.asarray(waveform, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("Waveform must be non-empty and finite")
        rms = float(np.sqrt(np.mean(arr ** 2)))
        flagged = rms > self.rms_threshold
        return InterlockDecision(
            step_index=command.step_index,
            halt_run=flagged,
            reason="broadband_rms_exceeded" if flagged else "nominal",
            raw_audio_saved=True,
            speech_band_dropped=False,
            cavitation_ratio=rms / max(self.rms_threshold, 1e-9),
            stall_ratio=rms / max(self.rms_threshold, 1e-9),
            score=rms,
        )


class CrossModalInterlockDetector:
    """Voice-stripped, command-synchronized cavitation & stepper-stall interlock."""

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        *,
        speech_rejection_mode: str = "bandstop_only",
    ) -> None:
        if isinstance(sample_rate_hz, bool) or int(sample_rate_hz) < 8000:
            raise ValueError(f"sample_rate_hz={sample_rate_hz} below 8000 Hz machine floor")
        min_nyquist_sr = int(math.ceil(2.0 * HIGH_CAVITATION_BAND_HZ[1]))
        if int(sample_rate_hz) < min_nyquist_sr:
            raise ValueError(
                f"sample_rate_hz={sample_rate_hz} insufficient for high cavitation band "
                f"{HIGH_CAVITATION_BAND_HZ} Hz (>= {min_nyquist_sr} Hz required)"
            )
        if speech_rejection_mode not in SPEECH_REJECTION_MODES:
            raise ValueError(
                f"Unsupported speech_rejection_mode={speech_rejection_mode!r}; "
                f"expected one of {SPEECH_REJECTION_MODES}"
            )
        self.sample_rate_hz = int(sample_rate_hz)
        self.speech_rejection_mode = str(speech_rejection_mode)
        self.base_lo_power: float = 1e-5
        self.base_hi_power: float = 1e-5
        self.calibrated: bool = False

    @property
    def _is_harmonic_aware(self) -> bool:
        return self.speech_rejection_mode == "harmonic_onset_aware"

    def calibrate(
        self,
        normal_clips: list[tuple[np.ndarray, PipettingCommandWindow]],
    ) -> None:
        lo_normed: list[float] = []
        hi_normed: list[float] = []
        lo_pct = 50.0 if self._is_harmonic_aware else 80.0
        hi_pct = 45.0 if self._is_harmonic_aware else 80.0

        for wav, cmd in normal_clips:
            if not cmd.is_active_transfer:
                continue
            feats = extract_voice_stripped_band_energies(
                wav,
                sample_rate_hz=self.sample_rate_hz,
                harmonic_comb_suppression=self._is_harmonic_aware,
            )
            in_win = (feats.frame_times_s >= cmd.start_s) & (feats.frame_times_s <= cmd.end_s)
            if not np.any(in_win):
                continue
            prior = RHEOLOGY_PRIORS.get(cmd.rheology, RHEOLOGY_PRIORS["aqueous"])
            lo_normed.append(
                float(np.percentile(feats.lo_band_power[in_win], lo_pct)) / prior["expected_lo_power"]
            )
            hi_normed.append(
                float(np.percentile(feats.hi_band_power[in_win], hi_pct)) / prior["expected_hi_power"]
            )

        if not lo_normed or not hi_normed:
            raise ValueError("Calibration requires at least one active normal transfer clip")
        base_lo = float(np.median(lo_normed)) + 1e-12
        base_hi = float(np.median(hi_normed)) + 1e-12
        if not (math.isfinite(base_lo) and math.isfinite(base_hi) and base_lo > 0.0 and base_hi > 0.0):
            raise ValueError(
                f"Non-finite or non-positive calibrated band baseline (lo={base_lo!r}, hi={base_hi!r})"
            )
        self.base_lo_power = base_lo
        self.base_hi_power = base_hi
        self.calibrated = True

    def evaluate(
        self,
        waveform: np.ndarray,
        command: PipettingCommandWindow,
    ) -> InterlockDecision:
        if not self.calibrated:
            raise RuntimeError("CrossModalInterlockDetector must be calibrated before evaluation")

        feats = extract_voice_stripped_band_energies(
            waveform,
            sample_rate_hz=self.sample_rate_hz,
            harmonic_comb_suppression=self._is_harmonic_aware,
        )
        if not command.is_active_transfer:
            return InterlockDecision(
                step_index=command.step_index,
                halt_run=False,
                reason="idle_transit_ignored",
                raw_audio_saved=False,
                speech_band_dropped=True,
                cavitation_ratio=0.0,
                stall_ratio=0.0,
                score=0.0,
            )

        in_win = (feats.frame_times_s >= command.start_s) & (feats.frame_times_s <= command.end_s)
        out_win = ~in_win
        if not np.any(in_win):
            return InterlockDecision(
                step_index=command.step_index,
                halt_run=False,
                reason="empty_command_window",
                raw_audio_saved=False,
                speech_band_dropped=True,
                cavitation_ratio=0.0,
                stall_ratio=0.0,
                score=0.0,
            )

        prior = RHEOLOGY_PRIORS.get(command.rheology, RHEOLOGY_PRIORS["aqueous"])
        vol_scale = max(0.85, min(1.25, (command.volume_ul / 50.0) ** 0.15))

        if self._is_harmonic_aware:
            # Sustained stroke percentiles (45th/50th) reject transient <90 ms consonant bursts,
            # while 75th-percentile out-of-window subtraction rejects asynchronous room speech.
            hi_in = float(np.percentile(feats.hi_band_power[in_win], 45))
            lo_in = float(np.percentile(feats.lo_band_power[in_win], 50))
            hi_bg = float(np.percentile(feats.hi_band_power[out_win], 75)) if np.any(out_win) else 0.0
            lo_bg = float(np.percentile(feats.lo_band_power[out_win], 75)) if np.any(out_win) else 0.0
            bg_weight = 0.85
        else:
            hi_in = float(np.percentile(feats.hi_band_power[in_win], 85))
            lo_in = float(np.percentile(feats.lo_band_power[in_win], 85))
            hi_bg = float(np.median(feats.hi_band_power[out_win])) if np.any(out_win) else 0.0
            lo_bg = float(np.median(feats.lo_band_power[out_win])) if np.any(out_win) else 0.0
            bg_weight = 0.50

        if not (math.isfinite(hi_in) and math.isfinite(lo_in) and math.isfinite(hi_bg) and math.isfinite(lo_bg)):
            raise ValueError("Non-finite window band power statistics encountered during interlock evaluation")

        cavitation_ratio = max(0.0, hi_in - bg_weight * hi_bg) / (
            self.base_hi_power * prior["expected_hi_power"] * vol_scale
        )
        stall_ratio = max(0.0, lo_in - bg_weight * lo_bg) / (
            self.base_lo_power * prior["expected_lo_power"] * vol_scale
        )
        if not (math.isfinite(cavitation_ratio) and math.isfinite(stall_ratio)):
            raise ValueError(
                f"Non-finite interlock ratio computed (cavitation={cavitation_ratio!r}, stall={stall_ratio!r})"
            )

        is_cavitation = cavitation_ratio > prior["cavitation_ratio_limit"]
        is_stall = stall_ratio > prior["stall_ratio_limit"]
        halt = bool(is_cavitation or is_stall)

        if is_cavitation and is_stall:
            reason = "dry_aspiration_cavitation_and_stepper_stall"
        elif is_cavitation:
            reason = "dry_aspiration_cavitation"
        elif is_stall:
            reason = "stepper_stall_harmonic"
        else:
            reason = "nominal"

        score = max(
            cavitation_ratio / prior["cavitation_ratio_limit"],
            stall_ratio / prior["stall_ratio_limit"],
        )
        if not math.isfinite(score):
            raise ValueError(f"Non-finite interlock score computed: {score!r}")
        return InterlockDecision(
            step_index=command.step_index,
            halt_run=halt,
            reason=reason,
            raw_audio_saved=feats.raw_audio_saved,
            speech_band_dropped=True,
            cavitation_ratio=float(cavitation_ratio),
            stall_ratio=float(stall_ratio),
            score=float(score),
        )


def _synthesize_speech_distractor(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Legacy stand-in: multi-tone signal strictly inside [380, 3150] Hz (zero out-of-band energy)."""
    signal = np.zeros_like(t)
    n_syllables = int(rng.integers(2, 5))
    duration = float(t[-1] - t[0])
    formants = [
        float(rng.uniform(380.0, 680.0)),
        float(rng.uniform(950.0, 1650.0)),
        float(rng.uniform(1950.0, 2950.0)),
        float(rng.uniform(2600.0, 3150.0)),
    ]
    for _ in range(n_syllables):
        center = float(rng.uniform(0.06, duration - 0.06))
        width = float(rng.uniform(0.035, 0.085))
        env = np.exp(-0.5 * ((t - center) / width) ** 2)
        for idx, f_hz in enumerate(formants):
            amp = float(rng.uniform(0.04, 0.11)) / (1.0 + 0.35 * idx)
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            fm = 4.0 * np.sin(2.0 * np.pi * 5.5 * t + phase)
            signal += env * amp * np.sin(2.0 * np.pi * f_hz * t + fm + phase)
    return signal


def synthesize_broadband_speech_distractor(
    t: np.ndarray,
    rng: np.random.Generator,
    *,
    sample_rate_hz: int = 16000,
) -> np.ndarray:
    """Synthesize realistic voiced pulse-train (`F0 100-220 Hz`, `1/h` harmonics to `7.8 kHz`)
    plus `4-7.8 kHz` sibilant/fricative noise bursts that bleed into both watched bands."""
    signal = np.zeros_like(t, dtype=np.float64)
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.5
    n_syllables = int(rng.integers(2, 5))
    freqs = np.fft.rfftfreq(t.size, d=1.0 / float(sample_rate_hz))
    fric_hi_hz = min(7800.0, 0.49 * float(sample_rate_hz))
    fricative_mask = (freqs >= 4000.0) & (freqs <= fric_hi_hz)
    max_harmonic_hz = min(7800.0, 0.49 * float(sample_rate_hz))

    for _ in range(n_syllables):
        center = float(rng.uniform(0.06, max(0.07, duration - 0.06)))
        width = float(rng.uniform(0.035, 0.085))
        env = np.exp(-0.5 * ((t - center) / width) ** 2)
        f0 = float(rng.uniform(100.0, 220.0))
        max_h = max(1, int(max_harmonic_hz // f0))
        pulse = np.zeros_like(t, dtype=np.float64)
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        for h in range(1, max_h + 1):
            pulse += (1.0 / float(h)) * np.sin(2.0 * np.pi * (float(h) * f0) * t + float(h) * phase0)
        pulse_rms = float(np.sqrt(np.mean(pulse ** 2))) + 1e-12
        pulse /= pulse_rms

        f_center = float(
            np.clip(center + float(rng.choice([-1.0, 1.0])) * 0.045, 0.03, max(0.04, duration - 0.03))
        )
        f_env = np.exp(-0.5 * ((t - f_center) / 0.022) ** 2)
        raw_noise = rng.normal(0.0, 1.0, size=t.size)
        fricative = np.fft.irfft(np.fft.rfft(raw_noise) * fricative_mask, n=t.size)
        fric_rms = float(np.sqrt(np.mean(fricative ** 2))) + 1e-12
        fricative /= fric_rms

        signal += env * 0.75 * pulse + f_env * 0.65 * fricative
    return signal


def _load_real_speech_segments(
    speech_dir: Path | str,
    *,
    target_sr: int = 16000,
) -> list[np.ndarray]:
    """Load mono float64 speech waveforms (>= `target_sr`) from `speech_dir` recursively."""
    root = Path(speech_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"speech_dir does not exist or is not a directory: {root}")
    wav_paths = sorted(root.rglob("*.wav"))
    if not wav_paths:
        raise ValueError(f"No .wav files found under speech_dir: {root}")

    segments: list[np.ndarray] = []
    for wp in wav_paths:
        sr, raw = wavfile.read(wp)
        src_sr = int(sr)
        if src_sr < target_sr or raw.size == 0:
            continue
        if raw.dtype == np.int16:
            x = raw.astype(np.float64) / 32768.0
        elif raw.dtype == np.int32:
            x = raw.astype(np.float64) / 2147483648.0
        elif raw.dtype == np.uint8:
            x = (raw.astype(np.float64) - 128.0) / 128.0
        else:
            x = raw.astype(np.float64)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if not np.isfinite(x).all():
            continue
        if src_sr != target_sr:
            g = math.gcd(target_sr, src_sr)
            x = resample_poly(x, target_sr // g, src_sr // g).astype(np.float64)
        if float(np.sqrt(np.mean(x ** 2))) > 1e-6:
            segments.append(np.ascontiguousarray(x, dtype=np.float64))

    if not segments:
        raise ValueError(f"No non-silent >= {target_sr} Hz .wav files found under speech_dir: {root}")
    return segments


def _sample_speech_segment(
    segments: Sequence[np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seg = segments[int(rng.integers(0, len(segments)))]
    if seg.size >= n_samples:
        max_start = seg.size - n_samples
        start = int(rng.integers(0, max_start + 1))
        chunk = seg[start : start + n_samples]
        if float(np.sqrt(np.mean(chunk ** 2))) > 1e-6:
            return chunk.copy()
        return seg[:n_samples].copy()
    reps = int(math.ceil(float(n_samples) / float(seg.size)))
    return np.tile(seg, reps)[:n_samples].copy()


def _mix_at_db(clean_wav: np.ndarray, speech_wav: np.ndarray, level_db: float, *, ref_rms: float) -> np.ndarray:
    sp_rms = float(np.sqrt(np.mean(speech_wav ** 2))) + 1e-12
    target_rms = float(ref_rms) * (10.0 ** (float(level_db) / 20.0))
    return clean_wav + speech_wav * (target_rms / sp_rms)


def _synthesize_pipetting_clip(
    rng: np.random.Generator,
    step_index: int,
    fault_mode: str,  # "nominal", "dry_aspiration", "stepper_stall", "idle_clink"
    with_speech: bool,
    rheology: str,
    sample_rate_hz: int = 16000,
    duration_s: float = 0.50,
    *,
    speech_mode: str = "inband_tones",  # "inband_tones" | "broadband_pulse_fricative"
    speech_snr_db: float | None = None,
) -> tuple[np.ndarray, PipettingCommandWindow]:
    """Generate a synthetic 16 kHz benchtop pipetting waveform + synchronized step window."""
    n_samples = int(round(sample_rate_hz * duration_s))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    prior = RHEOLOGY_PRIORS[rheology]

    action = "idle_transit" if fault_mode == "idle_clink" else str(rng.choice(["aspirate", "dispense"]))
    start_s = 0.10
    end_s = 0.38
    volume_ul = float(rng.choice([20.0, 50.0, 100.0, 200.0]))
    cmd = PipettingCommandWindow(
        step_index=step_index,
        action=action,
        start_s=start_s,
        end_s=end_s,
        rheology=rheology,
        volume_ul=volume_ul,
    )

    # Subtle broadband benchtop floor
    wav = rng.normal(0.0, 0.0035, size=n_samples)

    in_step = ((t >= start_s) & (t <= end_s)).astype(np.float64)
    if action != "idle_transit":
        lo_scale = np.sqrt(prior["expected_lo_power"])
        hi_scale = np.sqrt(prior["expected_hi_power"])
        # Normal microstepper drive fundamental + 2nd harmonic in [110, 220] Hz
        f_step = float(rng.uniform(115.0, 130.0))
        stepper_nom = lo_scale * (
            0.012 * np.sin(2.0 * np.pi * f_step * t + float(rng.uniform(0, 2 * np.pi)))
            + 0.006 * np.sin(2.0 * np.pi * (2.0 * f_step) * t + float(rng.uniform(0, 2 * np.pi)))
        )
        # Normal laminar orifice flow hiss in [4.0, 7.2] kHz
        raw_hiss = rng.normal(0.0, 0.0045 * hi_scale, size=n_samples)
        hiss_spec = np.fft.rfft(raw_hiss)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
        hiss_mask = (freqs >= 3900.0) & (freqs <= 7400.0)
        laminar_hiss = np.fft.irfft(hiss_spec * hiss_mask, n=n_samples)
        wav += in_step * (stepper_nom + laminar_hiss)

    if fault_mode == "dry_aspiration":
        # Meniscus air-break + cavitation bubble collapse burst in [4.2, 7.6] kHz during active window
        burst_env = ((t >= (start_s + 0.06)) & (t <= (end_s - 0.02))).astype(np.float64)
        cav_noise = rng.normal(0.0, 0.022, size=n_samples)
        cav_spec = np.fft.rfft(cav_noise)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
        cav_mask = (freqs >= 4200.0) & (freqs <= 7600.0)
        cav_burst = np.fft.irfft(cav_spec * cav_mask, n=n_samples)
        whistle = 0.013 * np.sin(2.0 * np.pi * float(rng.uniform(5100.0, 6800.0)) * t)
        wav += burst_env * (cav_burst + whistle)

    elif fault_mode == "stepper_stall":
        # Plunger bind / viscous cogging harmonics in [80, 240] Hz
        stall_env = ((t >= (start_s + 0.04)) & (t <= end_s)).astype(np.float64)
        f_cog = float(rng.uniform(82.0, 108.0))
        stall_wave = (
            0.030 * np.sin(2.0 * np.pi * f_cog * t)
            + 0.021 * np.sin(2.0 * np.pi * (2.0 * f_cog) * t + 0.7)
        )
        wav += stall_env * stall_wave

    elif fault_mode == "idle_clink":
        # Tube-rack clink while arm transits between steps (no active aspirate/dispense)
        clink_env = np.exp(-0.5 * ((t - 0.24) / 0.015) ** 2)
        wav += clink_env * 0.035 * np.sin(2.0 * np.pi * 5400.0 * t)

    if with_speech:
        if speech_mode == "broadband_pulse_fricative":
            sp = synthesize_broadband_speech_distractor(t, rng, sample_rate_hz=sample_rate_hz)
            clip_rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-12
            lvl = 0.0 if speech_snr_db is None else float(speech_snr_db)
            wav = _mix_at_db(wav, sp, lvl, ref_rms=clip_rms)
        else:
            sp = _synthesize_speech_distractor(t, rng)
            if speech_snr_db is not None:
                clip_rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-12
                wav = _mix_at_db(wav, sp, float(speech_snr_db), ref_rms=clip_rms)
            else:
                wav = wav + sp

    return wav, cmd


def _f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _evaluate_level_sweep(
    clean_clips: Sequence[tuple[np.ndarray, PipettingCommandWindow, str, bool]],
    speech_signals: Sequence[np.ndarray],
    det_bandstop: CrossModalInterlockDetector,
    det_harmonic: CrossModalInterlockDetector,
    speech_levels_db: Sequence[float],
    *,
    reference_mode: str,  # "clip_rms" | "conversation_standin_rms"
    standin_rms: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lvl_db in speech_levels_db:
        bs_tp = bs_fp = bs_fn = bs_speech_fp = bs_active_speech_fp = 0
        ha_tp = ha_fp = ha_fn = ha_speech_fp = ha_active_speech_fp = 0
        n_ff_speech = 0
        n_ff_active_speech = 0
        n_ff_total = 0

        for (clean_wav, cmd, fault_mode, with_speech), sp_wav in zip(clean_clips, speech_signals):
            is_fault = fault_mode in ("dry_aspiration", "stepper_stall")
            if with_speech:
                ref_rms = (
                    float(np.sqrt(np.mean(clean_wav ** 2))) + 1e-12
                    if reference_mode == "clip_rms"
                    else float(standin_rms)
                )
                mixed = _mix_at_db(clean_wav, sp_wav, float(lvl_db), ref_rms=ref_rms)
            else:
                mixed = clean_wav

            d_bs = det_bandstop.evaluate(mixed, cmd)
            d_ha = det_harmonic.evaluate(mixed, cmd)

            if is_fault:
                if d_bs.halt_run:
                    bs_tp += 1
                else:
                    bs_fn += 1
                if d_ha.halt_run:
                    ha_tp += 1
                else:
                    ha_fn += 1
            else:
                n_ff_total += 1
                if with_speech:
                    n_ff_speech += 1
                    if cmd.is_active_transfer:
                        n_ff_active_speech += 1
                if d_bs.halt_run:
                    bs_fp += 1
                    if with_speech:
                        bs_speech_fp += 1
                        if cmd.is_active_transfer:
                            bs_active_speech_fp += 1
                if d_ha.halt_run:
                    ha_fp += 1
                    if with_speech:
                        ha_speech_fp += 1
                        if cmd.is_active_transfer:
                            ha_active_speech_fp += 1

        bs_p, bs_r, bs_f1 = _f1_from_counts(bs_tp, bs_fp, bs_fn)
        ha_p, ha_r, ha_f1 = _f1_from_counts(ha_tp, ha_fp, ha_fn)
        rows.append(
            {
                "speech_level_db": float(lvl_db),
                "reference": reference_mode,
                "n_fault_free_with_speech": n_ff_speech,
                "n_active_fault_free_with_speech": n_ff_active_speech,
                "bandstop_only": {
                    "false_halts_under_speech": bs_speech_fp,
                    "false_halt_rate_under_speech": round(bs_speech_fp / max(1, n_ff_speech), 4),
                    "false_halt_rate_active_speech": round(bs_active_speech_fp / max(1, n_ff_active_speech), 4),
                    "false_alarm_rate_overall": round(bs_fp / max(1, n_ff_total), 4),
                    "fault_recall": round(bs_r, 4),
                    "precision": round(bs_p, 4),
                    "overall_fault_f1": round(bs_f1, 4),
                },
                "harmonic_onset_aware": {
                    "false_halts_under_speech": ha_speech_fp,
                    "false_halt_rate_under_speech": round(ha_speech_fp / max(1, n_ff_speech), 4),
                    "false_halt_rate_active_speech": round(ha_active_speech_fp / max(1, n_ff_active_speech), 4),
                    "false_alarm_rate_overall": round(ha_fp / max(1, n_ff_total), 4),
                    "fault_recall": round(ha_r, 4),
                    "precision": round(ha_p, 4),
                    "overall_fault_f1": round(ha_f1, 4),
                },
            }
        )
    return rows


def run_benchmark(
    seed: int = 20261123,
    *,
    speech_levels_db: Sequence[float] = DEFAULT_SPEECH_SNR_LEVELS_DB,
    speech_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run 120-clip 16 kHz pipetting benchmark under both legacy in-band tones and
    realistic broadband speech (`F0 100-220 Hz` `1/h` harmonics + `4-8 kHz` fricatives,
    plus optional local 16 kHz real-speech WAV directory `speech_dir`).

    Compares:
      1. `speech_unfiltered_rms` vs `cross_modal_voice_stripped_interlock` on the legacy
         in-band (`380-3150 Hz`) tone distractor (retaining historical `1.0` F1 comparison
         with explicit `inband_tones_only` scope note), and
      2. `bandstop_only` vs `harmonic_onset_aware` across `speech_levels_db` (`0, -10, -20, -30 dB`
         relative to clip RMS AND relative to the legacy conversation stand-in RMS) on
         broadband glottal pulse-train + sibilant speech and optional `speech_dir` WAVs.
    """
    rng = np.random.default_rng(seed)
    sample_rate_hz = 16000
    rheologies = ("aqueous", "viscous_glycerol", "volatile_solvent")

    # Separate calibration split (24 quiet normal clips across rheologies)
    calib_pairs: list[tuple[np.ndarray, PipettingCommandWindow]] = []
    calib_quiet_wavs: list[np.ndarray] = []
    for idx in range(24):
        rheo = rheologies[idx % len(rheologies)]
        wav, cmd = _synthesize_pipetting_clip(
            rng=rng,
            step_index=idx,
            fault_mode="nominal",
            with_speech=False,
            rheology=rheo,
            sample_rate_hz=sample_rate_hz,
        )
        calib_pairs.append((wav, cmd))
        calib_quiet_wavs.append(wav)

    baseline = SpeechUnfilteredRMSDetector(sigma_multiplier=3.0)
    baseline.calibrate(calib_quiet_wavs)

    interlock_bandstop = CrossModalInterlockDetector(
        sample_rate_hz=sample_rate_hz,
        speech_rejection_mode="bandstop_only",
    )
    interlock_bandstop.calibrate(calib_pairs)

    interlock_harmonic = CrossModalInterlockDetector(
        sample_rate_hz=sample_rate_hz,
        speech_rejection_mode="harmonic_onset_aware",
    )
    interlock_harmonic.calibrate(calib_pairs)

    # Construct 120 evaluation clip specifications
    eval_specs: list[tuple[str, bool, str]] = []
    for i in range(20):
        eval_specs.append(("nominal", False, rheologies[i % 3]))
    for i in range(44):
        eval_specs.append(("nominal", True, rheologies[i % 3]))
    for i in range(8):
        eval_specs.append(("idle_clink", True, rheologies[i % 3]))
    for i in range(32):
        eval_specs.append(("dry_aspiration", (i % 4) != 0, rheologies[i % 3]))
    for i in range(16):
        eval_specs.append(("stepper_stall", (i % 4) != 0, rheologies[i % 3]))
    assert len(eval_specs) == 120

    base_tp = base_fp = base_fn = 0
    base_dry_tp = base_dry_fn = 0
    base_speech_fp = 0

    cm_tp = cm_fp = cm_fn = 0
    cm_dry_tp = cm_dry_fn = 0
    cm_speech_fp = 0
    n_fault_free_with_speech = 0
    n_fault_free_total = 0

    clean_clips: list[tuple[np.ndarray, PipettingCommandWindow, str, bool]] = []
    standin_rms_samples: list[float] = []

    for step_idx, (fault_mode, with_speech, rheo) in enumerate(eval_specs):
        clean_wav, cmd = _synthesize_pipetting_clip(
            rng=rng,
            step_index=100 + step_idx,
            fault_mode=fault_mode,
            with_speech=False,
            rheology=rheo,
            sample_rate_hz=sample_rate_hz,
        )
        clean_clips.append((clean_wav, cmd, fault_mode, with_speech))

        t = np.arange(clean_wav.size, dtype=np.float64) / sample_rate_hz
        if with_speech:
            inband_sp = _synthesize_speech_distractor(t, rng)
            standin_rms_samples.append(float(np.sqrt(np.mean(inband_sp ** 2))))
            wav = clean_wav + inband_sp
        else:
            wav = clean_wav

        is_fault = fault_mode in ("dry_aspiration", "stepper_stall")
        is_dry = fault_mode == "dry_aspiration"

        b_dec = baseline.evaluate(wav, cmd)
        c_dec = interlock_bandstop.evaluate(wav, cmd)
        assert c_dec.raw_audio_saved is False
        assert c_dec.speech_band_dropped is True

        if is_fault:
            if b_dec.halt_run:
                base_tp += 1
            else:
                base_fn += 1
            if c_dec.halt_run:
                cm_tp += 1
            else:
                cm_fn += 1

            if is_dry:
                if b_dec.halt_run:
                    base_dry_tp += 1
                else:
                    base_dry_fn += 1
                if c_dec.halt_run:
                    cm_dry_tp += 1
                else:
                    cm_dry_fn += 1
        else:
            n_fault_free_total += 1
            if with_speech:
                n_fault_free_with_speech += 1
            if b_dec.halt_run:
                base_fp += 1
                if with_speech:
                    base_speech_fp += 1
            if c_dec.halt_run:
                cm_fp += 1
                if with_speech:
                    cm_speech_fp += 1

    base_prec, base_rec, base_f1 = _f1_from_counts(base_tp, base_fp, base_fn)
    _, _, base_dry_f1 = _f1_from_counts(base_dry_tp, base_fp, base_dry_fn)
    cm_prec, cm_rec, cm_f1 = _f1_from_counts(cm_tp, cm_fp, cm_fn)
    _, _, cm_dry_f1 = _f1_from_counts(cm_dry_tp, cm_fp, cm_dry_fn)

    standin_rms_mean = float(np.mean(standin_rms_samples)) if standin_rms_samples else 0.062

    # Realistic broadband speech (glottal pulse train F0 100-220 Hz + 4-7.8 kHz fricatives)
    rng_broad = np.random.default_rng(seed + 37)
    broadband_signals: list[np.ndarray] = []
    for clean_wav, _, _, _ in clean_clips:
        t = np.arange(clean_wav.size, dtype=np.float64) / sample_rate_hz
        broadband_signals.append(
            synthesize_broadband_speech_distractor(t, rng_broad, sample_rate_hz=sample_rate_hz)
        )

    levels_vs_clip = _evaluate_level_sweep(
        clean_clips,
        broadband_signals,
        interlock_bandstop,
        interlock_harmonic,
        speech_levels_db,
        reference_mode="clip_rms",
        standin_rms=standin_rms_mean,
    )
    levels_vs_conv = _evaluate_level_sweep(
        clean_clips,
        broadband_signals,
        interlock_bandstop,
        interlock_harmonic,
        speech_levels_db,
        reference_mode="conversation_standin_rms",
        standin_rms=standin_rms_mean,
    )

    real_speech_eval: dict[str, Any] | None = None
    if speech_dir is not None:
        real_segments = _load_real_speech_segments(speech_dir, target_sr=sample_rate_hz)
        rng_real = np.random.default_rng(seed + 73)
        real_signals = [
            _sample_speech_segment(real_segments, clean_wav.size, rng_real)
            for clean_wav, _, _, _ in clean_clips
        ]
        real_speech_eval = {
            "speech_dir": str(speech_dir),
            "n_wav_files_loaded": len(real_segments),
            "levels_vs_clip_rms_db": _evaluate_level_sweep(
                clean_clips,
                real_signals,
                interlock_bandstop,
                interlock_harmonic,
                speech_levels_db,
                reference_mode="clip_rms",
                standin_rms=standin_rms_mean,
            ),
            "levels_vs_conversation_rms_db": _evaluate_level_sweep(
                clean_clips,
                real_signals,
                interlock_bandstop,
                interlock_harmonic,
                speech_levels_db,
                reference_mode="conversation_standin_rms",
                standin_rms=standin_rms_mean,
            ),
        }

    return {
        "idea_id": "idea_17_cross_modal_cavitation_interlock",
        "n_clips": len(eval_specs),
        "sample_rate_hz": sample_rate_hz,
        "raw_audio_saved": False,
        "speech_band_stripped_hz": [int(SPEECH_BAND_LO_HZ), int(SPEECH_BAND_HI_HZ)],
        "retained_bands_hz": [list(LOW_STALL_BAND_HZ), list(HIGH_CAVITATION_BAND_HZ)],
        "scope_note": (
            "The 1.0 fault F1 under cross_modal_voice_stripped_interlock holds ONLY for the "
            "legacy synthetic pure-tone stand-in strictly inside [380, 3150] Hz. See "
            "broadband_speech_evaluation (pulse-train F0 100-220 Hz + 4-7.8 kHz fricatives) "
            "and real_speech_evaluation for realistic out-of-band speech false-halt rates."
        ),
        "speech_unfiltered_rms": {
            "speech_signal": "inband_tones_380_3150hz_only",
            "dry_aspiration_f1": round(base_dry_f1, 4),
            "overall_fault_f1": round(base_f1, 4),
            "precision": round(base_prec, 4),
            "recall": round(base_rec, 4),
            "false_alarm_rate_under_speech": round(base_speech_fp / n_fault_free_with_speech, 4),
            "false_alarm_rate_overall": round(base_fp / n_fault_free_total, 4),
        },
        "cross_modal_voice_stripped_interlock": {
            "speech_signal": "inband_tones_380_3150hz_only",
            "dry_aspiration_f1": round(cm_dry_f1, 4),
            "overall_fault_f1": round(cm_f1, 4),
            "precision": round(cm_prec, 4),
            "recall": round(cm_rec, 4),
            "false_alarm_rate_under_speech": round(cm_speech_fp / n_fault_free_with_speech, 4),
            "false_alarm_rate_overall": round(cm_fp / n_fault_free_total, 4),
            "dry_aspiration_halt_recall": round(cm_dry_tp / 32.0, 4),
        },
        "broadband_speech_evaluation": {
            "speech_signal": "glottal_pulse_100_220hz_1_over_h_plus_4_7.8khz_fricatives",
            "levels_vs_clip_rms_db": levels_vs_clip,
            "levels_vs_conversation_rms_db": levels_vs_conv,
        },
        "real_speech_evaluation": real_speech_eval,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2))
