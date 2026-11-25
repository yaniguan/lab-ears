"""Log-mel features with numpy and scipy only."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 64
SPEECH_BAND_HZ = (300.0, 3400.0)
STFT_SKIRT_HZ = 2.0 * SR / N_FFT  # Hann main-lobe half-width (2 * df = 31.25 Hz at 16 kHz / 1024)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def outward_round_band_hz(lo_hz: float, hi_hz: float, *, skirt_hz: float = 0.0) -> Tuple[int, int]:
    """Conservatively round continuous [lo_hz, hi_hz] outward to integer Hz (floor lo, ceil hi).

    Matches `LabEarsLean.outwardRoundBand`: subtracting `skirt_hz` on the lower bound and
    adding `skirt_hz` on the upper bound guarantees integer Hz certificates never
    under-approximate continuous HTK mel filter edges near `(300, 3400) Hz`.
    """
    lo_f, hi_f, sk_f = float(lo_hz), float(hi_hz), float(skirt_hz)
    if not (math.isfinite(lo_f) and math.isfinite(hi_f) and math.isfinite(sk_f)):
        raise ValueError("Frequency band endpoints and skirt_hz must be finite")
    if lo_f < 0.0 or lo_f > hi_f or sk_f < 0.0:
        raise ValueError(f"Invalid frequency band [{lo_hz}, {hi_hz}] or skirt_hz={skirt_hz}")
    return (max(0, int(math.floor(lo_f - sk_f))), int(math.ceil(hi_f + sk_f)))


def mel_filter_edges_hz(sr=SR, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Return (n_mels, 2) array of [lo_hz, hi_hz] support edges and (n_mels,) centers_hz."""
    fmin_f = float(fmin)
    fmax_f = sr / 2.0 if fmax is None else float(fmax)
    mels = np.linspace(hz_to_mel(fmin_f), hz_to_mel(fmax_f), n_mels + 2)
    hz = mel_to_hz(mels)
    hz[0] = fmin_f
    hz[-1] = fmax_f
    edges = np.column_stack([hz[:-2], hz[2:]])
    return edges, hz[1:-1]


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    fmax = sr / 2.0 if fmax is None else float(fmax)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(c - lo, 1e-12)
        down = (hi - bins) / max(hi - c, 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb, hz[1:-1]


_FB, MEL_CENTERS_HZ = mel_filterbank()
MEL_EDGES_HZ, _ = mel_filter_edges_hz()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def speech_band_mask(
    centers_hz=None,
    band=SPEECH_BAND_HZ,
    *,
    edges_hz=None,
    skirt_hz: float = STFT_SKIRT_HZ,
):
    """True for mel filters whose Hann-skirted support [lo_hz - skirt_hz, hi_hz + skirt_hz] lies outside `band`."""
    lo_cut, hi_cut = float(band[0]), float(band[1])
    sk = float(skirt_hz)
    if not math.isfinite(sk) or sk < 0.0:
        raise ValueError(f"skirt_hz must be finite and >= 0, got {skirt_hz!r}")

    if edges_hz is not None:
        edges = np.asarray(edges_hz, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges_hz must have shape (n_mels, 2), got {edges.shape}")
        return (edges[:, 1] + sk <= lo_cut) | (edges[:, 0] - sk >= hi_cut)

    if centers_hz is None or (
        isinstance(centers_hz, np.ndarray)
        and centers_hz.shape == MEL_CENTERS_HZ.shape
        and np.allclose(centers_hz, MEL_CENTERS_HZ)
    ):
        return (MEL_EDGES_HZ[:, 1] + sk <= lo_cut) | (MEL_EDGES_HZ[:, 0] - sk >= hi_cut)

    c = np.asarray(centers_hz, dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("centers_hz must be a non-empty 1D array")
    if c.size == 1:
        return (c + sk <= lo_cut) | (c - sk >= hi_cut)
    c_mel = hz_to_mel(c)
    step = np.empty(c.size + 2, dtype=np.float64)
    step[1:-1] = c_mel
    step[0] = max(0.0, c_mel[0] - (c_mel[1] - c_mel[0]))
    step[-1] = c_mel[-1] + (c_mel[-1] - c_mel[-2])
    hz = mel_to_hz(step)
    return (hz[2:] + sk <= lo_cut) | (hz[:-2] - sk >= hi_cut)


def retained_mel_bands_hz(
    *,
    drop_speech_band: bool = True,
    sr: int = SR,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    skirt_hz: Optional[float] = None,
) -> List[List[int]]:
    """Compute outward-rounded integer Hz support intervals for retained mel filters."""
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr!r}")
    edges, _ = mel_filter_edges_hz(sr=sr, n_mels=n_mels)
    if not drop_speech_band:
        lo_i, hi_i = outward_round_band_hz(float(edges[0, 0]), float(edges[-1, 1]), skirt_hz=0.0)
        return [[lo_i, hi_i]]

    eff_skirt = (2.0 * float(sr) / float(n_fft)) if skirt_hz is None else float(skirt_hz)
    keep = speech_band_mask(edges_hz=edges, skirt_hz=eff_skirt)
    bands: List[List[int]] = []
    idx = 0
    while idx < len(keep):
        if not keep[idx]:
            idx += 1
            continue
        end_idx = idx
        while end_idx + 1 < len(keep) and keep[end_idx + 1]:
            end_idx += 1
        lo_i, hi_i = outward_round_band_hz(float(edges[idx, 0]), float(edges[end_idx, 1]), skirt_hz=0.0)
        bands.append([lo_i, hi_i])
        idx = end_idx + 1
    return bands


def _pcm_to_float32(raw: np.ndarray) -> np.ndarray:
    dt = raw.dtype
    if dt == np.uint8:
        return ((raw.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if dt == np.int16:
        return (raw.astype(np.float32) / 32768.0).astype(np.float32)
    if dt == np.int32:
        return (raw.astype(np.float64) / 2147483648.0).astype(np.float32)
    if np.issubdtype(dt, np.signedinteger):
        scale = float(-np.iinfo(dt).min)
        return (raw.astype(np.float64) / scale).astype(np.float32)
    if np.issubdtype(dt, np.unsignedinteger):
        mid = float(np.iinfo(dt).max + 1) / 2.0
        return ((raw.astype(np.float64) - mid) / mid).astype(np.float32)
    if np.issubdtype(dt, np.floating):
        return raw.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported WAV sample dtype: {dt}")


def load_wav_info(
    path: Path | str,
    *,
    target_sr: int = SR,
    resample: bool = False,
    min_source_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load WAV file into mono float32 and return `(waveform, source_sample_rate_hz)`."""
    sr, raw = wavfile.read(path)
    src_sr = int(sr)
    if raw.size == 0:
        raise ValueError(f"{path}: WAV file contains zero samples")
    if min_source_sr is not None and src_sr < int(min_source_sr):
        raise ValueError(
            f"{path}: source sample rate {src_sr} Hz is below minimum required {min_source_sr} Hz "
            f"(upsampling cannot recover unrecorded high-band acoustic energy)"
        )
    x = _pcm_to_float32(raw)
    if x.ndim > 1:
        x = x.mean(axis=1, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: WAV file contains non-finite samples")
    if src_sr != target_sr:
        if not resample:
            raise ValueError(f"{path}: expected {target_sr} Hz, got {src_sr}")
        g = math.gcd(int(target_sr), src_sr)
        x = resample_poly(x, int(target_sr) // g, src_sr // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32), src_sr


def load_wav(
    path: Path | str,
    *,
    target_sr: int = SR,
    resample: bool = False,
    min_source_sr: Optional[int] = None,
) -> np.ndarray:
    """Load WAV file into mono float32 in [-1.0, 1.0] scaled deterministically by PCM dtype."""
    x, _ = load_wav_info(path, target_sr=target_sr, resample=resample, min_source_sr=min_source_sr)
    return x


def log_mel(x: np.ndarray, drop_speech_band: bool = False, *, pad_end: bool = True) -> np.ndarray:
    """Return log-mel frames of shape (n_frames, n_mels or fewer) without index-matrix copies."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D mono waveform, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("Cannot compute log_mel on empty waveform")
    if not np.isfinite(x).all():
        raise ValueError("Waveform contains NaN or Inf")

    if drop_speech_band:
        # Pre-STFT full-clip spectral notch removing [300, 3400] Hz before short-window framing
        clip_spec = np.fft.rfft(x.astype(np.float64, copy=False))
        clip_freqs = np.fft.rfftfreq(x.size, d=1.0 / SR)
        clip_spec[(clip_freqs >= SPEECH_BAND_HZ[0]) & (clip_freqs <= SPEECH_BAND_HZ[1])] = 0.0
        x = np.fft.irfft(clip_spec, n=x.size).astype(np.float32)

    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    elif pad_end:
        rem = (len(x) - N_FFT) % HOP
        if rem != 0:
            x = np.pad(x, (0, HOP - rem))

    frames = sliding_window_view(x, N_FFT)[::HOP]
    spec = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    mel = 10.0 * np.log10(spec @ _FB.T + 1e-10)
    if drop_speech_band:
        mel = mel[:, speech_band_mask()]
    return mel.astype(np.float32, copy=False)


def mel_windows(mel: np.ndarray, window_frames: int = 64, hop_frames: int = 32) -> List[np.ndarray]:
    """Slice causal temporal sub-windows from a 2D mel spectrogram without dropping tail frames."""
    if window_frames < 1 or hop_frames < 1:
        raise ValueError(f"window_frames ({window_frames}) and hop_frames ({hop_frames}) must be >= 1")
    arr = np.asarray(mel, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"Expected non-empty 2D mel array, got shape {arr.shape}")
    n_frames = int(arr.shape[0])
    if n_frames <= window_frames:
        return [arr]
    starts = list(range(0, n_frames - window_frames + 1, hop_frames))
    if starts[-1] != n_frames - window_frames:
        starts.append(n_frames - window_frames)
    return [arr[s : s + window_frames] for s in starts]


def clip_vector(mel: np.ndarray, *, pooling: str = "mean") -> np.ndarray:
    """Clip-level vector: temporal summary (mean or p95) and std of each mel bin over time."""
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] < 1 or mel.shape[1] < 1:
        raise ValueError(f"Expected non-empty 2D mel array, got shape {mel.shape}")
    if pooling == "mean":
        return np.concatenate([mel.mean(axis=0), mel.std(axis=0)])
    if pooling == "p95":
        return np.concatenate([np.percentile(mel, 95, axis=0).astype(np.float32), mel.std(axis=0)])
    raise ValueError(f"Unsupported clip_vector pooling={pooling!r}; expected 'mean' or 'p95'")


def stack_frames(mel: np.ndarray, context: int = 5) -> np.ndarray:
    """Stack `context` consecutive frames, edge-padding clips shorter than `context` frames."""
    if context < 1:
        raise ValueError(f"context must be >= 1, got {context}")
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] < 1 or mel.shape[1] < 1:
        raise ValueError(f"Expected non-empty 2D mel array, got shape {mel.shape}")
    if mel.shape[0] < context:
        mel = np.pad(mel, ((0, context - mel.shape[0]), (0, 0)), mode="edge")
    n = mel.shape[0] - context + 1
    return np.stack([mel[i : i + n] for i in range(context)], axis=1).reshape(n, -1)
