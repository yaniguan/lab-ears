"""Log-mel features with numpy and scipy only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 64
SPEECH_BAND_HZ = (300.0, 3400.0)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filter_edges_hz(sr=SR, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Return (n_mels, 2) array of [lo_hz, hi_hz] support edges and (n_mels,) centers_hz."""
    fmax = sr / 2.0 if fmax is None else float(fmax)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
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


def speech_band_mask(centers_hz=None, band=SPEECH_BAND_HZ, *, edges_hz=None):
    """True for mel filters whose full non-zero support [lo_hz, hi_hz] lies outside `band`."""
    lo_cut, hi_cut = float(band[0]), float(band[1])
    if edges_hz is not None:
        edges = np.asarray(edges_hz, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges_hz must have shape (n_mels, 2), got {edges.shape}")
        return (edges[:, 1] <= lo_cut) | (edges[:, 0] >= hi_cut)

    if centers_hz is None or (
        isinstance(centers_hz, np.ndarray)
        and centers_hz.shape == MEL_CENTERS_HZ.shape
        and np.allclose(centers_hz, MEL_CENTERS_HZ)
    ):
        return (MEL_EDGES_HZ[:, 1] <= lo_cut) | (MEL_EDGES_HZ[:, 0] >= hi_cut)

    c = np.asarray(centers_hz, dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("centers_hz must be a non-empty 1D array")
    if c.size == 1:
        return (c <= lo_cut) | (c >= hi_cut)
    c_mel = hz_to_mel(c)
    step = np.empty(c.size + 2, dtype=np.float64)
    step[1:-1] = c_mel
    step[0] = max(0.0, c_mel[0] - (c_mel[1] - c_mel[0]))
    step[-1] = c_mel[-1] + (c_mel[-1] - c_mel[-2])
    hz = mel_to_hz(step)
    return (hz[2:] <= lo_cut) | (hz[:-2] >= hi_cut)


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


def load_wav(path: Path | str, *, target_sr: int = SR, resample: bool = False) -> np.ndarray:
    """Load WAV file into mono float32 in [-1.0, 1.0] scaled deterministically by PCM dtype."""
    sr, raw = wavfile.read(path)
    if raw.size == 0:
        raise ValueError(f"{path}: WAV file contains zero samples")
    x = _pcm_to_float32(raw)
    if x.ndim > 1:
        x = x.mean(axis=1, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: WAV file contains non-finite samples")
    if sr != target_sr:
        if not resample:
            raise ValueError(f"{path}: expected {target_sr} Hz, got {sr}")
        g = math.gcd(int(target_sr), int(sr))
        x = resample_poly(x, int(target_sr) // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def log_mel(x: np.ndarray, drop_speech_band: bool = False, *, pad_end: bool = True) -> np.ndarray:
    """Return log-mel frames of shape (n_frames, n_mels or fewer) without index-matrix copies."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D mono waveform, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("Cannot compute log_mel on empty waveform")
    if not np.isfinite(x).all():
        raise ValueError("Waveform contains NaN or Inf")

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


def clip_vector(mel: np.ndarray) -> np.ndarray:
    """Clip-level vector: mean and std of each mel bin over time."""
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] < 1 or mel.shape[1] < 1:
        raise ValueError(f"Expected non-empty 2D mel array, got shape {mel.shape}")
    return np.concatenate([mel.mean(axis=0), mel.std(axis=0)])


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
