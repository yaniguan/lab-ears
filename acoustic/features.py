"""Log-mel features with numpy and scipy only."""

import numpy as np
from scipy.io import wavfile

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 64
SPEECH_BAND_HZ = (300.0, 3400.0)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb, hz[1:-1]


_FB, MEL_CENTERS_HZ = mel_filterbank()
_WINDOW = np.hanning(N_FFT)


def speech_band_mask(centers_hz=MEL_CENTERS_HZ, band=SPEECH_BAND_HZ):
    """True for mel bins to keep (outside the main speech band)."""
    return (centers_hz < band[0]) | (centers_hz > band[1])


def load_wav(path):
    sr, x = wavfile.read(path)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.5:
        x = x / 32768.0
    return x


def log_mel(x, drop_speech_band=False):
    """Return log-mel frames, shape (n_frames, n_mels or fewer)."""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _WINDOW, axis=1)) ** 2
    mel = 10.0 * np.log10(spec @ _FB.T + 1e-10)
    if drop_speech_band:
        mel = mel[:, speech_band_mask()]
    return mel.astype(np.float32)


def clip_vector(mel):
    """Clip-level vector: mean and std of each mel bin over time."""
    return np.concatenate([mel.mean(axis=0), mel.std(axis=0)])


def stack_frames(mel, context=5):
    """Stack `context` consecutive frames (the DCASE baseline input)."""
    n = mel.shape[0] - context + 1
    return np.stack([mel[i:i + n] for i in range(context)], axis=1).reshape(n, -1)
