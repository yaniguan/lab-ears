"""Machine sound anomaly detection baseline (log-mel + simple one-class detectors)."""

from .detect import (
    DETECTORS,
    AutoencoderDetector,
    GaussianDetector,
    KNNDetector,
    calibrate_threshold,
    featurize,
)
from .features import (
    HOP,
    MEL_CENTERS_HZ,
    MEL_EDGES_HZ,
    N_FFT,
    N_MELS,
    SPEECH_BAND_HZ,
    SR,
    clip_vector,
    load_wav,
    log_mel,
    mel_filterbank,
    speech_band_mask,
    stack_frames,
)
from .monitor import monitor_clip, monitor_steps
from .record import SCHEMA, AudioEvent, now_utc

__all__ = [
    "DETECTORS",
    "AutoencoderDetector",
    "GaussianDetector",
    "KNNDetector",
    "calibrate_threshold",
    "featurize",
    "HOP",
    "MEL_CENTERS_HZ",
    "MEL_EDGES_HZ",
    "N_FFT",
    "N_MELS",
    "SPEECH_BAND_HZ",
    "SR",
    "clip_vector",
    "load_wav",
    "log_mel",
    "mel_filterbank",
    "speech_band_mask",
    "stack_frames",
    "monitor_clip",
    "monitor_steps",
    "SCHEMA",
    "AudioEvent",
    "now_utc",
]
