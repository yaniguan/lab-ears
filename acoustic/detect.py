"""Unsupervised detectors. Each one is fit on normal sounds only."""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

from .features import N_MELS, clip_vector, log_mel, speech_band_mask, stack_frames

_SPEECH_DROPPED_BINS = int(speech_band_mask().sum())


def _infer_speech_band_dropped(n_mels: int) -> Optional[bool]:
    if n_mels == N_MELS:
        return False
    if n_mels == _SPEECH_DROPPED_BINS:
        return True
    return None


def _validate_fit_mels(mels: Iterable[np.ndarray], *, min_clips: int, detector_name: str) -> List[np.ndarray]:
    mel_list = [np.asarray(m, dtype=np.float32) for m in mels]
    if len(mel_list) < min_clips:
        raise ValueError(
            f"At least {min_clips} normal clip(s) required to fit {detector_name}, got {len(mel_list)}"
        )
    for idx, m in enumerate(mel_list):
        if m.ndim != 2 or m.shape[0] < 1 or m.shape[1] < 1:
            raise ValueError(f"{detector_name}.fit(): clip {idx} must be non-empty 2D mel array, got {m.shape}")
        if not np.isfinite(m).all():
            raise ValueError(f"{detector_name}.fit(): clip {idx} contains NaN or Inf")
    n_mels = int(mel_list[0].shape[1])
    for idx, m in enumerate(mel_list[1:], start=1):
        if int(m.shape[1]) != n_mels:
            raise ValueError(
                f"{detector_name}.fit(): inconsistent mel feature dimensions across training clips "
                f"(clip 0 has {n_mels} bins, clip {idx} has {m.shape[1]} bins)"
            )
    return mel_list


def _validate_score_mel(mel: np.ndarray, *, expected_n_mels: Optional[int], speech_band_dropped: Optional[bool], detector_name: str) -> np.ndarray:
    if expected_n_mels is None:
        raise RuntimeError(f"{detector_name} must be fit() before calling score()")
    arr = np.asarray(mel, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{detector_name}.score(): expected non-empty 2D mel array, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{detector_name}.score(): mel array contains NaN or Inf")
    if int(arr.shape[1]) != expected_n_mels:
        raise ValueError(
            f"{detector_name}.score(): feature dimension mismatch: detector was fit with n_mels={expected_n_mels} "
            f"(speech_band_dropped={speech_band_dropped}), but received {arr.shape[1]} bins"
        )
    return arr


class _Scaler:
    def fit(self, X: np.ndarray) -> "_Scaler":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


class GaussianDetector:
    """Mahalanobis distance of the clip vector to the normal clips."""

    name = "gaussian"

    def __init__(self, *, squared: bool = False):
        self.squared = squared
        self.n_mels_: Optional[int] = None
        self.speech_band_dropped_: Optional[bool] = None

    def fit(self, mels: Iterable[np.ndarray]) -> "GaussianDetector":
        mel_list = _validate_fit_mels(mels, min_clips=2, detector_name="GaussianDetector")
        self.n_mels_ = int(mel_list[0].shape[1])
        self.speech_band_dropped_ = _infer_speech_band_dropped(self.n_mels_)
        X = np.stack([clip_vector(m) for m in mel_list])
        self.cov = LedoitWolf().fit(X)
        return self

    def score(self, mel: np.ndarray) -> float:
        arr = _validate_score_mel(
            mel,
            expected_n_mels=self.n_mels_,
            speech_band_dropped=self.speech_band_dropped_,
            detector_name="GaussianDetector",
        )
        d2 = max(0.0, float(self.cov.mahalanobis(clip_vector(arr)[None, :])[0]))
        return d2 if self.squared else float(np.sqrt(d2))


class KNNDetector:
    """Mean distance to the k nearest normal clips (standardized clip vectors)."""

    name = "knn"

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.effective_k_: Optional[int] = None
        self.n_mels_: Optional[int] = None
        self.speech_band_dropped_: Optional[bool] = None

    def fit(self, mels: Iterable[np.ndarray]) -> "KNNDetector":
        mel_list = _validate_fit_mels(mels, min_clips=1, detector_name="KNNDetector")
        self.n_mels_ = int(mel_list[0].shape[1])
        self.speech_band_dropped_ = _infer_speech_band_dropped(self.n_mels_)
        X = np.stack([clip_vector(m) for m in mel_list])
        eff_k = min(self.k, len(X))
        if eff_k < self.k:
            warnings.warn(
                f"KNNDetector(k={self.k}) clamped to n_neighbors={eff_k} for {len(X)} normal clip(s)",
                UserWarning,
                stacklevel=2,
            )
        self.effective_k_ = eff_k
        self.scale = _Scaler().fit(X)
        self.nn = NearestNeighbors(n_neighbors=eff_k).fit(self.scale(X))
        return self

    def score(self, mel: np.ndarray) -> float:
        arr = _validate_score_mel(
            mel,
            expected_n_mels=self.n_mels_,
            speech_band_dropped=self.speech_band_dropped_,
            detector_name="KNNDetector",
        )
        d, _ = self.nn.kneighbors(self.scale(clip_vector(arr)[None, :]))
        return float(d.mean())


class AutoencoderDetector:
    """Dense autoencoder on 5 stacked frames; score = mean reconstruction error."""

    name = "autoencoder"

    def __init__(self, hidden=(128, 128, 8, 128, 128), max_frames: int = 60000, epochs: int = 30, seed: int = 0, context: int = 5):
        self.hidden = hidden
        self.max_frames = max_frames
        self.epochs = epochs
        self.seed = seed
        self.context = context
        self.n_mels_: Optional[int] = None
        self.speech_band_dropped_: Optional[bool] = None

    def fit(self, mels: Iterable[np.ndarray]) -> "AutoencoderDetector":
        mel_list = _validate_fit_mels(mels, min_clips=1, detector_name="AutoencoderDetector")
        self.n_mels_ = int(mel_list[0].shape[1])
        self.speech_band_dropped_ = _infer_speech_band_dropped(self.n_mels_)
        X = np.concatenate([stack_frames(m, context=self.context) for m in mel_list], axis=0)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_frames:
            X = X[rng.choice(len(X), self.max_frames, replace=False)]
        self.scale = _Scaler().fit(X)
        Xs = self.scale(X)
        batch_size = min(512, max(1, len(Xs)))
        self.ae = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            max_iter=self.epochs,
            batch_size=batch_size,
            learning_rate_init=1e-3,
            random_state=self.seed,
            early_stopping=False,
        )
        self.ae.fit(Xs, Xs)
        return self

    def score(self, mel: np.ndarray) -> float:
        arr = _validate_score_mel(
            mel,
            expected_n_mels=self.n_mels_,
            speech_band_dropped=self.speech_band_dropped_,
            detector_name="AutoencoderDetector",
        )
        Xs = self.scale(stack_frames(arr, context=self.context))
        return float(((self.ae.predict(Xs) - Xs) ** 2).mean())


DETECTORS = {"gaussian": GaussianDetector, "knn": KNNDetector, "autoencoder": AutoencoderDetector}


def calibrate_threshold(detector, val_mels: Iterable[np.ndarray], quantile: float = 0.99) -> float:
    """Calibrate an anomaly threshold from normal validation clips at `quantile` in (0, 1]."""
    if not (0.0 < quantile <= 1.0) or not np.isfinite(quantile):
        raise ValueError(f"quantile must be finite and in (0.0, 1.0], got {quantile}")
    val_list = list(val_mels)
    if not val_list:
        raise ValueError("val_mels must contain at least 1 validation clip")
    scores = np.asarray([detector.score(m) for m in val_list], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly score encountered during threshold calibration")
    return float(np.quantile(scores, quantile))


def featurize(wavs, drop_speech_band: bool = False):
    from .features import load_wav

    return [log_mel(load_wav(w), drop_speech_band) for w in wavs]
