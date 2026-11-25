"""Unsupervised detectors, out-of-fold conformal calibration, and non-pickle persistence."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

from .features import N_MELS, SR, clip_vector, log_mel, mel_windows, speech_band_mask, stack_frames

BUNDLE_SCHEMA = "lab-ears/detector-bundle/v1"
POOLING_MODES = ("mean", "p95", "max_window")


def _speech_dropped_bins() -> int:
    return int(speech_band_mask().sum())


def _infer_speech_band_dropped(n_mels: int) -> Optional[bool]:
    if n_mels == N_MELS:
        return False
    if n_mels == _speech_dropped_bins():
        return True
    return None


def _mel_fingerprint(mel: np.ndarray) -> Tuple[int, int, float, float, float, float]:
    arr = np.asarray(mel, dtype=np.float32)
    return (
        int(arr.shape[0]),
        int(arr.shape[1]),
        round(float(arr[0, 0]), 5),
        round(float(arr[-1, -1]), 5),
        round(float(arr.mean()), 5),
        round(float(arr.std()), 5),
    )


def _validate_pooling(pooling: str, window_frames: int, hop_frames: int) -> Tuple[str, int, int]:
    if pooling not in POOLING_MODES:
        raise ValueError(f"Unsupported pooling={pooling!r}; expected one of {POOLING_MODES}")
    if window_frames < 1 or hop_frames < 1:
        raise ValueError(f"window_frames ({window_frames}) and hop_frames ({hop_frames}) must be >= 1")
    return str(pooling), int(window_frames), int(hop_frames)


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


def _validate_score_mel(
    mel: np.ndarray,
    *,
    expected_n_mels: Optional[int],
    speech_band_dropped: Optional[bool],
    detector_name: str,
) -> np.ndarray:
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


def _aggregate_window_scores(win_scores: List[float], pooling: str) -> float:
    if len(win_scores) == 1 or pooling == "mean":
        return float(np.mean(win_scores))
    if pooling == "p95":
        return float(np.percentile(win_scores, 95))
    return float(np.max(win_scores))


class _Scaler:
    def __init__(self, mu: Optional[np.ndarray] = None, sd: Optional[np.ndarray] = None):
        self.mu = mu
        self.sd = sd

    def fit(self, X: np.ndarray) -> "_Scaler":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


class GaussianDetector:
    """Mahalanobis distance of clip/sub-window vectors to normal clips."""

    name = "gaussian"

    def __init__(
        self,
        *,
        squared: bool = False,
        pooling: str = "max_window",
        window_frames: int = 64,
        hop_frames: int = 32,
    ):
        self.squared = bool(squared)
        self.pooling, self.window_frames, self.hop_frames = _validate_pooling(
            pooling, window_frames, hop_frames
        )
        self.n_mels_: Optional[int] = None
        self.speech_band_dropped_: Optional[bool] = None
        self.location_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self._train_fingerprints_: set = set()

    def _clone_unfit(self) -> "GaussianDetector":
        return GaussianDetector(
            squared=self.squared,
            pooling=self.pooling,
            window_frames=self.window_frames,
            hop_frames=self.hop_frames,
        )

    def _extract_vectors(self, mel: np.ndarray) -> List[np.ndarray]:
        if self.pooling == "mean":
            return [clip_vector(mel, pooling="mean")]
        wins = mel_windows(mel, window_frames=self.window_frames, hop_frames=self.hop_frames)
        return [clip_vector(w, pooling="mean") for w in wins]

    def fit(self, mels: Iterable[np.ndarray]) -> "GaussianDetector":
        mel_list = _validate_fit_mels(mels, min_clips=2, detector_name="GaussianDetector")
        self.n_mels_ = int(mel_list[0].shape[1])
        self.speech_band_dropped_ = _infer_speech_band_dropped(self.n_mels_)
        self._train_fingerprints_ = {_mel_fingerprint(m) for m in mel_list}
        X = np.stack([v for m in mel_list for v in self._extract_vectors(m)])
        self.cov = LedoitWolf().fit(X)
        self.location_ = np.asarray(self.cov.location_, dtype=np.float64)
        self.precision_ = np.asarray(self.cov.precision_, dtype=np.float64)
        return self

    def _score_vector(self, v: np.ndarray) -> float:
        assert self.location_ is not None and self.precision_ is not None
        diff = np.asarray(v, dtype=np.float64) - self.location_
        d2 = max(0.0, float(diff @ self.precision_ @ diff))
        return d2 if self.squared else float(np.sqrt(d2))

    def score(self, mel: np.ndarray) -> float:
        arr = _validate_score_mel(
            mel,
            expected_n_mels=self.n_mels_,
            speech_band_dropped=self.speech_band_dropped_,
            detector_name="GaussianDetector",
        )
        win_scores = [self._score_vector(v) for v in self._extract_vectors(arr)]
        return _aggregate_window_scores(win_scores, self.pooling)


class KNNDetector:
    """Mean distance to the k nearest normal clips (standardized clip/sub-window vectors)."""

    name = "knn"

    def __init__(
        self,
        k: int = 5,
        *,
        pooling: str = "max_window",
        window_frames: int = 64,
        hop_frames: int = 32,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.pooling, self.window_frames, self.hop_frames = _validate_pooling(
            pooling, window_frames, hop_frames
        )
        self.effective_k_: Optional[int] = None
        self.n_mels_: Optional[int] = None
        self.speech_band_dropped_: Optional[bool] = None
        self.train_Xs_: Optional[np.ndarray] = None
        self._train_fingerprints_: set = set()

    def _clone_unfit(self) -> "KNNDetector":
        return KNNDetector(
            k=self.k,
            pooling=self.pooling,
            window_frames=self.window_frames,
            hop_frames=self.hop_frames,
        )

    def _extract_vectors(self, mel: np.ndarray) -> List[np.ndarray]:
        if self.pooling == "mean":
            return [clip_vector(mel, pooling="mean")]
        wins = mel_windows(mel, window_frames=self.window_frames, hop_frames=self.hop_frames)
        return [clip_vector(w, pooling="mean") for w in wins]

    def fit(self, mels: Iterable[np.ndarray], *, _warn_clamp: bool = True) -> "KNNDetector":
        mel_list = _validate_fit_mels(mels, min_clips=1, detector_name="KNNDetector")
        self.n_mels_ = int(mel_list[0].shape[1])
        self.speech_band_dropped_ = _infer_speech_band_dropped(self.n_mels_)
        self._train_fingerprints_ = {_mel_fingerprint(m) for m in mel_list}
        X = np.stack([v for m in mel_list for v in self._extract_vectors(m)])
        eff_k = min(self.k, len(X))
        if eff_k < self.k and _warn_clamp:
            warnings.warn(
                f"KNNDetector(k={self.k}) clamped to n_neighbors={eff_k} for {len(X)} normal clip(s)",
                UserWarning,
                stacklevel=2,
            )
        self.effective_k_ = eff_k
        self.scale = _Scaler().fit(X)
        self.train_Xs_ = np.asarray(self.scale(X), dtype=np.float32)
        self.nn = NearestNeighbors(n_neighbors=eff_k).fit(self.train_Xs_)
        return self

    def score(self, mel: np.ndarray) -> float:
        arr = _validate_score_mel(
            mel,
            expected_n_mels=self.n_mels_,
            speech_band_dropped=self.speech_band_dropped_,
            detector_name="KNNDetector",
        )
        vecs = np.stack(self._extract_vectors(arr))
        d, _ = self.nn.kneighbors(self.scale(vecs))
        win_scores = [float(row.mean()) for row in d]
        return _aggregate_window_scores(win_scores, self.pooling)


class AutoencoderDetector:
    """Dense autoencoder on stacked frames with causal sliding-window / p95 / mean aggregation."""

    name = "autoencoder"

    def __init__(
        self,
        hidden=(128, 128, 8, 128, 128),
        max_frames: int = 60000,
        epochs: int = 30,
        seed: int = 0,
        context: int = 5,
        *,
        pooling: str = "max_window",
        window_frames: int = 64,
        hop_frames: int = 32,
    ):
        self.hidden = tuple(int(h) for h in hidden)
        self.max_frames = int(max_frames)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.context = int(context)
        self.pooling, self.window_frames, self.hop_frames = _validate_pooling(
            pooling, window_frames, hop_frames
        )
        self.n_mels_: Optional[int] = None
        self.speech_band_dropped_: Optional[bool] = None
        self.coefs_: Optional[List[np.ndarray]] = None
        self.intercepts_: Optional[List[np.ndarray]] = None
        self._train_fingerprints_: set = set()

    def _clone_unfit(self) -> "AutoencoderDetector":
        return AutoencoderDetector(
            hidden=self.hidden,
            max_frames=self.max_frames,
            epochs=self.epochs,
            seed=self.seed,
            context=self.context,
            pooling=self.pooling,
            window_frames=self.window_frames,
            hop_frames=self.hop_frames,
        )

    def fit(self, mels: Iterable[np.ndarray]) -> "AutoencoderDetector":
        mel_list = _validate_fit_mels(mels, min_clips=1, detector_name="AutoencoderDetector")
        self.n_mels_ = int(mel_list[0].shape[1])
        self.speech_band_dropped_ = _infer_speech_band_dropped(self.n_mels_)
        self._train_fingerprints_ = {_mel_fingerprint(m) for m in mel_list}
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
        self.coefs_ = [np.asarray(w, dtype=np.float64) for w in self.ae.coefs_]
        self.intercepts_ = [np.asarray(b, dtype=np.float64) for b in self.ae.intercepts_]
        return self

    def _predict_scaled(self, Xs: np.ndarray) -> np.ndarray:
        if self.coefs_ is not None and self.intercepts_ is not None:
            h = np.asarray(Xs, dtype=np.float64)
            for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
                h = np.maximum(0.0, h @ W + b)
            return h @ self.coefs_[-1] + self.intercepts_[-1]
        return self.ae.predict(Xs)

    def score(self, mel: np.ndarray) -> float:
        arr = _validate_score_mel(
            mel,
            expected_n_mels=self.n_mels_,
            speech_band_dropped=self.speech_band_dropped_,
            detector_name="AutoencoderDetector",
        )
        Xs = self.scale(stack_frames(arr, context=self.context))
        pred = self._predict_scaled(Xs)
        frame_mse = ((pred - Xs) ** 2).mean(axis=1)
        if self.pooling == "mean":
            return float(frame_mse.mean())
        if self.pooling == "p95":
            return float(np.percentile(frame_mse, 95))

        w_len = max(1, self.window_frames - self.context + 1)
        n_stacks = int(frame_mse.shape[0])
        if n_stacks <= w_len:
            return float(frame_mse.mean())
        starts = list(range(0, n_stacks - w_len + 1, self.hop_frames))
        if starts[-1] != n_stacks - w_len:
            starts.append(n_stacks - w_len)
        return float(max(float(frame_mse[s : s + w_len].mean()) for s in starts))


DETECTORS = {"gaussian": GaussianDetector, "knn": KNNDetector, "autoencoder": AutoencoderDetector}


def compute_oof_scores(detector, normal_mels: Iterable[np.ndarray]) -> np.ndarray:
    """Compute leave-one-out / K-fold out-of-fold anomaly scores on normal training clips."""
    mel_list = [np.asarray(m, dtype=np.float32) for m in normal_mels]
    n = len(mel_list)
    if n == 0:
        raise ValueError("normal_mels must contain at least 1 clip")
    if n == 1:
        if isinstance(detector, GaussianDetector):
            raise ValueError("GaussianDetector requires at least 2 clips for OOF calibration")
        return np.array([float(detector.score(mel_list[0]))], dtype=np.float64)

    if isinstance(detector, GaussianDetector) and n == 2:
        v0 = clip_vector(mel_list[0], pooling="mean").astype(np.float64)
        v1 = clip_vector(mel_list[1], pooling="mean").astype(np.float64)
        d = float(np.linalg.norm(v0 - v1))
        val = d * d if detector.squared else d
        return np.array([val, val], dtype=np.float64)

    max_folds = 5 if isinstance(detector, AutoencoderDetector) else min(n, 20)
    if isinstance(detector, GaussianDetector):
        # Ensure every training fold has >= 2 clips
        max_folds = min(max_folds, n)

    folds = [f for f in np.array_split(np.arange(n), max_folds) if len(f) > 0]
    scores = np.empty(n, dtype=np.float64)
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        tr_fold = [mel_list[i] for i in np.where(mask)[0]]
        sub_det = detector._clone_unfit()
        if isinstance(sub_det, KNNDetector):
            sub_det.fit(tr_fold, _warn_clamp=False)
        else:
            sub_det.fit(tr_fold)
        for idx in fold:
            scores[idx] = float(sub_det.score(mel_list[idx]))
    return scores


def _conformal_threshold_from_scores(
    scores: np.ndarray,
    quantile: float,
    *,
    parametric_small_sample: bool = True,
) -> float:
    n = int(scores.size)
    if quantile >= 1.0:
        return float(np.max(scores))
    q_conf = (n + 1) * float(quantile) / float(n)
    if q_conf <= 1.0:
        return float(np.quantile(scores, q_conf, method="higher"))
    if parametric_small_sample and n >= 2:
        mu = float(np.mean(scores))
        sd = float(np.std(scores, ddof=1))
        t_crit = float(student_t.ppf(float(quantile), df=n - 1))
        pred_upper = mu + t_crit * math.sqrt(1.0 + 1.0 / float(n)) * sd
        return float(max(float(np.max(scores)), pred_upper))
    return float(scores[0] * 1.25 + 1e-6)


def calibrate_with_mode(
    detector,
    val_mels: Iterable[np.ndarray],
    quantile: float = 0.99,
    *,
    force_oof: bool = False,
) -> Tuple[float, str]:
    """Calibrate an anomaly threshold and report `('heldout_conformal' | 'oof_conformal')`.

    Automatically switches to leave-one-out / K-fold out-of-fold calibration when
    `val_mels` overlaps the training clips `detector` was fit on, preventing the
    in-sample self-neighbor / covariance-overfit false-alarm collapse.
    """
    if not (0.0 < quantile <= 1.0) or not np.isfinite(quantile):
        raise ValueError(f"quantile must be finite and in (0.0, 1.0], got {quantile}")
    val_list = [np.asarray(m, dtype=np.float32) for m in val_mels]
    if not val_list:
        raise ValueError("val_mels must contain at least 1 validation clip")

    train_fps = getattr(detector, "_train_fingerprints_", set())
    overlap = sum(1 for m in val_list if _mel_fingerprint(m) in train_fps) if train_fps else 0
    use_oof = bool(force_oof or (overlap > 0 and overlap >= 0.5 * len(val_list) and hasattr(detector, "_clone_unfit")))

    if use_oof:
        scores = compute_oof_scores(detector, val_list)
        mode = "oof_conformal"
    else:
        scores = np.asarray([detector.score(m) for m in val_list], dtype=np.float64)
        mode = "heldout_conformal"

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly score encountered during threshold calibration")

    thr = _conformal_threshold_from_scores(
        scores,
        float(quantile),
        parametric_small_sample=use_oof or (len(val_list) < 25),
    )
    return thr, mode


def calibrate_threshold(detector, val_mels: Iterable[np.ndarray], quantile: float = 0.99) -> float:
    """Calibrate an anomaly threshold from normal clips at `quantile` in (0, 1]."""
    thr, _ = calibrate_with_mode(detector, val_mels, quantile=quantile)
    return thr


def save_detector_bundle(
    bundle_path: Path | str,
    detector,
    *,
    threshold: float,
    calibration_mode: str = "heldout_conformal",
    quantile: float = 0.99,
) -> Path:
    """Persist fitted detector weights and calibrated threshold via safe non-pickle `.npz` + `.json`."""
    if getattr(detector, "n_mels_", None) is None:
        raise RuntimeError("Detector must be fit() before saving bundle")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got {threshold!r}")

    p = Path(bundle_path)
    npz_path = p if p.suffix == ".npz" else p.with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "detector": detector.name,
        "threshold": float(threshold),
        "quantile": float(quantile),
        "calibration_mode": str(calibration_mode),
        "n_mels": int(detector.n_mels_),
        "drop_speech_band": detector.speech_band_dropped_,
        "pooling": str(detector.pooling),
        "window_frames": int(detector.window_frames),
        "hop_frames": int(detector.hop_frames),
        "sample_rate_hz": SR,
    }
    arrays: Dict[str, np.ndarray] = {}

    if isinstance(detector, GaussianDetector):
        assert detector.location_ is not None and detector.precision_ is not None
        meta["squared"] = bool(detector.squared)
        arrays["location"] = detector.location_
        arrays["precision"] = detector.precision_
    elif isinstance(detector, KNNDetector):
        assert detector.train_Xs_ is not None and detector.effective_k_ is not None
        meta["k"] = int(detector.k)
        meta["effective_k"] = int(detector.effective_k_)
        arrays["scale_mu"] = np.asarray(detector.scale.mu, dtype=np.float32)
        arrays["scale_sd"] = np.asarray(detector.scale.sd, dtype=np.float32)
        arrays["train_Xs"] = np.asarray(detector.train_Xs_, dtype=np.float32)
    elif isinstance(detector, AutoencoderDetector):
        assert detector.coefs_ is not None and detector.intercepts_ is not None
        meta["hidden"] = list(detector.hidden)
        meta["context"] = int(detector.context)
        meta["n_layers"] = len(detector.coefs_)
        arrays["scale_mu"] = np.asarray(detector.scale.mu, dtype=np.float32)
        arrays["scale_sd"] = np.asarray(detector.scale.sd, dtype=np.float32)
        for idx, (w, b) in enumerate(zip(detector.coefs_, detector.intercepts_)):
            arrays[f"coef_{idx}"] = np.asarray(w, dtype=np.float64)
            arrays[f"intercept_{idx}"] = np.asarray(b, dtype=np.float64)
    else:
        raise TypeError(f"Unsupported detector type for safe serialization: {type(detector)!r}")

    meta_text = json.dumps(meta, indent=2, sort_keys=True)
    arrays["metadata_json"] = np.frombuffer(meta_text.encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(meta_text + "\n", encoding="utf-8")
    return npz_path


def load_detector_bundle(bundle_path: Path | str) -> Tuple[Any, Dict[str, Any]]:
    """Load a pre-calibrated detector + metadata with `allow_pickle=False`."""
    p = Path(bundle_path)
    npz_path = p if p.suffix == ".npz" else p.with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Detector bundle archive does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        if json_path.is_file():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        elif "metadata_json" in archive:
            meta = json.loads(bytes(archive["metadata_json"].tolist()).decode("utf-8"))
        else:
            raise ValueError(f"Missing detector metadata in {npz_path}")

        if meta.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"Unsupported detector bundle schema: {meta.get('schema')!r}")

        det_name = str(meta["detector"])
        pooling = str(meta.get("pooling", "max_window"))
        win_f = int(meta.get("window_frames", 64))
        hop_f = int(meta.get("hop_frames", 32))
        n_mels = int(meta["n_mels"])
        drop_speech = meta.get("drop_speech_band")

        if det_name == "gaussian":
            det: Any = GaussianDetector(
                squared=bool(meta.get("squared", False)),
                pooling=pooling,
                window_frames=win_f,
                hop_frames=hop_f,
            )
            det.n_mels_ = n_mels
            det.speech_band_dropped_ = drop_speech
            det.location_ = np.asarray(archive["location"], dtype=np.float64)
            det.precision_ = np.asarray(archive["precision"], dtype=np.float64)
        elif det_name == "knn":
            det = KNNDetector(
                k=int(meta["k"]),
                pooling=pooling,
                window_frames=win_f,
                hop_frames=hop_f,
            )
            det.n_mels_ = n_mels
            det.speech_band_dropped_ = drop_speech
            det.effective_k_ = int(meta["effective_k"])
            det.scale = _Scaler(
                mu=np.asarray(archive["scale_mu"], dtype=np.float32),
                sd=np.asarray(archive["scale_sd"], dtype=np.float32),
            )
            det.train_Xs_ = np.asarray(archive["train_Xs"], dtype=np.float32)
            det.nn = NearestNeighbors(n_neighbors=det.effective_k_).fit(det.train_Xs_)
        elif det_name == "autoencoder":
            det = AutoencoderDetector(
                hidden=tuple(meta["hidden"]),
                context=int(meta["context"]),
                pooling=pooling,
                window_frames=win_f,
                hop_frames=hop_f,
            )
            det.n_mels_ = n_mels
            det.speech_band_dropped_ = drop_speech
            det.scale = _Scaler(
                mu=np.asarray(archive["scale_mu"], dtype=np.float32),
                sd=np.asarray(archive["scale_sd"], dtype=np.float32),
            )
            n_layers = int(meta["n_layers"])
            det.coefs_ = [np.asarray(archive[f"coef_{i}"], dtype=np.float64) for i in range(n_layers)]
            det.intercepts_ = [np.asarray(archive[f"intercept_{i}"], dtype=np.float64) for i in range(n_layers)]
        else:
            raise ValueError(f"Unknown detector name in bundle: {det_name!r}")

    return det, meta


def featurize(wavs, drop_speech_band: bool = False):
    from .features import load_wav

    return [log_mel(load_wav(w), drop_speech_band) for w in wavs]
