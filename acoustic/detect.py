"""Unsupervised detectors. Each one is fit on normal sounds only."""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

from .features import clip_vector, log_mel, stack_frames


class _Scaler:
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class GaussianDetector:
    """Mahalanobis distance of the clip vector to the normal clips."""

    name = "gaussian"

    def fit(self, mels):
        X = np.stack([clip_vector(m) for m in mels])
        self.cov = LedoitWolf().fit(X)
        return self

    def score(self, mel):
        return float(self.cov.mahalanobis(clip_vector(mel)[None, :])[0])


class KNNDetector:
    """Mean distance to the k nearest normal clips (standardized clip vectors)."""

    name = "knn"

    def __init__(self, k=5):
        self.k = k

    def fit(self, mels):
        X = np.stack([clip_vector(m) for m in mels])
        self.scale = _Scaler().fit(X)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.scale(X))
        return self

    def score(self, mel):
        d, _ = self.nn.kneighbors(self.scale(clip_vector(mel)[None, :]))
        return float(d.mean())


class AutoencoderDetector:
    """Dense autoencoder on 5 stacked frames; score = mean reconstruction error."""

    name = "autoencoder"

    def __init__(self, hidden=(128, 128, 8, 128, 128), max_frames=60000, epochs=30, seed=0):
        self.hidden, self.max_frames, self.epochs, self.seed = hidden, max_frames, epochs, seed

    def fit(self, mels):
        X = np.concatenate([stack_frames(m) for m in mels])
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_frames:
            X = X[rng.choice(len(X), self.max_frames, replace=False)]
        self.scale = _Scaler().fit(X)
        Xs = self.scale(X)
        self.ae = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=self.epochs, batch_size=512,
                               learning_rate_init=1e-3, random_state=self.seed, early_stopping=False)
        self.ae.fit(Xs, Xs)
        return self

    def score(self, mel):
        Xs = self.scale(stack_frames(mel))
        return float(((self.ae.predict(Xs) - Xs) ** 2).mean())


DETECTORS = {"gaussian": GaussianDetector, "knn": KNNDetector, "autoencoder": AutoencoderDetector}


def featurize(wavs, drop_speech_band=False):
    from .features import load_wav
    return [log_mel(load_wav(w), drop_speech_band) for w in wavs]
