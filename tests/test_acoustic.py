import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from acoustic.detect import AutoencoderDetector, GaussianDetector, KNNDetector
from acoustic.features import MEL_CENTERS_HZ, N_MELS, SR, log_mel, speech_band_mask
from acoustic.record import AudioEvent, now_utc


def _clip(rng, knock=False, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    f0 = 120 * (1 + 0.01 * rng.standard_normal())
    x = sum(a * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) for k, a in [(1, 0.5), (2, 0.2), (3, 0.1)])
    x = x + 0.02 * rng.standard_normal(len(t))
    if knock:  # short broadband clicks, like a loose part
        for c in np.arange(0.1, seconds, 0.25):
            i = int(c * SR)
            x[i:i + 200] += 0.6 * rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    train = [log_mel(_clip(rng)) for _ in range(30)]
    test = [log_mel(_clip(rng)) for _ in range(10)] + [log_mel(_clip(rng, knock=True)) for _ in range(10)]
    y = [0] * 10 + [1] * 10
    return train, test, y


def test_log_mel_shape():
    m = log_mel(np.zeros(SR, dtype=np.float32))
    assert m.shape[1] == N_MELS and np.isfinite(m).all()


def test_speech_band_mask_drops_speech_bins():
    keep = speech_band_mask()
    assert not keep[(MEL_CENTERS_HZ > 300) & (MEL_CENTERS_HZ < 3400)].any()
    assert log_mel(np.zeros(SR, dtype=np.float32), drop_speech_band=True).shape[1] == keep.sum()


@pytest.mark.parametrize("det", [GaussianDetector(), KNNDetector(k=3), AutoencoderDetector(hidden=(32, 8, 32), epochs=20)])
def test_detectors_flag_knocks(data, det):
    train, test, y = data
    det.fit(train)
    assert roc_auc_score(y, [det.score(m) for m in test]) > 0.9


def test_audio_event_roundtrip():
    e = AudioEvent(protocol_id="pcr-setup-v2", step=4, instrument_id="centrifuge-01", start_utc=now_utc(),
                   duration_s=10.0, detector="gaussian", score=12.5, threshold=10.0, mic="contact")
    e.validate()
    line = e.to_json()
    assert json.loads(line)["flagged"] is True
    assert AudioEvent.from_json(line) == e


def test_raw_audio_with_speech_band_is_rejected():
    e = AudioEvent(protocol_id="p", step=0, instrument_id="i", start_utc=now_utc(), duration_s=1.0,
                   detector="knn", score=0.0, threshold=1.0, raw_audio_kept=True, speech_band_dropped=True)
    with pytest.raises(ValueError):
        e.validate()
