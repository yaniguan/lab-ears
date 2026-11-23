"""Audio event record: one line of JSON per scored clip, ready for the run ledger.

Raw audio is not stored by default. A record keeps the protocol step and instrument
it belongs to, the anomaly score, and (optionally) a path to the log-mel features.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

SCHEMA = "lab-ears/audio-event/v1"


def _is_real_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass
class AudioEvent:
    protocol_id: str
    step: int
    instrument_id: str
    start_utc: str
    duration_s: float
    detector: str
    score: float
    threshold: float
    sample_rate_hz: int = 16000
    mic: str = "unknown"  # e.g. "contact", "near-field", "room"
    features_path: Optional[str] = None  # log-mel .npy; no raw audio
    speech_band_dropped: bool = True
    raw_audio_kept: bool = False
    notes: str = ""
    schema: str = field(default=SCHEMA)

    @property
    def flagged(self) -> bool:
        if not (_is_real_number(self.score) and math.isfinite(float(self.score))):
            raise ValueError(f"score must be a finite number, got {self.score!r}")
        if not (_is_real_number(self.threshold) and math.isfinite(float(self.threshold))):
            raise ValueError(f"threshold must be a finite number, got {self.threshold!r}")
        return bool(float(self.score) > float(self.threshold))

    def to_json(self) -> str:
        self.validate()
        d = asdict(self)
        d["flagged"] = self.flagged
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "AudioEvent":
        d = json.loads(line)
        if not isinstance(d, dict):
            raise ValueError("AudioEvent JSON line must decode to an object")
        had_flagged = "flagged" in d
        raw_flagged = d.pop("flagged", None)
        event = cls(**d)
        event.validate()
        if had_flagged and raw_flagged is not event.flagged:
            raise ValueError(
                f"Serialized flagged={raw_flagged!r} disagrees with score ({event.score}) > threshold ({event.threshold})"
            )
        return event

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"Unsupported schema {self.schema!r}, expected {SCHEMA!r}")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("step must be an integer >= 0")
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise ValueError("protocol_id is required")
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise ValueError("instrument_id is required")
        if not isinstance(self.detector, str) or not self.detector.strip():
            raise ValueError("detector is required")
        if not _is_real_number(self.duration_s) or not math.isfinite(float(self.duration_s)) or float(self.duration_s) <= 0.0:
            raise ValueError(f"duration_s must be finite and > 0, got {self.duration_s!r}")
        if not isinstance(self.sample_rate_hz, int) or isinstance(self.sample_rate_hz, bool) or self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be an integer > 0, got {self.sample_rate_hz!r}")
        if not _is_real_number(self.score) or not math.isfinite(float(self.score)):
            raise ValueError(f"score must be finite, got {self.score!r}")
        if not _is_real_number(self.threshold) or not math.isfinite(float(self.threshold)):
            raise ValueError(f"threshold must be finite, got {self.threshold!r}")
        if not isinstance(self.start_utc, str) or not self.start_utc.strip():
            raise ValueError("start_utc must be a non-empty ISO-8601 string")
        iso = self.start_utc[:-1] + "+00:00" if self.start_utc.endswith("Z") else self.start_utc
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            raise ValueError("start_utc must be timezone-aware (e.g. trailing 'Z' or '+00:00')")
        if not isinstance(self.raw_audio_kept, bool) or not isinstance(self.speech_band_dropped, bool):
            raise ValueError("raw_audio_kept and speech_band_dropped must be booleans")
        if self.raw_audio_kept and self.speech_band_dropped:
            raise ValueError("raw audio still contains the speech band")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
