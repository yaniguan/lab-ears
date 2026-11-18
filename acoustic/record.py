"""Audio event record: one line of JSON per scored clip, ready for the run ledger.

Raw audio is not stored by default. A record keeps the protocol step and instrument
it belongs to, the anomaly score, and (optionally) a path to the log-mel features.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

SCHEMA = "lab-ears/audio-event/v1"


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
        return self.score > self.threshold

    def to_json(self) -> str:
        d = asdict(self)
        d["flagged"] = self.flagged
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "AudioEvent":
        d = json.loads(line)
        d.pop("flagged", None)
        return cls(**d)

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError("step must be >= 0")
        if not self.protocol_id or not self.instrument_id:
            raise ValueError("protocol_id and instrument_id are required")
        datetime.fromisoformat(self.start_utc.replace("Z", "+00:00"))
        if self.raw_audio_kept and self.speech_band_dropped:
            raise ValueError("raw audio still contains the speech band")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
