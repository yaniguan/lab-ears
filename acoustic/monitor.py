"""Live step acoustic monitor: fit/calibrate an anomaly detector on normal clips,
score protocol step WAV recordings, and emit validated `lab-ears/audio-event/v1`
JSONL ledger lines for `lab-rehearsal` and `lab-verify`.

Usage:
    python -m acoustic.monitor \\
        --train-dir path/to/normal_clips \\
        --wav path/to/step4.wav \\
        --protocol-id pcr-setup-v2 --step 4 --instrument-id centrifuge-01 \\
        --out-jsonl runs/audio_events.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .detect import DETECTORS, calibrate_threshold
from .features import SR, load_wav, log_mel
from .record import AudioEvent, now_utc


def _collect_wavs(directory: Optional[Path], files: Optional[Sequence[Path]], label: str) -> List[Path]:
    out: List[Path] = []
    if directory is not None:
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")
        out.extend(sorted(directory.glob("*.wav")))
    if files:
        for f in files:
            if not f.is_file():
                raise FileNotFoundError(f"{label} WAV file does not exist: {f}")
            out.append(f)
    if not out:
        raise ValueError(f"No {label} WAV files provided")
    return out


def monitor_clip(
    wav_path: Path | str,
    detector,
    *,
    threshold: float,
    protocol_id: str,
    step: int,
    instrument_id: str,
    mic: str = "contact",
    drop_speech_band: bool = True,
    resample: bool = False,
    save_features_path: Optional[Path | str] = None,
    start_utc: Optional[str] = None,
    notes: str = "",
) -> AudioEvent:
    """Score one WAV clip with a fitted detector and return a validated AudioEvent."""
    wav_path = Path(wav_path)
    x = load_wav(wav_path, target_sr=SR, resample=resample)
    mel = log_mel(x, drop_speech_band=drop_speech_band)
    score = float(detector.score(mel))

    feat_str: Optional[str] = None
    if save_features_path is not None:
        fpath = Path(save_features_path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(fpath, mel)
        feat_str = str(fpath)

    event = AudioEvent(
        protocol_id=protocol_id,
        step=step,
        instrument_id=instrument_id,
        start_utc=start_utc or now_utc(),
        duration_s=round(float(len(x)) / float(SR), 6),
        detector=getattr(detector, "name", type(detector).__name__.lower()),
        score=score,
        threshold=float(threshold),
        sample_rate_hz=SR,
        mic=mic,
        features_path=feat_str,
        speech_band_dropped=drop_speech_band,
        raw_audio_kept=False,
        notes=notes,
    )
    event.validate()
    return event


def monitor_steps(
    wav_paths: Iterable[Path | str],
    detector,
    *,
    threshold: float,
    protocol_id: str,
    start_step: int = 0,
    instrument_id: str,
    mic: str = "contact",
    drop_speech_band: bool = True,
    resample: bool = False,
    save_features_dir: Optional[Path] = None,
    out_jsonl: Optional[Path] = None,
    notes: str = "",
) -> List[AudioEvent]:
    """Score one or more protocol step WAV clips and optionally append JSONL events."""
    events: List[AudioEvent] = []
    for offset, wp in enumerate(wav_paths):
        wp_path = Path(wp)
        step_num = start_step + offset
        feat_path = (
            save_features_dir / f"{protocol_id}_step{step_num}_{wp_path.stem}.npy"
            if save_features_dir is not None
            else None
        )
        ev = monitor_clip(
            wp_path,
            detector,
            threshold=threshold,
            protocol_id=protocol_id,
            step=step_num,
            instrument_id=instrument_id,
            mic=mic,
            drop_speech_band=drop_speech_band,
            resample=resample,
            save_features_path=feat_path,
            notes=notes,
        )
        events.append(ev)

    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("a", encoding="utf-8") as fh:
            for ev in events:
                fh.write(ev.to_json() + "\n")
    return events


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, default=None, help="Directory of normal calibration WAV clips")
    ap.add_argument("--train-wavs", nargs="*", type=Path, default=None, help="Explicit normal calibration WAV files")
    ap.add_argument("--val-dir", type=Path, default=None, help="Optional normal validation WAV directory for threshold")
    ap.add_argument("--val-wavs", nargs="*", type=Path, default=None, help="Optional normal validation WAV files")
    ap.add_argument("--wav", nargs="+", type=Path, required=True, help="Step WAV clip(s) to score")
    ap.add_argument("--detector", choices=list(DETECTORS), default="gaussian")
    ap.add_argument("--threshold", type=float, default=None, help="Explicit threshold (overrides --quantile)")
    ap.add_argument("--quantile", type=float, default=0.99, help="Validation score quantile for threshold calibration")
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--step", type=int, default=0, help="Starting 0-based protocol step index")
    ap.add_argument("--instrument-id", required=True)
    ap.add_argument("--mic", default="contact")
    ap.add_argument("--speech-band", choices=["drop", "keep"], default="drop")
    ap.add_argument("--resample", action="store_true", help="Resample non-16kHz WAVs to 16 kHz")
    ap.add_argument("--save-features-dir", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None, help="Append validated AudioEvent JSONL lines to file")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    drop_speech = args.speech_band == "drop"
    try:
        train_wavs = _collect_wavs(args.train_dir, args.train_wavs, "training")
        step_wavs = _collect_wavs(None, args.wav, "step")
        train_mels = [log_mel(load_wav(w, resample=args.resample), drop_speech_band=drop_speech) for w in train_wavs]

        det = DETECTORS[args.detector]()
        det.fit(train_mels)

        if args.threshold is not None:
            threshold = float(args.threshold)
        else:
            if args.val_dir is not None or args.val_wavs:
                val_wavs = _collect_wavs(args.val_dir, args.val_wavs, "validation")
                val_mels = [
                    log_mel(load_wav(w, resample=args.resample), drop_speech_band=drop_speech) for w in val_wavs
                ]
            else:
                val_mels = train_mels
            threshold = calibrate_threshold(det, val_mels, quantile=args.quantile)

        events = monitor_steps(
            step_wavs,
            det,
            threshold=threshold,
            protocol_id=args.protocol_id,
            start_step=args.step,
            instrument_id=args.instrument_id,
            mic=args.mic,
            drop_speech_band=drop_speech,
            resample=args.resample,
            save_features_dir=args.save_features_dir,
            out_jsonl=args.out_jsonl,
            notes=args.notes,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for ev in events:
        print(ev.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
