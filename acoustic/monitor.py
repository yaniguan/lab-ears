"""Live step acoustic monitor: fit/calibrate or load a pre-calibrated anomaly detector,
score protocol step WAV recordings or live PCM streams, and emit validated
`lab-ears/audio-event/v1` JSONL ledger lines for `lab-rehearsal` and `lab-verify`.

Usage:
    # Fit + out-of-fold conformal calibrate once and persist a safe non-pickle bundle:
    python -m acoustic.monitor \\
        --train-dir path/to/normal_clips \\
        --save-model runs/centrifuge01_model.npz

    # Score live protocol steps without per-step retraining:
    python -m acoustic.monitor \\
        --load-model runs/centrifuge01_model.npz \\
        --wav path/to/step4.wav \\
        --protocol-id pcr-setup-v2 --step 4 --instrument-id centrifuge-01 \\
        --out-jsonl runs/audio_events.jsonl
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
from scipy.signal import resample_poly

from .detect import (
    DETECTORS,
    POOLING_MODES,
    calibrate_streaming_with_mode,
    calibrate_with_mode,
    load_detector_bundle,
    save_detector_bundle,
)
from .features import SR, load_wav_info, log_mel, retained_mel_bands_hz
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
    allow_upsample: bool = False,
    save_features_path: Optional[Path | str] = None,
    start_utc: Optional[str] = None,
    notes: str = "",
) -> AudioEvent:
    """Score one WAV clip with a fitted detector and return a validated AudioEvent."""
    wav_path = Path(wav_path)
    min_src_sr = None if allow_upsample else SR
    x, src_sr = load_wav_info(
        wav_path,
        target_sr=SR,
        resample=resample,
        min_source_sr=min_src_sr,
    )
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
        source_sample_rate_hz=src_sr,
        mic=mic,
        features_path=feat_str,
        speech_band_dropped=drop_speech_band,
        raw_audio_kept=False,
        raw_audio_saved=False,
        retained_bands_hz=retained_mel_bands_hz(drop_speech_band=drop_speech_band, sr=SR),
        notes=notes,
    )
    event.validate()
    return event


def monitor_stream(
    pcm_chunks: Iterable[np.ndarray],
    detector,
    *,
    threshold: float,
    protocol_id: str,
    step: int,
    instrument_id: str,
    sample_rate_hz: int = SR,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    mic: str = "contact",
    drop_speech_band: bool = True,
    allow_upsample: bool = False,
    allow_window_mismatch: bool = False,
    bundle_meta: Optional[Mapping[str, Any]] = None,
    flush_tail: bool = True,
    out_jsonl: Optional[Path] = None,
    notes: str = "",
) -> Iterator[AudioEvent]:
    """Causal PCM ring-buffer generator scoring live streaming chunks without raw-audio storage."""
    if isinstance(detector, tuple) and len(detector) == 2 and isinstance(detector[1], Mapping):
        det_obj, meta_obj = detector
    else:
        det_obj, meta_obj = detector, bundle_meta

    src_sr = int(sample_rate_hz)
    if src_sr <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz!r}")
    if src_sr < SR and not allow_upsample:
        raise ValueError(
            f"Streaming source sample rate {src_sr} Hz is below {SR} Hz; upsampling cannot recover high-band faults"
        )
    if window_s <= 0.0 or hop_s <= 0.0 or not (math.isfinite(window_s) and math.isfinite(hop_s)):
        raise ValueError(f"window_s ({window_s}) and hop_s ({hop_s}) must be finite and > 0")

    cal_win_s = (
        meta_obj.get("calibration_window_s")
        if meta_obj is not None
        else getattr(det_obj, "calibration_window_s_", None)
    )
    has_bundle = meta_obj is not None or bool(getattr(det_obj, "_loaded_from_bundle_", False))
    if not allow_window_mismatch:
        if cal_win_s is not None:
            if not math.isfinite(float(cal_win_s)) or abs(float(cal_win_s) - float(window_s)) > 1e-6:
                raise ValueError(
                    f"Detector calibration_window_s={cal_win_s}s does not match monitor_stream "
                    f"window_s={window_s}s; recalibrate with --window-s {window_s} or pass "
                    "allow_window_mismatch=True"
                )
        elif has_bundle:
            raise ValueError(
                f"Detector bundle lacks calibration_window_s metadata for monitor_stream(window_s={window_s}s); "
                f"recalibrate with --window-s {window_s} or pass allow_window_mismatch=True"
            )

    win_samples = max(1, int(round(window_s * SR)))
    hop_samples = max(1, int(round(hop_s * SR)))

    def _emit_window(buf: np.ndarray, idx: int) -> AudioEvent:
        mel = log_mel(buf, drop_speech_band=drop_speech_band)
        score = float(det_obj.score(mel))
        win_note = f"stream_window={idx}" if not notes else f"{notes}; stream_window={idx}"
        ev = AudioEvent(
            protocol_id=protocol_id,
            step=step,
            instrument_id=instrument_id,
            start_utc=now_utc(),
            duration_s=round(float(len(buf)) / float(SR), 6),
            detector=getattr(det_obj, "name", type(det_obj).__name__.lower()),
            score=score,
            threshold=float(threshold),
            sample_rate_hz=SR,
            source_sample_rate_hz=src_sr,
            mic=mic,
            features_path=None,
            speech_band_dropped=drop_speech_band,
            raw_audio_kept=False,
            raw_audio_saved=False,
            retained_bands_hz=retained_mel_bands_hz(drop_speech_band=drop_speech_band, sr=SR),
            notes=win_note,
        )
        ev.validate()
        if out_jsonl is not None:
            out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with out_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(ev.to_json() + "\n")
        return ev

    def _stream_iter() -> Iterator[AudioEvent]:
        ring = np.empty(0, dtype=np.float32)
        emitted = 0
        consumed_since_emit = 0
        for chunk in pcm_chunks:
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1, dtype=np.float32)
            arr = arr.ravel()
            if arr.size == 0:
                continue
            if not np.isfinite(arr).all():
                raise ValueError("Streaming PCM chunk contains NaN or Inf")
            if src_sr != SR:
                g = math.gcd(SR, src_sr)
                arr = resample_poly(arr, SR // g, src_sr // g).astype(np.float32)

            ring = np.concatenate([ring, arr])
            consumed_since_emit += int(arr.size)

            while ring.size >= win_samples:
                ev = _emit_window(ring[:win_samples], emitted)
                emitted += 1
                consumed_since_emit = 0
                yield ev
                ring = ring[hop_samples:]

        if flush_tail and ring.size > 0 and (emitted == 0 or consumed_since_emit > 0):
            yield _emit_window(ring, emitted)

    return _stream_iter()


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
    allow_upsample: bool = False,
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
            allow_upsample=allow_upsample,
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
    ap.add_argument("--save-model", type=Path, default=None, help="Save calibrated detector bundle (.npz + .json)")
    ap.add_argument("--load-model", type=Path, default=None, help="Load pre-calibrated detector bundle (.npz)")
    ap.add_argument("--wav", nargs="*", type=Path, default=None, help="Step WAV clip(s) to score")
    ap.add_argument("--detector", choices=list(DETECTORS), default="gaussian")
    ap.add_argument("--pooling", choices=list(POOLING_MODES), default="max_window")
    ap.add_argument("--threshold", type=float, default=None, help="Explicit threshold (overrides --quantile)")
    ap.add_argument("--quantile", type=float, default=0.99, help="Validation score quantile for threshold calibration")
    ap.add_argument(
        "--window-s",
        type=float,
        default=None,
        help="Optional causal streaming window in seconds for chunk-matched calibration",
    )
    ap.add_argument(
        "--hop-s",
        type=float,
        default=1.0,
        help="Causal streaming hop in seconds when --window-s or --stream is set",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Score --wav files via causal monitor_stream() chunks instead of full-clip monitor_steps()",
    )
    ap.add_argument(
        "--allow-window-mismatch",
        action="store_true",
        help="Allow monitor_stream() to score a bundle calibrated with a different window_s",
    )
    ap.add_argument("--protocol-id", default="calibration-only")
    ap.add_argument("--step", type=int, default=0, help="Starting 0-based protocol step index")
    ap.add_argument("--instrument-id", default="benchtop-01")
    ap.add_argument("--mic", default="contact")
    ap.add_argument("--speech-band", choices=["drop", "keep"], default="drop")
    ap.add_argument("--resample", action="store_true", help="Resample non-16kHz WAVs (>= 16 kHz) to 16 kHz")
    ap.add_argument("--allow-upsample", action="store_true", help="Allow resampling source WAVs recorded below 16 kHz")
    ap.add_argument("--save-features-dir", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None, help="Append validated AudioEvent JSONL lines to file")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    drop_speech = args.speech_band == "drop"
    min_src_sr = None if args.allow_upsample else SR

    try:
        if not args.wav and args.save_model is None:
            raise ValueError("Provide --wav step clip(s) to score and/or --save-model to persist a calibration bundle")

        if args.load_model is not None:
            det, meta = load_detector_bundle(args.load_model)
            loaded_drop = meta.get("drop_speech_band")
            if loaded_drop is not None and bool(loaded_drop) != drop_speech:
                raise ValueError(
                    f"Loaded model bundle has drop_speech_band={loaded_drop}, which conflicts with --speech-band {args.speech_band}"
                )
            threshold = float(args.threshold) if args.threshold is not None else float(meta["threshold"])
            calib_mode = "explicit" if args.threshold is not None else str(meta.get("calibration_mode", "bundle"))
            cal_win_s = meta.get("calibration_window_s")
            cal_hop_s = meta.get("calibration_hop_s")
        else:
            train_wavs = _collect_wavs(args.train_dir, args.train_wavs, "training")
            train_wavs_arr = [
                load_wav_info(w, resample=args.resample, min_source_sr=min_src_sr)[0]
                for w in train_wavs
            ]
            train_mels = [log_mel(x, drop_speech_band=drop_speech) for x in train_wavs_arr]
            det = DETECTORS[args.detector](pooling=args.pooling)
            det.fit(train_mels)

            eff_stream_win_s = args.window_s if args.window_s is not None else (2.0 if args.stream else None)
            if args.threshold is not None:
                threshold = float(args.threshold)
                calib_mode = "explicit"
                cal_win_s = (
                    float(eff_stream_win_s)
                    if eff_stream_win_s is not None
                    else round(float(np.median([len(x) / float(SR) for x in train_wavs_arr])), 6)
                )
                cal_hop_s = float(args.hop_s) if eff_stream_win_s is not None else None
                det.calibration_window_s_ = cal_win_s
                det.calibration_hop_s_ = cal_hop_s
            elif eff_stream_win_s is not None:
                if args.val_dir is not None or args.val_wavs:
                    val_wavs = _collect_wavs(args.val_dir, args.val_wavs, "validation")
                    val_wavs_arr = [
                        load_wav_info(w, resample=args.resample, min_source_sr=min_src_sr)[0]
                        for w in val_wavs
                    ]
                    threshold, calib_mode = calibrate_streaming_with_mode(
                        det,
                        window_s=eff_stream_win_s,
                        hop_s=args.hop_s,
                        quantile=args.quantile,
                        drop_speech_band=drop_speech,
                        val_wavs=val_wavs_arr,
                    )
                else:
                    threshold, calib_mode = calibrate_streaming_with_mode(
                        det,
                        window_s=eff_stream_win_s,
                        hop_s=args.hop_s,
                        quantile=args.quantile,
                        drop_speech_band=drop_speech,
                        train_mels=train_mels,
                        train_wavs=train_wavs_arr,
                    )
                cal_win_s = float(eff_stream_win_s)
                cal_hop_s = float(args.hop_s)
            elif args.val_dir is not None or args.val_wavs:
                val_wavs = _collect_wavs(args.val_dir, args.val_wavs, "validation")
                val_wavs_arr = [
                    load_wav_info(w, resample=args.resample, min_source_sr=min_src_sr)[0]
                    for w in val_wavs
                ]
                val_mels = [log_mel(x, drop_speech_band=drop_speech) for x in val_wavs_arr]
                threshold, calib_mode = calibrate_with_mode(det, val_mels, quantile=args.quantile, force_oof=False)
                cal_win_s = round(float(np.median([len(x) / float(SR) for x in val_wavs_arr])), 6)
                cal_hop_s = None
                det.calibration_window_s_ = cal_win_s
                det.calibration_hop_s_ = None
            else:
                threshold, calib_mode = calibrate_with_mode(det, train_mels, quantile=args.quantile, force_oof=True)
                cal_win_s = round(float(np.median([len(x) / float(SR) for x in train_wavs_arr])), 6)
                cal_hop_s = None
                det.calibration_window_s_ = cal_win_s
                det.calibration_hop_s_ = None

        if args.save_model is not None:
            save_detector_bundle(
                args.save_model,
                det,
                threshold=threshold,
                calibration_mode=calib_mode,
                quantile=args.quantile,
                calibration_window_s=cal_win_s,
                calibration_hop_s=cal_hop_s,
            )

        events: List[AudioEvent] = []
        if args.wav:
            step_wavs = _collect_wavs(None, args.wav, "step")
            if args.stream:
                stream_win_s = float(args.window_s) if args.window_s is not None else 2.0
                for offset, wp in enumerate(step_wavs):
                    x_step, src_sr_step = load_wav_info(
                        wp,
                        target_sr=SR,
                        resample=args.resample,
                        min_source_sr=min_src_sr,
                    )
                    events.extend(
                        monitor_stream(
                            [x_step],
                            det,
                            threshold=threshold,
                            protocol_id=args.protocol_id,
                            step=args.step + offset,
                            instrument_id=args.instrument_id,
                            sample_rate_hz=SR if args.resample else src_sr_step,
                            window_s=stream_win_s,
                            hop_s=args.hop_s,
                            mic=args.mic,
                            drop_speech_band=drop_speech,
                            allow_upsample=args.allow_upsample,
                            allow_window_mismatch=args.allow_window_mismatch,
                            out_jsonl=args.out_jsonl,
                            notes=args.notes,
                        )
                    )
            else:
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
                    allow_upsample=args.allow_upsample,
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
