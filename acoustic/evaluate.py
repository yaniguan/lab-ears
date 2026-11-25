"""Score DCASE 2023 Task 2 dev machines with the unsupervised detectors.

    python -m acoustic.fetch_dcase2023 --machines fan valve bearing
    python -m acoustic.evaluate --data data/dcase2023 --machines fan valve bearing
    python -m acoustic.evaluate --data data/dcase2023 --streaming-window-s 2.0

Training uses only normal clips from each machine's train/ folder. Test clips are
scored and compared with their file-name labels. Metrics follow the DCASE 2023
Task 2 rules: AUC per domain (source, target) and pAUC (max FPR 0.1) over both.
Latency is measured on one CPU thread for one 10 s clip: read, log-mel, score.
When `--streaming-window-s` is enabled, causal sub-window chunk latency, median
detection delay on detected anomalies, and false-alarms per hour on normal test
streams are also evaluated using out-of-fold conformal threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import hmean  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from .detect import DETECTORS, POOLING_MODES, calibrate_with_mode, featurize  # noqa: E402
from .features import SR, load_wav, log_mel  # noqa: E402


def _require_binary_labels(y_sub: np.ndarray, split_name: str) -> None:
    if y_sub.size == 0:
        raise ValueError(
            f"No test clips matched {split_name}; expected DCASE filenames containing '_source_' or '_target_' "
            f"and '_normal_' or '_anomaly_'."
        )
    classes = set(np.unique(y_sub).tolist())
    if classes != {0, 1}:
        present = "anomaly only" if classes == {1} else "normal only"
        raise ValueError(
            f"Cannot compute ROC-AUC for {split_name}: found {present} ({y_sub.size} clips). "
            f"Both normal and '_anomaly_' test WAVs are required."
        )


def metrics(names, scores):
    if len(names) == 0:
        raise ValueError("Cannot compute metrics on an empty test set")
    if len(names) != len(scores):
        raise ValueError(f"Length mismatch: {len(names)} names vs {len(scores)} scores")
    y = np.array(["_anomaly_" in n for n in names], dtype=int)
    dom = np.array(["_source_" in n for n in names], dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("Non-finite anomaly scores encountered in metrics()")
    _require_binary_labels(y, "overall test set")
    _require_binary_labels(y[dom], "source domain ('_source_' clips)")
    _require_binary_labels(y[~dom], "target domain (non-'_source_' clips)")
    return {
        "auc_source": float(roc_auc_score(y[dom], s[dom])),
        "auc_target": float(roc_auc_score(y[~dom], s[~dom])),
        "pauc": float(roc_auc_score(y, s, max_fpr=0.1)),
    }


def _validate_machine_dirs(data: Path, machine: str):
    if not data.exists() or not data.is_dir():
        raise FileNotFoundError(
            f"Dataset root '{data}' does not exist. Fetch DCASE 2023 clips first:\n"
            f"  python -m acoustic.fetch_dcase2023 --machines {machine} --out {data}"
        )
    train_dir = data / machine / "train"
    test_dir = data / machine / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Missing '{train_dir}' or '{test_dir}'. Fetch machine '{machine}' first:\n"
            f"  python -m acoustic.fetch_dcase2023 --machines {machine} --out {data}"
        )
    train = sorted(train_dir.glob("*.wav"))
    test = sorted(test_dir.glob("*.wav"))
    if not train:
        raise ValueError(f"No training .wav files found in '{train_dir}'.")
    if not test:
        raise ValueError(f"No test .wav files found in '{test_dir}'.")
    test_names = [p.name for p in test]
    _require_binary_labels(np.array(["_anomaly_" in n for n in test_names], dtype=int), f"{machine}/test")
    dom = np.array(["_source_" in n for n in test_names], dtype=bool)
    _require_binary_labels(np.array(["_anomaly_" in n for n in test_names], dtype=int)[dom], f"{machine}/test source")
    _require_binary_labels(np.array(["_anomaly_" in n for n in test_names], dtype=int)[~dom], f"{machine}/test target")
    return train, test


def _evaluate_streaming_chunks(
    det,
    tr_mel,
    test_paths,
    drop_speech_band: bool,
    window_s: float,
    hop_s: float,
    quantile: float = 0.99,
) -> dict:
    if window_s <= 0.0 or hop_s <= 0.0:
        raise ValueError("streaming_window_s and streaming_hop_s must be > 0")
    cal_subset = tr_mel[: min(len(tr_mel), 40)]
    thr, _ = calibrate_with_mode(det, cal_subset, quantile=quantile, force_oof=True)

    win_n = max(1, int(round(window_s * SR)))
    hop_n = max(1, int(round(hop_s * SR)))
    chunk_lats = []
    delays_s = []
    false_alarm_chunks = 0
    normal_duration_s = 0.0
    stream_scores = []

    for p in test_paths:
        wav = load_wav(p)
        is_anom = "_anomaly_" in p.name
        if wav.size <= win_n:
            starts = [0]
        else:
            starts = list(range(0, wav.size - win_n + 1, hop_n))
            if starts[-1] != wav.size - win_n:
                starts.append(wav.size - win_n)

        clip_chunk_scores = []
        first_flag_s: Optional[float] = None
        for s_idx in starts:
            chunk = wav[s_idx : s_idx + win_n]
            t0 = time.perf_counter()
            sc = float(det.score(log_mel(chunk, drop_speech_band)))
            chunk_lats.append(1000.0 * (time.perf_counter() - t0))
            clip_chunk_scores.append(sc)
            end_time_s = float(min(wav.size, s_idx + win_n)) / float(SR)
            if sc > thr:
                if is_anom and first_flag_s is None:
                    first_flag_s = end_time_s
                elif not is_anom:
                    false_alarm_chunks += 1

        if is_anom and first_flag_s is not None:
            delays_s.append(first_flag_s)
        if not is_anom:
            normal_duration_s += float(wav.size) / float(SR)
        stream_scores.append(float(max(clip_chunk_scores)))

    fa_per_hour = (
        round(3600.0 * float(false_alarm_chunks) / normal_duration_s, 3)
        if normal_duration_s > 0.0
        else 0.0
    )
    return {
        "streaming_window_s": float(window_s),
        "streaming_hop_s": float(hop_s),
        "chunk_latency_ms_median": round(float(np.median(chunk_lats)), 2),
        "chunk_latency_ms_p95": round(float(np.percentile(chunk_lats, 95)), 2),
        "detection_delay_s_median": round(float(np.median(delays_s)), 3) if delays_s else None,
        "false_alarms_per_hour": fa_per_hour,
        "streaming_scores": stream_scores,
    }


def run(
    data: Path,
    machines,
    detectors,
    drop_speech_band,
    latency_clips=50,
    *,
    pooling: str = "mean",
    streaming_window_s: Optional[float] = None,
    streaming_hop_s: float = 1.0,
):
    rows = []
    with threadpool_limits(limits=1):
        for m in machines:
            train, test = _validate_machine_dirs(data, m)
            t0 = time.perf_counter()
            tr_mel = featurize(train, drop_speech_band)
            te_mel = featurize(test, drop_speech_band)
            feat_s = time.perf_counter() - t0
            for name in detectors:
                det = DETECTORS[name](pooling=pooling)
                t0 = time.perf_counter()
                det.fit(tr_mel)
                fit_s = time.perf_counter() - t0
                scores = [det.score(x) for x in te_mel]
                r = {k: round(float(v), 4) for k, v in metrics([p.name for p in test], scores).items()}
                lat = []
                for p in test[:latency_clips]:
                    t0 = time.perf_counter()
                    det.score(log_mel(load_wav(p), drop_speech_band))
                    lat.append(1000 * (time.perf_counter() - t0))
                r.update(
                    machine=m,
                    detector=name,
                    pooling=pooling,
                    drop_speech_band=drop_speech_band,
                    n_train=len(train),
                    n_test=len(test),
                    fit_s=round(fit_s, 1),
                    feature_s_all_clips=round(feat_s, 1),
                    latency_ms_median=round(float(np.median(lat)), 1),
                    latency_ms_p95=round(float(np.percentile(lat, 95)), 1),
                )
                if streaming_window_s is not None:
                    s_info = _evaluate_streaming_chunks(
                        det,
                        tr_mel,
                        test,
                        drop_speech_band=drop_speech_band,
                        window_s=streaming_window_s,
                        hop_s=streaming_hop_s,
                    )
                    s_metrics = metrics([p.name for p in test], s_info.pop("streaming_scores"))
                    r.update(
                        s_info,
                        streaming_pauc=round(float(s_metrics["pauc"]), 4),
                    )
                rows.append(r)
                print(json.dumps(r), flush=True)
    return rows


def summary(rows):
    out = []
    for key in sorted({(r["detector"], r["drop_speech_band"]) for r in rows}):
        rs = [r for r in rows if (r["detector"], r["drop_speech_band"]) == key]
        vals = [v for r in rs for v in (r["auc_source"], r["auc_target"], r["pauc"])]
        out.append({"detector": key[0], "drop_speech_band": key[1], "hmean_all": round(float(hmean(vals)), 4)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/dcase2023"))
    ap.add_argument("--machines", nargs="+", default=["fan", "valve", "bearing"])
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    ap.add_argument("--speech-band", choices=["keep", "drop", "both"], default="both")
    ap.add_argument("--pooling", choices=list(POOLING_MODES), default="mean")
    ap.add_argument("--streaming-window-s", type=float, default=None, help="Optional causal streaming window in seconds")
    ap.add_argument("--streaming-hop-s", type=float, default=1.0, help="Causal streaming hop in seconds")
    ap.add_argument("--out", type=Path, default=Path("results/dcase2023_cpu.json"))
    args = ap.parse_args(argv)
    modes = {"keep": [False], "drop": [True], "both": [False, True]}[args.speech_band]
    rows = []
    try:
        for mode in modes:
            rows += run(
                args.data,
                args.machines,
                args.detectors,
                mode,
                pooling=args.pooling,
                streaming_window_s=args.streaming_window_s,
                streaming_hop_s=args.streaming_hop_s,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    res = {
        "dataset": "DCASE 2023 Task 2 dev (Zenodo 7882613, CC BY 4.0)",
        "cpu": platform.processor() or platform.machine(),
        "threads": 1,
        "pooling": args.pooling,
        "speech_band_filter_mode": "full_hann_skirt_plus_pre_stft_notch",
        "rows": rows,
        "summary": summary(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
