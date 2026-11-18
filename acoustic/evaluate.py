"""Score DCASE 2023 Task 2 dev machines with the unsupervised detectors.

    python -m acoustic.fetch_dcase2023 --machines fan valve bearing
    python -m acoustic.evaluate --data data/dcase2023 --machines fan valve bearing

Training uses only normal clips from each machine's train/ folder. Test clips are
scored and compared with their file-name labels. Metrics follow the DCASE 2023
Task 2 rules: AUC per domain (source, target) and pAUC (max FPR 0.1) over both.
Latency is measured on one CPU thread for one 10 s clip: read, log-mel, score.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import hmean  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .detect import DETECTORS, featurize  # noqa: E402
from .features import load_wav, log_mel  # noqa: E402


def metrics(names, scores):
    y = np.array(["_anomaly_" in n for n in names], dtype=int)
    dom = np.array(["_source_" in n for n in names])
    s = np.asarray(scores)
    return {
        "auc_source": roc_auc_score(y[dom], s[dom]),
        "auc_target": roc_auc_score(y[~dom], s[~dom]),
        "pauc": roc_auc_score(y, s, max_fpr=0.1),
    }


def run(data: Path, machines, detectors, drop_speech_band, latency_clips=50):
    rows = []
    for m in machines:
        train = sorted((data / m / "train").glob("*.wav"))
        test = sorted((data / m / "test").glob("*.wav"))
        t0 = time.perf_counter()
        tr_mel = featurize(train, drop_speech_band)
        te_mel = featurize(test, drop_speech_band)
        feat_s = time.perf_counter() - t0
        for name in detectors:
            det = DETECTORS[name]()
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
            r.update(machine=m, detector=name, drop_speech_band=drop_speech_band, n_train=len(train),
                     n_test=len(test), fit_s=round(fit_s, 1), feature_s_all_clips=round(feat_s, 1),
                     latency_ms_median=round(float(np.median(lat)), 1),
                     latency_ms_p95=round(float(np.percentile(lat, 95)), 1))
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
    ap.add_argument("--out", type=Path, default=Path("results/dcase2023_cpu.json"))
    args = ap.parse_args(argv)
    modes = {"keep": [False], "drop": [True], "both": [False, True]}[args.speech_band]
    rows = []
    for mode in modes:
        rows += run(args.data, args.machines, args.detectors, mode)
    res = {"dataset": "DCASE 2023 Task 2 dev (Zenodo 7882613, CC BY 4.0)", "cpu": platform.processor() or platform.machine(),
           "threads": 1, "rows": rows, "summary": summary(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
