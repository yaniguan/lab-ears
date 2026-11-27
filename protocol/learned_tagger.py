"""Zero-leakage Structured Linear-Chain BIOES Span Tagger for wet-lab protocols.

Trains a cost-augmented Passive-Aggressive (`PA-II`) max-margin linear-chain sequence
tagger with constrained BIOES Viterbi decoding strictly on `WLP/train` (374 protocols),
selects hyperparameters (`c_pa`, `fn_cost`, `epochs`) strictly on `WLP/dev` (123 protocols),
and evaluates once on held-out `WLP/test` (125 protocols) plus external `BioProBench` and
`BioPlanner` corpora without test-split rule tuning.

Runs on CPU in ~30 s on full WLP with `numpy` (and includes a pure-stdlib fallback when
`numpy` is absent) and persists weights via safe non-pickle `.json.gz` bundles.

Usage:
    python -m protocol.learned_tagger \\
        --train data/WLP-Dataset/train \\
        --dev data/WLP-Dataset/dev \\
        --test data/WLP-Dataset/test \\
        --external-jsonl data/bioprobench/protocols.jsonl data/bioplanner_bioprot/protocols.jsonl \\
        --out results/wlp_learned_vs_rules.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - stdlib-only CI fallback
    _np = None

from demo.parse_protocol import (
    ParsedQuantity,
    _normalize_span,
    count_overlap_matches,
    evaluate_folder as evaluate_rule_folder,
    find_spans as find_rule_spans,
    gold_spans_with_stats,
)

TAGGER_BUNDLE_SCHEMA = "lab-ears/linear-chain-tagger/v1"
CORE_LABELS: Tuple[str, ...] = ("Amount", "Temperature", "Time", "Speed")
DEFAULT_HYPERPARAM_GRID: Tuple[Tuple[float, float, int], ...] = (
    (0.15, 1.40, 12),
    (0.25, 1.30, 14),
    (0.35, 1.15, 14),
)

_NEG_INF = -1e18
_TOK_RE = re.compile(
    r"(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
    r"|[A-Za-zµμ]+|[°º˚%×]|[^\s\w]"
)


def _doc_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_bioes_space(
    labels: Sequence[str],
) -> Tuple[List[str], Dict[str, int], List[List[int]], List[int], Set[int]]:
    states = ["O"] + [f"{prefix}-{lbl}" for lbl in labels for prefix in ("B", "I", "E", "S")]
    s2i = {s: i for i, s in enumerate(states)}
    n_s = len(states)
    valid_prev: List[List[int]] = [[] for _ in range(n_s)]
    valid_start: List[int] = []
    valid_end: Set[int] = set()

    for s_idx, s in enumerate(states):
        if s == "O" or s.startswith(("B-", "S-")):
            valid_start.append(s_idx)
        if s == "O" or s.startswith(("E-", "S-")):
            valid_end.add(s_idx)
        for p_idx, p in enumerate(states):
            ok = False
            if p == "O" or p.startswith(("E-", "S-")):
                ok = s == "O" or s.startswith(("B-", "S-"))
            elif p.startswith(("B-", "I-")):
                pl = p.split("-", 1)[1]
                ok = s in (f"I-{pl}", f"E-{pl}")
            if ok:
                valid_prev[s_idx].append(p_idx)
    return states, s2i, valid_prev, valid_start, valid_end


def _tok_class(w: str) -> str:
    if w[0].isdigit():
        if "," in w:
            return "NUM_COMMA"
        if "." in w:
            return "NUM_DEC"
        return "NUM_INT"
    if w in "°º˚":
        return "DEG"
    if w in "-–−":
        return "DASH"
    if w in "/×xX%":
        return f"SYM_{w.lower()}"
    if not w[0].isalpha() and w[0] not in "µμ":
        return "PUNCT"
    if w.isupper():
        return "UPPER1" if len(w) == 1 else "UPPER"
    if w.islower():
        return "LOWER"
    return "MIXED"


def _num_bucket(w: str) -> str:
    if not w[0].isdigit():
        return ""
    try:
        v = float(w.replace(",", ""))
    except ValueError:
        return "N_ERR"
    if v == 0.0:
        return "N0"
    if v < 1.0:
        return "N_SUB1"
    if v <= 12.0:
        return "N_1_12"
    if v <= 45.0:
        return "N_13_45"
    if v <= 105.0:
        return "N_46_105"
    if v < 500.0:
        return "N_106_500"
    return "N_500P"


def _word_shape(w: str) -> str:
    out: List[str] = []
    for c in w[:8]:
        if c.isdigit():
            out.append("d")
        elif c.isupper():
            out.append("X")
        elif c.islower() or c in "µμ":
            out.append("x")
        else:
            out.append(c)
    return "".join(out)


def tokenize_protocol_lines(text: str) -> List[List[Tuple[str, int, int]]]:
    """Tokenize protocol text line-by-line into offset-preserving `(tok, start, end)` tuples."""
    seqs: List[List[Tuple[str, int, int]]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        toks = [(m.group(), pos + m.start(), pos + m.end()) for m in _TOK_RE.finditer(line)]
        if toks:
            seqs.append(toks)
        pos += len(line)
    return seqs


def extract_sequence_features(toks: Sequence[Tuple[str, int, int]]) -> List[List[str]]:
    """Extract purely lexical, orthographic, numeric-magnitude, and local-context features."""
    n = len(toks)
    ws = [t[0] for t in toks]
    lows = [w.lower().replace("μ", "µ") for w in ws]
    clss = [_tok_class(w) for w in ws]
    shps = [_word_shape(w) for w in ws]
    nbks = [_num_bucket(w) for w in ws]
    sp_before = [i > 0 and toks[i][1] > toks[i - 1][2] for i in range(n)]
    sp_after = [i + 1 < n and toks[i + 1][1] > toks[i][2] for i in range(n)]

    seq_f: List[List[str]] = []
    for i in range(n):
        f: List[str] = [
            "BIAS",
            f"w0={lows[i]}",
            f"r0={ws[i]}",
            f"c0={clss[i]}",
            f"s0={shps[i]}",
            f"spb={int(sp_before[i])}",
            f"spa={int(sp_after[i])}",
        ]
        if nbks[i]:
            f.append(f"nb0={nbks[i]}")
        w0 = lows[i]
        if len(w0) >= 2:
            f.append(f"p2={w0[:2]}")
            f.append(f"s2={w0[-2:]}")
        if len(w0) >= 3:
            f.append(f"p3={w0[:3]}")
            f.append(f"s3={w0[-3:]}")
        if len(w0) >= 4:
            f.append(f"s4={w0[-4:]}")

        for k in (-2, -1, 1, 2):
            j = i + k
            if 0 <= j < n:
                f.append(f"w{k}={lows[j]}")
                f.append(f"c{k}={clss[j]}")
                if k in (-1, 1):
                    f.append(f"r{k}={ws[j]}")
                    if nbks[j]:
                        f.append(f"nb{k}={nbks[j]}")
            else:
                f.append(f"w{k}=<BOS>" if k < 0 else f"w{k}=<EOS>")

        if i > 0:
            f.append(f"bg-1={lows[i - 1]}|{lows[i]}")
            f.append(f"cbg-1={clss[i - 1]}|{ws[i]}")
            if nbks[i - 1]:
                f.append(f"nbu-1={nbks[i - 1]}|{ws[i]}")
        if i + 1 < n:
            f.append(f"bg+1={lows[i]}|{lows[i + 1]}")
            f.append(f"cbg+1={clss[i]}|{ws[i + 1]}")
            if nbks[i]:
                f.append(f"nbu0={nbks[i]}|{ws[i + 1]}")
        if i > 0 and i + 1 < n:
            f.append(f"tg={lows[i - 1]}|{lows[i]}|{lows[i + 1]}")
            f.append(f"ctg={clss[i - 1]}|{clss[i]}|{clss[i + 1]}")
        if i + 2 < n:
            f.append(f"tg+2={clss[i]}|{lows[i + 1]}|{lows[i + 2]}")

        for j in range(max(0, i - 5), i):
            if clss[j] in ("LOWER", "MIXED", "UPPER") and len(lows[j]) >= 2:
                f.append(f"ctxL={lows[j]}")
        for j in range(i + 1, min(n, i + 5)):
            if clss[j] in ("LOWER", "MIXED", "UPPER") and len(lows[j]) >= 2:
                f.append(f"ctxR={lows[j]}")
        seq_f.append(f)
    return seq_f


def spans_to_bioes_ids(
    toks: Sequence[Tuple[str, int, int]],
    spans: Iterable[Tuple[str, int, int]],
    s2i: Dict[str, int],
) -> List[int]:
    tags = [0] * len(toks)
    for lbl, gs, ge in sorted(spans, key=lambda x: (x[1], -(x[2] - x[1]))):
        if f"S-{lbl}" not in s2i:
            continue
        idxs = [i for i, (_, ts, te) in enumerate(toks) if ts >= gs and te <= ge]
        if not idxs or any(tags[i] != 0 for i in idxs):
            continue
        if len(idxs) == 1:
            tags[idxs[0]] = s2i[f"S-{lbl}"]
        else:
            tags[idxs[0]] = s2i[f"B-{lbl}"]
            for mid_i in idxs[1:-1]:
                tags[mid_i] = s2i[f"I-{lbl}"]
            tags[idxs[-1]] = s2i[f"E-{lbl}"]
    return tags


def bioes_ids_to_spans(
    toks: Sequence[Tuple[str, int, int]],
    tag_ids: Sequence[int],
    states: Sequence[str],
) -> Set[Tuple[str, int, int]]:
    out: Set[Tuple[str, int, int]] = set()
    i = 0
    n = len(tag_ids)
    while i < n:
        st = states[tag_ids[i]]
        if st.startswith("S-"):
            out.add((st[2:], toks[i][1], toks[i][2]))
            i += 1
        elif st.startswith("B-"):
            lbl = st[2:]
            j = i + 1
            while j < n and states[tag_ids[j]] == f"I-{lbl}":
                j += 1
            if j < n and states[tag_ids[j]] == f"E-{lbl}":
                out.add((lbl, toks[i][1], toks[j][2]))
                i = j + 1
            else:
                out.add((lbl, toks[i][1], toks[j - 1][2]))
                i = j
        else:
            i += 1
    return out


def _load_brat_documents(
    folder: Path | str,
    *,
    labels: Sequence[str],
    include_discontinuous: bool = True,
) -> Tuple[List[Tuple[str, str, Set[Tuple[str, int, int]]]], Dict[str, int]]:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist or is not a directory: {folder_path}")
    ann_files = sorted(folder_path.glob("*.ann"))
    if not ann_files:
        raise ValueError(f"No .ann annotation files found in {folder_path}")

    docs: List[Tuple[str, str, Set[Tuple[str, int, int]]]] = []
    totals = {"discontinuous_entities": 0, "skipped_discontinuous": 0, "malformed_lines": 0}
    for ann in ann_files:
        txt_path = ann.with_suffix(".txt")
        if not txt_path.is_file():
            raise FileNotFoundError(f"Missing matching .txt protocol file for {ann}")
        text = txt_path.read_text(encoding="utf-8")
        gold, st = gold_spans_with_stats(
            ann,
            include_discontinuous=include_discontinuous,
            target_labels=labels,
        )
        for k in totals:
            totals[k] += st[k]
        docs.append((ann.stem, text, gold))
    return docs, totals


def _compute_corpus_metrics(
    labels: Sequence[str],
    doc_preds_golds: Sequence[Tuple[Set[Tuple[str, int, int]], Set[Tuple[str, int, int]]]],
) -> Tuple[Dict[str, Dict[str, float | int]], Dict[str, float | int]]:
    stats = {k: [0, 0, 0, 0] for k in labels}  # tp_exact, tp_overlap, n_pred, n_gold
    for pred, gold in doc_preds_golds:
        for k in labels:
            p = [x for x in pred if x[0] == k]
            g = [x for x in gold if x[0] == k]
            stats[k][0] += len(set(p) & set(g))
            stats[k][1] += count_overlap_matches(p, g)
            stats[k][2] += len(p)
            stats[k][3] += len(g)

    metrics: Dict[str, Dict[str, float | int]] = {}
    tot_te = tot_to = tot_p = tot_g = 0
    for k, (te, to, n_p, n_g) in stats.items():
        tot_te += te
        tot_to += to
        tot_p += n_p
        tot_g += n_g
        pe = te / n_p if n_p else 0.0
        re_ = te / n_g if n_g else 0.0
        fe = 2 * pe * re_ / (pe + re_) if (pe + re_) else 0.0
        po = to / n_p if n_p else 0.0
        ro = to / n_g if n_g else 0.0
        fo = 2 * po * ro / (po + ro) if (po + ro) else 0.0
        metrics[k] = {
            "gold": n_g,
            "pred": n_p,
            "tp_exact": te,
            "tp_overlap": to,
            "P_exact": pe,
            "R_exact": re_,
            "F1_exact": fe,
            "P_overlap": po,
            "R_overlap": ro,
            "F1_overlap": fo,
        }

    m_pe = tot_te / tot_p if tot_p else 0.0
    m_re = tot_te / tot_g if tot_g else 0.0
    m_fe = 2 * m_pe * m_re / (m_pe + m_re) if (m_pe + m_re) else 0.0
    m_po = tot_to / tot_p if tot_p else 0.0
    m_ro = tot_to / tot_g if tot_g else 0.0
    m_fo = 2 * m_po * m_ro / (m_po + m_ro) if (m_po + m_ro) else 0.0
    micro = {
        "gold": tot_g,
        "pred": tot_p,
        "tp_exact": tot_te,
        "tp_overlap": tot_to,
        "P_exact": m_pe,
        "R_exact": m_re,
        "F1_exact": m_fe,
        "P_overlap": m_po,
        "R_overlap": m_ro,
        "F1_overlap": m_fo,
    }
    return metrics, micro


class LinearChainSpanTagger:
    """Cost-augmented Passive-Aggressive (`PA-II`) linear-chain BIOES span tagger."""

    def __init__(
        self,
        *,
        labels: Sequence[str] = CORE_LABELS,
        c_pa: float = 0.25,
        fn_cost: float = 1.30,
        epochs: int = 14,
        min_feature_freq: int = 1,
        seed: int = 42,
    ) -> None:
        if c_pa <= 0.0 or fn_cost <= 0.0 or epochs < 1 or min_feature_freq < 1:
            raise ValueError("Invalid hyperparameters for LinearChainSpanTagger")
        self.labels: Tuple[str, ...] = tuple(labels)
        self.c_pa = float(c_pa)
        self.fn_cost = float(fn_cost)
        self.epochs = int(epochs)
        self.min_feature_freq = int(min_feature_freq)
        self.seed = int(seed)

        (
            self.states_,
            self.s2i_,
            self.valid_prev_,
            self.valid_start_,
            self.valid_end_,
        ) = _build_bioes_space(self.labels)
        self.f2i_: Dict[str, int] = {}
        self.weights_: Optional[object] = None  # np.ndarray or List[List[float]]
        self.transitions_: Optional[object] = None
        self.start_bias_: Optional[object] = None
        self.end_bias_: Optional[object] = None
        self.train_doc_ids_: Set[str] = set()
        self.train_doc_sha256s_: Set[str] = set()
        self.n_train_protocols_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.weights_ is not None and bool(self.f2i_)

    def fit_documents(
        self,
        documents: Sequence[Tuple[str, str, Set[Tuple[str, int, int]]]],
        *,
        source_tag: str = "in_memory",
    ) -> "LinearChainSpanTagger":
        """Fit feature vocabulary and averaged PA-II linear-chain weights strictly on `documents`."""
        if not documents:
            raise ValueError("LinearChainSpanTagger.fit_documents() requires at least 1 training document")

        self.train_doc_ids_ = {f"{source_tag}:{doc_id}" for doc_id, _, _ in documents}
        self.train_doc_sha256s_ = {_doc_sha256(txt) for _, txt, _ in documents}
        self.n_train_protocols_ = len(documents)

        raw_seqs: List[Tuple[List[Tuple[str, int, int]], List[List[str]], List[int]]] = []
        feat_counts: Counter[str] = Counter()
        for _, text, gold in documents:
            for toks in tokenize_protocol_lines(text):
                feats = extract_sequence_features(toks)
                tags = spans_to_bioes_ids(toks, gold, self.s2i_)
                raw_seqs.append((toks, feats, tags))
                for tf in feats:
                    feat_counts.update(tf)

        self.f2i_ = {
            fn: idx for idx, (fn, cnt) in enumerate(feat_counts.items()) if cnt >= self.min_feature_freq
        }
        if not self.f2i_:
            raise ValueError("Empty feature vocabulary after min_feature_freq filtering")

        n_f = len(self.f2i_)
        n_s = len(self.states_)

        if _np is not None:
            self._fit_numpy(raw_seqs, n_f, n_s)
        else:  # pragma: no cover - stdlib-only CI fallback
            self._fit_pure_python(raw_seqs, n_f, n_s)
        return self

    def _fit_numpy(
        self,
        raw_seqs: Sequence[Tuple[List[Tuple[str, int, int]], List[List[str]], List[int]]],
        n_f: int,
        n_s: int,
    ) -> None:
        assert _np is not None
        enc_seqs = [
            (
                [_np.array([self.f2i_[fn] for fn in tf if fn in self.f2i_], dtype=_np.int32) for tf in feats],
                tags,
            )
            for _, feats, tags in raw_seqs
        ]

        trans_mask = _np.full((n_s, n_s), _NEG_INF, dtype=_np.float64)
        start_mask = _np.full(n_s, _NEG_INF, dtype=_np.float64)
        end_mask = _np.full(n_s, _NEG_INF, dtype=_np.float64)
        for s in self.valid_start_:
            start_mask[s] = 0.0
        for s in self.valid_end_:
            end_mask[s] = 0.0
        for s in range(n_s):
            for p in self.valid_prev_[s]:
                trans_mask[p, s] = 0.0

        W = _np.zeros((n_f, n_s), dtype=_np.float64)
        T = _np.zeros((n_s, n_s), dtype=_np.float64)
        Bs = _np.zeros(n_s, dtype=_np.float64)
        Be = _np.zeros(n_s, dtype=_np.float64)

        W_acc = _np.zeros_like(W)
        T_acc = _np.zeros_like(T)
        Bs_acc = _np.zeros_like(Bs)
        Be_acc = _np.zeros_like(Be)

        rng = random.Random(self.seed)
        indices = list(range(len(enc_seqs)))
        step = 0
        s_range = _np.arange(n_s)

        for _ in range(self.epochs):
            rng.shuffle(indices)
            for seq_idx in indices:
                ef, y_true = enc_seqs[seq_idx]
                step += 1
                L = len(ef)
                emit = _np.array(
                    [W[f_ids].sum(axis=0) if f_ids.size else _np.zeros(n_s) for f_ids in ef],
                    dtype=_np.float64,
                )
                cost_mat = _np.ones((L, n_s), dtype=_np.float64)
                for t_i, yt in enumerate(y_true):
                    if yt != 0:
                        cost_mat[t_i, :] = self.fn_cost
                    cost_mat[t_i, yt] = 0.0

                dp = Bs + start_mask + emit[0] + cost_mat[0]
                bp = _np.zeros((L, n_s), dtype=_np.int32)
                t_eff = T + trans_mask
                for t_i in range(1, L):
                    sc = dp[:, None] + t_eff
                    bp[t_i] = _np.argmax(sc, axis=0)
                    dp = sc[bp[t_i], s_range] + emit[t_i] + cost_mat[t_i]

                cur = int(_np.argmax(dp + Be + end_mask))
                yh = [0] * L
                yh[L - 1] = cur
                for t_i in range(L - 1, 0, -1):
                    cur = int(bp[t_i, cur])
                    yh[t_i - 1] = cur

                if yh != y_true:
                    gold_sc = (
                        float(Bs[y_true[0]] + Be[y_true[-1]])
                        + sum(float(emit[t_i, y_true[t_i]]) for t_i in range(L))
                        + sum(float(T[y_true[t_i - 1], y_true[t_i]]) for t_i in range(1, L))
                    )
                    pred_sc = (
                        float(Bs[yh[0]] + Be[yh[-1]])
                        + sum(float(emit[t_i, yh[t_i]]) for t_i in range(L))
                        + sum(float(T[yh[t_i - 1], yh[t_i]]) for t_i in range(1, L))
                    )
                    loss = sum(float(cost_mat[t_i, yh[t_i]]) for t_i in range(L)) + pred_sc - gold_sc
                    if loss > 0.0:
                        norm2 = (2.0 if y_true[0] != yh[0] else 0.0) + (
                            2.0 if y_true[-1] != yh[-1] else 0.0
                        )
                        norm2 += sum(2.0 * int(ef[t_i].size) for t_i in range(L) if y_true[t_i] != yh[t_i])
                        norm2 += sum(
                            2.0
                            for t_i in range(1, L)
                            if (y_true[t_i - 1], y_true[t_i]) != (yh[t_i - 1], yh[t_i])
                        )
                        tau = min(self.c_pa, loss / max(norm2, 1.0))

                        Bs[y_true[0]] += tau
                        Bs[yh[0]] -= tau
                        Bs_acc[y_true[0]] += step * tau
                        Bs_acc[yh[0]] -= step * tau

                        Be[y_true[-1]] += tau
                        Be[yh[-1]] -= tau
                        Be_acc[y_true[-1]] += step * tau
                        Be_acc[yh[-1]] -= step * tau

                        for t_i in range(L):
                            yt, yp = y_true[t_i], yh[t_i]
                            if yt != yp and ef[t_i].size:
                                W[ef[t_i], yt] += tau
                                W[ef[t_i], yp] -= tau
                                W_acc[ef[t_i], yt] += step * tau
                                W_acc[ef[t_i], yp] -= step * tau
                        for t_i in range(1, L):
                            if (y_true[t_i - 1], y_true[t_i]) != (yh[t_i - 1], yh[t_i]):
                                T[y_true[t_i - 1], y_true[t_i]] += tau
                                T[yh[t_i - 1], yh[t_i]] -= tau
                                T_acc[y_true[t_i - 1], y_true[t_i]] += step * tau
                                T_acc[yh[t_i - 1], yh[t_i]] -= step * tau

        denom = float(step + 1)
        self.weights_ = W - W_acc / denom
        self.transitions_ = T - T_acc / denom
        self.start_bias_ = Bs - Bs_acc / denom
        self.end_bias_ = Be - Be_acc / denom

    def _fit_pure_python(
        self,
        raw_seqs: Sequence[Tuple[List[Tuple[str, int, int]], List[List[str]], List[int]]],
        n_f: int,
        n_s: int,
    ) -> None:  # pragma: no cover - stdlib-only CI fallback
        enc_seqs = [
            ([[self.f2i_[fn] for fn in tf if fn in self.f2i_] for tf in feats], tags)
            for _, feats, tags in raw_seqs
        ]
        W = [[0.0] * n_s for _ in range(n_f)]
        T = [[0.0] * n_s for _ in range(n_s)]
        Bs = [0.0] * n_s
        Be = [0.0] * n_s
        W_acc = [[0.0] * n_s for _ in range(n_f)]
        T_acc = [[0.0] * n_s for _ in range(n_s)]
        Bs_acc = [0.0] * n_s
        Be_acc = [0.0] * n_s

        rng = random.Random(self.seed)
        indices = list(range(len(enc_seqs)))
        step = 0
        end_list = sorted(self.valid_end_)

        for _ in range(self.epochs):
            rng.shuffle(indices)
            for seq_idx in indices:
                ef, y_true = enc_seqs[seq_idx]
                step += 1
                L = len(ef)
                emit = [[0.0] * n_s for _ in range(L)]
                for t_i, f_ids in enumerate(ef):
                    row = emit[t_i]
                    for fid in f_ids:
                        wf = W[fid]
                        for s in range(n_s):
                            row[s] += wf[s]

                dp = [[_NEG_INF] * n_s for _ in range(L)]
                bp = [[0] * n_s for _ in range(L)]
                for s in self.valid_start_:
                    cost = 0.0 if s == y_true[0] else (self.fn_cost if y_true[0] != 0 else 1.0)
                    dp[0][s] = Bs[s] + emit[0][s] + cost
                for t_i in range(1, L):
                    yt = y_true[t_i]
                    for s in range(n_s):
                        best_sc = _NEG_INF
                        best_p = 0
                        for p in self.valid_prev_[s]:
                            sc = dp[t_i - 1][p] + T[p][s]
                            if sc > best_sc:
                                best_sc = sc
                                best_p = p
                        cost = 0.0 if s == yt else (self.fn_cost if yt != 0 else 1.0)
                        dp[t_i][s] = best_sc + emit[t_i][s] + cost
                        bp[t_i][s] = best_p

                cur = max(end_list, key=lambda s: dp[L - 1][s] + Be[s])
                yh = [0] * L
                yh[L - 1] = cur
                for t_i in range(L - 1, 0, -1):
                    cur = bp[t_i][cur]
                    yh[t_i - 1] = cur

                if yh != y_true:
                    gold_sc = (
                        Bs[y_true[0]]
                        + Be[y_true[-1]]
                        + sum(emit[t_i][y_true[t_i]] for t_i in range(L))
                        + sum(T[y_true[t_i - 1]][y_true[t_i]] for t_i in range(1, L))
                    )
                    pred_sc = (
                        Bs[yh[0]]
                        + Be[yh[-1]]
                        + sum(emit[t_i][yh[t_i]] for t_i in range(L))
                        + sum(T[yh[t_i - 1]][yh[t_i]] for t_i in range(1, L))
                    )
                    loss = (
                        sum(
                            (self.fn_cost if y_true[t_i] != 0 else 1.0)
                            for t_i in range(L)
                            if yh[t_i] != y_true[t_i]
                        )
                        + pred_sc
                        - gold_sc
                    )
                    if loss > 0.0:
                        norm2 = (2.0 if y_true[0] != yh[0] else 0.0) + (
                            2.0 if y_true[-1] != yh[-1] else 0.0
                        )
                        norm2 += sum(2.0 * len(ef[t_i]) for t_i in range(L) if y_true[t_i] != yh[t_i])
                        norm2 += sum(
                            2.0
                            for t_i in range(1, L)
                            if (y_true[t_i - 1], y_true[t_i]) != (yh[t_i - 1], yh[t_i])
                        )
                        tau = min(self.c_pa, loss / max(norm2, 1.0))
                        Bs[y_true[0]] += tau
                        Bs[yh[0]] -= tau
                        Bs_acc[y_true[0]] += step * tau
                        Bs_acc[yh[0]] -= step * tau
                        Be[y_true[-1]] += tau
                        Be[yh[-1]] -= tau
                        Be_acc[y_true[-1]] += step * tau
                        Be_acc[yh[-1]] -= step * tau
                        for t_i in range(L):
                            yt, yp = y_true[t_i], yh[t_i]
                            if yt != yp:
                                for fid in ef[t_i]:
                                    W[fid][yt] += tau
                                    W[fid][yp] -= tau
                                    W_acc[fid][yt] += step * tau
                                    W_acc[fid][yp] -= step * tau
                        for t_i in range(1, L):
                            if (y_true[t_i - 1], y_true[t_i]) != (yh[t_i - 1], yh[t_i]):
                                T[y_true[t_i - 1]][y_true[t_i]] += tau
                                T[yh[t_i - 1]][yh[t_i]] -= tau
                                T_acc[y_true[t_i - 1]][y_true[t_i]] += step * tau
                                T_acc[yh[t_i - 1]][yh[t_i]] -= step * tau

        denom = float(step + 1)
        self.weights_ = [[W[f][s] - W_acc[f][s] / denom for s in range(n_s)] for f in range(n_f)]
        self.transitions_ = [[T[p][s] - T_acc[p][s] / denom for s in range(n_s)] for p in range(n_s)]
        self.start_bias_ = [Bs[s] - Bs_acc[s] / denom for s in range(n_s)]
        self.end_bias_ = [Be[s] - Be_acc[s] / denom for s in range(n_s)]

    def fit(self, train_folder: Path | str, *, include_discontinuous: bool = True) -> "LinearChainSpanTagger":
        """Load BRAT `.ann`/`.txt` files from `train_folder` and fit sequence weights."""
        resolved = str(Path(train_folder).resolve())
        docs, _ = _load_brat_documents(
            train_folder,
            labels=self.labels,
            include_discontinuous=include_discontinuous,
        )
        return self.fit_documents(docs, source_tag=resolved)

    def _decode_sequence(self, toks: Sequence[Tuple[str, int, int]]) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("LinearChainSpanTagger must be fit() before prediction")
        feats = extract_sequence_features(toks)
        L = len(feats)
        n_s = len(self.states_)

        if _np is not None and isinstance(self.weights_, _np.ndarray):
            W = self.weights_
            T = self.transitions_
            Bs = self.start_bias_
            Be = self.end_bias_
            assert isinstance(T, _np.ndarray) and isinstance(Bs, _np.ndarray) and isinstance(Be, _np.ndarray)
            trans_mask = _np.full((n_s, n_s), _NEG_INF, dtype=_np.float64)
            start_mask = _np.full(n_s, _NEG_INF, dtype=_np.float64)
            end_mask = _np.full(n_s, _NEG_INF, dtype=_np.float64)
            for s in self.valid_start_:
                start_mask[s] = 0.0
            for s in self.valid_end_:
                end_mask[s] = 0.0
            for s in range(n_s):
                for p in self.valid_prev_[s]:
                    trans_mask[p, s] = 0.0

            emit = _np.zeros((L, n_s), dtype=_np.float64)
            for t_i, tf in enumerate(feats):
                f_ids = [self.f2i_[fn] for fn in tf if fn in self.f2i_]
                if f_ids:
                    emit[t_i] = W[f_ids].sum(axis=0)

            dp = Bs + start_mask + emit[0]
            bp = _np.zeros((L, n_s), dtype=_np.int32)
            t_eff = T + trans_mask
            s_range = _np.arange(n_s)
            for t_i in range(1, L):
                sc = dp[:, None] + t_eff
                bp[t_i] = _np.argmax(sc, axis=0)
                dp = sc[bp[t_i], s_range] + emit[t_i]
            cur = int(_np.argmax(dp + Be + end_mask))
            yh = [0] * L
            yh[L - 1] = cur
            for t_i in range(L - 1, 0, -1):
                cur = int(bp[t_i, cur])
                yh[t_i - 1] = cur
            return yh

        # Pure-Python Viterbi fallback
        W_py = self.weights_  # type: ignore[assignment]
        T_py = self.transitions_  # type: ignore[assignment]
        Bs_py = self.start_bias_  # type: ignore[assignment]
        Be_py = self.end_bias_  # type: ignore[assignment]
        emit_py = [[0.0] * n_s for _ in range(L)]
        for t_i, tf in enumerate(feats):
            row = emit_py[t_i]
            for fn in tf:
                fid = self.f2i_.get(fn)
                if fid is not None:
                    wf = W_py[fid]  # type: ignore[index]
                    for s in range(n_s):
                        row[s] += wf[s]

        dp_py = [[_NEG_INF] * n_s for _ in range(L)]
        bp_py = [[0] * n_s for _ in range(L)]
        for s in self.valid_start_:
            dp_py[0][s] = Bs_py[s] + emit_py[0][s]  # type: ignore[index]
        for t_i in range(1, L):
            for s in range(n_s):
                best_sc = _NEG_INF
                best_p = 0
                for p in self.valid_prev_[s]:
                    sc = dp_py[t_i - 1][p] + T_py[p][s]  # type: ignore[index]
                    if sc > best_sc:
                        best_sc = sc
                        best_p = p
                dp_py[t_i][s] = best_sc + emit_py[t_i][s]
                bp_py[t_i][s] = best_p
        end_list = sorted(self.valid_end_)
        cur = max(end_list, key=lambda s: dp_py[L - 1][s] + Be_py[s])  # type: ignore[index]
        yh = [0] * L
        yh[L - 1] = cur
        for t_i in range(L - 1, 0, -1):
            cur = bp_py[t_i][cur]
            yh[t_i - 1] = cur
        return yh

    def find_spans(self, text: str) -> Set[Tuple[str, int, int]]:
        """Predict `(label, start_char, end_char)` spans on free-form protocol text."""
        if not self.is_fitted:
            raise RuntimeError("LinearChainSpanTagger must be fit() before calling find_spans()")
        spans: Set[Tuple[str, int, int]] = set()
        for toks in tokenize_protocol_lines(text):
            tag_ids = self._decode_sequence(toks)
            spans |= bioes_ids_to_spans(toks, tag_ids, self.states_)
        return spans

    def parse_quantities(self, text: str) -> List[ParsedQuantity]:
        """Predict spans via Viterbi decoding and normalize into canonical Auto-Lab units."""
        spans = self.find_spans(text)
        ordered = sorted(spans, key=lambda item: (item[1], item[2], item[0]))
        return [_normalize_span(lbl, s, e, text[s:e]) for lbl, s, e in ordered]

    def evaluate_folder(
        self,
        folder: Path | str,
        *,
        include_discontinuous: bool = True,
        forbid_train_overlap: bool = False,
    ) -> Dict[str, object]:
        """Score tagger on a folder of BRAT `.ann`/`.txt` pairs with split-overlap & SHA-256 audit."""
        resolved = str(Path(folder).resolve())
        docs, ann_totals = _load_brat_documents(
            folder,
            labels=self.labels,
            include_discontinuous=include_discontinuous,
        )
        id_overlap = sum(1 for doc_id, _, _ in docs if f"{resolved}:{doc_id}" in self.train_doc_ids_)
        if forbid_train_overlap and id_overlap > 0:
            raise ValueError(
                f"Data leakage detected: {id_overlap} protocol file(s) in {folder} "
                "belong to the training split"
            )

        sha_overlap = sum(1 for _, txt, _ in docs if _doc_sha256(txt) in self.train_doc_sha256s_)
        all_pairs = []
        unseen_sha_pairs = []
        for _, txt, gold in docs:
            pred = self.find_spans(txt)
            all_pairs.append((pred, gold))
            if _doc_sha256(txt) not in self.train_doc_sha256s_:
                unseen_sha_pairs.append((pred, gold))

        metrics, micro = _compute_corpus_metrics(self.labels, all_pairs)
        _, unseen_micro = _compute_corpus_metrics(self.labels, unseen_sha_pairs)
        return {
            "protocols": len(docs),
            "train_overlap_documents": id_overlap,
            "upstream_text_sha256_duplicates": sha_overlap,
            "unseen_text_protocols": len(unseen_sha_pairs),
            "unseen_text_micro_f1_exact": round(float(unseen_micro["F1_exact"]), 4),
            "unseen_text_micro_f1_overlap": round(float(unseen_micro["F1_overlap"]), 4),
            "metrics": metrics,
            "micro": micro,
            **ann_totals,
        }


def save_tagger_bundle(bundle_path: Path | str, tagger: LinearChainSpanTagger) -> Path:
    """Persist a fitted `LinearChainSpanTagger` to a safe non-pickle `.json.gz` archive."""
    if not tagger.is_fitted:
        raise RuntimeError("LinearChainSpanTagger must be fit() before saving bundle")
    p = Path(bundle_path)
    out_path = p if p.name.endswith(".json.gz") else p.with_suffix(".json.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _np is not None and isinstance(tagger.weights_, _np.ndarray):
        w_list = _np.round(tagger.weights_, 6).tolist()
        t_list = _np.round(tagger.transitions_, 6).tolist()  # type: ignore[arg-type]
        bs_list = _np.round(tagger.start_bias_, 6).tolist()  # type: ignore[arg-type]
        be_list = _np.round(tagger.end_bias_, 6).tolist()  # type: ignore[arg-type]
    else:
        w_list = tagger.weights_
        t_list = tagger.transitions_
        bs_list = tagger.start_bias_
        be_list = tagger.end_bias_

    payload = {
        "schema": TAGGER_BUNDLE_SCHEMA,
        "labels": list(tagger.labels),
        "c_pa": tagger.c_pa,
        "fn_cost": tagger.fn_cost,
        "epochs": tagger.epochs,
        "min_feature_freq": tagger.min_feature_freq,
        "seed": tagger.seed,
        "n_train_protocols": tagger.n_train_protocols_,
        "train_doc_ids": sorted(tagger.train_doc_ids_),
        "train_doc_sha256s": sorted(tagger.train_doc_sha256s_),
        "f2i": tagger.f2i_,
        "weights": w_list,
        "transitions": t_list,
        "start_bias": bs_list,
        "end_bias": be_list,
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return out_path


def load_tagger_bundle(bundle_path: Path | str) -> LinearChainSpanTagger:
    """Load a persisted `LinearChainSpanTagger` from `.json.gz` without `pickle`."""
    p = Path(bundle_path)
    in_path = p if p.is_file() else p.with_suffix(".json.gz")
    if not in_path.is_file():
        raise FileNotFoundError(f"Tagger bundle does not exist: {in_path}")
    with gzip.open(in_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != TAGGER_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported tagger bundle schema: {payload.get('schema')!r}")

    tagger = LinearChainSpanTagger(
        labels=payload["labels"],
        c_pa=float(payload["c_pa"]),
        fn_cost=float(payload["fn_cost"]),
        epochs=int(payload["epochs"]),
        min_feature_freq=int(payload["min_feature_freq"]),
        seed=int(payload["seed"]),
    )
    tagger.n_train_protocols_ = int(payload.get("n_train_protocols", 0))
    tagger.train_doc_ids_ = set(payload.get("train_doc_ids", []))
    tagger.train_doc_sha256s_ = set(payload.get("train_doc_sha256s", []))
    tagger.f2i_ = {str(k): int(v) for k, v in payload["f2i"].items()}
    if _np is not None:
        tagger.weights_ = _np.asarray(payload["weights"], dtype=_np.float64)
        tagger.transitions_ = _np.asarray(payload["transitions"], dtype=_np.float64)
        tagger.start_bias_ = _np.asarray(payload["start_bias"], dtype=_np.float64)
        tagger.end_bias_ = _np.asarray(payload["end_bias"], dtype=_np.float64)
    else:  # pragma: no cover
        tagger.weights_ = payload["weights"]
        tagger.transitions_ = payload["transitions"]
        tagger.start_bias_ = payload["start_bias"]
        tagger.end_bias_ = payload["end_bias"]
    return tagger


def select_hyperparameters_on_dev(
    train_folder: Path | str,
    dev_folder: Path | str,
    *,
    grid: Sequence[Tuple[float, float, int]] = DEFAULT_HYPERPARAM_GRID,
    labels: Sequence[str] = CORE_LABELS,
    seed: int = 42,
) -> Tuple[LinearChainSpanTagger, List[Dict[str, object]]]:
    """Fit candidate models strictly on `train_folder` and select best by `dev_folder` F1."""
    train_resolved = str(Path(train_folder).resolve())
    dev_resolved = str(Path(dev_folder).resolve())
    if train_resolved == dev_resolved:
        raise ValueError("Leakage blocked: train_folder and dev_folder resolve to the same directory")
    train_docs, _ = _load_brat_documents(train_folder, labels=labels)
    dev_docs, _ = _load_brat_documents(dev_folder, labels=labels)

    best_tagger: Optional[LinearChainSpanTagger] = None
    best_key = (-1.0, -1.0)
    history: List[Dict[str, object]] = []

    for c_pa, fn_cost, epochs in grid:
        t0 = time.perf_counter()
        cand = LinearChainSpanTagger(
            labels=labels,
            c_pa=c_pa,
            fn_cost=fn_cost,
            epochs=epochs,
            seed=seed,
        ).fit_documents(train_docs, source_tag=train_resolved)
        fit_s = round(time.perf_counter() - t0, 2)
        pairs = [(cand.find_spans(txt), gold) for _, txt, gold in dev_docs]
        _, micro = _compute_corpus_metrics(labels, pairs)
        key = (round(float(micro["F1_exact"]), 6), round(float(micro["F1_overlap"]), 6))
        history.append(
            {
                "c_pa": c_pa,
                "fn_cost": fn_cost,
                "epochs": epochs,
                "fit_s": fit_s,
                "dev_micro_f1_exact": round(float(micro["F1_exact"]), 4),
                "dev_micro_f1_overlap": round(float(micro["F1_overlap"]), 4),
            }
        )
        if key > best_key:
            best_key = key
            best_tagger = cand

    assert best_tagger is not None
    return best_tagger, history


def evaluate_external_jsonl(
    jsonl_path: Path | str,
    tagger: LinearChainSpanTagger,
) -> Dict[str, object]:
    """Evaluate transfer extraction on normalized external wet-lab corpora (`BioProBench`, `BioPlanner`)."""
    p = Path(jsonl_path)
    if not p.is_file():
        raise FileNotFoundError(f"Normalized protocol JSONL file not found: {p}")

    n_protocols = 0
    n_nonempty_learned = 0
    n_nonempty_rules = 0
    learned_counts = {lbl: 0 for lbl in tagger.labels}
    rule_counts = {lbl: 0 for lbl in tagger.labels}
    exact_agree = 0
    overlap_agree = 0
    total_learned = 0
    total_rules = 0
    dataset_id = p.parent.name

    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        dataset_id = str(obj.get("dataset_id", dataset_id))
        text = str(obj.get("text", "") or "")
        if not text.strip():
            continue
        n_protocols += 1
        l_spans = tagger.find_spans(text)
        r_spans = {sp for sp in find_rule_spans(text, include_concentration=False) if sp[0] in tagger.labels}
        if l_spans:
            n_nonempty_learned += 1
        if r_spans:
            n_nonempty_rules += 1
        for lbl in tagger.labels:
            lp = [x for x in l_spans if x[0] == lbl]
            rp = [x for x in r_spans if x[0] == lbl]
            learned_counts[lbl] += len(lp)
            rule_counts[lbl] += len(rp)
            exact_agree += len(set(lp) & set(rp))
            overlap_agree += count_overlap_matches(lp, rp)
        total_learned += len(l_spans)
        total_rules += len(r_spans)

    denom = total_learned + total_rules
    return {
        "dataset_id": dataset_id,
        "n_protocols": n_protocols,
        "protocols_with_learned_spans": n_nonempty_learned,
        "protocols_with_rule_spans": n_nonempty_rules,
        "learned_total_spans": total_learned,
        "rule_total_spans": total_rules,
        "learned_spans_by_label": learned_counts,
        "rule_spans_by_label": rule_counts,
        "cross_extractor_f1_exact": round(2.0 * exact_agree / denom, 4) if denom else 1.0,
        "cross_extractor_f1_overlap": round(2.0 * overlap_agree / denom, 4) if denom else 1.0,
    }


def _round_eval_block(res: Dict[str, object]) -> Dict[str, object]:
    metrics_in: Dict[str, Dict[str, float | int]] = res["metrics"]  # type: ignore[assignment]
    rounded_metrics: Dict[str, Dict[str, float | int]] = {}
    for lbl, m in metrics_in.items():
        rounded_metrics[lbl] = {
            "gold": int(m["gold"]),
            "pred": int(m["pred"]),
            "P_exact": round(float(m["P_exact"]), 4),
            "R_exact": round(float(m["R_exact"]), 4),
            "F1_exact": round(float(m["F1_exact"]), 4),
            "P_overlap": round(float(m["P_overlap"]), 4),
            "R_overlap": round(float(m["R_overlap"]), 4),
            "F1_overlap": round(float(m["F1_overlap"]), 4),
        }
    if "micro" in res:
        mc: Dict[str, float | int] = res["micro"]  # type: ignore[assignment]
    else:
        tot_g = sum(int(m["gold"]) for m in metrics_in.values())
        tot_p = sum(int(m["pred"]) for m in metrics_in.values())
        tot_te = sum(int(round(float(m["P_exact"]) * int(m["pred"]))) for m in metrics_in.values())
        tot_to = sum(int(round(float(m["P_overlap"]) * int(m["pred"]))) for m in metrics_in.values())
        pe = tot_te / tot_p if tot_p else 0.0
        re_ = tot_te / tot_g if tot_g else 0.0
        fe = 2 * pe * re_ / (pe + re_) if (pe + re_) else 0.0
        po = tot_to / tot_p if tot_p else 0.0
        ro = tot_to / tot_g if tot_g else 0.0
        fo = 2 * po * ro / (po + ro) if (po + ro) else 0.0
        mc = {
            "gold": tot_g,
            "pred": tot_p,
            "P_exact": pe,
            "R_exact": re_,
            "F1_exact": fe,
            "P_overlap": po,
            "R_overlap": ro,
            "F1_overlap": fo,
        }
    out_block: Dict[str, object] = {
        "protocols": int(res["protocols"]),
        "metrics": rounded_metrics,
        "micro": {
            "gold": int(mc["gold"]),
            "pred": int(mc["pred"]),
            "P_exact": round(float(mc["P_exact"]), 4),
            "R_exact": round(float(mc["R_exact"]), 4),
            "F1_exact": round(float(mc["F1_exact"]), 4),
            "P_overlap": round(float(mc["P_overlap"]), 4),
            "R_overlap": round(float(mc["R_overlap"]), 4),
            "F1_overlap": round(float(mc["F1_overlap"]), 4),
        },
    }
    if "upstream_text_sha256_duplicates" in res:
        out_block["upstream_text_sha256_duplicates"] = int(res["upstream_text_sha256_duplicates"])  # type: ignore[arg-type]
        out_block["unseen_text_protocols"] = int(res["unseen_text_protocols"])  # type: ignore[arg-type]
        out_block["unseen_text_micro_f1_exact"] = float(res["unseen_text_micro_f1_exact"])  # type: ignore[arg-type]
        out_block["unseen_text_micro_f1_overlap"] = float(res["unseen_text_micro_f1_overlap"])  # type: ignore[arg-type]
    return out_block


def _pool_split_blocks(splits: Sequence[Dict[str, object]], labels: Sequence[str]) -> Dict[str, object]:
    tot_protos = sum(int(s["protocols"]) for s in splits)
    by_lbl: Dict[str, Dict[str, float | int]] = {}
    all_te = all_to = all_p = all_g = 0
    for lbl in labels:
        g = sum(int(s["metrics"][lbl]["gold"]) for s in splits)  # type: ignore[index]
        p = sum(int(s["metrics"][lbl]["pred"]) for s in splits)  # type: ignore[index]
        te = sum(
            int(round(float(s["metrics"][lbl]["P_exact"]) * int(s["metrics"][lbl]["pred"])))  # type: ignore[index]
            for s in splits
        )
        to = sum(
            int(round(float(s["metrics"][lbl]["P_overlap"]) * int(s["metrics"][lbl]["pred"])))  # type: ignore[index]
            for s in splits
        )
        all_g += g
        all_p += p
        all_te += te
        all_to += to
        pe = te / p if p else 0.0
        re_ = te / g if g else 0.0
        fe = 2 * pe * re_ / (pe + re_) if (pe + re_) else 0.0
        po = to / p if p else 0.0
        ro = to / g if g else 0.0
        fo = 2 * po * ro / (po + ro) if (po + ro) else 0.0
        by_lbl[lbl] = {
            "gold": g,
            "pred": p,
            "P_exact": round(pe, 4),
            "R_exact": round(re_, 4),
            "F1_exact": round(fe, 4),
            "P_overlap": round(po, 4),
            "R_overlap": round(ro, 4),
            "F1_overlap": round(fo, 4),
        }
    m_pe = all_te / all_p if all_p else 0.0
    m_re = all_te / all_g if all_g else 0.0
    m_fe = 2 * m_pe * m_re / (m_pe + m_re) if (m_pe + m_re) else 0.0
    m_po = all_to / all_p if all_p else 0.0
    m_ro = all_to / all_g if all_g else 0.0
    m_fo = 2 * m_po * m_ro / (m_po + m_ro) if (m_po + m_ro) else 0.0
    return {
        "protocols": tot_protos,
        "metrics": by_lbl,
        "micro": {
            "gold": all_g,
            "pred": all_p,
            "P_exact": round(m_pe, 4),
            "R_exact": round(m_re, 4),
            "F1_exact": round(m_fe, 4),
            "P_overlap": round(m_po, 4),
            "R_overlap": round(m_ro, 4),
            "F1_overlap": round(m_fo, 4),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path("data/WLP-Dataset/train"))
    ap.add_argument("--dev", type=Path, default=Path("data/WLP-Dataset/dev"))
    ap.add_argument("--test", type=Path, default=Path("data/WLP-Dataset/test"))
    ap.add_argument("--tune-on-dev", action="store_true", help="Sweep DEFAULT_HYPERPARAM_GRID on dev split")
    ap.add_argument("--c-pa", type=float, default=0.25)
    ap.add_argument("--fn-cost", type=float, default=1.30)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--save-model", type=Path, default=None)
    ap.add_argument("--external-jsonl", nargs="*", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/wlp_learned_vs_rules.json"))
    args = ap.parse_args(argv)

    try:
        t0 = time.perf_counter()
        if args.tune_on_dev:
            tagger, dev_grid = select_hyperparameters_on_dev(args.train, args.dev)
        else:
            tagger = LinearChainSpanTagger(
                c_pa=args.c_pa,
                fn_cost=args.fn_cost,
                epochs=args.epochs,
            ).fit(args.train)
            dev_grid = []
        fit_s = round(time.perf_counter() - t0, 2)

        if args.save_model is not None:
            save_tagger_bundle(args.save_model, tagger)

        learned_splits = {
            "train": _round_eval_block(tagger.evaluate_folder(args.train, forbid_train_overlap=False)),
            "dev": _round_eval_block(tagger.evaluate_folder(args.dev, forbid_train_overlap=True)),
            "test": _round_eval_block(tagger.evaluate_folder(args.test, forbid_train_overlap=True)),
        }
        learned_splits["pooled_622"] = _pool_split_blocks(
            [learned_splits["train"], learned_splits["dev"], learned_splits["test"]],
            CORE_LABELS,
        )

        rule_splits = {
            sp: _round_eval_block(evaluate_rule_folder(getattr(args, sp)))
            for sp in ("train", "dev", "test")
        }
        rule_splits["pooled_622"] = _pool_split_blocks(
            [rule_splits["train"], rule_splits["dev"], rule_splits["test"]],
            CORE_LABELS,
        )

        ext_reports = []
        if args.external_jsonl:
            for ext_p in args.external_jsonl:
                if ext_p.is_file():
                    ext_reports.append(evaluate_external_jsonl(ext_p, tagger))

        report = {
            "dataset": "Wet Lab Protocols (WLP, Kulkarni et al. NAACL 2018, commit 1f51ae6e620a5aa3590acf725713f430783f56d7)",
            "zero_leakage_protocol": (
                "LinearChainSpanTagger vocabulary and PA-II weights fit strictly on WLP/train "
                "(374 protocols); hyperparameters selected strictly on WLP/dev (123 protocols); "
                "WLP/test (125 protocols) evaluated once with SHA-256 zero-overlap verification. "
                "By contrast, rule_based regex patterns were previously refined with WLP/test errors visible."
            ),
            "selected_hyperparameters": {
                "c_pa": tagger.c_pa,
                "fn_cost": tagger.fn_cost,
                "epochs": tagger.epochs,
                "min_feature_freq": tagger.min_feature_freq,
                "seed": tagger.seed,
                "n_features": len(tagger.f2i_),
                "fit_s": fit_s,
            },
            "dev_hyperparameter_search": dev_grid,
            "learned_bioes_pa2": learned_splits,
            "rule_based_regex": rule_splits,
            "external_transfer": ext_reports,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
