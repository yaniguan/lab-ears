"""Rule-based baseline: find Amount, Temperature, Time and Speed spans in wet-lab
protocol text, normalize them into canonical Auto-Lab units, and score them against
the WLP gold annotations (Kulkarni et al., 2018).

Usage:
    python demo/parse_protocol.py path/to/WLP-Dataset/test [--min-f1 0.75]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_MANTISSA = r"(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:[.,]\d+)?"
_SCI_EXP = r"(?:[eE][-−+]?\d+|\s*[x×]\s*10\s*\^\s*[-−+]?\d+|\^\s*[-−+]?\d+)"
SCALAR = rf"{_MANTISSA}(?:{_SCI_EXP})?"
RANGE_SEP = r"(?:\s*(?:-|–|−|to)\s*)"
_NUM_LEFT_GUARD = r"(?<![\w.^])(?<![eE][-−+])(?<!\^[-−+])"
NUM = rf"{_NUM_LEFT_GUARD}{SCALAR}(?:{RANGE_SEP}{SCALAR})?"
TEMP_NUM = (
    rf"{_NUM_LEFT_GUARD}(?:(?<![\w.^])[-+−–]{SCALAR}|{SCALAR})"
    rf"(?:{RANGE_SEP}[-+−–]?{SCALAR})?"
)

CONCENTRATION_RULE = (
    NUM + r"\s*(?:"
    r"(?i:[mµμunpf]?g\s*/\s*(?:ml|µl|μl|ul|l)\b)|"
    r"(?i:(?:mU|U|units?)\s*/\s*(?:ml|µl|μl|ul)\b)|"
    r"(?:[mµμunpf]?M)\b|"
    r"%(?:\s*(?i:\(?[wv]\s*/\s*[wv]\)?))?|"
    r"(?<=\d)\s*[xX]\b(?!\s*(?:g\b|10\s*\^))"
    r")"
)

RULES: Dict[str, str] = {
    "Amount": (
        NUM + r"\s*(?:"
        r"(?i:(?:µl|μl|ul|ml|nl)(?:\s*/\s*well)?|liters?|litres?|volumes?|vol)\b|"
        r"L(?:\s*/\s*(?i:well))?\b|"
        r"(?i:fg|pg|ng|µg|μg|ug|mg|kg)\b|"
        r"g\b|"
        r"(?i:fmol|pmol|nmol|µmol|μmol|umol|mmol|mol)\b|"
        r"(?:mU|U)\b|(?i:units?)\b"
        r")"
    ),
    "Temperature": (
        TEMP_NUM
        + r"\s*(?:[°º˚]\s*[CFK]\b|oC\b|(?i:deg(?:rees?)?\s*(?:C(?:elsius)?|F(?:ahrenheit)?)\b))"
        r"|(?i:\b(?:room\s+temp(?:erature)?|on\s+dry\s+ice|dry\s+ice|on\s+ice|ice)\b(?!\s*[-‐]\s*(?i:cold)))"
        r"|(?<![-\w])RT\b(?!\s*(?:[-/]\s*(?i:PCR|qPCR|LAMP)|"
        r"(?i:reaction|buffer|enzyme|mix|master\s*mix|primer|step|kit|product|superscript|polymerase|transcriptase)\b))"
    ),
    "Time": (
        NUM
        + r"\s*(?:(?i:hrs?|hours?|mins?|minutes?|secs?|seconds?|ms|days?|overnight)|h|s)\b"
        r"|(?i:\bovernight\b)"
    ),
    "Speed": NUM + r"\s*(?:[x×]\s*g|xg|k\s*g|(?i:rpm|rcf))\b",
}

_TEMP_BARE_CK = re.compile(TEMP_NUM + r"\s*([CK])\b")
_THERMAL_CTX = re.compile(
    r"\b(?:incubat\w*|stor\w*|heat\w*|preheat\w*|cool\w*|chill\w*|freez\w*|thaw\w*|"
    r"autoclave\w*|boil\w*|warm\w*|anneal\w*|denatur\w*|extend\w*|extension|oven|"
    r"water\s*bath|thermocycler|temp(?:erature)?|grow\w*)\b",
    re.IGNORECASE,
)
_THERMAL_PREP = re.compile(r"(?:\b(?:at|to|in)\s+|\(\s*)$", re.IGNORECASE)
_NON_THERMAL_AFTER = re.compile(
    r"^\s*(?:MWCO|label\w*|NMR\b|isotop\w*|enrich\w*|spectrum\b|spectra\b|shift\w*)",
    re.IGNORECASE,
)

_SPEED_BARE_G = re.compile(NUM + r"\s*g\b")
_CENTRIFUGE_CTX = re.compile(
    r"\b(?:centrifug\w*|spin\w*|pellet\w*|microfug\w*|rotor|supernatant)\b",
    re.IGNORECASE,
)
_AT_PREFIX = re.compile(r"(?:\bat\s+|@\s*)$", re.IGNORECASE)
_MASS_VERB_BEFORE = re.compile(
    r"\b(?:add|dissolve|weigh|measure|suspend\s+in|resuspend\s+in)\s+$",
    re.IGNORECASE,
)
_OF_AFTER = re.compile(r"^\s+of\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedQuantity:
    """Normalized quantitative parameter extracted from protocol text."""

    label: str
    start: int
    end: int
    raw: str
    value_min: Optional[float]
    value_max: Optional[float]
    unit: str
    canonical_value: Optional[float]
    canonical_unit: str
    canonical_min: Optional[float] = None
    canonical_max: Optional[float] = None


def _parse_mantissa_str(token: str) -> float:
    s = token.strip().replace("−", "-").replace("–", "-")
    if re.fullmatch(r"[-+]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    return float(s)


def _parse_scalar_str(token: str) -> float:
    s = token.strip().replace("−", "-").replace("–", "-")
    m_pow10 = re.fullmatch(r"(?P<mant>.+?)\s*[x×]\s*10\s*\^\s*(?P<exp>[-+]?\d+)", s)
    if m_pow10 is not None:
        return _parse_mantissa_str(m_pow10.group("mant")) * (10.0 ** int(m_pow10.group("exp")))
    m_caret = re.fullmatch(r"(?P<base>.+?)\^\s*(?P<exp>[-+]?\d+)", s)
    if m_caret is not None:
        return _parse_mantissa_str(m_caret.group("base")) ** int(m_caret.group("exp"))
    m_sci = re.fullmatch(r"(?P<mant>.+?)[eE](?P<exp>[-+]?\d+)", s)
    if m_sci is not None:
        return _parse_mantissa_str(m_sci.group("mant")) * (10.0 ** int(m_sci.group("exp")))
    return _parse_mantissa_str(s)


def _first_scalar_value(raw: str) -> float:
    m = re.match(r"[-+−–]?" + SCALAR, raw.strip())
    if not m:
        return 0.0
    return _parse_scalar_str(m.group())


def _dedupe_nested(spans: Set[Tuple[str, int, int]], label: str) -> Set[Tuple[str, int, int]]:
    lbl_spans = [(s, e) for l, s, e in spans if l == label]
    return {
        (l, s, e)
        for l, s, e in spans
        if not (
            l == label
            and any(s2 <= s and e <= e2 and (s2, e2) != (s, e) for s2, e2 in lbl_spans)
        )
    }


def find_spans(text: str, *, include_concentration: bool = False) -> Set[Tuple[str, int, int]]:
    """Extract (label, start_char, end_char) entity spans from protocol text."""
    spans: Set[Tuple[str, int, int]] = set()
    active_rules = dict(RULES)
    if include_concentration:
        active_rules["Concentration"] = CONCENTRATION_RULE

    for label, pat in active_rules.items():
        for m in re.finditer(pat, text):
            spans.add((label, m.start(), m.end()))

    # Contextual bare uppercase C and K for Temperature (e.g. "at 37C", "store at 4 C", "-80C")
    for m in _TEMP_BARE_CK.finditer(text):
        s, e = m.start(), m.end()
        raw = m.group()
        unit_char = m.group(1)
        before = text[max(0, s - 50) : s]
        after = text[e : min(len(text), e + 25)]
        if _NON_THERMAL_AFTER.search(after):
            continue
        is_neg = raw[0] in "-−–"
        has_thermal = bool(_THERMAL_CTX.search(before) or _THERMAL_CTX.search(after))
        has_prep = unit_char == "C" and bool(_THERMAL_PREP.search(before))
        if is_neg or has_thermal or has_prep:
            spans.add(("Temperature", s, e))

    spans = _dedupe_nested(spans, "Temperature")
    spans = _dedupe_nested(spans, "Time")

    # Contextual bare '<num> g' for Speed vs Amount:
    # Only classify '<num> g' as centrifugal Speed when val >= 50 and centrifugation context applies.
    for m in _SPEED_BARE_G.finditer(text):
        s, e = m.start(), m.end()
        if any(s2 <= s and e <= e2 for l2, s2, e2 in spans if l2 == "Speed"):
            continue
        val = _first_scalar_value(m.group())
        if val < 50.0:
            continue
        before = text[max(0, s - 60) : s]
        after = text[e : min(len(text), e + 40)]
        if _MASS_VERB_BEFORE.search(before) or _OF_AFTER.search(after):
            continue
        if _CENTRIFUGE_CTX.search(before) or _CENTRIFUGE_CTX.search(after) or (
            val >= 100.0 and _AT_PREFIX.search(before)
        ):
            spans.add(("Speed", s, e))

    speed_ranges = [(s, e) for l, s, e in spans if l == "Speed"]
    spans = {
        (l, s, e)
        for l, s, e in spans
        if not (
            l in ("Amount", "Concentration")
            and any(s2 <= s and e <= e2 for s2, e2 in speed_ranges)
        )
    }
    if include_concentration:
        spans = _dedupe_nested(spans, "Concentration")
        conc_ranges = [(s, e) for l, s, e in spans if l == "Concentration"]
        spans = {
            (l, s, e)
            for l, s, e in spans
            if not (l == "Amount" and any(s2 <= s and e <= e2 for s2, e2 in conc_ranges))
        }
    return spans


_VOLUME_TO_UL = {
    "nl": 1e-3,
    "ul": 1.0,
    "µl": 1.0,
    "μl": 1.0,
    "ml": 1e3,
    "l": 1e6,
    "liter": 1e6,
    "liters": 1e6,
    "litre": 1e6,
    "litres": 1e6,
}
_MASS_TO_MG = {
    "fg": 1e-12,
    "pg": 1e-9,
    "ng": 1e-6,
    "ug": 1e-3,
    "µg": 1e-3,
    "μg": 1e-3,
    "mg": 1.0,
    "g": 1e3,
    "kg": 1e6,
}
_MOLES_TO_MOL = {
    "fmol": 1e-15,
    "pmol": 1e-12,
    "nmol": 1e-9,
    "umol": 1e-6,
    "µmol": 1e-6,
    "μmol": 1e-6,
    "mmol": 1e-3,
    "mol": 1.0,
}
_TIME_TO_S = {
    "ms": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
}
_MOLAR_TO_MM = {
    "fM": 1e-12,
    "pM": 1e-9,
    "nM": 1e-6,
    "uM": 1e-3,
    "µM": 1e-3,
    "μM": 1e-3,
    "mM": 1.0,
    "M": 1e3,
}

_NUM_RANGE_PREFIX = re.compile(
    rf"^\s*(?P<v1>[-+−–]?{SCALAR})(?:{RANGE_SEP}(?P<v2>[-+−–]?{SCALAR}))?\s*(?P<unit>.*?)\s*$"
)


def _normalize_span(label: str, start: int, end: int, raw: str) -> ParsedQuantity:
    low = " ".join(raw.lower().split())

    # Qualitative Temperature & Time forms
    if label == "Temperature":
        if "dry ice" in low:
            return ParsedQuantity(label, start, end, raw, -78.5, -78.5, raw.strip(), -78.5, "C", -78.5, -78.5)
        if low in ("ice", "on ice"):
            return ParsedQuantity(label, start, end, raw, 0.0, 0.0, raw.strip(), 0.0, "C", 0.0, 0.0)
        if low == "rt" or low.startswith("room temp"):
            return ParsedQuantity(label, start, end, raw, 20.0, 25.0, raw.strip(), 22.0, "C", 20.0, 25.0)
    if label == "Time" and low == "overnight":
        return ParsedQuantity(label, start, end, raw, 12.0, 16.0, "overnight", 43200.0, "s", 43200.0, 57600.0)

    m = _NUM_RANGE_PREFIX.match(raw)
    if not m:
        return ParsedQuantity(label, start, end, raw, None, None, raw.strip(), None, "")

    v1 = _parse_scalar_str(m.group("v1"))
    v2 = _parse_scalar_str(m.group("v2")) if m.group("v2") is not None else v1
    vmin, vmax = (v1, v2) if v1 <= v2 else (v2, v1)
    unit = re.sub(r"\s+", " ", m.group("unit").strip())
    unit_low = unit.lower()

    if label == "Amount":
        base_u = re.sub(r"\s*/\s*well$", "", unit_low)
        if base_u in _VOLUME_TO_UL:
            scale = _VOLUME_TO_UL[base_u]
            cmin, cmax = vmin * scale, vmax * scale
            cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "uL", cmin, cmax)
        if unit_low in _MASS_TO_MG:
            scale = _MASS_TO_MG[unit_low]
            cmin, cmax = vmin * scale, vmax * scale
            cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "mg", cmin, cmax)
        if unit_low in _MOLES_TO_MOL:
            scale = _MOLES_TO_MOL[unit_low]
            cmin, cmax = vmin * scale, vmax * scale
            cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "mol", cmin, cmax)
        if unit in ("U", "mU") or unit_low in ("unit", "units"):
            scale = 1e-3 if unit == "mU" else 1.0
            cmin, cmax = vmin * scale, vmax * scale
            cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "U", cmin, cmax)
        cval = vmin if vmin == vmax else 0.5 * (vmin + vmax)
        return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "vol", vmin, vmax)

    if label == "Time":
        if unit_low == "overnight":
            cmin, cmax = vmin * 3600.0, vmax * 3600.0
        else:
            scale = _TIME_TO_S.get(unit_low, 1.0)
            cmin, cmax = vmin * scale, vmax * scale
        cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
        return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "s", cmin, cmax)

    if label == "Temperature":
        if "f" in unit_low:
            cmin = (vmin - 32.0) * 5.0 / 9.0
            cmax = (vmax - 32.0) * 5.0 / 9.0
        elif unit.endswith("K"):
            cmin = vmin - 273.15
            cmax = vmax - 273.15
        else:
            cmin, cmax = vmin, vmax
        cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
        return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "C", cmin, cmax)

    if label == "Speed":
        if "rpm" in unit_low:
            cmin, cmax = vmin, vmax
            cunit = "rpm"
        elif re.fullmatch(r"k\s*g", unit_low):
            cmin, cmax = vmin * 1000.0, vmax * 1000.0
            cunit = "xg"
        else:
            cmin, cmax = vmin, vmax
            cunit = "xg"
        cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
        return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, cunit, cmin, cmax)

    if label == "Concentration":
        if unit in _MOLAR_TO_MM:
            scale = _MOLAR_TO_MM[unit]
            cmin, cmax = vmin * scale, vmax * scale
            cval = cmin if vmin == vmax else 0.5 * (cmin + cmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "mM", cmin, cmax)
        if unit.startswith("%"):
            cval = vmin if vmin == vmax else 0.5 * (vmin + vmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "%", vmin, vmax)
        if unit_low == "x":
            cval = vmin if vmin == vmax else 0.5 * (vmin + vmax)
            return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, "X", vmin, vmax)
        cval = vmin if vmin == vmax else 0.5 * (vmin + vmax)
        return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, unit, vmin, vmax)

    cval = vmin if vmin == vmax else 0.5 * (vmin + vmax)
    return ParsedQuantity(label, start, end, raw, vmin, vmax, unit, cval, unit, vmin, vmax)


def parse_quantities(
    text: str, *, include_concentration: bool = True
) -> List[ParsedQuantity]:
    """Extract and unit-normalize quantitative spans for downstream stepbot / YLM use."""
    spans = find_spans(text, include_concentration=include_concentration)
    ordered = sorted(spans, key=lambda item: (item[1], item[2], item[0]))
    return [_normalize_span(lbl, s, e, text[s:e]) for lbl, s, e in ordered]


def gold_spans_with_stats(
    ann_path: Path,
    *,
    include_discontinuous: bool = True,
    target_labels: Optional[Iterable[str]] = None,
) -> Tuple[Set[Tuple[str, int, int]], Dict[str, int]]:
    """Parse BRAT .ann entity spans with discontinuous-span support and malformed-line guards."""
    allowed = set(target_labels) if target_labels is not None else set(RULES)
    out: Set[Tuple[str, int, int]] = set()
    stats = {"discontinuous_entities": 0, "skipped_discontinuous": 0, "malformed_lines": 0}

    for raw_line in Path(ann_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("T"):
            continue
        tab_parts = raw_line.split("\t", 2)
        if len(tab_parts) < 2 or not re.fullmatch(r"T\d+", tab_parts[0].strip()):
            stats["malformed_lines"] += 1
            continue
        info = tab_parts[1].strip()
        tokens = info.split(None, 1)
        if len(tokens) < 2:
            stats["malformed_lines"] += 1
            continue
        label, coord_spec = tokens[0], tokens[1]
        if label not in allowed:
            continue
        is_disc = ";" in coord_spec
        if is_disc:
            stats["discontinuous_entities"] += 1
            if not include_discontinuous:
                stats["skipped_discontinuous"] += 1
                continue
        parsed_segments: List[Tuple[int, int]] = []
        malformed = False
        for seg in coord_spec.split(";"):
            seg_parts = seg.strip().split()
            if len(seg_parts) != 2:
                malformed = True
                break
            try:
                s_idx, e_idx = int(seg_parts[0]), int(seg_parts[1])
            except ValueError:
                malformed = True
                break
            if s_idx < 0 or e_idx <= s_idx:
                malformed = True
                break
            parsed_segments.append((s_idx, e_idx))
        if malformed or not parsed_segments:
            stats["malformed_lines"] += 1
            continue
        for s_idx, e_idx in parsed_segments:
            out.add((label, s_idx, e_idx))
    return out, stats


def gold_spans(
    ann_path: Path,
    *,
    include_discontinuous: bool = True,
    target_labels: Optional[Iterable[str]] = None,
) -> Set[Tuple[str, int, int]]:
    """Return BRAT gold (label, start, end) spans for `ann_path`."""
    spans, _ = gold_spans_with_stats(
        ann_path,
        include_discontinuous=include_discontinuous,
        target_labels=target_labels,
    )
    return spans


def overlaps(a: Tuple[str, int, int], b: Tuple[str, int, int]) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def _span_iou(a: Tuple[str, int, int], b: Tuple[str, int, int]) -> float:
    if not overlaps(a, b):
        return 0.0
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    union = max(a[2], b[2]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def count_overlap_matches(
    preds: Sequence[Tuple[str, int, int]],
    golds: Sequence[Tuple[str, int, int]],
) -> int:
    """Maximum 1-to-1 bipartite overlap matching between predicted and gold spans."""
    p_list = sorted(set(preds), key=lambda x: (x[1], x[2], x[0]))
    g_list = sorted(set(golds), key=lambda x: (x[1], x[2], x[0]))
    if not p_list or not g_list:
        return 0

    adj: List[List[int]] = []
    for p in p_list:
        neighbors = [j for j, g in enumerate(g_list) if overlaps(p, g)]
        neighbors.sort(key=lambda j: -_span_iou(p, g_list[j]))
        adj.append(neighbors)

    match_g = [-1] * len(g_list)

    def dfs(u: int, seen: List[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_g[v] == -1 or dfs(match_g[v], seen):
                match_g[v] = u
                return True
        return False

    matched = 0
    for u in range(len(p_list)):
        if dfs(u, [False] * len(g_list)):
            matched += 1
    return matched


def evaluate_folder(
    folder: Path | str,
    *,
    include_discontinuous: bool = True,
    include_concentration: bool = False,
) -> Dict[str, object]:
    """Evaluate span extraction on a folder of BRAT .ann/.txt protocol pairs."""
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist or is not a directory: {folder_path}")

    files = sorted(folder_path.glob("*.ann"))
    if not files:
        raise ValueError(f"No .ann annotation files found in {folder_path}")

    labels = list(RULES) + (["Concentration"] if include_concentration else [])
    stats = {k: [0, 0, 0, 0] for k in labels}  # tp_exact, tp_overlap, n_pred, n_gold
    ann_totals = {"discontinuous_entities": 0, "skipped_discontinuous": 0, "malformed_lines": 0}

    for ann in files:
        txt_path = ann.with_suffix(".txt")
        if not txt_path.is_file():
            raise FileNotFoundError(f"Missing matching .txt protocol file for {ann}")
        text = txt_path.read_text(encoding="utf-8")
        pred = find_spans(text, include_concentration=include_concentration)
        gold, ann_stats = gold_spans_with_stats(
            ann,
            include_discontinuous=include_discontinuous,
            target_labels=labels,
        )
        for key in ann_totals:
            ann_totals[key] += ann_stats[key]
        for k in labels:
            p = [x for x in pred if x[0] == k]
            g = [x for x in gold if x[0] == k]
            stats[k][0] += len(set(p) & set(g))
            stats[k][1] += count_overlap_matches(p, g)
            stats[k][2] += len(p)
            stats[k][3] += len(g)

    metrics: Dict[str, Dict[str, float | int]] = {}
    for k, (te, to, n_p, n_g) in stats.items():
        pe = te / n_p if n_p else 0.0
        re_ = te / n_g if n_g else 0.0
        fe = 2 * pe * re_ / (pe + re_) if (pe + re_) else 0.0
        po = to / n_p if n_p else 0.0
        ro = to / n_g if n_g else 0.0
        fo = 2 * po * ro / (po + ro) if (po + ro) else 0.0
        metrics[k] = {
            "gold": n_g,
            "pred": n_p,
            "P_exact": pe,
            "R_exact": re_,
            "F1_exact": fe,
            "P_overlap": po,
            "R_overlap": ro,
            "F1_overlap": fo,
        }
    return {
        "protocols": len(files),
        "metrics": metrics,
        **ann_totals,
    }


def main(
    folder: Path | str = "data/WLP-Dataset/test",
    *,
    min_f1: float = 0.0,
    include_discontinuous: bool = True,
    include_concentration: bool = False,
) -> Dict[str, object]:
    try:
        res = evaluate_folder(
            folder,
            include_discontinuous=include_discontinuous,
            include_concentration=include_concentration,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        f"protocols: {res['protocols']} "
        f"(discontinuous_gold: {res['discontinuous_entities']}, "
        f"skipped_discontinuous: {res['skipped_discontinuous']})"
    )
    header = (
        f"{'label':<14}{'gold':>6}{'pred':>6}"
        f"{'P_exact':>9}{'R_exact':>9}{'F1_exact':>9}"
        f"{'P_overlap':>11}{'R_overlap':>11}{'F1_overlap':>11}"
    )
    print(header)
    metrics: Dict[str, Dict[str, float | int]] = res["metrics"]  # type: ignore[assignment]
    failed_gates: List[str] = []
    for k, m in metrics.items():
        print(
            f"{k:<14}{int(m['gold']):>6}{int(m['pred']):>6}"
            f"{float(m['P_exact']):>9.3f}{float(m['R_exact']):>9.3f}{float(m['F1_exact']):>9.3f}"
            f"{float(m['P_overlap']):>11.3f}{float(m['R_overlap']):>11.3f}{float(m['F1_overlap']):>11.3f}"
        )
        if k in RULES and float(m["F1_exact"]) < min_f1:
            failed_gates.append(f"{k} F1_exact={float(m['F1_exact']):.3f} < {min_f1:.3f}")
    if failed_gates:
        print("error: F1 regression gate failed: " + "; ".join(failed_gates), file=sys.stderr)
        raise SystemExit(1)
    return res


def cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="data/WLP-Dataset/test")
    ap.add_argument("--min-f1", type=float, default=0.0, help="Minimum exact F1 required per core label")
    ap.add_argument(
        "--skip-discontinuous",
        action="store_true",
        help="Skip semicolon-separated discontinuous BRAT annotations instead of splitting segments",
    )
    ap.add_argument(
        "--include-concentration",
        action="store_true",
        help="Also extract and score WLP Concentration entities",
    )
    args = ap.parse_args(argv)
    main(
        args.folder,
        min_f1=args.min_f1,
        include_discontinuous=not args.skip_discontinuous,
        include_concentration=args.include_concentration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
