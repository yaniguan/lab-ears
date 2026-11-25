"""Auto-Lab 15-repository wire node for lab-ears (Layer L2 Sensory & Protocol Cortex).

Extracts step-indexed, range-aware wet-lab protocol spans (Amount, Temperature, Time,
Speed, Concentration) from `goal_text` via `demo/parse_protocol.py`, enforces the
Lean-verified `L2` speech-band privacy + Nyquist + Hann-skirt invariants, and propagates
calibrated `AudioEvent` anomalies and `CrossModalInterlockDetector` cavitation/stall halts.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ID = "lab-ears"
LAYER = "L2"
SCHEMA_VERSION = "autolab-wire/v1"
AUDIO_EVENT_SCHEMA = "lab-ears/audio-event/v1"

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

SPEECH_BAND_LO_HZ = 300.0
SPEECH_BAND_HI_HZ = 3400.0
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_RETAINED_BANDS_HZ = [[60, 260], [3600, 7800]]
DEFAULT_VOLUME_UL = 100.0
DEFAULT_TEMP_C = 25.0
DEFAULT_DURATION_MIN = 15.0

_PARSE_MODULE_KEY = "_lab_ears_demo_parse_protocol"
_BARE_LIST_TEMP_RE = re.compile(
    r"(?:^|[,;:=]\s*)(?P<raw>(?P<val>[-+]?\d+(?:\.\d+)?)\s*C)\b"
)
_NON_THERMAL_AFTER_RE = re.compile(
    r"^\s*[-/]?\s*(?:MWCO|label\w*|NMR\b|isotop\w*|enrich\w*|spectrum\b|spectra\b|"
    r"chemical\s+shift\w*|ppm\b|tracer\w*)",
    re.IGNORECASE,
)
_ISOTOPE_CTX_RE = re.compile(
    r"\b(?:NMR|13C|14C|isotop\w*|enrich\w*|spectrum|spectra|chemical\s+shift\w*|ppm|tracer\w*)\b",
    re.IGNORECASE,
)

_STEP_SPLIT_RE = re.compile(
    r"(?:[;\n]+|"
    r"(?:,\s*|\.\s+|\s+)\b(?:then|next|afterwards?|subsequently|finally)\b\s*,?|"
    r"\.\s+(?=[A-Z])|"
    r",\s*(?:and\s+)?(?=(?:incubat|react|transfer|dispense|add|mix|spin|centrifug|"
    r"heat|warm|cool|chill|thaw|store|dilute|wash|elute|dissolv|resuspend|aliquot|"
    r"vortex|equilibrat)\w*\b))",
    re.IGNORECASE,
)
_PREP_VERB_RE = re.compile(
    r"\b(?:thaw\w*|pre-?warm\w*|pre-?heat\w*|pre-?chill\w*|pre-?cool\w*|prime\w*|equilibrat\w*|rinse\w*)\b",
    re.IGNORECASE,
)
_PRIMARY_VERB_RE = re.compile(
    r"\b(?:react\w*|incubat\w*|synthesi[sz]\w*|digest\w*|ligat\w*|amplif\w*|"
    r"polymeriz\w*|ferment\w*|cultur\w*|couple\w*|reflux\w*|stir\w*)\b",
    re.IGNORECASE,
)


def _load_protocol_parser():
    """Load `demo/parse_protocol.py` relative to `REPO_DIR` without package collisions."""
    cached = sys.modules.get(_PARSE_MODULE_KEY)
    if cached is not None:
        return cached
    module_path = REPO_DIR / "demo" / "parse_protocol.py"
    spec = importlib.util.spec_from_file_location(_PARSE_MODULE_KEY, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load protocol parser from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PARSE_MODULE_KEY] = module
    spec.loader.exec_module(module)
    return module


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def outward_round_band_hz(lo_hz: float, hi_hz: float, *, skirt_hz: float = 0.0) -> list[int]:
    """Conservatively round `[lo_hz, hi_hz]` outward to integer Hz (`[floor(lo - skirt), ceil(hi + skirt)]`)."""
    if not (_is_finite_number(lo_hz) and _is_finite_number(hi_hz) and _is_finite_number(skirt_hz)):
        raise ValueError("Band bounds and skirt_hz must be finite numbers")
    lo_f, hi_f, sk_f = float(lo_hz), float(hi_hz), float(skirt_hz)
    if lo_f < 0.0 or lo_f > hi_f or sk_f < 0.0:
        raise ValueError(f"Invalid frequency band [{lo_hz}, {hi_hz}] or skirt_hz={skirt_hz}")
    return [max(0, int(math.floor(lo_f - sk_f))), int(math.ceil(hi_f + sk_f))]


def verify_speech_privacy(
    retained_bands_hz: Any,
    raw_audio_saved: Any,
    *,
    raw_audio_kept: Any = False,
    speech_band_dropped: Any = True,
    sample_rate_hz: Any = DEFAULT_SAMPLE_RATE_HZ,
    window_skirt_hz: Any = 0.0,
) -> bool:
    """Verify Lean L2 privacy, non-empty band, Nyquist, and Hann-skirt invariants."""
    if raw_audio_saved is not False or raw_audio_kept is not False or speech_band_dropped is not True:
        return False
    if not _is_finite_number(sample_rate_hz) or float(sample_rate_hz) < 8000.0:
        return False
    if not _is_finite_number(window_skirt_hz) or float(window_skirt_hz) < 0.0:
        return False
    if not isinstance(retained_bands_hz, (list, tuple)) or len(retained_bands_hz) == 0:
        return False

    sr_f = float(sample_rate_hz)
    skirt_f = float(window_skirt_hz)
    for band in retained_bands_hz:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            return False
        lo_raw, hi_raw = band[0], band[1]
        if not (_is_finite_number(lo_raw) and _is_finite_number(hi_raw)):
            return False
        lo_hz, hi_hz = float(lo_raw), float(hi_raw)
        if lo_hz < 0.0 or lo_hz > hi_hz:
            return False
        if 2.0 * hi_hz > sr_f:
            return False
        if not ((hi_hz + skirt_f) <= SPEECH_BAND_LO_HZ or (lo_hz - skirt_f) >= SPEECH_BAND_HI_HZ):
            return False
    return True


def _extract_protocol_quantities(goal_text: str):
    parser = _load_protocol_parser()
    quantities = list(parser.parse_quantities(goal_text, include_concentration=True))
    if not any(q.label == "Temperature" for q in quantities):
        for match in _BARE_LIST_TEMP_RE.finditer(goal_text):
            start_idx, end_idx = match.span("raw")
            raw_tok = match.group("raw")
            trailing = goal_text[end_idx : min(len(goal_text), end_idx + 35)]
            if _NON_THERMAL_AFTER_RE.search(trailing):
                continue
            val = float(match.group("val"))
            # Guard carbon-13 / carbon-14 isotope tokens (e.g. "13C NMR", "14C-enriched")
            if val in (13.0, 14.0) and " " not in raw_tok:
                window = goal_text[max(0, start_idx - 25) : min(len(goal_text), end_idx + 40)]
                if _ISOTOPE_CTX_RE.search(window):
                    continue
            quantities.append(
                parser.ParsedQuantity(
                    label="Temperature",
                    start=start_idx,
                    end=end_idx,
                    raw=raw_tok,
                    value_min=val,
                    value_max=val,
                    unit="C",
                    canonical_value=val,
                    canonical_unit="C",
                    canonical_min=val,
                    canonical_max=val,
                )
            )
        quantities.sort(key=lambda item: (item.start, item.end, item.label))
    return quantities


def _span_to_dict(q: Any) -> dict[str, Any]:
    return {
        "label": q.label,
        "start": q.start,
        "end": q.end,
        "raw": q.raw,
        "unit": q.unit,
        "value_min": q.value_min,
        "value_max": q.value_max,
        "canonical_value": q.canonical_value,
        "canonical_min": q.canonical_min,
        "canonical_max": q.canonical_max,
        "canonical_unit": q.canonical_unit,
    }


def _classify_clause_role(clause_text: str) -> str:
    has_prep = bool(_PREP_VERB_RE.search(clause_text))
    has_primary = bool(_PRIMARY_VERB_RE.search(clause_text))
    if has_primary and not has_prep:
        return "primary"
    if has_prep:
        return "prep"
    return "step"


def _segment_protocol_steps(goal_text: str, quantities: list[Any]) -> list[dict[str, Any]]:
    if not goal_text.strip():
        return []

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for m in _STEP_SPLIT_RE.finditer(goal_text):
        if m.start() > cursor:
            bounds.append((cursor, m.start()))
        cursor = m.end()
    if cursor < len(goal_text):
        bounds.append((cursor, len(goal_text)))
    if not bounds:
        bounds = [(0, len(goal_text))]

    steps: list[dict[str, Any]] = []
    step_idx = 0
    for idx, (c_start, c_end) in enumerate(bounds):
        clause = goal_text[c_start:c_end].strip()
        if not clause:
            continue
        is_last = idx == len(bounds) - 1
        step_qs = [
            q
            for q in quantities
            if (c_start <= 0.5 * (q.start + q.end) < c_end)
            or (is_last and 0.5 * (q.start + q.end) >= c_start)
        ]

        vol_ul = vol_min = vol_max = None
        mass_mg = None
        temp_c = temp_min = temp_max = None
        dur_min = dur_min_lo = dur_min_hi = None
        speed_val = None
        speed_unit = None
        conc_mm = None

        for q in step_qs:
            if q.label == "Amount" and q.canonical_unit == "uL" and q.canonical_value is not None and vol_ul is None:
                vol_ul = float(q.canonical_value)
                vol_min = float(q.canonical_min) if q.canonical_min is not None else vol_ul
                vol_max = float(q.canonical_max) if q.canonical_max is not None else vol_ul
            elif q.label == "Amount" and q.canonical_unit == "mg" and q.canonical_value is not None and mass_mg is None:
                mass_mg = float(q.canonical_value)
            elif q.label == "Temperature" and q.canonical_unit == "C" and q.canonical_value is not None and temp_c is None:
                temp_c = float(q.canonical_value)
                temp_min = float(q.canonical_min) if q.canonical_min is not None else temp_c
                temp_max = float(q.canonical_max) if q.canonical_max is not None else temp_c
            elif q.label == "Time" and q.canonical_unit == "s" and q.canonical_value is not None and dur_min is None:
                dur_min = round(float(q.canonical_value) / 60.0, 6)
                dur_min_lo = (
                    round(float(q.canonical_min) / 60.0, 6) if q.canonical_min is not None else dur_min
                )
                dur_min_hi = (
                    round(float(q.canonical_max) / 60.0, 6) if q.canonical_max is not None else dur_min
                )
            elif q.label == "Speed" and q.canonical_value is not None and speed_val is None:
                speed_val = float(q.canonical_value)
                speed_unit = q.canonical_unit
            elif q.label == "Concentration" and q.canonical_unit == "mM" and q.canonical_value is not None and conc_mm is None:
                conc_mm = float(q.canonical_value)

        steps.append(
            {
                "step_index": step_idx,
                "role": _classify_clause_role(clause),
                "clause": clause,
                "volume_ul": vol_ul,
                "volume_ul_range": [vol_min, vol_max] if vol_ul is not None else None,
                "mass_mg": mass_mg,
                "temp_c": temp_c,
                "temp_c_range": [temp_min, temp_max] if temp_c is not None else None,
                "duration_min": dur_min,
                "duration_min_range": [dur_min_lo, dur_min_hi] if dur_min is not None else None,
                "speed_rpm_or_xg": speed_val,
                "speed_unit": speed_unit,
                "concentration_mm": conc_mm,
                "spans": [_span_to_dict(q) for q in step_qs],
            }
        )
        step_idx += 1
    return steps


def _select_primary_slot(
    steps: list[dict[str, Any]],
    key: str,
    range_key: str,
) -> tuple[float | None, list[float] | None, bool]:
    candidates = [s for s in steps if s.get(key) is not None]
    if not candidates:
        return None, None, False
    distinct_vals = {round(float(s[key]), 6) for s in candidates}
    ambiguous = len(distinct_vals) > 1
    role_rank = {"primary": 0, "step": 1, "prep": 2}
    chosen = min(candidates, key=lambda s: (role_rank.get(str(s.get("role", "step")), 1), int(s["step_index"])))
    return float(chosen[key]), chosen.get(range_key), ambiguous


def _coerce_obj_dict(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    if hasattr(item, "to_wire_dict") and callable(item.to_wire_dict):
        return dict(item.to_wire_dict())
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        d = dataclasses.asdict(item)
        if hasattr(item, "flagged"):
            try:
                d["flagged"] = bool(item.flagged)
            except ValueError:
                d["flagged"] = True
        return d
    if isinstance(item, dict):
        return dict(item)
    return None


def _inspect_acoustic_inputs(packet: dict[str, Any]) -> tuple[bool, bool, bool, dict[str, Any]]:
    """Inspect packet for AudioEvent and InterlockDecision records.

    Returns `(events_privacy_ok, interlock_halted, anomaly_halted, acoustic_telemetry)`.
    """
    raw_events: list[dict[str, Any]] = []
    raw_interlocks: list[dict[str, Any]] = []

    if packet.get("schema") == AUDIO_EVENT_SCHEMA or (
        "score" in packet and "threshold" in packet and "detector" in packet
    ):
        raw_events.append(dict(packet))

    if "audio_event" in packet and packet["audio_event"] is not None:
        coerced = _coerce_obj_dict(packet["audio_event"])
        if coerced is not None:
            raw_events.append(coerced)

    if isinstance(packet.get("audio_events"), (list, tuple)):
        for item in packet["audio_events"]:
            coerced = _coerce_obj_dict(item)
            if coerced is not None:
                raw_events.append(coerced)

    if "halt_run" in packet and ("cavitation_ratio" in packet or "stall_ratio" in packet):
        raw_interlocks.append(dict(packet))

    if "interlock_decision" in packet and packet["interlock_decision"] is not None:
        coerced = _coerce_obj_dict(packet["interlock_decision"])
        if coerced is not None:
            raw_interlocks.append(coerced)

    if isinstance(packet.get("interlock_decisions"), (list, tuple)):
        for item in packet["interlock_decisions"]:
            coerced = _coerce_obj_dict(item)
            if coerced is not None:
                raw_interlocks.append(coerced)

    events_privacy_ok = True
    interlock_halted = False
    anomaly_halted = False
    normalized_records: list[dict[str, Any]] = []

    for ev in raw_events:
        ev_raw_saved = ev.get("raw_audio_saved", False)
        ev_raw_kept = ev.get("raw_audio_kept", False)
        ev_speech_dropped = ev.get("speech_band_dropped", True)
        ev_sr = ev.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ)
        ev_bands = ev.get("retained_bands_hz", DEFAULT_RETAINED_BANDS_HZ)
        if not verify_speech_privacy(
            ev_bands,
            ev_raw_saved,
            raw_audio_kept=ev_raw_kept,
            speech_band_dropped=ev_speech_dropped,
            sample_rate_hz=ev_sr,
        ):
            events_privacy_ok = False

        score_raw = ev.get("score", 0.0)
        thr_raw = ev.get("threshold", 0.0)
        if not (_is_finite_number(score_raw) and _is_finite_number(thr_raw)):
            anomaly_halted = True
            flagged = True
            score_f = float("inf")
            thr_f = 0.0
            reason = "non_finite_anomaly_score"
        else:
            score_f = float(score_raw)
            thr_f = float(thr_raw)
            flagged = bool(ev.get("flagged") is True or score_f > thr_f)
            if flagged:
                anomaly_halted = True
            reason = "acoustic_anomaly_exceeded_threshold" if flagged else "nominal"

        normalized_records.append(
            {
                "kind": "audio_event",
                "step": int(ev.get("step", 0)),
                "instrument_id": str(ev.get("instrument_id", "unknown")),
                "detector": str(ev.get("detector", "unknown")),
                "score": score_f if math.isfinite(score_f) else None,
                "threshold": thr_f,
                "flagged": flagged,
                "cavitation_ratio": 0.0,
                "stall_ratio": 0.0,
                "fault_reason": reason,
            }
        )

    for dec in raw_interlocks:
        dec_raw_saved = dec.get("raw_audio_saved", False)
        dec_raw_kept = dec.get("raw_audio_kept", False)
        dec_speech_dropped = dec.get("speech_band_dropped", True)
        dec_sr = dec.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ)
        dec_bands = dec.get("retained_bands_hz", DEFAULT_RETAINED_BANDS_HZ)
        if not verify_speech_privacy(
            dec_bands,
            dec_raw_saved,
            raw_audio_kept=dec_raw_kept,
            speech_band_dropped=dec_speech_dropped,
            sample_rate_hz=dec_sr,
        ):
            events_privacy_ok = False

        cav_raw = dec.get("cavitation_ratio", 0.0)
        stall_raw = dec.get("stall_ratio", 0.0)
        score_raw = dec.get("score", 0.0)
        if not (_is_finite_number(cav_raw) and _is_finite_number(stall_raw) and _is_finite_number(score_raw)):
            interlock_halted = True
            halt_flag = True
            cav_f = 0.0
            stall_f = 0.0
            score_f = None
            reason = "non_finite_interlock_telemetry"
        else:
            cav_f = float(cav_raw)
            stall_f = float(stall_raw)
            score_f = float(score_raw)
            halt_flag = bool(dec.get("halt_run", False))
            if halt_flag:
                interlock_halted = True
            reason = str(dec.get("reason", "interlock_halt" if halt_flag else "nominal"))

        normalized_records.append(
            {
                "kind": "cross_modal_interlock",
                "step": int(dec.get("step_index", dec.get("step", 0))),
                "instrument_id": str(dec.get("instrument_id", "liquid-handler-01")),
                "detector": "cross_modal_interlock",
                "score": score_f,
                "threshold": 1.0,
                "flagged": halt_flag,
                "cavitation_ratio": cav_f,
                "stall_ratio": stall_f,
                "fault_reason": reason,
            }
        )

    flagged_records = [r for r in normalized_records if r["flagged"]]
    primary = flagged_records[0] if flagged_records else (normalized_records[0] if normalized_records else None)

    telemetry: dict[str, Any] = {
        "step": primary["step"] if primary is not None else None,
        "instrument_id": primary["instrument_id"] if primary is not None else None,
        "detector": primary["detector"] if primary is not None else None,
        "score": primary["score"] if primary is not None else None,
        "threshold": primary["threshold"] if primary is not None else None,
        "cavitation_ratio": primary["cavitation_ratio"] if primary is not None else 0.0,
        "stall_ratio": primary["stall_ratio"] if primary is not None else 0.0,
        "fault_reason": primary["fault_reason"] if primary is not None else "nominal",
        "flagged": bool(flagged_records),
        "flagged_steps": [r["step"] for r in flagged_records],
        "events": normalized_records,
    }
    return events_privacy_ok, interlock_halted, anomaly_halted, telemetry


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Process an `autolab-wire/v1` packet through `lab-ears` (Layer L2)."""
    incoming = packet if isinstance(packet, dict) else {}
    out: dict[str, Any] = copy.deepcopy(incoming)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if out["halted"] is True:
        return out

    goal_text = str(out.get("goal_text", "") or "")
    retained_bands_hz = out.get("retained_bands_hz", copy.deepcopy(DEFAULT_RETAINED_BANDS_HZ))
    sample_rate_hz = out.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ)
    window_skirt_hz = out.get("window_skirt_hz", 0.0)

    raw_audio_saved_in = out.get("raw_audio_saved", False)
    raw_audio_kept_in = out.get("raw_audio_kept", False)
    speech_dropped_in = incoming.get("speech_band_dropped", True)

    quantities = _extract_protocol_quantities(goal_text)
    spans_list = [_span_to_dict(q) for q in quantities]
    steps_list = _segment_protocol_steps(goal_text, quantities)

    amounts = [q.raw for q in quantities if q.label == "Amount"]
    temperatures = [q.raw for q in quantities if q.label == "Temperature"]
    times = [q.raw for q in quantities if q.label == "Time"]
    speeds = [q.raw for q in quantities if q.label == "Speed"]
    concentrations = [q.raw for q in quantities if q.label == "Concentration"]

    vol_sel, vol_range, vol_amb = _select_primary_slot(steps_list, "volume_ul", "volume_ul_range")
    temp_sel, temp_range, temp_amb = _select_primary_slot(steps_list, "temp_c", "temp_c_range")
    dur_sel, dur_range, dur_amb = _select_primary_slot(steps_list, "duration_min", "duration_min_range")

    defaults_applied: list[str] = []
    if vol_sel is None:
        volume_ul = DEFAULT_VOLUME_UL
        defaults_applied.append("volume_ul")
    else:
        volume_ul = vol_sel

    if temp_sel is None:
        temp_c = DEFAULT_TEMP_C
        defaults_applied.append("temp_c")
    else:
        temp_c = temp_sel

    if dur_sel is None:
        duration_min = DEFAULT_DURATION_MIN
        defaults_applied.append("duration_min")
    else:
        duration_min = dur_sel

    events_privacy_ok, interlock_halted, anomaly_halted, telemetry = _inspect_acoustic_inputs(incoming)

    top_privacy_ok = verify_speech_privacy(
        retained_bands_hz,
        raw_audio_saved_in,
        raw_audio_kept=raw_audio_kept_in,
        speech_band_dropped=speech_dropped_in,
        sample_rate_hz=sample_rate_hz,
        window_skirt_hz=window_skirt_hz,
    )
    privacy_ok = bool(top_privacy_ok and events_privacy_ok)

    # Unify raw_audio_saved and raw_audio_kept on the outgoing packet
    any_raw_retained = bool(raw_audio_saved_in is not False or raw_audio_kept_in is not False)
    out["raw_audio_saved"] = any_raw_retained
    out["raw_audio_kept"] = any_raw_retained

    out["parsed_spans"] = {
        "spans": spans_list,
        "steps": steps_list,
        "amounts": amounts,
        "temperatures": temperatures,
        "times": times,
        "speeds": speeds,
        "concentrations": concentrations,
        "volume_ul": volume_ul,
        "volume_ul_range": vol_range,
        "temp_c": temp_c,
        "temp_c_range": temp_range,
        "duration_min": duration_min,
        "duration_min_range": dur_range,
        "defaults_applied": defaults_applied,
        "multi_step_ambiguous": bool(vol_amb or temp_amb or dur_amb),
    }
    out["acoustic_telemetry"] = telemetry

    if not privacy_ok:
        out["halted"] = True
        out["halt_reason"] = "L2_SPEECH_PRIVACY_BLOCK"
        out["speech_band_dropped"] = False
        status = "BLOCK"
        summary = (
            "Speech-band privacy check failed: retained bands overlap (300, 3400) Hz, "
            "exceed Nyquist, or raw audio was retained."
        )
    elif interlock_halted:
        out["halted"] = True
        out["halt_reason"] = "L2_CAVITATION_INTERLOCK_HALT"
        out["speech_band_dropped"] = True
        status = "BLOCK"
        summary = (
            f"Cross-modal pipetting interlock halted at step {telemetry['step']} "
            f"({telemetry['fault_reason']}: cavitation_ratio={telemetry['cavitation_ratio']:.3f}, "
            f"stall_ratio={telemetry['stall_ratio']:.3f})."
        )
    elif anomaly_halted:
        out["halted"] = True
        out["halt_reason"] = "L2_ACOUSTIC_ANOMALY_HALT"
        out["speech_band_dropped"] = True
        status = "BLOCK"
        summary = (
            f"Calibrated acoustic monitor halted at step {telemetry['step']} on "
            f"{telemetry['instrument_id']} (score={telemetry['score']} > threshold={telemetry['threshold']})."
        )
    else:
        out["halted"] = False
        out["halt_reason"] = None
        out["speech_band_dropped"] = True
        status = "PASS"
        summary = (
            f"Extracted {len(spans_list)} span(s) across {len(steps_list)} step(s) "
            f"(volume_ul={volume_ul}, temp_c={temp_c}, duration_min={duration_min}); "
            "speech band (300, 3400) Hz dropped."
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": status,
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "invariant": "full_skirt_speech_band_drop",
            "first_principles": (
                "Human formant bands [300, 3400] Hz are stripped before feature storage "
                "while preserving sub-300 Hz rotor harmonics and >3.4 kHz cavitation cues."
            ),
        }
    )
    if privacy_ok and (interlock_halted or anomaly_halted):
        out["bottom_reasoning_traces"].append(
            {
                "repo": REPO_ID,
                "invariant": (
                    "cross_modal_cavitation_or_stall_interlock"
                    if interlock_halted
                    else "calibrated_anomaly_threshold"
                ),
                "first_principles": (
                    "High-band (3.6-7.8 kHz) meniscus air-break cavitation and low-band "
                    "(60-260 Hz) stepper-stall harmonics halt execution before air bubbles "
                    "or mechanical faults corrupt downstream assay steps."
                ),
            }
        )
    return out


if __name__ == "__main__":
    sample = {
        "goal_text": "Transfer 150 uL aspirin stock, incubate at 37 C for 20 min, and spin at 1200 rpm."
    }
    print(json.dumps(process_packet(sample), indent=2))
