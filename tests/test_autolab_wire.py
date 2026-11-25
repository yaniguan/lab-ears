"""Unit tests for the Auto-Lab 15-repository wire node (`autolab_wire.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import autolab_wire  # noqa: E402
from autolab_wire import LAYER, REPO_DIR, REPO_ID, process_packet  # noqa: E402


def test_process_packet_normal_protocol_extracts_ul_c_min_rpm() -> None:
    assert REPO_ID == "lab-ears"
    assert LAYER == "L2"
    assert REPO_DIR == _ROOT

    packet = {
        "goal_text": (
            "Transfer 150 uL aspirin solution, incubate at 37 C for 20 min, "
            "and spin at 1200 rpm."
        )
    }
    out = process_packet(packet)

    assert "parsed_spans" not in packet
    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["halt_reason"] is None
    assert out["speech_band_dropped"] is True

    parsed = out["parsed_spans"]
    assert parsed["amounts"] == ["150 uL"]
    assert parsed["temperatures"] == ["37 C"]
    assert parsed["times"] == ["20 min"]
    assert parsed["speeds"] == ["1200 rpm"]
    assert parsed["volume_ul"] == pytest.approx(150.0)
    assert parsed["temp_c"] == pytest.approx(37.0)
    assert parsed["duration_min"] == pytest.approx(20.0)
    assert len(parsed["spans"]) == 4

    assert len(out["trace"]) == 1
    assert out["trace"][0]["repo"] == "lab-ears"
    assert out["trace"][0]["layer"] == "L2"
    assert out["trace"][0]["status"] == "PASS"

    assert out["bottom_reasoning_traces"] == [
        {
            "repo": "lab-ears",
            "invariant": "full_skirt_speech_band_drop",
            "first_principles": (
                "Human formant bands [300, 3400] Hz are stripped before feature storage "
                "while preserving sub-300 Hz rotor harmonics and >3.4 kHz cavitation cues."
            ),
        }
    ]


def test_process_packet_defaults_and_unit_normalization() -> None:
    empty_out = process_packet({})
    assert empty_out["halted"] is False
    assert empty_out["parsed_spans"]["volume_ul"] == pytest.approx(100.0)
    assert empty_out["parsed_spans"]["temp_c"] == pytest.approx(25.0)
    assert empty_out["parsed_spans"]["duration_min"] == pytest.approx(15.0)

    unit_out = process_packet(
        {
            "goal_text": (
                "Dispense 0.25 mL buffer, warm to 68 degrees F for 90 s, "
                "and centrifuge at 2500 rpm."
            )
        }
    )
    assert unit_out["parsed_spans"]["volume_ul"] == pytest.approx(250.0)
    assert unit_out["parsed_spans"]["temp_c"] == pytest.approx(20.0)
    assert unit_out["parsed_spans"]["duration_min"] == pytest.approx(1.5)
    assert unit_out["parsed_spans"]["speeds"] == ["2500 rpm"]


@pytest.mark.parametrize(
    "bad_packet",
    [
        {"goal_text": "Add 100 uL buffer", "retained_bands_hz": [[250, 500], [3500, 7800]]},
        {"goal_text": "Add 100 uL buffer", "retained_bands_hz": [[1000, 2400]]},
        {"goal_text": "Add 100 uL buffer", "retained_bands_hz": [[50, 280], [3200, 7800]]},
        {"goal_text": "Add 100 uL buffer", "retained_bands_hz": [[50, 7800]]},
        {"goal_text": "Add 100 uL buffer", "retained_bands_hz": [[280, 50]]},
        {
            "goal_text": "Add 100 uL buffer",
            "retained_bands_hz": [[50, 280], [3500, 7800]],
            "raw_audio_saved": True,
        },
    ],
)
def test_process_packet_blocks_speech_privacy_violations(bad_packet: dict) -> None:
    out = process_packet(bad_packet)
    assert out["halted"] is True
    assert out["halt_reason"] == "L2_SPEECH_PRIVACY_BLOCK"
    assert out["speech_band_dropped"] is False
    assert out["trace"][-1]["repo"] == "lab-ears"
    assert out["trace"][-1]["layer"] == "L2"
    assert out["trace"][-1]["status"] == "BLOCK"
    assert out["bottom_reasoning_traces"][-1]["invariant"] == "full_skirt_speech_band_drop"


def test_process_packet_allows_exact_outer_speech_band_boundaries() -> None:
    out = process_packet(
        {
            "goal_text": "Transfer 80 uL sample at 4 °C for 5 min.",
            "retained_bands_hz": [[50, 300], [3400, 7800]],
            "raw_audio_saved": False,
        }
    )
    assert out["halted"] is False
    assert out["speech_band_dropped"] is True


def test_process_packet_halted_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    upstream = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "UPSTREAM_BLOCK",
        "goal_text": "Transfer 150 uL at 37 C for 20 min at 1200 rpm.",
        "trace": [{"repo": "upstream-node", "layer": "L6", "status": "BLOCK", "summary": "halted"}],
        "bottom_reasoning_traces": [],
    }
    out = autolab_wire.process_packet(upstream)
    assert out == upstream
    assert out is not upstream
    assert "parsed_spans" not in out
    assert len(out["trace"]) == 1


def test_issue31_multistep_segmentation_ranges_concentrations_and_13c_nmr() -> None:
    p_multi = process_packet(
        {
            "goal_text": (
                "Thaw 10 uL enzyme on ice for 2 min, then react 150 uL aspirin substrate "
                "at 37 C for 45 min."
            )
        }
    )
    ps = p_multi["parsed_spans"]
    assert ps["volume_ul"] == pytest.approx(150.0)
    assert ps["temp_c"] == pytest.approx(37.0)
    assert ps["duration_min"] == pytest.approx(45.0)
    assert ps["multi_step_ambiguous"] is True
    assert ps["defaults_applied"] == []
    assert len(ps["steps"]) == 2
    assert ps["steps"][0]["role"] == "prep"
    assert ps["steps"][0]["volume_ul"] == pytest.approx(10.0)
    assert ps["steps"][0]["temp_c"] == pytest.approx(0.0)
    assert ps["steps"][0]["duration_min"] == pytest.approx(2.0)
    assert ps["steps"][1]["role"] == "primary"
    assert ps["steps"][1]["volume_ul"] == pytest.approx(150.0)

    p_nmr = process_packet(
        {"goal_text": "Assay: 13C NMR verification; dissolve 25 mg aspirin in 10 mM Tris."}
    )
    ps_nmr = p_nmr["parsed_spans"]
    assert ps_nmr["temperatures"] == []
    assert ps_nmr["temp_c"] == pytest.approx(25.0)
    assert ps_nmr["concentrations"] == ["10 mM"]
    assert set(ps_nmr["defaults_applied"]) == {"volume_ul", "temp_c", "duration_min"}

    p_range = process_packet(
        {"goal_text": "Incubate 50-100 uL reaction at RT for 10-20 min."}
    )
    ps_r = p_range["parsed_spans"]
    assert ps_r["volume_ul_range"] == pytest.approx([50.0, 100.0])
    assert ps_r["temp_c_range"] == pytest.approx([20.0, 25.0])
    assert ps_r["duration_min_range"] == pytest.approx([10.0, 20.0])
    assert all("canonical_min" in sp and "canonical_max" in sp for sp in ps_r["spans"])


def test_issue32_audio_event_privacy_unification_and_fault_halts() -> None:
    import json
    from acoustic.record import AudioEvent, now_utc
    from community_extensions.cross_modal_interlock import InterlockDecision

    # 1. AudioEvent with raw_audio_kept=True must trigger L2_SPEECH_PRIVACY_BLOCK
    ev_raw = AudioEvent(
        protocol_id="pcr-v2",
        step=4,
        instrument_id="centrifuge-01",
        start_utc=now_utc(),
        duration_s=10.0,
        detector="gaussian",
        score=25.0,
        threshold=10.0,
        speech_band_dropped=False,
        raw_audio_kept=True,
    )
    pkt_raw = json.loads(ev_raw.to_json())
    pkt_raw["goal_text"] = "Spin at 12000 rpm for 10 min."
    out_raw = process_packet(pkt_raw)
    assert out_raw["halted"] is True
    assert out_raw["halt_reason"] == "L2_SPEECH_PRIVACY_BLOCK"
    assert out_raw["speech_band_dropped"] is False
    assert out_raw["raw_audio_saved"] is True
    assert out_raw["raw_audio_kept"] is True

    # 2. Privacy-clean flagged AudioEvent halts with L2_ACOUSTIC_ANOMALY_HALT
    ev_anom = AudioEvent(
        protocol_id="pcr-v2",
        step=4,
        instrument_id="centrifuge-01",
        start_utc=now_utc(),
        duration_s=10.0,
        detector="gaussian",
        score=25.0,
        threshold=10.0,
    )
    out_anom = process_packet({"goal_text": "Spin at 12000 rpm for 10 min.", "audio_events": [ev_anom]})
    assert out_anom["halted"] is True
    assert out_anom["halt_reason"] == "L2_ACOUSTIC_ANOMALY_HALT"
    assert out_anom["speech_band_dropped"] is True
    assert out_anom["acoustic_telemetry"]["flagged"] is True
    assert out_anom["acoustic_telemetry"]["step"] == 4

    # 3. CrossModalInterlock cavitation halt propagates L2_CAVITATION_INTERLOCK_HALT
    dec_cav = InterlockDecision(
        step_index=7,
        halt_run=True,
        reason="dry_aspiration_cavitation",
        raw_audio_saved=False,
        speech_band_dropped=True,
        cavitation_ratio=3.8,
        stall_ratio=0.9,
        score=1.55,
    )
    out_cav = process_packet({"goal_text": "Aspirate 50 uL buffer.", "interlock_decision": dec_cav})
    assert out_cav["halted"] is True
    assert out_cav["halt_reason"] == "L2_CAVITATION_INTERLOCK_HALT"
    assert out_cav["speech_band_dropped"] is True
    assert out_cav["acoustic_telemetry"]["fault_reason"] == "dry_aspiration_cavitation"
    assert out_cav["acoustic_telemetry"]["cavitation_ratio"] == pytest.approx(3.8)

    # 4. Nyquist, empty-band, Hann-skirt, and outward rounding guards (#30)
    assert autolab_wire.outward_round_band_hz(50.0, 300.4) == [50, 301]
    assert process_packet({"retained_bands_hz": []})["halt_reason"] == "L2_SPEECH_PRIVACY_BLOCK"
    assert (
        process_packet({"sample_rate_hz": 8000, "retained_bands_hz": [[60, 260], [3600, 7800]]})["halt_reason"]
        == "L2_SPEECH_PRIVACY_BLOCK"
    )
    assert (
        process_packet({"window_skirt_hz": 32.0, "retained_bands_hz": [[60, 285], [3600, 7800]]})["halt_reason"]
        == "L2_SPEECH_PRIVACY_BLOCK"
    )

