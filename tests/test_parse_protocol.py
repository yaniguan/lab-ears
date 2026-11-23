"""Offline pytest suite for demo/parse_protocol.py covering issues #3..#12, #24, #25."""

from __future__ import annotations

from pathlib import Path

import pytest

from demo.parse_protocol import (
    ParsedQuantity,
    cli,
    count_overlap_matches,
    evaluate_folder,
    find_spans,
    gold_spans,
    gold_spans_with_stats,
    main,
    overlaps,
    parse_quantities,
)


def _extract_texts(text: str, *, include_concentration: bool = False):
    return {(lbl, text[s:e]) for lbl, s, e in find_spans(text, include_concentration=include_concentration)}


# --- Issue #3: Gram mass vs. centrifuge Speed disambiguation ---
def test_gram_mass_vs_centrifuge_speed():
    assert _extract_texts("Add 5 g NaCl and dissolve 0.5 g agarose.") == {
        ("Amount", "5 g"),
        ("Amount", "0.5 g"),
    }
    assert _extract_texts("Resuspend the cell pellet in 2.5 g sucrose.") == {
        ("Amount", "2.5 g"),
    }
    assert _extract_texts("Spin at 12000 x g for 10 min.") == {
        ("Speed", "12000 x g"),
        ("Time", "10 min"),
    }
    assert _extract_texts("Centrifuge at 1000g for 5 minutes.") == {
        ("Speed", "1000g"),
        ("Time", "5 minutes"),
    }
    assert _extract_texts("Spin down @ 16,000g for 1 min.") == {
        ("Speed", "16,000g"),
        ("Time", "1 min"),
    }


# --- Issue #4: Negative temperatures and sub-zero ranges ---
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Store aliquots at -20 °C.", "-20 °C"),
        ("Snap-freeze and transfer to -80°C freezer.", "-80°C"),
        ("Keep between -20 to -70 °C.", "-20 to -70 °C"),
        ("Freeze at −196 °C in liquid nitrogen.", "−196 °C"),
        ("Keep at –20°C overnight.", "–20°C"),
        ("Store buffer at 4-8 °C.", "4-8 °C"),
    ],
)
def test_negative_and_range_temperatures(text: str, expected: str):
    temps = {text[s:e] for lbl, s, e in find_spans(text) if lbl == "Temperature"}
    assert temps == {expected}


# --- Issue #5: Case-sensitive bare C and K without false positives ---
def test_bare_c_k_case_sensitivity_and_context():
    assert ("Temperature", "10 k") not in _extract_texts("Sequence 10 k reads per sample.")
    assert ("Temperature", "2 c") not in _extract_texts("Inspect lane 2 c on the gel.")
    assert ("Temperature", "10K") not in _extract_texts("Concentrate with a 10K MWCO spin column.")
    assert ("Temperature", "13C") not in _extract_texts("Add 13C labeled glucose.")
    assert ("Temperature", "37C") in _extract_texts("Incubate at 37C for 1 h.")
    assert ("Temperature", "298 K") in _extract_texts("Equilibrate chamber temperature at 298 K.")
    assert ("Temperature", "37 degrees Celsius") in _extract_texts("Warm to 37 degrees Celsius.")


# --- Issue #6: RT-PCR exclusion and dry ice vs. wet ice ---
def test_qualitative_temperature_rt_pcr_and_dry_ice():
    assert _extract_texts("Perform one-step RT-PCR and RT-qPCR.") == set()
    assert _extract_texts("Prepare RT reaction buffer and RT enzyme mix.") == set()
    assert _extract_texts("Incubate at RT for 15 min.") == {
        ("Temperature", "RT"),
        ("Time", "15 min"),
    }
    assert _extract_texts("Place tubes on dry ice immediately.") == {
        ("Temperature", "on dry ice"),
    }
    assert _extract_texts("Keep samples chilled on ice.") == {
        ("Temperature", "on ice"),
    }


# --- Issue #7: Amount units, Unicode micro symbols, enzyme units, and Concentration ---
def test_amount_unicode_units_and_concentration_mode():
    txt = "Mix 10 µL primer, 10 μL template, 200 μl/well buffer, 1 liter H2O, and 5 U Taq."
    got = _extract_texts(txt)
    assert got == {
        ("Amount", "10 µL"),
        ("Amount", "10 μL"),
        ("Amount", "200 μl/well"),
        ("Amount", "1 liter"),
        ("Amount", "5 U"),
    }
    # Bare lowercase 'l' should not falsely match list enumerations
    assert _extract_texts("Step 1 l: vortex briefly.") == set()

    conc_txt = "Wash with 70% ethanol in 10 mM Tris (1X stock, 20 mg/mL BSA)."
    with_conc = _extract_texts(conc_txt, include_concentration=True)
    assert ("Concentration", "70%") in with_conc
    assert ("Concentration", "10 mM") in with_conc
    assert ("Concentration", "1X") in with_conc
    assert ("Concentration", "20 mg/mL") in with_conc


# --- Issue #8: Speed trailing word boundaries and RCF support ---
def test_speed_word_boundaries_and_rcf():
    assert ("Speed", "2 xg") not in _extract_texts("Add 2 xgal substrate.")
    got_xgal = _extract_texts("Add 20 mg/mL X-gal solution.", include_concentration=True)
    assert not any(lbl == "Speed" for lbl, _ in got_xgal)
    assert _extract_texts("Centrifuge at 5000 RCF or 3000 rpm.") == {
        ("Speed", "5000 RCF"),
        ("Speed", "3000 rpm"),
    }


# --- Issue #9: Discontinuous BRAT annotations and malformed-line guards ---
def test_gold_spans_discontinuous_and_malformed_guards(tmp_path: Path):
    ann = tmp_path / "sample.ann"
    ann.write_text(
        "\n".join(
            [
                "# comment starting with T? no, comment",
                "T_not_entity_id",
                "T1\tAmount",
                "T2\tAmount bad coords\tbad",
                "T3\tAmount 20 10\tinverted",
                "T4\tAmount 4 6;11 16\t10 20 µL",
                "T5\tTemperature 20 25\t-80°C",
                "R1\tArg1:T4 Arg2:T5",
            ]
        ),
        encoding="utf-8",
    )
    spans_inc, stats_inc = gold_spans_with_stats(ann, include_discontinuous=True)
    assert spans_inc == {("Amount", 4, 6), ("Amount", 11, 16), ("Temperature", 20, 25)}
    assert stats_inc["discontinuous_entities"] == 1
    assert stats_inc["skipped_discontinuous"] == 0
    assert stats_inc["malformed_lines"] == 4

    spans_skip, stats_skip = gold_spans_with_stats(ann, include_discontinuous=False)
    assert spans_skip == {("Temperature", 20, 25)}
    assert stats_skip["skipped_discontinuous"] == 1
    assert gold_spans(ann) == spans_inc


# --- Issue #10: 1-to-1 bipartite overlap matching + R_overlap / F1_overlap ---
def test_bipartite_overlap_matching_no_double_counting():
    assert overlaps(("Amount", 0, 5), ("Amount", 3, 8))
    assert not overlaps(("Amount", 0, 3), ("Amount", 3, 8))
    assert not overlaps(("Amount", 0, 5), ("Time", 0, 5))

    # Two fragmented predicted spans inside a single gold span must count as at most 1 match
    preds = [("Amount", 0, 2), ("Amount", 6, 11)]
    golds = [("Amount", 0, 11)]
    assert count_overlap_matches(preds, golds) == 1


# --- Issue #11 & #12 & #24: Missing/empty folder non-zero exit & synthetic benchmark F1 gate ---
def test_missing_or_empty_folder_exits_nonzero(tmp_path: Path):
    with pytest.raises(SystemExit) as exc_missing:
        main(tmp_path / "does_not_exist")
    assert exc_missing.value.code == 1

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as exc_empty:
        main(empty_dir)
    assert exc_empty.value.code == 1

    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    (orphan_dir / "p1.ann").write_text("T1\tAmount 0 5\t10 uL\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_orphan:
        main(orphan_dir)
    assert exc_orphan.value.code == 1


def test_synthetic_corpus_evaluation_and_f1_gate(tmp_path: Path):
    txt1 = "Add 0.5 mL lysis buffer, incubate at -20 °C for 15 min, then spin at 12000 x g."
    ann1_lines = []
    for idx, (lbl, sub) in enumerate(
        [("Amount", "0.5 mL"), ("Temperature", "-20 °C"), ("Time", "15 min"), ("Speed", "12000 x g")],
        start=1,
    ):
        s = txt1.index(sub)
        ann1_lines.append(f"T{idx}\t{lbl} {s} {s + len(sub)}\t{sub}")
    (tmp_path / "p1.txt").write_text(txt1, encoding="utf-8")
    (tmp_path / "p1.ann").write_text("\n".join(ann1_lines) + "\n", encoding="utf-8")

    res = evaluate_folder(tmp_path)
    assert res["protocols"] == 1
    for lbl in ("Amount", "Temperature", "Time", "Speed"):
        m = res["metrics"][lbl]
        assert m["F1_exact"] == pytest.approx(1.0)
        assert m["P_overlap"] == pytest.approx(1.0)
        assert m["R_overlap"] == pytest.approx(1.0)
        assert m["F1_overlap"] == pytest.approx(1.0)

    assert cli([str(tmp_path), "--min-f1", "0.95"]) == 0


# --- Issue #25: Canonical unit normalization for stepbot / YLM ---
def test_parse_quantities_canonical_normalization():
    text = (
        "Transfer 0.5 mL buffer and 10-20 µL enzyme, incubate at 37 °C for 15 min, "
        "chill on dry ice, store at 68 °F overnight, and spin at 12,000 x g or 3000 rpm."
    )
    qs = {q.raw: q for q in parse_quantities(text)}

    q_ml: ParsedQuantity = qs["0.5 mL"]
    assert q_ml.label == "Amount"
    assert q_ml.canonical_unit == "uL"
    assert q_ml.canonical_value == pytest.approx(500.0)

    q_range = qs["10-20 µL"]
    assert q_range.value_min == pytest.approx(10.0)
    assert q_range.value_max == pytest.approx(20.0)
    assert q_range.canonical_unit == "uL"
    assert q_range.canonical_value == pytest.approx(15.0)

    q_temp = qs["37 °C"]
    assert q_temp.canonical_unit == "C"
    assert q_temp.canonical_value == pytest.approx(37.0)

    q_dry = qs["on dry ice"]
    assert q_dry.canonical_unit == "C"
    assert q_dry.canonical_value == pytest.approx(-78.5)

    q_f = qs["68 °F"]
    assert q_f.canonical_unit == "C"
    assert q_f.canonical_value == pytest.approx(20.0)

    q_min = qs["15 min"]
    assert q_min.canonical_unit == "s"
    assert q_min.canonical_value == pytest.approx(900.0)

    q_on = qs["overnight"]
    assert q_on.canonical_unit == "s"
    assert q_on.canonical_value == pytest.approx(43200.0)

    q_xg = qs["12,000 x g"]
    assert q_xg.canonical_unit == "xg"
    assert q_xg.canonical_value == pytest.approx(12000.0)

    q_rpm = qs["3000 rpm"]
    assert q_rpm.canonical_unit == "rpm"
    assert q_rpm.canonical_value == pytest.approx(3000.0)
