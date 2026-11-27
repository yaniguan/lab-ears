"""Offline unit tests for `protocol/learned_tagger.py` and README/doc metric parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.parse_protocol import main as parse_protocol_main
from protocol import (
    CORE_LABELS,
    LinearChainSpanTagger,
    evaluate_external_jsonl,
    load_tagger_bundle,
    save_tagger_bundle,
    select_hyperparameters_on_dev,
    tokenize_protocol_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_brat_doc(folder: Path, stem: str, text: str, spans: list[tuple[str, str]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.txt").write_text(text, encoding="utf-8")
    ann_lines: list[str] = []
    cursor = 0
    for idx, (lbl, sub) in enumerate(spans, start=1):
        s = text.index(sub, cursor)
        e = s + len(sub)
        cursor = e
        ann_lines.append(f"T{idx}\t{lbl} {s} {e}\t{sub}")
    (folder / f"{stem}.ann").write_text("\n".join(ann_lines) + "\n", encoding="utf-8")


def test_tokenizer_preserves_exact_offsets_and_subtoken_boundaries() -> None:
    text = "Incubate (37°C) with 100µl buffer, spin at 14,000xg for 5 min.\n"
    seqs = tokenize_protocol_lines(text)
    assert len(seqs) == 1
    toks = seqs[0]
    for tok, s, e in toks:
        assert text[s:e] == tok
    words = [t[0] for t in toks]
    assert ["(", "37", "°", "C", ")"] == words[1:6]
    assert ["100", "µl"] == words[7:9]
    assert ["14,000", "xg"] == words[13:15]


def test_linear_chain_tagger_zero_leakage_bundle_roundtrip_and_transfer(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    dev_dir = tmp_path / "dev"
    test_dir = tmp_path / "test"

    train_samples = [
        (
            "Add 200 uL lysis buffer, incubate at 37 °C for 15 min, and spin at 12,000 x g.",
            [("Amount", "200 uL"), ("Temperature", "37 °C"), ("Time", "15 min"), ("Speed", "12,000 x g")],
        ),
        (
            "Transfer 50 uL sample, heat at 95 °C for 5 min, then centrifuge at 3000 rpm.",
            [("Amount", "50 uL"), ("Temperature", "95 °C"), ("Time", "5 min"), ("Speed", "3000 rpm")],
        ),
        (
            "Mix equal volume ethanol, chill at -20 °C overnight, and spin at maximum speed.",
            [
                ("Amount", "equal volume"),
                ("Temperature", "-20 °C"),
                ("Time", "overnight"),
                ("Speed", "maximum speed"),
            ],
        ),
        (
            "Perform RT-PCR setup: add 10 uL master mix at room temperature for 10 min.",
            [("Amount", "10 uL"), ("Temperature", "room temperature"), ("Time", "10 min")],
        ),
    ]
    for i, (txt, sps) in enumerate(train_samples):
        _write_brat_doc(train_dir, f"tr_{i}", txt, sps)

    _write_brat_doc(
        dev_dir,
        "dv_0",
        "Add 100 uL elution buffer, incubate at 65 °C for 10 min, and spin at 14,000 x g.",
        [("Amount", "100 uL"), ("Temperature", "65 °C"), ("Time", "10 min"), ("Speed", "14,000 x g")],
    )
    _write_brat_doc(
        test_dir,
        "te_0",
        "Dispense 25 uL enzyme, incubate at 37 °C for 20 min, and centrifuge at 5000 rpm.",
        [("Amount", "25 uL"), ("Temperature", "37 °C"), ("Time", "20 min"), ("Speed", "5000 rpm")],
    )

    tagger, history = select_hyperparameters_on_dev(
        train_dir,
        dev_dir,
        grid=[(0.2, 1.3, 10), (0.3, 1.2, 12)],
        labels=CORE_LABELS,
    )
    assert len(history) == 2
    assert tagger.is_fitted

    # Held-out test evaluation passes zero-overlap check and achieves high F1
    test_res = tagger.evaluate_folder(test_dir, forbid_train_overlap=True)
    assert test_res["train_overlap_documents"] == 0
    assert float(test_res["micro"]["F1_exact"]) >= 0.95  # type: ignore[index]

    # Data-leakage guard blocks evaluating train split when forbid_train_overlap=True
    with pytest.raises(ValueError, match="Data leakage detected"):
        tagger.evaluate_folder(train_dir, forbid_train_overlap=True)

    # Canonical unit normalization via parse_quantities
    qs = {q.label: q for q in tagger.parse_quantities("Add 25 uL buffer at 37 °C for 20 min.")}
    assert qs["Amount"].canonical_value == pytest.approx(25.0)
    assert qs["Temperature"].canonical_value == pytest.approx(37.0)
    assert qs["Time"].canonical_value == pytest.approx(1200.0)

    # Non-pickle .json.gz bundle persistence round-trip
    bundle_path = save_tagger_bundle(tmp_path / "tagger_model.json.gz", tagger)
    reloaded = load_tagger_bundle(bundle_path)
    assert reloaded.find_spans("Add 25 uL buffer at 37 °C for 20 min.") == tagger.find_spans(
        "Add 25 uL buffer at 37 °C for 20 min."
    )

    # Pure-Python fallback produces identical decoding when weights are converted to nested lists
    if hasattr(reloaded.weights_, "tolist"):
        reloaded.weights_ = reloaded.weights_.tolist()  # type: ignore[union-attr]
        reloaded.transitions_ = reloaded.transitions_.tolist()  # type: ignore[union-attr]
        reloaded.start_bias_ = reloaded.start_bias_.tolist()  # type: ignore[union-attr]
        reloaded.end_bias_ = reloaded.end_bias_.tolist()  # type: ignore[union-attr]
    assert reloaded.find_spans("Add 25 uL buffer at 37 °C for 20 min.") == tagger.find_spans(
        "Add 25 uL buffer at 37 °C for 20 min."
    )

    # External JSONL evaluation (BioProBench / BioPlanner normalized schema)
    ext_jsonl = tmp_path / "bioprobench.jsonl"
    ext_jsonl.write_text(
        json.dumps(
            {
                "dataset_id": "bioprobench",
                "record_id": "1",
                "text": "Add 50 uL buffer, incubate at 37 °C for 15 min, and spin at 3000 rpm.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ext_res = evaluate_external_jsonl(ext_jsonl, tagger)
    assert ext_res["dataset_id"] == "bioprobench"
    assert ext_res["n_protocols"] == 1
    assert ext_res["learned_total_spans"] >= 3


def test_committed_benchmark_results_and_output_txt_consistency() -> None:
    out_txt = REPO_ROOT / "demo" / "output.txt"
    wlp_json = REPO_ROOT / "results" / "wlp_learned_vs_rules.json"
    dcase_json = REPO_ROOT / "results" / "dcase2023_cpu.json"

    assert out_txt.is_file()
    assert dcase_json.is_file()

    # Verify DCASE 2023 streaming valve rows meet Issue #36 acceptance criteria
    dcase = json.loads(dcase_json.read_text(encoding="utf-8"))
    valve_stream_rows = {
        r["detector"]: r
        for r in dcase["rows"]
        if r["machine"] == "valve" and r["drop_speech_band"] is True and "normal_chunk_fpr" in r
    }
    assert set(valve_stream_rows) >= {"gaussian", "knn"}
    for det_name in ("gaussian", "knn"):
        row = valve_stream_rows[det_name]
        target_fpr = float(row["target_chunk_fpr"])
        actual_fpr = float(row["normal_chunk_fpr"])
        assert 0.5 * target_fpr <= actual_fpr <= 3.0 * target_fpr
        assert "time_to_first_flag_s_median" in row
        assert "detection_delay_s_median" not in row

    if wlp_json.is_file():
        wlp = json.loads(wlp_json.read_text(encoding="utf-8"))
        test_rules = wlp["rule_based_regex"]["test"]["metrics"]
        out_lines = out_txt.read_text(encoding="utf-8").strip().splitlines()
        assert len(out_lines) == 6
        for line in out_lines[2:]:
            parts = line.split()
            lbl = parts[0]
            assert int(parts[1]) == int(test_rules[lbl]["gold"])
            assert int(parts[2]) == int(test_rules[lbl]["pred"])
            assert float(parts[5]) == pytest.approx(float(test_rules[lbl]["F1_exact"]), abs=1e-3)

        # Learned tagger outperforms rule-based baseline on held-out dev and test micro-F1
        assert (
            wlp["learned_bioes_pa2"]["dev"]["micro"]["F1_exact"]
            > wlp["rule_based_regex"]["dev"]["micro"]["F1_exact"]
        )
        assert (
            wlp["learned_bioes_pa2"]["test"]["micro"]["F1_exact"]
            > wlp["rule_based_regex"]["test"]["micro"]["F1_exact"]
        )

        # Verify README.md and docs/ARCHITECTURE_AND_HISTORY.md match committed JSONs & output.txt
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        arch_text = (REPO_ROOT / "docs" / "ARCHITECTURE_AND_HISTORY.md").read_text(encoding="utf-8")
        assert out_txt.read_text(encoding="utf-8").strip() in readme_text
        for summary_row in dcase["summary"]:
            val_str = f"{float(summary_row['hmean_all']):.4f}"
            assert val_str in readme_text
            assert val_str in arch_text
        for det_name in ("gaussian", "knn"):
            fpr_str = f"{float(valve_stream_rows[det_name]['normal_chunk_fpr']):.4f}"
            assert fpr_str in readme_text
            assert fpr_str in arch_text
        for split_key in ("train", "dev", "test", "pooled_622"):
            for model_key in ("rule_based_regex", "learned_bioes_pa2"):
                f1_str = f"{float(wlp[model_key][split_key]['micro']['F1_exact']):.4f}"
                assert f1_str in readme_text
                assert f1_str in arch_text

    wlp_test_dir = REPO_ROOT / "data" / "WLP-Dataset" / "test"
    if wlp_test_dir.is_dir() and any(wlp_test_dir.glob("*.ann")):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            parse_protocol_main(wlp_test_dir)
        assert buf.getvalue() == out_txt.read_text(encoding="utf-8")
