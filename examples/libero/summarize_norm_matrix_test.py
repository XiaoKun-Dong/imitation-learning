import importlib.util
import json
import pathlib

import pytest

_MODULE_PATH = pathlib.Path(__file__).with_name("summarize_norm_matrix.py")
_SPEC = importlib.util.spec_from_file_location("summarize_norm_matrix", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
summarize_norm_matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarize_norm_matrix)


def _write_metrics(root, label, values):
    output_dir = root / label
    output_dir.mkdir(parents=True)
    rows = [
        {"benchmark_task_id": task, "episode": episode, "success": success}
        for (task, episode), success in values.items()
    ]
    (output_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_equivalence_summary_detects_matching_wrappers(tmp_path):
    official = {(0, 0): True, (0, 1): False, (1, 0): True, (1, 1): True}
    local = {(0, 0): False, (0, 1): False, (1, 0): True, (1, 1): False}
    _write_metrics(tmp_path, "pi05_official_stats", official)
    _write_metrics(tmp_path, "demovla_off_official_stats", official)
    _write_metrics(tmp_path, "pi05_local_stats", local)
    _write_metrics(tmp_path, "demovla_off_local_stats", local)

    result, markdown = summarize_norm_matrix.summarize(tmp_path, "equivalence")

    assert result["same_stats_outcome_equivalence"] == {"official": True, "local": True}
    assert result["conditions"]["pi05_official_stats"]["success_rate"] == pytest.approx(0.75)
    assert "Outcome-equivalence gate" in markdown


def test_causal_summary_computes_paired_differences(tmp_path):
    demovla_off_official = {(0, 0): True, (0, 1): False, (1, 0): True, (1, 1): False}
    demovla_off_local = {(0, 0): False, (0, 1): False, (1, 0): True, (1, 1): False}
    dynamic_official = {(0, 0): True, (0, 1): True, (1, 0): True, (1, 1): False}
    dynamic_local = {(0, 0): True, (0, 1): True, (1, 0): True, (1, 1): True}
    for label, values in (
        ("demovla_off_official_stats", demovla_off_official),
        ("demovla_off_local_stats", demovla_off_local),
        ("dynamic_official_stats", dynamic_official),
        ("dynamic_local_stats", dynamic_local),
    ):
        _write_metrics(tmp_path, label, values)

    result, markdown = summarize_norm_matrix.summarize(tmp_path, "causal")

    assert result["paired_contrasts"][0]["success_rate_delta"] == pytest.approx(0.25)
    assert result["paired_contrasts"][1]["success_rate_delta"] == pytest.approx(0.75)
    assert result["difference_in_differences"]["estimate"] == pytest.approx(0.5)
    assert "Difference-in-differences" in markdown


def test_summary_rejects_unpaired_episode_keys(tmp_path):
    base = {(0, 0): True}
    for label in summarize_norm_matrix.PHASE_LABELS["equivalence"]:
        _write_metrics(tmp_path, label, base if label != "pi05_local_stats" else {(0, 1): True})

    with pytest.raises(ValueError, match="not paired"):
        summarize_norm_matrix.summarize(tmp_path, "equivalence")
