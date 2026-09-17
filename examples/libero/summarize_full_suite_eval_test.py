import importlib.util
import json
import pathlib

_MODULE_PATH = pathlib.Path(__file__).with_name("summarize_full_suite_eval.py")
_SPEC = importlib.util.spec_from_file_location("summarize_full_suite_eval", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _write_run(root, successes):
    for suite in summary.SUITES:
        suite_root = root / suite
        suite_root.mkdir(parents=True)
        rows = [
            {"benchmark_task_id": 1, "episode": episode, "success": success}
            for episode, success in enumerate(successes[suite])
        ]
        (suite_root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_summarize_full_suite_paired_comparison(tmp_path):
    reference_root = tmp_path / "reference"
    contender_root = tmp_path / "contender"
    reference = {suite: [True, False] for suite in summary.SUITES}
    contender = {suite: [True, True] for suite in summary.SUITES}
    _write_run(reference_root, reference)
    _write_run(contender_root, contender)

    result, markdown = summary.summarize(contender_root, reference_root)

    assert result["run"]["successes"] == 8
    assert result["run"]["total"] == 8
    comparison = result["comparison_to_reference"]["overall"]
    assert comparison["success_rate_delta"] == 0.5
    assert comparison["reference_only"] == 0
    assert comparison["contender_only"] == 4
    assert comparison["mcnemar_exact_p"] == 0.125
    assert "Paired comparison" in markdown
