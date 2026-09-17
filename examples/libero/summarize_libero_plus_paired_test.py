import json
import pathlib

import summarize_libero_plus_paired


def _write_metrics(root: pathlib.Path, condition: str, seed: int, outcomes: list[bool]) -> None:
    output = root / condition / f"seed_{seed}"
    output.mkdir(parents=True)
    rows = [
        {
            "benchmark_task_id": index + 1,
            "episode": 0,
            "task": f"pick object level{index + 1}",
            "success": success,
        }
        for index, success in enumerate(outcomes)
    ]
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_metrics_and_paired_statistics(tmp_path: pathlib.Path) -> None:
    _write_metrics(tmp_path, "correct", 7, [True, True, False])
    _write_metrics(tmp_path, "zero_memory", 7, [False, True, True])

    correct = summarize_libero_plus_paired._load_metrics(tmp_path, "correct", [7])  # noqa: SLF001
    zero = summarize_libero_plus_paired._load_metrics(tmp_path, "zero_memory", [7])  # noqa: SLF001

    assert set(correct) == set(zero)
    assert sum(bool(row["success"]) for row in correct.values()) == 2
    assert summarize_libero_plus_paired._mcnemar_exact_p(1, 1) == 1.0  # noqa: SLF001
    assert summarize_libero_plus_paired._level("pick object level3 sample1") == "level3"  # noqa: SLF001


def test_cluster_bootstrap_zero_difference() -> None:
    rows = {
        (7, 1, 0): {"success": True},
        (42, 1, 0): {"success": False},
        (7, 2, 0): {"success": True},
        (42, 2, 0): {"success": True},
    }
    assert summarize_libero_plus_paired._cluster_bootstrap_ci(rows, rows) == (0.0, 0.0)  # noqa: SLF001
