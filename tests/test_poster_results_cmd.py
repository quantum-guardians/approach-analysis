"""Tests for the poster-results command cache behavior."""

from __future__ import annotations

import math
import json
from types import SimpleNamespace
from typing import Any

from src.commands import poster_results as pr


def _fake_trial(task: tuple[int, int, int | None]) -> tuple[int, int, dict[str, Any]]:
    n, trial, seed = task
    value = float(n + trial + (seed or 0))
    return n, trial, {
        "raw_sa": {"apsp": value, "flow": value + 1},
        "global": {
            "apsp": value + 2,
            "flow": value + 3,
            "qvars": value + 4,
            "sg": value + 5,
            "pt": value + 6,
        },
        "mr2s": {
            "apsp": value + 7,
            "flow": value + 8,
            "qvars": value + 9,
            "sg": value + 10,
            "phys_total": value + 11,
            "phys_max": value + 12,
            "phys_mean": value + 13,
            "phys_min": value + 14,
        },
        "random": {"apsp": value + 15, "flow": value + 16},
        "timings": {
            "raw_sa": 0.0,
            "global_solve": 0.0,
            "global_embed": 0.0,
            "clustered_solve": 0.0,
            "clustered_embed": 0.0,
            "random": 0.0,
        },
    }


def test_poster_trial_cache_key_is_stable() -> None:
    key = pr._poster_trial_cache_key(n=20, trial=3, seed=42)
    assert key == (
        'poster-results-trial:{"n": 20, "seed": 42, '
        '"trial": 3, "version": 2}'
    )


def test_run_reuses_poster_trial_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_plot_results", lambda results, output_dir: None)

    call_count = 0

    def counted_trial(task):
        nonlocal call_count
        call_count += 1
        return _fake_trial(task)

    monkeypatch.setattr(pr, "_run_trial", counted_trial)

    kwargs = dict(
        sizes=[8],
        num_graphs=2,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
    )

    pr.run(**kwargs)
    assert call_count == 2
    assert (tmp_path / "poster_results.json").exists()
    assert len(list((tmp_path / "poster_trial_cache").glob("*.pkl"))) == 2

    call_count = 0
    pr.run(**kwargs)
    assert call_count == 0


def test_run_can_disable_poster_trial_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_plot_results", lambda results, output_dir: None)
    monkeypatch.setattr(pr, "_run_trial", _fake_trial)

    pr.run(
        sizes=[8],
        num_graphs=1,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
        use_cache=False,
    )

    assert not (tmp_path / "poster_trial_cache").exists()


def test_run_mr2s_only_merges_with_existing_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_plot_results", lambda results, output_dir: None)

    existing = {
        "sizes": [8],
        "mr2s": {},
        "global": {"apsp": [1.0], "flow": [2.0]},
        "raw_sa": {"apsp": [3.0], "flow": [4.0]},
        "random": {"apsp": [5.0], "flow": [6.0]},
    }
    results_path = tmp_path / "poster_results.json"
    results_path.write_text(json.dumps(existing))

    def fake_mr2s_trial(task):
        n, trial, seed = task
        value = float(n + trial + (seed or 0))
        return n, trial, {
            "mr2s": {
                "apsp": value,
                "flow": value + 1,
                "qvars": value + 2,
                "sg": value + 3,
                "phys_total": value + 4,
                "phys_max": value + 5,
                "phys_mean": value + 6,
                "phys_min": value + 7,
                "partition": {"selected_reason": "test"},
            },
            "timings": {
                "clustered_solve": 0.0,
                "clustered_embed": 0.0,
            },
        }

    monkeypatch.setattr(pr, "_run_mr2s_trial", fake_mr2s_trial)

    pr.run_mr2s_only(
        sizes=[8],
        num_graphs=1,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
        use_cache=False,
    )

    merged = json.loads(results_path.read_text())
    assert merged["raw_sa"] == existing["raw_sa"]
    assert merged["global"] == existing["global"]
    assert merged["random"] == existing["random"]
    assert merged["mr2s"]["apsp"] == [8.0]
    assert merged["mr2s"]["partition"] == [[{"selected_reason": "test"}]]


def test_normalize_random_baseline_converts_legacy_zero_to_nan() -> None:
    normalized = pr._normalize_random_baseline({"apsp": 0.0, "flow": 0.0})

    assert math.isnan(normalized["apsp"])
    assert math.isnan(normalized["flow"])
    assert normalized["sample_count"] == 0


def test_divide_graph_with_diagnostics_records_selected_partition(monkeypatch) -> None:
    class DummyGraph:
        def __init__(self, edge_count: int):
            self.edges = list(range(edge_count))

    def fake_probe(_solver, graph):
        can_embed = len(graph.edges) <= 3
        return {
            "can_embed": can_embed,
            "qvars": len(graph.edges),
            "physical_qubits": float(len(graph.edges) * 10) if can_embed else float("nan"),
            "error": None if can_embed else "too large",
        }

    def fake_partition(_face_cycle, _graph, target_k):
        if target_k >= 8:
            sub_graphs = [DummyGraph(3), DummyGraph(3), DummyGraph(3)]
        else:
            sub_graphs = [DummyGraph(4), DummyGraph(4)]
        return SimpleNamespace(sub_graphs=sub_graphs, remaining_edges=[])

    monkeypatch.setattr(pr, "_probe_embedding", fake_probe)
    monkeypatch.setattr(pr, "_partition_with_target_k", fake_partition)

    solver = SimpleNamespace(mr2s_solver=object(), face_cycle=object())
    sub_graphs, diagnostics = pr._divide_graph_with_diagnostics(solver, DummyGraph(10))

    assert [len(sub_graph.edges) for sub_graph in sub_graphs] == [3, 3, 3]
    assert diagnostics["whole_graph"]["can_embed"] is False
    assert diagnostics["selected_reason"] == "partition_found"
    assert len(diagnostics["selected_probes"]) == 3
    assert any(attempt["accepted"] for attempt in diagnostics["attempts"])
