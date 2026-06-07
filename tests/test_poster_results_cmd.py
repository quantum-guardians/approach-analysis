"""Tests for the poster-results command cache behavior."""

from __future__ import annotations

import math
import json
from types import SimpleNamespace
from typing import Any

import networkx as nx

from src.commands import poster_results as pr
from src.commands.poster_results import _cli as pr_cli
from src.commands.poster_results.runner import (
    PosterResultsAggregator,
    PosterResultsRunner,
    PosterRunConfig,
    RESULT_SERIES_KEYS,
    TrialScheduler,
    _merge_nested_results_by_size,
    _merge_results_by_size,
)
from src.commands.poster_results.partition_strategy import DncPartitionStrategy
from src.commands.poster_results.models import (
    BasicSolverResult,
    GlobalSolverResult,
    Mr2sSolverResult,
    RandomBaselineResult,
    TrialTimings,
)
from src.commands.poster_results.solvers.dnc_strategy import DncStrategySolver
from src.commands.poster_results import plotting as pr_plotting
from src.cache import SimpleCache


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
            "graph": value + 20,
            "raw_sa": value + 21,
            "global_solve": value + 22,
            "global_embed": value + 23,
            "clustered_solve": value + 24,
            "clustered_embed": value + 25,
            "random": value + 26,
        },
    }


def _fake_algorithm(
    n: int,
    trial: int,
    seed: int | None,
    algorithm: str,
) -> tuple[Any, TrialTimings]:
    value = float(n + trial + (seed or 0))
    if algorithm == "raw_sa":
        return BasicSolverResult(apsp=value, flow=value + 1), TrialTimings({
            "graph": value + 20,
            "raw_sa": value + 21,
        })
    if algorithm == "global":
        return GlobalSolverResult(
            apsp=value + 2,
            flow=value + 3,
            qvars=value + 4,
            subgraph_size=value + 5,
            physical_total=value + 6,
        ), TrialTimings({
            "graph": value + 20,
            "global_solve": value + 22,
            "global_embed": value + 23,
        })
    if algorithm == "random":
        return RandomBaselineResult(
            apsp=value + 15,
            flow=value + 16,
            sample_count=1,
            strong_sample_count=1,
        ), TrialTimings({
            "graph": value + 20,
            "random": value + 26,
        })
    if algorithm in {"robbin_mr2s", "iterated_local_search_mr2s"}:
        return GlobalSolverResult(
            apsp=value + 30,
            flow=value + 31,
            qvars=value + 32,
            subgraph_size=value + 33,
            physical_total=value + 34,
        ), TrialTimings({
            "graph": value + 20,
            f"{algorithm}_solve": value + 35,
            f"{algorithm}_embed": value + 36,
        })
    return Mr2sSolverResult(
        apsp=value + 7,
        flow=value + 8,
        qvars=value + 9,
        subgraph_size=value + 10,
        phys_total=value + 11,
        phys_max=value + 12,
        phys_mean=value + 13,
        phys_min=value + 14,
        partition={"selected_reason": algorithm},
    ), TrialTimings({
        "graph": value + 20,
        f"dnc_{algorithm}_solve": value + 24,
        f"dnc_{algorithm}_embed": value + 25,
        **({
            "clustered_solve": value + 24,
            "clustered_embed": value + 25,
        } if algorithm == "embedding_aware" else {}),
    })


def test_poster_trial_cache_key_is_stable() -> None:
    key = pr._poster_trial_cache_key(n=20, trial=3, seed=42)
    assert key == (
        'poster-results-trial:{"n": 20, "seed": 42, '
        '"trial": 3, "version": 5}'
    )


def test_trial_scheduler_uses_spawn_on_macos(monkeypatch) -> None:
    monkeypatch.setattr("src.commands.poster_results.runner.sys.platform", "darwin")

    context = TrialScheduler()._process_pool_context()

    assert context.get_start_method() == "spawn"


def test_run_reuses_poster_trial_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr_cli, "_plot_results", lambda results, output_dir: None)

    call_count = 0

    def counted_algorithm(n, trial, seed, algorithm):
        nonlocal call_count
        call_count += 1
        return _fake_algorithm(n, trial, seed, algorithm)

    monkeypatch.setattr(pr._solver_helpers, "_run_poster_algorithm", counted_algorithm)

    kwargs = dict(
        sizes=[8],
        num_graphs=2,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
    )

    pr.run(**kwargs)
    assert call_count == 12
    assert (tmp_path / "poster_results.json").exists()
    assert len(list((tmp_path / "poster_trial_cache").glob("*/*.pkl"))) == 12

    (tmp_path / "poster_results.json").unlink()
    call_count = 0
    pr.run(**kwargs)
    assert call_count == 0


def test_run_migrates_legacy_full_trial_cache_to_solver_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr_cli, "_plot_results", lambda results, output_dir: None)
    cache_dir = tmp_path / "poster_trial_cache"
    legacy_cache = SimpleCache(str(cache_dir))
    legacy_key = pr._poster_trial_cache_key(n=8, trial=0, seed=0)
    _, _, legacy_result = _fake_trial((8, 0, 0))
    legacy_cache.set(legacy_key, legacy_result)

    call_count = 0

    def counted_algorithm(n, trial, seed, algorithm):
        nonlocal call_count
        call_count += 1
        return _fake_algorithm(n, trial, seed, algorithm)

    monkeypatch.setattr(pr._solver_helpers, "_run_poster_algorithm", counted_algorithm)

    pr.run(
        sizes=[8],
        num_graphs=1,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
    )

    assert call_count == 3
    assert (cache_dir / "raw_sa").exists()
    assert len(list(cache_dir.glob("*/*.pkl"))) == 6


def test_run_can_disable_poster_trial_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr_cli, "_plot_results", lambda results, output_dir: None)
    monkeypatch.setattr(pr._solver_helpers, "_run_trial", _fake_trial)

    pr.run(
        sizes=[8],
        num_graphs=1,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
        use_cache=False,
    )

    assert not (tmp_path / "poster_trial_cache").exists()


def test_run_reuses_existing_aggregate_results_for_all_solvers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr_cli, "_plot_results", lambda results, output_dir: None)

    existing = {
        "sizes": [8],
        "mr2s": {
            "apsp": [80.0],
            "flow": [81.0],
            "qubo_vars": [82.0],
            "subgraph_size": [83.0],
            "phys_total": [84.0],
            "phys_max": [85.0],
            "phys_mean": [86.0],
            "phys_min": [87.0],
            "partition": [[{"selected_reason": "existing"}]],
        },
        "global": {
            "apsp": [70.0],
            "flow": [71.0],
            "qubo_vars": [72.0],
            "subgraph_size": [73.0],
            "phys_total": [74.0],
            "phys_max": [],
            "phys_mean": [],
            "phys_min": [],
        },
        "raw_sa": {"apsp": [60.0], "flow": [61.0]},
        "random": {"apsp": [50.0], "flow": [51.0]},
    }
    existing["timings"] = {
        key: [1.0]
        for key in RESULT_SERIES_KEYS["timings"]
    }
    existing["dnc_strategies"] = {
        name: {
            "apsp": [80.0],
            "flow": [81.0],
            "qubo_vars": [82.0],
            "subgraph_size": [83.0],
            "phys_total": [84.0],
            "phys_max": [85.0],
            "phys_mean": [86.0],
            "phys_min": [87.0],
            "partition": [[{"selected_reason": name}]],
        }
        for name in ("poster", "embedding_aware", "degeneracy_pruning")
    }
    existing["mr2s_variants"] = {
        name: {
            "apsp": [90.0],
            "flow": [91.0],
            "qubo_vars": [92.0],
            "subgraph_size": [93.0],
            "phys_total": [94.0],
        }
        for name in ("robbin_mr2s", "iterated_local_search_mr2s")
    }
    results_path = tmp_path / "poster_results.json"
    results_path.write_text(json.dumps(existing))

    call_count = 0

    def counted_algorithm(n, trial, seed, algorithm):
        nonlocal call_count
        call_count += 1
        return _fake_algorithm(n, trial, seed, algorithm)

    monkeypatch.setattr(pr._solver_helpers, "_run_poster_algorithm", counted_algorithm)

    pr.run(
        sizes=[8, 9],
        num_graphs=1,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
    )

    merged = json.loads(results_path.read_text())
    assert call_count == 6
    assert merged["sizes"] == [8, 9]
    assert merged["raw_sa"]["apsp"][0] == 60.0
    assert merged["global"]["apsp"][0] == 70.0
    assert merged["mr2s"]["apsp"][0] == 80.0
    assert merged["random"]["apsp"][0] == 50.0
    assert merged["raw_sa"]["apsp"][1] == 9.0
    assert merged["timings"]["graph"][0] == 1.0
    assert merged["timings"]["graph"][1] == 29.0


def test_merge_results_pads_missing_series_for_old_result_schema() -> None:
    existing = {
        "sizes": [5],
        "mr2s": {"apsp": [1.0]},
        "timings": {"global_solve": [2.0]},
    }
    updates = {
        "sizes": [10],
        "mr2s": {
            "apsp": [10.0],
            "partition": [[{"selected_reason": "new"}]],
        },
        "timings": {
            "global_solve": [20.0],
            "robbin_mr2s_solve": [30.0],
        },
    }

    merged = _merge_results_by_size(
        existing,
        updates,
        replace_sections=set(RESULT_SERIES_KEYS),
    )

    assert merged["sizes"] == [5, 10]
    assert merged["mr2s"]["apsp"] == [1.0, 10.0]
    assert merged["mr2s"]["partition"] == [[], [{"selected_reason": "new"}]]
    assert merged["timings"]["global_solve"] == [2.0, 20.0]
    assert math.isnan(merged["timings"]["robbin_mr2s_solve"][0])
    assert merged["timings"]["robbin_mr2s_solve"][1] == 30.0


def test_merge_nested_results_pads_missing_strategy_and_variant_sizes() -> None:
    existing = {
        "sizes": [5],
        "dnc_strategies": {},
        "mr2s_variants": {},
    }
    updates = {
        "sizes": [10],
        "dnc_strategies": {
            "poster": {
                "apsp": [10.0],
                "partition": [[{"selected_reason": "new"}]],
            },
        },
        "mr2s_variants": {
            "robbin_mr2s": {
                "apsp": [20.0],
                "phys_total": [21.0],
            },
        },
    }
    merged = {"sizes": [5, 10]}

    merged = _merge_nested_results_by_size(
        merged,
        existing,
        updates,
        "dnc_strategies",
        replace_existing=True,
    )
    merged = _merge_nested_results_by_size(
        merged,
        existing,
        updates,
        "mr2s_variants",
        replace_existing=True,
    )

    assert math.isnan(merged["dnc_strategies"]["poster"]["apsp"][0])
    assert merged["dnc_strategies"]["poster"]["apsp"][1] == 10.0
    assert merged["dnc_strategies"]["poster"]["partition"] == [
        [],
        [{"selected_reason": "new"}],
    ]
    assert math.isnan(merged["mr2s_variants"]["robbin_mr2s"]["apsp"][0])
    assert merged["mr2s_variants"]["robbin_mr2s"]["apsp"][1] == 20.0
    assert len(merged["mr2s_variants"]["robbin_mr2s"]["phys_total"]) == 2


def test_runner_accepts_custom_aggregator_and_plotter(tmp_path) -> None:
    class CustomAggregator(PosterResultsAggregator):
        def __init__(self) -> None:
            self.received_sizes = []
            self.received_trial_count = 0

        def aggregate_full(self, sizes, trial_results):
            self.received_sizes = sizes
            self.received_trial_count = sum(len(items) for items in trial_results.values())
            return {
                "sizes": sizes,
                "custom": {"trial_count": self.received_trial_count},
            }

    plotted = {}

    def fake_plotter(results, output_dir):
        plotted["results"] = results
        plotted["output_dir"] = output_dir

    aggregator = CustomAggregator()
    runner = PosterResultsRunner(
        config=PosterRunConfig(
            sizes=[8],
            num_graphs=1,
            seed=0,
            output_dir=str(tmp_path),
            num_workers=0,
            use_cache=False,
        ),
        worker=lambda task: _fake_trial((task[0], task[1], task[2])),
        progress_printer=lambda *_args: None,
        plotter=fake_plotter,
        aggregator=aggregator,
    )

    results = runner.run()

    assert aggregator.received_sizes == [8]
    assert aggregator.received_trial_count == 1
    assert results["custom"]["trial_count"] == 1
    assert plotted["results"] == results
    assert plotted["output_dir"] == str(tmp_path)


def test_plot_results_writes_publication_series_summary(tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_apsp(sizes, random_apsp, raw_sa_apsp, global_apsp, clustered_apsp, **kwargs):
        del random_apsp, raw_sa_apsp, clustered_apsp, kwargs
        captured["apsp"] = (sizes, global_apsp)

    def fake_flow(sizes, random_flow, raw_sa_flow, global_flow, clustered_flow, **kwargs):
        del random_flow, raw_sa_flow, clustered_flow, kwargs
        captured["flow"] = (sizes, global_flow)

    def fake_scalability(*_args, **_kwargs):
        captured["scalability"] = True

    def fake_spent_time(
        sizes,
        graph_time,
        raw_sa_time,
        global_solve_time,
        global_embed_time,
        *_args,
        **_kwargs,
    ):
        del graph_time, raw_sa_time
        captured["time"] = (sizes, global_solve_time, global_embed_time)

    monkeypatch.setattr(pr_plotting, "plot_apsp_reduction", fake_apsp)
    monkeypatch.setattr(pr_plotting, "plot_flow_stability", fake_flow)
    monkeypatch.setattr(pr_plotting, "plot_preprocessing_scalability", fake_scalability)
    monkeypatch.setattr(pr_plotting, "plot_spent_time", fake_spent_time)

    results = {
        "sizes": [100, 200],
        "random": {"apsp": [5.0, 6.0], "flow": [7.0, 8.0]},
        "raw_sa": {"apsp": [4.0, 5.0], "flow": [6.0, 7.0]},
        "global": {
            "apsp": [3.0, 4.0],
            "flow": [5.0, 6.0],
            "qubo_vars": [30.0, 40.0],
            "subgraph_size": [30.0, 40.0],
            "phys_total": [300.0, 400.0],
        },
        "mr2s": {
            "apsp": [2.0, 3.0],
            "flow": [4.0, 5.0],
            "qubo_vars": [20.0, 30.0],
            "subgraph_size": [20.0, 30.0],
            "phys_total": [200.0, 300.0],
            "phys_max": [120.0, 130.0],
            "phys_mean": [100.0, 110.0],
            "phys_min": [80.0, 90.0],
        },
        "timings": {
            "graph": [0.1, 0.2],
            "raw_sa": [1.0, 2.0],
            "global_solve": [3.0, 4.0],
            "global_embed": [0.3, 0.4],
            "clustered_solve": [2.0, 3.0],
            "clustered_embed": [0.2, 0.3],
            "random": [0.01, 0.02],
        },
    }

    pr_plotting._plot_results(results, str(tmp_path))

    summary = json.loads((tmp_path / "plotted_data_summary.json").read_text())
    assert [item["key"] for item in summary["plots"]["apsp_reduction"]["series"]] == [
        "raw_sa",
        "embedding_aware",
    ]
    assert [item["key"] for item in summary["plots"]["flow_stability"]["series"]] == [
        "raw_sa",
        "embedding_aware",
    ]
    assert [item["key"] for item in summary["plots"]["scalability"]["series"]] == [
        "global",
        "embedding_aware_sum",
        "embedding_aware_max",
        "embedding_aware_avg",
        "embedding_aware_min",
    ]
    assert [item["display_name"] for item in summary["plots"]["scalability"]["series"]] == [
        "Mono",
        "Cluster sum",
        "Cluster max",
        "Cluster avg",
        "Cluster min",
    ]
    assert summary["plots"]["apsp_reduction"]["series"][1]["display_name"] == "Ours"
    assert summary["plots"]["apsp_reduction"]["series"][1]["color"] == pr_plotting.SERIES_COLORS["embedding_aware"]
    assert summary["plots"]["spent_time"]["series"][1]["y"] == [2.2, 3.3]
    assert captured["apsp"] == ([100, 200], [3.0, 4.0])
    assert captured["flow"] == ([100, 200], [5.0, 6.0])
    assert captured["time"] == ([100, 200], [3.0, 4.0], [0.3, 0.4])


def test_run_mr2s_trial_records_graph_generation_time(monkeypatch) -> None:
    def fake_dnc_run(self, graph, n, seed):
        return (
            Mr2sSolverResult(
                apsp=1.0,
                flow=2.0,
                qvars=3.0,
                subgraph_size=4.0,
                phys_total=5.0,
                phys_max=6.0,
                phys_mean=7.0,
                phys_min=8.0,
            ),
            TrialTimings({"clustered_solve": 0.0, "clustered_embed": 0.0}),
        )

    monkeypatch.setattr(DncStrategySolver, "run", fake_dnc_run)

    n, trial, result = pr._run_mr2s_trial((3, 0, 1))

    assert n == 3
    assert trial == 0
    assert "graph" in result.timings.values
    assert result.timings.values["graph"] >= 0.0


def test_run_mr2s_only_merges_with_existing_results(tmp_path, monkeypatch) -> None:
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

        monkeypatch.setattr(pr._solver_helpers, "_run_mr2s_trial", fake_mr2s_trial)
        trial_result = fake_mr2s_trial((8, 0, 0))
        _, _, raw_result = trial_result
        merged = pr._aggregate_mr2s_results(existing, {8: [raw_result]})
        assert merged["raw_sa"] == existing["raw_sa"]
        assert merged["global"] == existing["global"]
        assert merged["random"] == existing["random"]
        assert merged["mr2s"]["apsp"] == [8.0]
        assert merged["mr2s"]["partition"] == [[{"selected_reason": "test"}]]


def test_run_mr2s_only_preserves_unrequested_existing_sizes(tmp_path, monkeypatch) -> None:
    existing = {
        "sizes": [8, 9],
        "mr2s": {
            "apsp": [18.0, 19.0],
            "flow": [28.0, 29.0],
            "qubo_vars": [38.0, 39.0],
            "subgraph_size": [48.0, 49.0],
            "phys_total": [58.0, 59.0],
            "phys_max": [68.0, 69.0],
            "phys_mean": [78.0, 79.0],
            "phys_min": [88.0, 89.0],
            "partition": [[{"old": 8}], [{"old": 9}]],
        },
        "global": {"apsp": [1.0, 2.0], "flow": [3.0, 4.0]},
        "raw_sa": {"apsp": [5.0, 6.0], "flow": [7.0, 8.0]},
        "random": {"apsp": [9.0, 10.0], "flow": [11.0, 12.0]},
        "timings": {
            "graph": [1.0, 2.0],
            "raw_sa": [3.0, 4.0],
            "global_solve": [5.0, 6.0],
            "global_embed": [7.0, 8.0],
            "clustered_solve": [9.0, 10.0],
            "clustered_embed": [11.0, 12.0],
            "random": [13.0, 14.0],
        },
    }

    trial_result = {
        "mr2s": {
            "apsp": 800.0,
            "flow": 801.0,
            "qvars": 802.0,
            "sg": 803.0,
            "phys_total": 804.0,
            "phys_max": 805.0,
            "phys_mean": 806.0,
            "phys_min": 807.0,
            "partition": {"selected_reason": "updated"},
        },
        "timings": {
            "graph": 100.0,
            "clustered_solve": 101.0,
            "clustered_embed": 102.0,
        },
    }

    merged = pr._aggregate_mr2s_results(existing, {8: [trial_result]})
    assert merged["sizes"] == [8, 9]
    assert merged["global"] == existing["global"]
    assert merged["raw_sa"] == existing["raw_sa"]
    assert merged["random"] == existing["random"]
    assert merged["mr2s"]["apsp"] == [800.0, 19.0]
    assert merged["mr2s"]["partition"] == [[{"selected_reason": "updated"}], [{"old": 9}]]
    assert merged["timings"]["graph"] == [100.0, 2.0]
    assert merged["timings"]["clustered_solve"] == [101.0, 10.0]
    assert merged["timings"]["raw_sa"] == [3.0, 4.0]


def test_normalize_random_baseline_converts_legacy_zero_to_nan() -> None:
    normalized = pr._normalize_random_baseline({"apsp": 0.0, "flow": 0.0})

    assert math.isnan(normalized["apsp"])
    assert math.isnan(normalized["flow"])
    assert normalized["sample_count"] == 0


def test_random_baseline_scores_flow_for_non_strong_orientation(monkeypatch) -> None:
    graph = nx.path_graph(3)
    orientation = nx.DiGraph([(0, 1), (1, 2)])
    orientation.add_nodes_from(graph.nodes())

    monkeypatch.setattr(
        pr._solver_helpers,
        "_sample_random_orientations",
        lambda _graph, max_samples, seed: [orientation],
    )

    result = pr._calculate_random_baseline(graph, n=3, seed=0, max_samples=1)

    assert math.isnan(result["apsp"])
    assert result["flow"] == 2.0
    assert result["sample_count"] == 1
    assert result["strong_sample_count"] == 0


def test_random_baseline_averages_flow_across_all_orientations(monkeypatch) -> None:
    graph = nx.cycle_graph(3)
    strong_orientation = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    non_strong_orientation = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    for orientation in (strong_orientation, non_strong_orientation):
        orientation.add_nodes_from(graph.nodes())

    monkeypatch.setattr(
        pr._solver_helpers,
        "_sample_random_orientations",
        lambda _graph, max_samples, seed: [strong_orientation, non_strong_orientation],
    )

    result = pr._calculate_random_baseline(graph, n=3, seed=0, max_samples=2)

    assert result["apsp"] == 1.5
    assert result["flow"] == 4.0
    assert result["sample_count"] == 2
    assert result["strong_sample_count"] == 1


def test_divide_graph_with_diagnostics_records_selected_partition(monkeypatch) -> None:
    class DummyGraph:
        def __init__(self, edge_count: int):
            self.edges = list(range(edge_count))

    def fake_probe(self, _solver, graph):
        can_embed = len(graph.edges) <= 3
        probe = {
            "can_embed": can_embed,
            "qvars": len(graph.edges),
            "physical_qubits": float(len(graph.edges) * 10) if can_embed else float("nan"),
            "error": None if can_embed else "too large",
        }
        estimate = SimpleNamespace(num_logical_variables=len(graph.edges), num_physical_qubits=len(graph.edges) * 10) if can_embed else None
        return probe, estimate

    def fake_partition(self, _face_cycle, _graph, target_k):
        if target_k >= 8:
            sub_graphs = [DummyGraph(3), DummyGraph(3), DummyGraph(3)]
        else:
            sub_graphs = [DummyGraph(4), DummyGraph(4)]
        return SimpleNamespace(sub_graphs=sub_graphs, remaining_edges=[])

    monkeypatch.setattr(DncPartitionStrategy, "probe_embedding_with_estimate", fake_probe)
    monkeypatch.setattr(DncPartitionStrategy, "partition_with_target_k", fake_partition)

    solver = SimpleNamespace(mr2s_solver=object(), face_cycle=object())
    sub_graphs, diagnostics = pr._divide_graph_with_diagnostics(solver, DummyGraph(10))

    assert [len(sub_graph.edges) for sub_graph in sub_graphs] == [3, 3, 3]
    assert diagnostics["whole_graph"]["can_embed"] is False
    assert diagnostics["selected_reason"] == "partition_found"
    assert len(diagnostics["selected_probes"]) == 3
    assert any(attempt["accepted"] for attempt in diagnostics["attempts"])


def test_build_dnc_qubo_solver_sets_subgraph_processes_for_qa(monkeypatch) -> None:
    from src.commands.poster_results.solvers import dnc_strategy

    # Mock _build_qubo_solver to avoid actual D-Wave connection/initialization
    monkeypatch.setattr(dnc_strategy, "_build_qubo_solver", lambda use_qa: None)

    solver_qa = dnc_strategy._build_dnc_qubo_solver(use_qa=True)
    assert solver_qa.subgraph_processes == 1

    solver_sa = dnc_strategy._build_dnc_qubo_solver(use_qa=False)
    assert solver_sa.subgraph_processes is None


def test_dnc_strategy_solver_failure_returns_nan(monkeypatch) -> None:
    from src.commands.poster_results.solvers import dnc_strategy

    class SolverFailure(Exception):
        pass

    class FailingSolver:
        def run(self, _graph):
            raise SolverFailure("chain for e_0_162 is not connected")

    monkeypatch.setattr(
        dnc_strategy,
        "_build_dnc_qubo_solver",
        lambda use_qa: FailingSolver(),
    )
    monkeypatch.setattr(
        dnc_strategy,
        "_build_partition_strategy",
        lambda strategy_name, solver: SimpleNamespace(),
    )

    graph = nx.cycle_graph(3)
    result, timings = DncStrategySolver("embedding_aware").run(graph, n=3, seed=0)

    assert math.isnan(result.apsp)
    assert math.isnan(result.flow)
    assert result.partition["selected_reason"] == "failed"
    assert result.partition["error"] == "chain for e_0_162 is not connected"
    assert timings.values["clustered_solve"] == 0.0
    assert timings.values["clustered_embed"] >= 0.0
