"""``poster-results`` subcommand - MR2S poster visualization data generation."""

from __future__ import annotations

import argparse
from typing import Any

import networkx as nx

from src.cache import generate_cache_key
from src.commands.poster_results_cache import cached_trial_result
from src.commands.poster_results_models import (
    Mr2sTrialResult,
    PosterTrialResult,
)
from src.commands.poster_results_plotting import _plot_results
from src.commands.poster_results_runner import (
    Mr2sOnlyPosterResultsRunner,
    PosterResultsAggregator,
    PosterResultsRunner,
    PosterRunConfig,
    TrialScheduler,
)
from src.commands import poster_results_solvers as _solver_helpers
from src.commands.poster_results_solvers import (
    _mean_finite,
    _normalize_random_baseline,
    _run_mr2s_trial,
    _run_trial,
)
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts

POSTER_CACHE_VERSION = 5
TrialTask = tuple[int, int, int | None, str | None]


# 테스트와 기존 내부 호출자가 이 모듈의 private helper를 직접 monkeypatch하므로
# solver 모듈로 위임하는 얇은 호환 래퍼를 유지한다.
def _sample_random_orientations(
    graph: nx.Graph,
    max_samples: int,
    seed: int | None = None,
) -> list[nx.DiGraph]:
    return _solver_helpers._sample_random_orientations(graph, max_samples, seed)


def _flow_imbalance_score(graph: nx.DiGraph) -> int:
    return _solver_helpers._flow_imbalance_score(graph)


def _calculate_random_baseline(
    graph: nx.Graph,
    n: int,
    seed: int | None,
    max_samples: int = 10,
) -> dict[str, Any]:
    random_samples = _sample_random_orientations(graph, max_samples=max_samples, seed=seed)
    trial_apsp = []
    trial_flow = []

    for orient in random_samples:
        # APSP는 강연결 방향 그래프에서만 의미 있게 비교하고,
        # flow imbalance는 강연결 여부와 무관하게 모든 샘플의 기준값으로 남긴다.
        if nx.is_strongly_connected(orient):
            apsp, _ = calculate_apsp_sum_and_nhop_neighbor_counts(orient, hops=[])
            trial_apsp.append(apsp / (n * (n - 1)))
        trial_flow.append(_flow_imbalance_score(orient))

    return {
        "apsp": _mean_finite(trial_apsp),
        "flow": _mean_finite(trial_flow),
        "sample_count": len(random_samples),
        "strong_sample_count": len(trial_apsp),
    }


def _probe_embedding(mr2s_solver: Any, graph: Any) -> dict[str, Any]:
    return _solver_helpers._probe_embedding(mr2s_solver, graph)


def _partition_with_target_k(face_cycle: Any, graph: Any, target_k: int) -> Any:
    return _solver_helpers._partition_with_target_k(face_cycle, graph, target_k)


def _find_partition_by_target_k_with_diagnostics(
    mr2s_solver: Any,
    face_cycle: Any,
    graph: Any,
) -> tuple[list[Any], dict[str, Any]]:
    left = 2
    right = max(2, len(graph.edges))
    best_sub_graphs: list[Any] | None = None
    best_probes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    while left <= right:
        # target_k가 작을수록 큰 subgraph가 생길 수 있으므로,
        # embedding 가능한 가장 작은 target_k를 이진 탐색한다.
        target_k = (left + right) // 2
        result = _partition_with_target_k(face_cycle, graph, target_k)
        sub_graphs = result.sub_graphs
        can_recurse = _solver_helpers._can_recurse_partition(graph, sub_graphs)
        probes = (
            [_probe_embedding(mr2s_solver, sub_graph) for sub_graph in sub_graphs]
            if can_recurse
            else []
        )
        accepted = can_recurse and all(probe["can_embed"] for probe in probes)
        attempts.append(
            _solver_helpers._summarize_partition_attempt(
                target_k=target_k,
                result=result,
                can_recurse=can_recurse,
                probes=probes,
                accepted=accepted,
            )
        )

        if accepted:
            best_sub_graphs = sub_graphs
            best_probes = probes
            right = target_k - 1
        else:
            left = target_k + 1

    diagnostics = {
        "attempts": attempts,
        "selected_reason": "partition_found" if best_sub_graphs else "partition_not_found",
        "selected_probes": best_probes,
    }
    return best_sub_graphs or [], diagnostics


def _divide_graph_with_diagnostics(solver: Any, graph: Any) -> tuple[list[Any], dict[str, Any]]:
    # 전체 그래프가 바로 embedding 가능하면 분할하지 않는다.
    whole_graph_probe = _probe_embedding(solver.mr2s_solver, graph)
    diagnostics = {
        "whole_graph": whole_graph_probe,
        "attempts": [],
        "selected_reason": "whole_graph_embeddable",
        "selected_probes": [whole_graph_probe],
    }
    if whole_graph_probe["can_embed"]:
        return [graph], diagnostics

    sub_graphs, partition_diagnostics = _find_partition_by_target_k_with_diagnostics(
        solver.mr2s_solver,
        solver.face_cycle,
        graph,
    )
    diagnostics.update(partition_diagnostics)
    if not sub_graphs:
        # 분할도 실패하면 기존 solver 동작을 유지하기 위해 전체 그래프로 fallback한다.
        diagnostics["selected_reason"] = "fallback_whole_graph"
        diagnostics["selected_probes"] = [whole_graph_probe]
        return [graph], diagnostics
    return sub_graphs, diagnostics


def _poster_trial_cache_key(n: int, trial: int, seed: int | None) -> str:
    """Return the stable cache key for one poster-result graph trial."""
    return generate_cache_key(
        "poster-results-trial",
        version=POSTER_CACHE_VERSION,
        n=n,
        trial=trial,
        seed=seed,
    )


def _poster_mr2s_trial_cache_key(n: int, trial: int, seed: int | None) -> str:
    """Return the stable cache key for one MR2S-only poster trial."""
    return generate_cache_key(
        "poster-results-mr2s-trial",
        version=POSTER_CACHE_VERSION,
        n=n,
        trial=trial,
        seed=seed,
    )


def _coerce_full_trial_result(result: PosterTrialResult | dict[str, Any]) -> PosterTrialResult:
    if isinstance(result, PosterTrialResult):
        return result
    return PosterTrialResult.from_dict(result)


def _coerce_mr2s_trial_result(result: Mr2sTrialResult | dict[str, Any]) -> Mr2sTrialResult:
    if isinstance(result, Mr2sTrialResult):
        return result
    return Mr2sTrialResult.from_dict(result)


def _run_trial_worker(
    task: tuple[int, int, int | None],
) -> tuple[int, int, PosterTrialResult | dict[str, Any]]:
    # decorator 안쪽에서 실제 cache hit/miss 처리를 공통화하되,
    # multiprocessing pickle이 찾을 수 있는 별도 top-level worker 이름을 유지한다.
    return _run_trial(task)


_run_trial_with_cache = cached_trial_result(
    cache_key=_poster_trial_cache_key,
    from_dict=PosterTrialResult.from_dict,
    coerce_result=_coerce_full_trial_result,
)(_run_trial_worker)


def _run_mr2s_trial_worker(
    task: tuple[int, int, int | None],
) -> tuple[int, int, Mr2sTrialResult | dict[str, Any]]:
    # MR2S-only도 동일한 cache decorator를 공유한다.
    return _run_mr2s_trial(task)


_run_mr2s_trial_with_cache = cached_trial_result(
    cache_key=_poster_mr2s_trial_cache_key,
    from_dict=Mr2sTrialResult.from_dict,
    coerce_result=_coerce_mr2s_trial_result,
)(
    _run_mr2s_trial_worker
)


def _process_pool_context() -> Any:
    """Prefer fork where available; fall back to the platform default."""
    return TrialScheduler()._process_pool_context()


def _iter_completed_trials(
    worker: Any,
    tasks: list[TrialTask],
    num_workers: int,
) -> Any:
    """Yield trial results from non-daemonic worker processes as they finish."""
    yield from TrialScheduler().iter_completed(worker, tasks, num_workers)


def _aggregate_mr2s_results(results: dict[str, Any], trial_results: dict[int, list[dict[str, Any]]]) -> None:
    PosterResultsAggregator().merge_mr2s_only(results, trial_results)


def run(
    sizes: list[int],
    num_graphs: int,
    seed: int | None,
    output_dir: str,
    num_workers: int | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> None:
    config = PosterRunConfig(
        sizes=sizes,
        num_graphs=num_graphs,
        seed=seed,
        output_dir=output_dir,
        num_workers=num_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    PosterResultsRunner(
        config=config,
        worker=_run_trial_with_cache,
        progress_printer=_print_trial_progress,
        plotter=_plot_results,
    ).run()


def run_mr2s_only(
    sizes: list[int],
    num_graphs: int,
    seed: int | None,
    output_dir: str,
    num_workers: int | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
    source_results_path: str | None = None,
) -> None:
    config = PosterRunConfig(
        sizes=sizes,
        num_graphs=num_graphs,
        seed=seed,
        output_dir=output_dir,
        num_workers=num_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    Mr2sOnlyPosterResultsRunner(
        config=config,
        worker=_run_mr2s_trial_with_cache,
        progress_printer=_print_mr2s_trial_progress,
        plotter=_plot_results,
        source_results_path=source_results_path,
    ).run()


def _print_trial_progress(
    index: int,
    total: int,
    n: int,
    trial: int,
    timings: dict[str, float],
) -> None:
    if timings.get("cache_hit"):
        print(f"[{index}/{total}] n={n}, trial={trial}: cache hit")
        return

    print(
        f"[{index}/{total}] n={n}, trial={trial}: "
        f"Graph {timings.get('graph', 0.0):.2f}s, "
        f"Raw SA {timings['raw_sa']:.2f}s, "
        f"Global {timings['global_solve']:.2f}s + {timings['global_embed']:.2f}s, "
        f"Clustered {timings['clustered_solve']:.2f}s + {timings['clustered_embed']:.2f}s, "
        f"Random {timings['random']:.2f}s"
    )


def _print_mr2s_trial_progress(
    index: int,
    total: int,
    n: int,
    trial: int,
    timings: dict[str, float],
) -> None:
    if timings.get("cache_hit"):
        print(f"[{index}/{total}] n={n}, trial={trial}: MR2S-only cache hit")
        return

    print(
        f"[{index}/{total}] n={n}, trial={trial}: "
        f"Graph {timings.get('graph', 0.0):.2f}s, "
        f"Clustered {timings['clustered_solve']:.2f}s + "
        f"{timings['clustered_embed']:.2f}s"
    )


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("poster-results", help="Generate visualization data for MR2S poster.")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    p.add_argument("--num-graphs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/poster")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for per-trial cache files; defaults to OUTPUT_DIR/poster_trial_cache.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the per-trial cache.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes to use; omit for auto, 0 for sequential.",
    )
    p.add_argument(
        "--mr2s-only",
        action="store_true",
        help="Recompute only DnCMr2sSolver results and merge into existing poster_results.json.",
    )
    p.add_argument(
        "--source-results",
        type=str,
        default=None,
        help="Existing poster_results.json to merge in MR2S-only mode.",
    )
    p.set_defaults(func=_dispatch)


def _dispatch(args: argparse.Namespace) -> None:
    if args.mr2s_only:
        run_mr2s_only(
            args.sizes,
            args.num_graphs,
            args.seed,
            args.output_dir,
            args.num_workers,
            args.cache_dir,
            not args.no_cache,
            args.source_results,
        )
        return

    run(
        args.sizes,
        args.num_graphs,
        args.seed,
        args.output_dir,
        args.num_workers,
        args.cache_dir,
        not args.no_cache,
    )
