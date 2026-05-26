"""Solver and trial helpers for the ``poster-results`` command."""

from __future__ import annotations

import time
from typing import Any

import mr2s_module.domain.graph
import networkx as nx
import numpy as np
from mr2s_module import (
    ApspSumRanker,
    Evaluator,
    FlowPolyGenerator,
    NHop,
    NHopPolyGenerator,
    QuboMR2SSolver,
    Robbin,
    SAMR2SSolver,
    SAQuboSolver,
    SmallWorldSpec,
)
from mr2s_module.edge_orient.iterated_local_search import IteratedLocalSearch
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.dnc_graph_partition_strategy import (
    DegeneracyPruningFaceCyclePartitionStrategy,
    EmbeddingAwareFaceCyclePartitionStrategy,
)

from src.commands.face_k_analysis import _generate_delaunay_graph, _nx_to_mr2s_graph
from src.commands.poster_results_models import (
    BasicSolverResult,
    GlobalSolverResult,
    Mr2sSolverResult,
    Mr2sTrialResult,
    PosterTrialResult,
    RandomBaselineResult,
    TrialTimings,
)
from src.commands.poster_results_partition_strategy import (
    DncPartitionStrategy,
    TimedPartitionStrategy,
    _can_recurse_partition,
    _divide_graph_with_diagnostics,
    _estimate_physical_qubits,
    _estimate_physical_qubits_with_status,
    _find_partition_by_target_k_with_diagnostics,
    _partition_with_target_k,
    _probe_embedding,
    _summarize_partition_attempt,
)
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts

# mr2s_module의 Graph 구현에 is_empty가 없는 버전이 있어 런타임에서 보강한다.
if not hasattr(mr2s_module.domain.graph.Graph, "is_empty"):
    def is_empty(self):
        return len(self.edges) == 0

    mr2s_module.domain.graph.Graph.is_empty = is_empty


spec = SmallWorldSpec([NHop(2, 1), NHop(3, 1)])
DNC_STRATEGIES = ["poster", "embedding_aware", "degeneracy_pruning"]
MR2S_VARIANTS = ["robbin_mr2s", "iterated_local_search_mr2s"]


def _as_finite_or_nan(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _mean_finite(values: list[float]) -> float:
    # embedding 실패 등으로 생긴 NaN/Infinity는 평균에서 제외한다.
    finite_values = [_as_finite_or_nan(value) for value in values]
    finite_values = [value for value in finite_values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _normalize_random_baseline(result: RandomBaselineResult | dict[str, Any]) -> RandomBaselineResult | dict[str, Any]:
    """Treat missing random samples as unavailable, not as a zero score."""
    if isinstance(result, RandomBaselineResult):
        return result.normalized()

    sample_count = result.get("sample_count")
    # 예전 캐시는 random sample이 없을 때 0점처럼 저장된 경우가 있어 NaN으로 보정한다.
    missing_legacy_sample = sample_count is None and result.get("apsp") == 0 and result.get("flow") == 0
    if sample_count == 0 or missing_legacy_sample:
        normalized = dict(result)
        normalized["apsp"] = float("nan")
        normalized["flow"] = float("nan")
        normalized["sample_count"] = 0
        return normalized
    return result


def _sample_random_orientations(
    graph: nx.Graph,
    max_samples: int,
    seed: int | None = None,
) -> list[nx.DiGraph]:
    """Sample arbitrary edge orientations without filtering by connectivity."""
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    edges = list(graph.edges())
    nodes = list(graph.nodes())
    rng = np.random.default_rng(seed)
    orientations: list[nx.DiGraph] = []

    for _ in range(max_samples):
        # 각 undirected edge마다 난수 bit 하나로 방향을 선택한다.
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        bits = rng.integers(0, 2, size=len(edges))
        for bit, (u, v) in zip(bits, edges):
            if bit == 0:
                dg.add_edge(u, v)
            else:
                dg.add_edge(v, u)
        orientations.append(dg)

    return orientations


def _flow_imbalance_score(graph: nx.DiGraph) -> int:
    return sum(
        (graph.in_degree(node) - graph.out_degree(node)) ** 2
        for node in graph.nodes()
    )


def _calculate_random_baseline(
    graph: nx.Graph,
    n: int,
    seed: int | None,
    max_samples: int = 10,
) -> RandomBaselineResult:
    random_samples = _sample_random_orientations(graph, max_samples=max_samples, seed=seed)
    trial_apsp = []
    trial_flow = []

    for orient in random_samples:
        # APSP는 강연결 샘플만 평균에 넣고, flow는 모든 random orientation의 분포를 본다.
        if nx.is_strongly_connected(orient):
            apsp, _ = calculate_apsp_sum_and_nhop_neighbor_counts(orient, hops=[])
            trial_apsp.append(apsp / (n * (n - 1)))
        trial_flow.append(_flow_imbalance_score(orient))

    return RandomBaselineResult(
        apsp=_mean_finite(trial_apsp),
        flow=_mean_finite(trial_flow),
        sample_count=len(random_samples),
        strong_sample_count=len(trial_apsp),
    )


def _build_sa_solver(seed: int | None = None) -> SAMR2SSolver:
    return SAMR2SSolver(
        evaluator=Evaluator(),
        random_seed=seed,
    )


def _build_qubo_solver(edge_orienter: Any | None = None) -> QuboMR2SSolver:
    # poster 실험에서는 2-hop/3-hop small-world 항과 flow 항을 함께 사용한다.
    n_hop_poly = NHopPolyGenerator()
    n_hop_poly.small_world_spec = spec
    return QuboMR2SSolver(
        edge_orienter=edge_orienter,
        qubo_solver=SAQuboSolver(ApspSumRanker()),
        evaluator=Evaluator(),
        poly_generators=[n_hop_poly, FlowPolyGenerator()],
    )


def _build_edge_orienter(variant_name: str) -> Any:
    if variant_name == "robbin_mr2s":
        return Robbin()
    if variant_name == "iterated_local_search_mr2s":
        return IteratedLocalSearch()
    raise ValueError(f"Unknown MR2S variant: {variant_name}")


def _run_oriented_mr2s_solver(
    graph: nx.Graph,
    n: int,
    variant_name: str,
) -> tuple[GlobalSolverResult, TrialTimings]:
    timings = TrialTimings()

    start = time.monotonic()
    solver = _build_qubo_solver(edge_orienter=_build_edge_orienter(variant_name))
    sol = solver.run(_nx_to_mr2s_graph(graph))
    timings.values[f"{variant_name}_solve"] = time.monotonic() - start

    start = time.monotonic()
    embed_solver = _build_qubo_solver(edge_orienter=_build_edge_orienter(variant_name))
    bqm = embed_solver.build_bqm(_nx_to_mr2s_graph(graph))
    physical_qubits = _estimate_physical_qubits(bqm)
    timings.values[f"{variant_name}_embed"] = time.monotonic() - start

    return GlobalSolverResult(
        apsp=sol.score.apsp_sum / (n * (n - 1)),
        flow=sol.score.flow_score,
        qvars=len(bqm.variables),
        subgraph_size=len(bqm.variables),
        physical_total=physical_qubits,
    ), timings


def _build_dnc_qubo_solver(
    partition_strategy: Any | None = None,
) -> DnCMr2sSolver:
    solver = DnCMr2sSolver(
        mr2s_solver=_build_qubo_solver(),
    )
    if partition_strategy is not None:
        # mr2s-module 0.0.8의 graph_partition_strategy hook에 poster용 전략을 주입한다.
        partition_strategy.mr2s_solver = solver.mr2s_solver
        partition_strategy.face_cycle = solver.face_cycle
        solver.graph_partition_strategy = partition_strategy
    return solver


def _build_partition_strategy(strategy_name: str, solver: DnCMr2sSolver) -> Any:
    if strategy_name == "poster":
        strategy = DncPartitionStrategy()
        strategy.mr2s_solver = solver.mr2s_solver
        strategy.face_cycle = solver.face_cycle
        return strategy
    if strategy_name == "embedding_aware":
        return TimedPartitionStrategy(
            EmbeddingAwareFaceCyclePartitionStrategy(
                mr2s_solver=solver.mr2s_solver,
                face_cycle=solver.face_cycle,
                target_graph=solver.target_graph,
            )
        )
    if strategy_name == "degeneracy_pruning":
        return TimedPartitionStrategy(
            DegeneracyPruningFaceCyclePartitionStrategy(
                mr2s_solver=solver.mr2s_solver,
                face_cycle=solver.face_cycle,
                target_graph=solver.target_graph,
            )
        )
    raise ValueError(f"Unknown DnC partition strategy: {strategy_name}")


def _physical_qubit_stats(values: list[float]) -> dict[str, float]:
    # embedding 실패 NaN은 통계에서 제외하고, 모두 실패했을 때만 NaN 통계를 낸다.
    finite_values = [value for value in values if not np.isnan(value)]
    if not finite_values:
        return {
            "total": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
        }
    return {
        "total": float(sum(finite_values)),
        "max": float(max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "min": float(min(finite_values)),
    }


def _diagnostics_from_solution(strategy_name: str, sol_cls: Any) -> dict[str, Any]:
    estimates = list(getattr(sol_cls, "embedding_estimates", []) or [])
    sub_graphs = list(getattr(sol_cls, "sub_graphs", []) or [])
    probes = [
        {
            "can_embed": True,
            "qvars": estimate.num_logical_variables,
            "physical_qubits": float(estimate.num_physical_qubits),
            "error": None,
        }
        for estimate in estimates
    ]
    return {
        "strategy": strategy_name,
        "selected_reason": "module_strategy",
        "partition_target_k": getattr(sol_cls, "partition_target_k", None),
        "selected_probes": probes,
        "subgraph_edge_counts": [len(sub_graph.edges) for sub_graph in sub_graphs],
    }


def _failed_dnc_result(strategy_name: str, error: Exception) -> Mr2sSolverResult:
    return Mr2sSolverResult(
        apsp=float("nan"),
        flow=float("nan"),
        qvars=float("nan"),
        subgraph_size=float("nan"),
        phys_total=float("nan"),
        phys_max=float("nan"),
        phys_mean=float("nan"),
        phys_min=float("nan"),
        partition={
            "strategy": strategy_name,
            "selected_reason": "failed",
            "error": str(error),
        },
    )


def _run_dnc_strategy_solver(
    graph: Any,
    n: int,
    strategy_name: str,
) -> tuple[Mr2sSolverResult, TrialTimings]:
    timings = TrialTimings()

    # strategy별 DnC solver를 별도로 실행해 같은 graph에서 partition 정책을 비교한다.
    start = time.monotonic()
    solver_cls = _build_dnc_qubo_solver()
    partition_strategy = _build_partition_strategy(strategy_name, solver_cls)
    solver_cls.graph_partition_strategy = partition_strategy
    graph_cls = _nx_to_mr2s_graph(graph)
    try:
        sol_cls = solver_cls.run(graph_cls)
    except RuntimeError as exc:
        elapsed = time.monotonic() - start
        timings.values[f"dnc_{strategy_name}_solve"] = 0.0
        timings.values[f"dnc_{strategy_name}_embed"] = elapsed
        if strategy_name == "poster":
            timings.values["clustered_solve"] = 0.0
            timings.values["clustered_embed"] = elapsed
        return _failed_dnc_result(strategy_name, exc), timings

    partition_diagnostics = getattr(partition_strategy, "last_diagnostics", None)
    if not partition_diagnostics:
        partition_diagnostics = _diagnostics_from_solution(strategy_name, sol_cls)
    clustered_total = time.monotonic() - start
    clustered_embed = getattr(partition_strategy, "last_elapsed_seconds", 0.0)
    # DnCMr2sSolver.run() 안에서 partition/embed가 먼저 실행되므로 solve 시간에서 분리한다.
    solve_time = max(0.0, clustered_total - clustered_embed)
    timings.values[f"dnc_{strategy_name}_solve"] = solve_time
    timings.values[f"dnc_{strategy_name}_embed"] = clustered_embed
    if strategy_name == "poster":
        timings.values["clustered_solve"] = solve_time
        timings.values["clustered_embed"] = clustered_embed

    selected_probes = partition_diagnostics.get("selected_probes", [])
    # qvars가 0인 trivial subgraph는 resource 통계에서 제외한다.
    cluster_phys = [
        probe["physical_qubits"]
        for probe in selected_probes
        if probe["qvars"] > 0
    ]
    cluster_qvars = [
        probe["qvars"]
        for probe in selected_probes
        if probe["qvars"] > 0
    ]

    if not cluster_qvars:
        cluster_qvars = [0]
    if not cluster_phys:
        cluster_phys = [0]

    phys_cls = _physical_qubit_stats(cluster_phys)

    return Mr2sSolverResult(
        apsp=sol_cls.score.apsp_sum / (n * (n - 1)),
        flow=sol_cls.score.flow_score,
        qvars=sum(cluster_qvars),
        subgraph_size=max(cluster_qvars),
        phys_total=phys_cls["total"],
        phys_max=phys_cls["max"],
        phys_mean=phys_cls["mean"],
        phys_min=phys_cls["min"],
        partition=partition_diagnostics,
    ), timings


def _run_clustered_solver(graph: Any, n: int) -> tuple[Mr2sSolverResult, TrialTimings]:
    return _run_dnc_strategy_solver(graph, n, "poster")


def _run_trial(task: tuple[int, int, int | None]) -> tuple[int, int, PosterTrialResult]:
    """Run all poster-result solvers for one graph trial."""
    n, trial, seed = task
    # size와 trial index를 seed에 섞어 size별 그래프가 재현 가능하면서도 서로 달라지게 한다.
    trial_seed = (seed + trial * 100 + n) if seed is not None else None
    timings = TrialTimings()

    # graph 생성 시간도 solver 시간과 분리해서 기록한다.
    start = time.monotonic()
    graph = _generate_delaunay_graph(n, trial_seed)
    timings.values["graph"] = time.monotonic() - start

    # 1. Raw SA baseline
    start = time.monotonic()
    solver_rsa = _build_sa_solver(seed=trial_seed)
    solver_rsa.face_cycle = None
    sol_rsa = solver_rsa.run(_nx_to_mr2s_graph(graph))
    timings.values["raw_sa"] = time.monotonic() - start
    res_rsa = BasicSolverResult(
        apsp=sol_rsa.score.apsp_sum / (n * (n - 1)),
        flow=sol_rsa.score.flow_score,
    )

    # 2. 전체 그래프를 하나의 QUBO로 푸는 global baseline
    start = time.monotonic()
    solver_glb = _build_qubo_solver()
    solver_glb.face_cycle = None
    sol_glb = solver_glb.run(_nx_to_mr2s_graph(graph))
    timings.values["global_solve"] = time.monotonic() - start

    start = time.monotonic()
    bqm_glb = solver_glb.build_bqm(_nx_to_mr2s_graph(graph))
    phys_glb = _estimate_physical_qubits(bqm_glb)
    timings.values["global_embed"] = time.monotonic() - start

    res_glb = GlobalSolverResult(
        apsp=sol_glb.score.apsp_sum / (n * (n - 1)),
        flow=sol_glb.score.flow_score,
        qvars=len(bqm_glb.variables),
        subgraph_size=len(bqm_glb.variables),
        physical_total=phys_glb,
    )

    # 3. 분할-병합 기반 MR2S solver를 strategy별로 실행한다.
    mr2s_variant_results: dict[str, GlobalSolverResult] = {}
    for variant_name in MR2S_VARIANTS:
        result, variant_timings = _run_oriented_mr2s_solver(graph, n, variant_name)
        mr2s_variant_results[variant_name] = result
        timings.values.update(variant_timings.values)

    # 4. 분할-병합 기반 MR2S solver를 strategy별로 실행한다.
    dnc_strategy_results: dict[str, Mr2sSolverResult] = {}
    for strategy_name in DNC_STRATEGIES:
        result, strategy_timings = _run_dnc_strategy_solver(graph, n, strategy_name)
        dnc_strategy_results[strategy_name] = result
        timings.values.update(strategy_timings.values)
    res_cls = dnc_strategy_results["poster"]

    # 5. 임의 방향 baseline
    start = time.monotonic()
    res_rnd = _calculate_random_baseline(graph, n, trial_seed)
    timings.values["random"] = time.monotonic() - start

    return n, trial, PosterTrialResult(
        raw_sa=res_rsa,
        global_result=res_glb,
        mr2s=res_cls,
        mr2s_variants=mr2s_variant_results,
        dnc_strategies=dnc_strategy_results,
        random=res_rnd,
        timings=timings,
    )


def _run_mr2s_trial(task: tuple[int, int, int | None]) -> tuple[int, int, Mr2sTrialResult]:
    n, trial, seed = task
    # MR2S-only 모드는 같은 seed 규칙으로 그래프를 재생성해 기존 결과와 병합한다.
    trial_seed = (seed + trial * 100 + n) if seed is not None else None
    start = time.monotonic()
    graph = _generate_delaunay_graph(n, trial_seed)
    graph_time = time.monotonic() - start
    res_cls, timings = _run_clustered_solver(graph, n)
    timings.values["graph"] = graph_time
    return n, trial, Mr2sTrialResult(
        mr2s=res_cls,
        timings=timings,
    )
