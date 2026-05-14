"""``poster-results`` subcommand – MR2S poster visualization data generation.

Four categories:
1. Raw SA (SAMR2SSolver)
2. Global (QuboMR2SSolver)
3. Clustered (DnCMr2sSolver)
4. Random baseline
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from typing import Any

import numpy as np
import mr2s_module.domain.graph

# Patch missing is_empty method in mr2s_module.domain.graph.Graph class directly
if not hasattr(mr2s_module.domain.graph.Graph, 'is_empty'):
    def is_empty(self):
        return len(self.edges) == 0
    mr2s_module.domain.graph.Graph.is_empty = is_empty

from mr2s_module import SAMR2SSolver, Evaluator, estimate_required_qubits, \
  QuboMR2SSolver, SAQuboSolver, ApspSumRanker, NHopPolyGenerator, \
  FlowPolyGenerator, SmallWorldSpec, NHop
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver

from src.commands.face_k_analysis import _generate_delaunay_graph, _nx_to_mr2s_graph
from src.case_generator import sample_strongly_connected_orientations
from src.cache import SimpleCache, generate_cache_key
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts
from src.visualizer import plot_apsp_reduction, plot_flow_stability, plot_preprocessing_scalability

spec = SmallWorldSpec([NHop(2, 1), NHop(3, 1)])
POSTER_CACHE_VERSION = 2

def _as_finite_or_nan(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")

def _mean_finite(values: list[float]) -> float:
    finite_values = [_as_finite_or_nan(value) for value in values]
    finite_values = [value for value in finite_values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))

def _normalize_random_baseline(result: dict[str, Any]) -> dict[str, Any]:
    """Treat missing random samples as unavailable, not as a zero score."""
    sample_count = result.get("sample_count")
    missing_legacy_sample = sample_count is None and result.get("apsp") == 0 and result.get("flow") == 0
    if sample_count == 0 or missing_legacy_sample:
        normalized = dict(result)
        normalized["apsp"] = float("nan")
        normalized["flow"] = float("nan")
        normalized["sample_count"] = 0
        return normalized
    return result

def _build_sa_solver(seed: int | None = None) -> SAMR2SSolver:
    return SAMR2SSolver(
        evaluator=Evaluator(),
        random_seed=seed
    )

def _build_qubo_solver() -> QuboMR2SSolver:
    n_hop_poly = NHopPolyGenerator()
    n_hop_poly.small_world_spec = spec
    return QuboMR2SSolver(
        qubo_solver=SAQuboSolver(ApspSumRanker()),
        evaluator=Evaluator(),
        poly_generators=[n_hop_poly, FlowPolyGenerator()],
    )

def _build_dnc_qubo_solver() -> DnCMr2sSolver:
    return DnCMr2sSolver(
      mr2s_solver=_build_qubo_solver()
    )

def _estimate_physical_qubits_with_status(bqm: Any) -> tuple[float, bool, str | None]:
    try:
        return float(estimate_required_qubits(bqm).num_physical_qubits), True, None
    except RuntimeError as exc:
        return float("nan"), False, str(exc)

def _estimate_physical_qubits(bqm: Any) -> float:
    """Return physical qubits for embeddable BQMs; NaN when embedding fails."""
    physical_qubits, _, _ = _estimate_physical_qubits_with_status(bqm)
    return physical_qubits

def _probe_embedding(mr2s_solver: QuboMR2SSolver, graph: Any) -> dict[str, Any]:
    bqm = mr2s_solver.build_bqm(graph)
    physical_qubits, can_embed, error = _estimate_physical_qubits_with_status(bqm)
    return {
        "can_embed": can_embed,
        "qvars": len(bqm.variables),
        "physical_qubits": physical_qubits,
        "error": error,
    }

def _can_recurse_partition(parent: Any, sub_graphs: list[Any]) -> bool:
    if not sub_graphs:
        return False

    parent_edge_count = len(parent.edges)
    return all(
        0 < len(sub_graph.edges) < parent_edge_count
        for sub_graph in sub_graphs
    )

def _partition_with_target_k(face_cycle: Any, graph: Any, target_k: int) -> Any:
    previous_target_k = face_cycle.target_k
    face_cycle.target_k = target_k
    try:
        return face_cycle.run(graph)
    finally:
        face_cycle.target_k = previous_target_k

def _summarize_partition_attempt(
    target_k: int,
    result: Any,
    can_recurse: bool,
    probes: list[dict[str, Any]],
    accepted: bool,
) -> dict[str, Any]:
    sub_graphs = result.sub_graphs
    finite_physical = [
        probe["physical_qubits"]
        for probe in probes
        if np.isfinite(probe["physical_qubits"])
    ]
    qvars = [probe["qvars"] for probe in probes]
    return {
        "target_k": target_k,
        "subgraph_count": len(sub_graphs),
        "remaining_edge_count": len(result.remaining_edges),
        "edge_counts": [len(sub_graph.edges) for sub_graph in sub_graphs],
        "can_recurse": can_recurse,
        "all_embed": bool(probes) and all(probe["can_embed"] for probe in probes),
        "accepted": accepted,
        "embed_failures": sum(not probe["can_embed"] for probe in probes),
        "qvars": qvars,
        "max_qvars": max(qvars) if qvars else 0,
        "total_qvars": sum(qvars),
        "physical_qubits": [probe["physical_qubits"] for probe in probes],
        "physical_total": float(sum(finite_physical)) if finite_physical else float("nan"),
    }

def _find_partition_by_target_k_with_diagnostics(
    mr2s_solver: QuboMR2SSolver,
    face_cycle: Any,
    graph: Any,
) -> tuple[list[Any], dict[str, Any]]:
    left = 2
    right = max(2, len(graph.edges))
    best_sub_graphs: list[Any] | None = None
    best_probes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    while left <= right:
        target_k = (left + right) // 2
        result = _partition_with_target_k(face_cycle, graph, target_k)
        sub_graphs = result.sub_graphs
        can_recurse = _can_recurse_partition(graph, sub_graphs)
        probes = (
            [_probe_embedding(mr2s_solver, sub_graph) for sub_graph in sub_graphs]
            if can_recurse
            else []
        )
        accepted = can_recurse and all(probe["can_embed"] for probe in probes)
        attempts.append(
            _summarize_partition_attempt(
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

def _divide_graph_with_diagnostics(solver: DnCMr2sSolver, graph: Any) -> tuple[list[Any], dict[str, Any]]:
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
        diagnostics["selected_reason"] = "fallback_whole_graph"
        diagnostics["selected_probes"] = [whole_graph_probe]
        return [graph], diagnostics
    return sub_graphs, diagnostics

def _physical_qubit_stats(values: list[float]) -> dict[str, float]:
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

def _run_clustered_solver(graph: Any, n: int) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}

    start = time.monotonic()
    solver_cls = _build_dnc_qubo_solver()
    graph_cls = _nx_to_mr2s_graph(graph)
    sub_graphs_cls, partition_diagnostics = _divide_graph_with_diagnostics(
        solver_cls,
        graph_cls,
    )
    if len(sub_graphs_cls) == 1 and sub_graphs_cls[0] is graph_cls:
        sol_cls = solver_cls.mr2s_solver.run(graph_cls)
    else:
        sub_solutions = solver_cls._solve_subgraphs(sub_graphs_cls)
        merged_solution = solver_cls.merge_solutions(
            solutions=sub_solutions,
            graph=graph_cls,
        )
        solver_cls._apply_merged_directions(graph_cls, merged_solution)
        sol_cls = solver_cls.mr2s_solver.run(graph_cls)
        sol_cls.score = solver_cls.score_merged_solution(sol_cls, sub_solutions)
    timings["clustered_solve"] = time.monotonic() - start

    start = time.monotonic()
    selected_probes = partition_diagnostics.get("selected_probes", [])
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

    timings["clustered_embed"] = time.monotonic() - start
    phys_cls = _physical_qubit_stats(cluster_phys)

    return {
        "apsp": sol_cls.score.apsp_sum / (n * (n - 1)),
        "flow": sol_cls.score.flow_score,
        "qvars": sum(cluster_qvars),
        "sg": max(cluster_qvars),
        "phys_total": phys_cls["total"],
        "phys_max": phys_cls["max"],
        "phys_mean": phys_cls["mean"],
        "phys_min": phys_cls["min"],
        "partition": partition_diagnostics,
    }, timings

def _run_trial(task: tuple[int, int, int | None]) -> tuple[int, int, dict[str, Any]]:
    """Run all poster-result solvers for one graph trial."""
    n, trial, seed = task
    trial_seed = (seed + trial * 100 + n) if seed is not None else None
    graph = _generate_delaunay_graph(n, trial_seed)
    timings: dict[str, float] = {}

    # 1. Raw SA
    start = time.monotonic()
    solver_rsa = _build_sa_solver(seed=trial_seed)
    solver_rsa.face_cycle = None
    sol_rsa = solver_rsa.run(_nx_to_mr2s_graph(graph))
    timings["raw_sa"] = time.monotonic() - start
    res_rsa = {
        "apsp": sol_rsa.score.apsp_sum / (n * (n - 1)),
        "flow": sol_rsa.score.flow_score,
    }

    # 2. Global
    start = time.monotonic()
    solver_glb = _build_qubo_solver()
    solver_glb.face_cycle = None
    sol_glb = solver_glb.run(_nx_to_mr2s_graph(graph))
    timings["global_solve"] = time.monotonic() - start

    start = time.monotonic()
    bqm_glb = solver_glb.build_bqm(_nx_to_mr2s_graph(graph))
    phys_glb = _estimate_physical_qubits(bqm_glb)
    timings["global_embed"] = time.monotonic() - start

    res_glb = {
        "apsp": sol_glb.score.apsp_sum / (n * (n - 1)),
        "flow": sol_glb.score.flow_score,
        "qvars": len(bqm_glb.variables),
        "sg": len(bqm_glb.variables),
        "pt": phys_glb,
    }

    # 3. Clustered
    res_cls, clustered_timings = _run_clustered_solver(graph, n)
    timings.update(clustered_timings)

    # 4. Random
    start = time.monotonic()
    random_samples = list(
        sample_strongly_connected_orientations(
            graph, max_samples=10, seed=trial_seed
        )
    )
    if random_samples:
        trial_apsp = []
        trial_flow = []
        for orient in random_samples:
            apsp, _ = calculate_apsp_sum_and_nhop_neighbor_counts(orient, hops=[])
            trial_apsp.append(apsp / (n * (n - 1)))
            imbalance = sum(
                (orient.in_degree(node) - orient.out_degree(node)) ** 2
                for node in orient.nodes()
            )
            trial_flow.append(imbalance)
        res_rnd = {
            "apsp": np.mean(trial_apsp),
            "flow": np.mean(trial_flow),
            "sample_count": len(random_samples),
        }
    else:
        res_rnd = {
            "apsp": float("nan"),
            "flow": float("nan"),
            "sample_count": 0,
        }
    timings["random"] = time.monotonic() - start

    return n, trial, {
        "raw_sa": res_rsa,
        "global": res_glb,
        "mr2s": res_cls,
        "random": res_rnd,
        "timings": timings,
    }

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

def _run_trial_with_cache(
    task: tuple[int, int, int | None, str | None]
) -> tuple[int, int, dict[str, Any]]:
    """Run one poster trial, reusing the on-disk cache when configured."""
    n, trial, seed, cache_dir = task
    if cache_dir is None:
        n, trial, result = _run_trial((n, trial, seed))
        result["cache_hit"] = False
        result.setdefault("timings", {})["cache_hit"] = False
        return n, trial, result

    cache = SimpleCache(cache_dir)
    cache_key = _poster_trial_cache_key(n, trial, seed)
    cached_result = cache.get(cache_key)
    if isinstance(cached_result, dict):
        cached_result["cache_hit"] = True
        cached_result.setdefault("timings", {})["cache_hit"] = True
        return n, trial, cached_result

    n, trial, result = _run_trial((n, trial, seed))
    result["cache_hit"] = False
    result.setdefault("timings", {})["cache_hit"] = False
    cache.set(cache_key, result)
    return n, trial, result

def _run_mr2s_trial(task: tuple[int, int, int | None]) -> tuple[int, int, dict[str, Any]]:
    n, trial, seed = task
    trial_seed = (seed + trial * 100 + n) if seed is not None else None
    graph = _generate_delaunay_graph(n, trial_seed)
    res_cls, timings = _run_clustered_solver(graph, n)
    return n, trial, {
        "mr2s": res_cls,
        "timings": timings,
    }

def _run_mr2s_trial_with_cache(
    task: tuple[int, int, int | None, str | None]
) -> tuple[int, int, dict[str, Any]]:
    n, trial, seed, cache_dir = task
    if cache_dir is None:
        n, trial, result = _run_mr2s_trial((n, trial, seed))
        result["cache_hit"] = False
        result.setdefault("timings", {})["cache_hit"] = False
        return n, trial, result

    cache = SimpleCache(cache_dir)
    cache_key = _poster_mr2s_trial_cache_key(n, trial, seed)
    cached_result = cache.get(cache_key)
    if isinstance(cached_result, dict):
        cached_result["cache_hit"] = True
        cached_result.setdefault("timings", {})["cache_hit"] = True
        return n, trial, cached_result

    n, trial, result = _run_mr2s_trial((n, trial, seed))
    result["cache_hit"] = False
    result.setdefault("timings", {})["cache_hit"] = False
    cache.set(cache_key, result)
    return n, trial, result

def _aggregate_mr2s_results(results: dict[str, Any], trial_results: dict[int, list[dict[str, Any]]]) -> None:
    results["mr2s"] = {
        "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
        "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": [],
        "partition": [],
    }

    for n in results["sizes"]:
        a_cls, f_cls, v_cls, s_cls = [], [], [], []
        pt_cls, pmax_cls, pmean_cls, pmin_cls = [], [], [], []
        partitions = []

        print(f"\n>>> Aggregating MR2S-only size n={n}")

        for result in trial_results[n]:
            res_cls = result["mr2s"]
            a_cls.append(res_cls["apsp"])
            f_cls.append(res_cls["flow"])
            v_cls.append(res_cls["qvars"])
            s_cls.append(res_cls["sg"])
            pt_cls.append(res_cls["phys_total"])
            pmax_cls.append(res_cls["phys_max"])
            pmean_cls.append(res_cls["phys_mean"])
            pmin_cls.append(res_cls["phys_min"])
            partitions.append(res_cls.get("partition", {}))

        results["mr2s"]["apsp"].append(_mean_finite(a_cls))
        results["mr2s"]["flow"].append(_mean_finite(f_cls))
        results["mr2s"]["qubo_vars"].append(_mean_finite(v_cls))
        results["mr2s"]["subgraph_size"].append(_mean_finite(s_cls))
        results["mr2s"]["phys_total"].append(_mean_finite(pt_cls))
        results["mr2s"]["phys_max"].append(_mean_finite(pmax_cls))
        results["mr2s"]["phys_mean"].append(_mean_finite(pmean_cls))
        results["mr2s"]["phys_min"].append(_mean_finite(pmin_cls))
        results["mr2s"]["partition"].append(partitions)

def run(
    sizes: list[int],
    num_graphs: int,
    seed: int | None,
    output_dir: str,
    num_workers: int | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir is None and use_cache:
        cache_dir = os.path.join(output_dir, "poster_trial_cache")
    elif not use_cache:
        cache_dir = None

    results = {
        "sizes": sizes,
        "mr2s": {
            "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
            "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": [],
            "partition": [],
        },
        "global": {
            "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
            "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": []
        },
        "raw_sa": {"apsp": [], "flow": []},
        "random": {"apsp": [], "flow": []},
    }

    tasks = [(n, trial, seed, cache_dir) for n in sizes for trial in range(num_graphs)]
    total_tasks = len(tasks)
    effective_workers = (
        num_workers
        if num_workers is not None
        else min(total_tasks, max((os.cpu_count() or 2) - 1, 1))
    )

    print(
        f"Starting poster results: {len(sizes)} size(s), "
        f"{num_graphs} graph(s) each, {effective_workers} worker process(es)."
    )
    if cache_dir is not None:
        print(f"Using poster trial cache: {cache_dir}")

    trial_results: dict[int, list[dict[str, Any]]] = {n: [] for n in sizes}

    if effective_workers == 0:
        for index, task in enumerate(tasks, start=1):
            n, trial, result = _run_trial_with_cache(task)
            _print_trial_progress(index, total_tasks, n, trial, result["timings"])
            trial_results[n].append(result)
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=effective_workers) as pool:
            for index, (n, trial, result) in enumerate(
                pool.imap_unordered(_run_trial_with_cache, tasks),
                start=1,
            ):
                _print_trial_progress(index, total_tasks, n, trial, result["timings"])
                trial_results[n].append(result)

    for n in sizes:
        a_rsa, f_rsa = [], []
        a_glb, f_glb, v_glb, s_glb, p_glb = [], [], [], [], []
        a_cls, f_cls, v_cls, s_cls, pt_cls, pmax_cls, pmean_cls, pmin_cls = [], [], [], [], [], [], [], []
        a_rnd, f_rnd = [], []
        partitions = []

        print(f"\n>>> Aggregating size n={n}")

        for result in trial_results[n]:
            res_rsa = result["raw_sa"]
            res_glb = result["global"]
            res_cls = result["mr2s"]
            res_rnd = _normalize_random_baseline(result["random"])
            # Accumulate
            a_rsa.append(res_rsa["apsp"]); f_rsa.append(res_rsa["flow"])
            a_glb.append(res_glb["apsp"]); f_glb.append(res_glb["flow"]); v_glb.append(res_glb["qvars"]); s_glb.append(res_glb["sg"]); p_glb.append(res_glb["pt"])
            a_cls.append(res_cls["apsp"]); f_cls.append(res_cls["flow"]); v_cls.append(res_cls["qvars"]); s_cls.append(res_cls["sg"])
            pt_cls.append(res_cls["phys_total"]); pmax_cls.append(res_cls["phys_max"]); pmean_cls.append(res_cls["phys_mean"]); pmin_cls.append(res_cls["phys_min"])
            partitions.append(res_cls.get("partition", {}))
            a_rnd.append(res_rnd["apsp"]); f_rnd.append(res_rnd["flow"])

        # Store averages
        results["raw_sa"]["apsp"].append(_mean_finite(a_rsa))
        results["raw_sa"]["flow"].append(_mean_finite(f_rsa))
        results["global"]["apsp"].append(_mean_finite(a_glb))
        results["global"]["flow"].append(_mean_finite(f_glb))
        results["global"]["qubo_vars"].append(_mean_finite(v_glb))
        results["global"]["subgraph_size"].append(_mean_finite(s_glb))
        results["global"]["phys_total"].append(_mean_finite(p_glb))
        results["mr2s"]["apsp"].append(_mean_finite(a_cls))
        results["mr2s"]["flow"].append(_mean_finite(f_cls))
        results["mr2s"]["qubo_vars"].append(_mean_finite(v_cls))
        results["mr2s"]["subgraph_size"].append(_mean_finite(s_cls))
        results["mr2s"]["phys_total"].append(_mean_finite(pt_cls))
        results["mr2s"]["phys_max"].append(_mean_finite(pmax_cls))
        results["mr2s"]["phys_mean"].append(_mean_finite(pmean_cls))
        results["mr2s"]["phys_min"].append(_mean_finite(pmin_cls))
        results["mr2s"]["partition"].append(partitions)
        results["random"]["apsp"].append(_mean_finite(a_rnd))
        results["random"]["flow"].append(_mean_finite(f_rnd))

    with open(os.path.join(output_dir, "poster_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_results(results, output_dir)

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
    os.makedirs(output_dir, exist_ok=True)
    if source_results_path is None:
        source_results_path = os.path.join(output_dir, "poster_results.json")
    if not os.path.exists(source_results_path):
        raise FileNotFoundError(
            f"MR2S-only mode needs existing results to merge: {source_results_path}"
        )

    with open(source_results_path) as f:
        results = json.load(f)
    results["sizes"] = sizes

    if cache_dir is None and use_cache:
        cache_dir = os.path.join(output_dir, "poster_mr2s_trial_cache")
    elif not use_cache:
        cache_dir = None

    tasks = [(n, trial, seed, cache_dir) for n in sizes for trial in range(num_graphs)]
    total_tasks = len(tasks)
    effective_workers = (
        num_workers
        if num_workers is not None
        else min(total_tasks, max((os.cpu_count() or 2) - 1, 1))
    )

    print(
        f"Starting MR2S-only poster results: {len(sizes)} size(s), "
        f"{num_graphs} graph(s) each, {effective_workers} worker process(es)."
    )
    print(f"Merging into: {source_results_path}")
    if cache_dir is not None:
        print(f"Using MR2S-only trial cache: {cache_dir}")

    trial_results: dict[int, list[dict[str, Any]]] = {n: [] for n in sizes}

    if effective_workers == 0:
        for index, task in enumerate(tasks, start=1):
            n, trial, result = _run_mr2s_trial_with_cache(task)
            _print_mr2s_trial_progress(index, total_tasks, n, trial, result["timings"])
            trial_results[n].append(result)
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=effective_workers) as pool:
            for index, (n, trial, result) in enumerate(
                pool.imap_unordered(_run_mr2s_trial_with_cache, tasks),
                start=1,
            ):
                _print_mr2s_trial_progress(index, total_tasks, n, trial, result["timings"])
                trial_results[n].append(result)

    _aggregate_mr2s_results(results, trial_results)

    with open(os.path.join(output_dir, "poster_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_results(results, output_dir)

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
        f"Clustered {timings['clustered_solve']:.2f}s + "
        f"{timings['clustered_embed']:.2f}s"
    )

def _plot_results(results: dict, output_dir: str):
    sizes = results["sizes"]
    plot_apsp_reduction(sizes, results["random"]["apsp"], results["raw_sa"]["apsp"], results["global"]["apsp"], results["mr2s"]["apsp"], save_path=os.path.join(output_dir, "apsp_reduction.png"))
    plot_flow_stability(sizes, results["random"]["flow"], results["raw_sa"]["flow"], results["global"]["flow"], results["mr2s"]["flow"], save_path=os.path.join(output_dir, "flow_stability.png"))
    plot_preprocessing_scalability(sizes, results["global"]["qubo_vars"], results["mr2s"]["qubo_vars"], results["global"]["subgraph_size"], results["mr2s"]["subgraph_size"], global_physical=results["global"].get("phys_total"), clustered_physical_total=results["mr2s"].get("phys_total"), clustered_physical_max=results["mr2s"].get("phys_max"), clustered_physical_mean=results["mr2s"].get("phys_mean"), clustered_physical_min=results["mr2s"].get("phys_min"), save_path=os.path.join(output_dir, "scalability.png"))

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
