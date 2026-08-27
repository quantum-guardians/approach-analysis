"""Trial runner - orchestrates all solvers for a single graph trial."""

from __future__ import annotations

import time
from typing import Any

import networkx as nx

from src.commands.poster_results.models import (
    BasicSolverResult,
    GlobalSolverResult,
    Mr2sSolverResult,
    Mr2sTrialResult,
    PosterTrialResult,
    RandomBaselineResult,
    TrialTimings,
)
from src.commands.poster_results.solvers.base import (
    _generate_trial_graph,
    _trial_seed,
)
from src.commands.poster_results.solvers.raw_sa import RawSASolver
from src.commands.poster_results.solvers.global_qubo import GlobalQuboSolver
from src.commands.poster_results.solvers.mr2s_variant import Mr2sVariantSolver, MR2S_VARIANTS
from src.commands.poster_results.solvers.dnc_strategy import (
    CLUSTER_DNC_STRATEGY,
    DncStrategySolver,
    DNC_STRATEGIES,
)
from src.commands.poster_results.solvers.random_baseline import RandomBaselineSolver


def _run_trial(task: tuple[int, int, int | None]) -> tuple[int, int, PosterTrialResult]:
    n, trial, seed = task
    trial_seed = _trial_seed(n, trial, seed)
    timings = TrialTimings()

    start = time.monotonic()
    graph, _, _ = _generate_trial_graph(n, trial, seed)
    timings.values["graph"] = time.monotonic() - start

    # 1. Raw SA baseline
    print(f"[{n=}, {trial=}] Running Raw SA...")
    res_rsa, rsa_timings = RawSASolver().run(graph, n, seed=trial_seed)
    timings.values.update(rsa_timings.values)

    # 2. Global QUBO baseline
    print(f"[{n=}, {trial=}] Running Global QUBO...")
    res_glb, glb_timings = GlobalQuboSolver().run(graph, n, seed=trial_seed)
    timings.values.update(glb_timings.values)

    # 3. MR2S variants
    mr2s_variant_results: dict[str, GlobalSolverResult] = {}
    for variant_name in MR2S_VARIANTS:
        print(f"[{n=}, {trial=}] Running MR2S variant: {variant_name}...")
        result, variant_timings = Mr2sVariantSolver(variant_name).run(
            graph, n, seed=trial_seed
        )
        mr2s_variant_results[variant_name] = result
        timings.values.update(variant_timings.values)

    # 4. DnC strategies
    dnc_strategy_results: dict[str, Mr2sSolverResult] = {}
    for strategy_name in DNC_STRATEGIES:
        print(f"[{n=}, {trial=}] Running DnC strategy: {strategy_name}...")
        result, strategy_timings = DncStrategySolver(strategy_name).run(
            graph, n, seed=trial_seed
        )
        dnc_strategy_results[strategy_name] = result
        timings.values.update(strategy_timings.values)

    res_cls = dnc_strategy_results.get(CLUSTER_DNC_STRATEGY, Mr2sSolverResult(
        apsp=float("nan"),
        flow=float("nan"),
        qvars=float("nan"),
        subgraph_size=float("nan"),
        phys_total=float("nan"),
        phys_max=float("nan"),
        phys_mean=float("nan"),
        phys_min=float("nan"),
    ))

    # 5. Random baseline
    print(f"[{n=}, {trial=}] Running Random baseline...")
    start = time.monotonic()
    res_rnd = RandomBaselineSolver()._calculate(graph, n, trial_seed)
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
    trial_seed = (seed + trial * 100 + n) if seed is not None else None
    start = time.monotonic()
    graph, _, _ = _generate_trial_graph(n, trial, seed)
    graph_time = time.monotonic() - start

    print(f"[{n=}, {trial=}] Running DnC strategy: {CLUSTER_DNC_STRATEGY}...")
    res_cls, timings = DncStrategySolver(CLUSTER_DNC_STRATEGY).run(graph, n, seed=trial_seed)
    timings.values["graph"] = graph_time
    return n, trial, Mr2sTrialResult(
        mr2s=res_cls,
        timings=timings,
    )


ALL_ALGORITHMS = (
    "raw_sa", "global", "mr2s", "random",
    *MR2S_VARIANTS,
    *DNC_STRATEGIES,
)

ALGORITHM_SOLVER_MAP: dict[str, type] = {
    "raw_sa": RawSASolver,
    "global": GlobalQuboSolver,
    "mr2s": DncStrategySolver,
    "random": RandomBaselineSolver,
}
ALGORITHM_SOLVER_MAP.update({name: Mr2sVariantSolver for name in MR2S_VARIANTS})
ALGORITHM_SOLVER_MAP.update({name: DncStrategySolver for name in DNC_STRATEGIES})


def _run_poster_algorithm(
    n: int,
    trial: int,
    seed: int | None,
    algorithm: str,
) -> tuple[BasicSolverResult | GlobalSolverResult | Mr2sSolverResult | RandomBaselineResult, TrialTimings]:
    trial_seed = _trial_seed(n, trial, seed)
    start = time.monotonic()
    graph, _, _ = _generate_trial_graph(n, trial, seed)
    graph_time = time.monotonic() - start

    solver_cls = ALGORITHM_SOLVER_MAP.get(algorithm)
    if solver_cls is None:
        raise ValueError(f"Unknown poster algorithm: {algorithm}")

    if algorithm == "mr2s":
        solver = DncStrategySolver(CLUSTER_DNC_STRATEGY)
    elif algorithm in MR2S_VARIANTS:
        solver = Mr2sVariantSolver(algorithm)
    elif algorithm in DNC_STRATEGIES:
        solver = DncStrategySolver(algorithm)
    else:
        solver = solver_cls()

    print(f"[{n=}, {trial=}] Running algorithm: {algorithm}...")
    result, timings = solver.run(graph, n, seed=trial_seed)
    timings.values["graph"] = graph_time
    return result, timings
