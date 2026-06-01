"""Solver package for poster results."""

from src.commands.poster_results.solvers.base import (
    BaseSolver,
    _as_finite_or_nan,
    _mean_finite,
    _trial_seed,
    _build_sa_solver,
    _build_qubo_solver,
    _extract_directed_edges_from_solution,
)
from src.commands.poster_results.solvers.raw_sa import RawSASolver
from src.commands.poster_results.solvers.global_qubo import GlobalQuboSolver
from src.commands.poster_results.solvers.mr2s_variant import (
    Mr2sVariantSolver,
    MR2S_VARIANTS,
    _build_edge_orienter,
)
from src.commands.poster_results.solvers.dnc_strategy import (
    DncStrategySolver,
    DNC_STRATEGIES,
    _build_dnc_qubo_solver,
    _build_partition_strategy,
)
from src.commands.poster_results.solvers.random_baseline import (
    RandomBaselineSolver,
    _sample_random_orientations,
    _flow_imbalance_score,
)
from src.commands.poster_results.solvers.trial import (
    _run_trial,
    _run_mr2s_trial,
    _run_poster_algorithm,
    ALL_ALGORITHMS,
    ALGORITHM_SOLVER_MAP,
)

__all__ = [
    "BaseSolver",
    "RawSASolver",
    "GlobalQuboSolver",
    "Mr2sVariantSolver",
    "DncStrategySolver",
    "RandomBaselineSolver",
    "MR2S_VARIANTS",
    "DNC_STRATEGIES",
    "ALL_ALGORITHMS",
    "ALGORITHM_SOLVER_MAP",
    "_run_trial",
    "_run_mr2s_trial",
    "_run_poster_algorithm",
    "_mean_finite",
    "_trial_seed",
    "_build_sa_solver",
    "_build_qubo_solver",
    "_build_edge_orienter",
    "_build_dnc_qubo_solver",
    "_build_partition_strategy",
    "_sample_random_orientations",
    "_flow_imbalance_score",
    "_as_finite_or_nan",
    "_extract_directed_edges_from_solution",
]
