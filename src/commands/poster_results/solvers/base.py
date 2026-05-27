"""Base solver class for poster results."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
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
    SAMR2SSolver,
    SAQuboSolver,
    SmallWorldSpec,
)

from src.commands.face_k_analysis import _generate_delaunay_graph, _nx_to_mr2s_graph
from src.commands.poster_results.models import TrialTimings

# mr2s_moduleのGraph実装にis_emptyがないバージョンがあるためランタイムで補強する。
if not hasattr(mr2s_module.domain.graph.Graph, "is_empty"):

    def is_empty(self):
        return len(self.edges) == 0

    mr2s_module.domain.graph.Graph.is_empty = is_empty


spec = SmallWorldSpec([NHop(2, 1), NHop(3, 1)])


def _as_finite_or_nan(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _mean_finite(values: list[float]) -> float:
    """embedding 실패 등으로 생긴 NaN/Infinity는 평균에서 제외한다."""
    finite_values = [_as_finite_or_nan(value) for value in values]
    finite_values = [value for value in finite_values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _trial_seed(n: int, trial: int, seed: int | None) -> int | None:
    return (seed + trial * 100 + n) if seed is not None else None


def _generate_trial_graph(n: int, trial: int, seed: int | None) -> tuple[nx.Graph, int | None, float]:
    trial_seed = _trial_seed(n, trial, seed)
    start = time.monotonic()
    graph = _generate_delaunay_graph(n, trial_seed)
    return graph, trial_seed, time.monotonic() - start


def _build_sa_solver(seed: int | None = None) -> SAMR2SSolver:
    return SAMR2SSolver(
        evaluator=Evaluator(),
        random_seed=seed,
    )


def _build_qubo_solver(edge_orienter: Any | None = None) -> QuboMR2SSolver:
    n_hop_poly = NHopPolyGenerator()
    n_hop_poly.small_world_spec = spec
    return QuboMR2SSolver(
        edge_orienter=edge_orienter,
        qubo_solver=SAQuboSolver(ApspSumRanker()),
        evaluator=Evaluator(),
        poly_generators=[n_hop_poly, FlowPolyGenerator()],
    )


def _extract_directed_edges_from_solution(sol: Any) -> list[tuple[int, int]] | None:
    for attr in ("directed_graph", "dir_graph", "graph", "_graph"):
        candidate = getattr(sol, attr, None)
        if candidate is None or not hasattr(candidate, "edges"):
            continue
        edges = candidate.edges
        if not edges:
            continue
        if isinstance(edges, dict):
            return [(e.u, e.v) for e in edges.values()]
        try:
            first = next(iter(edges))
        except (StopIteration, TypeError):
            continue
        if hasattr(first, "u"):
            return [(e.u, e.v) for e in edges]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return [(e[0], e[1]) for e in edges]
    return None


class BaseSolver(ABC):
    """Abstract base class for all poster-result solvers."""

    name: str = "base"

    @classmethod
    def algorithm_name(cls) -> str:
        return cls.name

    @abstractmethod
    def run(
        self,
        graph: nx.Graph,
        n: int,
        seed: int | None,
    ) -> tuple[Any, TrialTimings]:
        """Run solver and return (result, timings)."""
        ...
