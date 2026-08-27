"""Raw SA solver for poster results."""

from __future__ import annotations

import time

import networkx as nx

from src.commands.poster_results.solvers.base import (
    BaseSolver,
    TrialTimings,
    _build_sa_solver,
    _extract_directed_edges_from_solution,
    _nx_to_mr2s_graph,
)
from src.commands.poster_results.models import BasicSolverResult


class RawSASolver(BaseSolver):
    name = "raw_sa"

    def run(
        self,
        graph: nx.Graph,
        n: int,
        seed: int | None,
    ) -> tuple[BasicSolverResult, TrialTimings]:
        timings = TrialTimings()
        start = time.monotonic()
        solver = _build_sa_solver(seed=seed)
        sol = solver.run(_nx_to_mr2s_graph(graph))
        timings.values["raw_sa"] = time.monotonic() - start
        directed_edges = _extract_directed_edges_from_solution(sol)

        return BasicSolverResult(
            apsp=sol.score.apsp_sum / (n * (n - 1)),
            flow=sol.score.flow_score,
            directed_edges=directed_edges,
        ), timings
