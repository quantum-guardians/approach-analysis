"""Global QUBO solver for poster results."""

from __future__ import annotations

import time

import networkx as nx

from src.commands.poster_results.solvers.base import (
    BaseSolver,
    TrialTimings,
    _build_qubo_solver,
    _extract_directed_edges_from_solution,
    _nx_to_mr2s_graph,
)
from src.commands.poster_results.models import GlobalSolverResult
from src.commands.poster_results.partition_strategy import _estimate_physical_qubits


class GlobalQuboSolver(BaseSolver):
    name = "global"

    def run(
        self,
        graph: nx.Graph,
        n: int,
        seed: int | None,
    ) -> tuple[GlobalSolverResult, TrialTimings]:
        timings = TrialTimings()
        start = time.monotonic()
        solver = _build_qubo_solver()
        solver.face_cycle = None
        sol = solver.run(_nx_to_mr2s_graph(graph))
        timings.values["global_solve"] = time.monotonic() - start
        directed_edges = _extract_directed_edges_from_solution(sol)

        start = time.monotonic()
        bqm = solver.build_bqm(_nx_to_mr2s_graph(graph))
        phys = _estimate_physical_qubits(bqm)
        timings.values["global_embed"] = time.monotonic() - start

        return GlobalSolverResult(
            apsp=sol.score.apsp_sum / (n * (n - 1)),
            flow=sol.score.flow_score,
            qvars=len(bqm.variables),
            subgraph_size=len(bqm.variables),
            physical_total=phys,
            directed_edges=directed_edges,
        ), timings
