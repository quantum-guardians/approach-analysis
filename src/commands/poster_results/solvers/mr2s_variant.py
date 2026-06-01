"""MR2S variant solver (Robbin, IteratedLocalSearch edge orienters)."""

from __future__ import annotations

import time
from typing import Any

import networkx as nx
from mr2s_module import Robbin
from mr2s_module.edge_orient.iterated_local_search import IteratedLocalSearch

from src.commands.poster_results.solvers.base import (
    BaseSolver,
    TrialTimings,
    _build_qubo_solver,
    _extract_directed_edges_from_solution,
    _nx_to_mr2s_graph,
)
from src.commands.poster_results.models import GlobalSolverResult
from src.commands.poster_results.partition_strategy import _estimate_physical_qubits

MR2S_VARIANTS = ["robbin_mr2s", "iterated_local_search_mr2s"]


def _build_edge_orienter(variant_name: str) -> Any:
    if variant_name == "robbin_mr2s":
        return Robbin()
    if variant_name == "iterated_local_search_mr2s":
        return IteratedLocalSearch()
    raise ValueError(f"Unknown MR2S variant: {variant_name}")


class Mr2sVariantSolver(BaseSolver):
    def __init__(self, variant_name: str) -> None:
        if variant_name not in MR2S_VARIANTS:
            raise ValueError(f"Unknown MR2S variant: {variant_name}")
        self.variant_name = variant_name
        self.name = variant_name

    def run(
        self,
        graph: nx.Graph,
        n: int,
        seed: int | None,
    ) -> tuple[GlobalSolverResult, TrialTimings]:
        timings = TrialTimings()
        start = time.monotonic()
        solver = _build_qubo_solver(edge_orienter=_build_edge_orienter(self.variant_name))
        sol = solver.run(_nx_to_mr2s_graph(graph))
        timings.values[f"{self.variant_name}_solve"] = time.monotonic() - start
        directed_edges = _extract_directed_edges_from_solution(sol)

        start = time.monotonic()
        embed_solver = _build_qubo_solver(edge_orienter=_build_edge_orienter(self.variant_name))
        bqm = embed_solver.build_bqm(_nx_to_mr2s_graph(graph))
        physical_qubits = _estimate_physical_qubits(bqm)
        timings.values[f"{self.variant_name}_embed"] = time.monotonic() - start

        return GlobalSolverResult(
            apsp=sol.score.apsp_sum / (n * (n - 1)),
            flow=sol.score.flow_score,
            qvars=len(bqm.variables),
            subgraph_size=len(bqm.variables),
            physical_total=physical_qubits,
            directed_edges=directed_edges,
        ), timings
