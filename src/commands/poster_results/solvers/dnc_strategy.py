"""DnC strategy solver for poster results."""

from __future__ import annotations

import time
from typing import Any

import networkx as nx
import numpy as np
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver

from src.commands.poster_results.solvers.base import (
    BaseSolver,
    TrialTimings,
    _build_qubo_solver,
    _extract_directed_edges_from_solution,
    _nx_to_mr2s_graph,
)
from src.commands.poster_results.models import Mr2sSolverResult
from src.commands.poster_results.partition_strategy import (
    DncPartitionStrategy,
    TimedPartitionStrategy,
    _physical_qubit_stats,
)
from mr2s_module.solver.dnc_graph_partition_strategy import (
    DegeneracyPruningFaceCyclePartitionStrategy,
    EmbeddingAwareFaceCyclePartitionStrategy,
)

DNC_STRATEGIES = ["poster", "embedding_aware", "degeneracy_pruning"]


def _build_dnc_qubo_solver(
    partition_strategy: Any | None = None,
) -> DnCMr2sSolver:
    solver = DnCMr2sSolver(
        mr2s_solver=_build_qubo_solver(),
    )
    if partition_strategy is not None:
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


class DncStrategySolver(BaseSolver):
    def __init__(self, strategy_name: str) -> None:
        if strategy_name not in DNC_STRATEGIES:
            raise ValueError(f"Unknown DnC strategy: {strategy_name}")
        self.strategy_name = strategy_name
        self.name = strategy_name

    def run(
        self,
        graph: Any,
        n: int,
        seed: int | None,
    ) -> tuple[Mr2sSolverResult, TrialTimings]:
        del seed
        timings = TrialTimings()
        start = time.monotonic()
        solver_cls = _build_dnc_qubo_solver()
        partition_strategy = _build_partition_strategy(self.strategy_name, solver_cls)
        solver_cls.graph_partition_strategy = partition_strategy
        graph_cls = _nx_to_mr2s_graph(graph)

        try:
            sol_cls = solver_cls.run(graph_cls)
        except RuntimeError as exc:
            elapsed = time.monotonic() - start
            timings.values[f"dnc_{self.strategy_name}_solve"] = 0.0
            timings.values[f"dnc_{self.strategy_name}_embed"] = elapsed
            if self.strategy_name == "poster":
                timings.values["clustered_solve"] = 0.0
                timings.values["clustered_embed"] = elapsed
            return self._failed_result(exc), timings

        directed_edges = _extract_directed_edges_from_solution(sol_cls)
        partition_diagnostics = getattr(partition_strategy, "last_diagnostics", None)
        if not partition_diagnostics:
            partition_diagnostics = _diagnostics_from_solution(self.strategy_name, sol_cls)

        clustered_total = time.monotonic() - start
        clustered_embed = getattr(partition_strategy, "last_elapsed_seconds", 0.0)
        solve_time = max(0.0, clustered_total - clustered_embed)
        timings.values[f"dnc_{self.strategy_name}_solve"] = solve_time
        timings.values[f"dnc_{self.strategy_name}_embed"] = clustered_embed
        if self.strategy_name == "poster":
            timings.values["clustered_solve"] = solve_time
            timings.values["clustered_embed"] = clustered_embed

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
            directed_edges=directed_edges,
        ), timings

    def _failed_result(self, error: Exception) -> Mr2sSolverResult:
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
                "strategy": self.strategy_name,
                "selected_reason": "failed",
                "error": str(error),
            },
        )
