"""DnC partition strategies for poster-result MR2S runs."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from mr2s_module import QuboMR2SSolver, estimate_required_qubits
from mr2s_module.domain.embeddable_graph_partition import EmbeddableGraphPartition
from mr2s_module.domain.score import EmbeddingEstimate
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver


def _embedding_estimate_with_status(bqm: Any) -> tuple[EmbeddingEstimate | None, bool, str | None]:
    try:
        return estimate_required_qubits(bqm), True, None
    except RuntimeError as exc:
        # embedding 실패는 fallback/partition 판단에 필요한 진단값으로 남긴다.
        return None, False, str(exc)


def _estimate_physical_qubits_with_status(bqm: Any) -> tuple[float, bool, str | None]:
    estimate, can_embed, error = _embedding_estimate_with_status(bqm)
    physical_qubits = (
        float(estimate.num_physical_qubits)
        if estimate is not None
        else float("nan")
    )
    return physical_qubits, can_embed, error


def _estimate_physical_qubits(bqm: Any) -> float:
    """Return physical qubits for embeddable BQMs; NaN when embedding fails."""
    physical_qubits, _, _ = _estimate_physical_qubits_with_status(bqm)
    return physical_qubits


def _probe_embedding(mr2s_solver: QuboMR2SSolver, graph: Any) -> dict[str, Any]:
    probe, _ = _probe_embedding_with_estimate(mr2s_solver, graph)
    return probe


def _probe_embedding_with_estimate(
    mr2s_solver: QuboMR2SSolver,
    graph: Any,
) -> tuple[dict[str, Any], EmbeddingEstimate | None]:
    bqm = mr2s_solver.build_bqm(graph)
    estimate, can_embed, error = _embedding_estimate_with_status(bqm)
    physical_qubits = (
        float(estimate.num_physical_qubits)
        if estimate is not None
        else float("nan")
    )
    return {
        "can_embed": can_embed,
        "qvars": len(bqm.variables),
        "physical_qubits": physical_qubits,
        "error": error,
    }, estimate


def _can_recurse_partition(parent: Any, sub_graphs: list[Any]) -> bool:
    if not sub_graphs:
        return False

    # 자기 자신과 같은 크기의 subgraph가 나오면 재귀 분할이 진행되지 않는다.
    parent_edge_count = len(parent.edges)
    return all(
        0 < len(sub_graph.edges) < parent_edge_count
        for sub_graph in sub_graphs
    )


def _partition_with_target_k(face_cycle: Any, graph: Any, target_k: int) -> Any:
    # face_cycle 객체의 target_k를 임시로 바꾸고 반드시 원래 값으로 복구한다.
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
    # 실패한 embedding은 NaN으로 보존하고, 합산 통계는 유한값만 사용한다.
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


class DncPartitionStrategy:
    """Find an embeddable DnC partition by varying face-cycle ``target_k``."""

    def __init__(self) -> None:
        self.last_diagnostics: dict[str, Any] = {}
        self.last_elapsed_seconds: float = 0.0

    def run(self, graph: Any) -> EmbeddableGraphPartition:
        if not hasattr(self, "mr2s_solver") or not hasattr(self, "face_cycle"):
            raise RuntimeError("DncPartitionStrategy requires mr2s_solver and face_cycle")

        start = time.monotonic()
        partition, diagnostics = self.find_partition(
            self.mr2s_solver,
            self.face_cycle,
            graph,
        )
        self.last_elapsed_seconds = time.monotonic() - start
        self.last_diagnostics = diagnostics
        return partition

    def divide(self, solver: DnCMr2sSolver, graph: Any) -> tuple[list[Any], dict[str, Any]]:
        self.mr2s_solver = solver.mr2s_solver
        self.face_cycle = solver.face_cycle
        partition = self.run(graph)
        return partition.sub_graphs, self.last_diagnostics

    def find_partition(
        self,
        mr2s_solver: QuboMR2SSolver,
        face_cycle: Any,
        graph: Any,
    ) -> tuple[EmbeddableGraphPartition, dict[str, Any]]:
        # 먼저 전체 그래프 QUBO가 하드웨어 embedding 가능한지 확인한다.
        whole_graph_probe, whole_graph_estimate = self.probe_embedding_with_estimate(
            mr2s_solver,
            graph,
        )
        diagnostics = {
            "whole_graph": whole_graph_probe,
            "attempts": [],
            "selected_reason": "whole_graph_embeddable",
            "selected_probes": [whole_graph_probe],
        }
        if whole_graph_estimate is not None:
            return EmbeddableGraphPartition(
                sub_graphs=[graph],
                embedding_estimates=[whole_graph_estimate],
                target_k=None,
            ), diagnostics

        partition, partition_diagnostics = self.find_embeddable_partition_by_target_k(
            mr2s_solver,
            face_cycle,
            graph,
        )
        diagnostics.update(partition_diagnostics)
        if partition is None:
            # partition을 찾지 못하면 solver가 기존 경로로 처리하도록 전체 그래프를 반환한다.
            diagnostics["selected_reason"] = "fallback_whole_graph"
            diagnostics["selected_probes"] = [whole_graph_probe]
            return EmbeddableGraphPartition(
                sub_graphs=[graph],
                embedding_estimates=[],
                target_k=None,
            ), diagnostics
        return partition, diagnostics

    def find_embeddable_partition_by_target_k(
        self,
        mr2s_solver: QuboMR2SSolver,
        face_cycle: Any,
        graph: Any,
    ) -> tuple[EmbeddableGraphPartition | None, dict[str, Any]]:
        left = 2
        right = max(2, len(graph.edges))
        best_partition: EmbeddableGraphPartition | None = None
        best_probes: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []

        while left <= right:
            # 가능한 partition 중 embedding이 되는 가장 작은 target_k를 찾는다.
            target_k = (left + right) // 2
            result = self.partition_with_target_k(face_cycle, graph, target_k)
            sub_graphs = result.sub_graphs
            can_recurse = self.can_recurse_partition(graph, sub_graphs)
            probes: list[dict[str, Any]] = []
            estimates: list[EmbeddingEstimate] = []
            if can_recurse:
                for sub_graph in sub_graphs:
                    probe, estimate = self.probe_embedding_with_estimate(
                        mr2s_solver,
                        sub_graph,
                    )
                    probes.append(probe)
                    if estimate is not None:
                        estimates.append(estimate)
            accepted = can_recurse and all(probe["can_embed"] for probe in probes)
            attempts.append(
                self.summarize_partition_attempt(
                    target_k=target_k,
                    result=result,
                    can_recurse=can_recurse,
                    probes=probes,
                    accepted=accepted,
                )
            )

            if accepted:
                best_partition = EmbeddableGraphPartition(
                    sub_graphs=sub_graphs,
                    embedding_estimates=estimates,
                    target_k=target_k,
                )
                best_probes = probes
                right = target_k - 1
            else:
                left = target_k + 1

        diagnostics = {
            "attempts": attempts,
            "selected_reason": "partition_found" if best_partition else "partition_not_found",
            "selected_probes": best_probes,
        }
        return best_partition, diagnostics

    def probe_embedding(self, mr2s_solver: QuboMR2SSolver, graph: Any) -> dict[str, Any]:
        return _probe_embedding(mr2s_solver, graph)

    def probe_embedding_with_estimate(
        self,
        mr2s_solver: QuboMR2SSolver,
        graph: Any,
    ) -> tuple[dict[str, Any], EmbeddingEstimate | None]:
        return _probe_embedding_with_estimate(mr2s_solver, graph)

    def find_partition_by_target_k(
        self,
        mr2s_solver: QuboMR2SSolver,
        face_cycle: Any,
        graph: Any,
    ) -> tuple[list[Any], dict[str, Any]]:
        partition, diagnostics = self.find_embeddable_partition_by_target_k(
            mr2s_solver,
            face_cycle,
            graph,
        )
        return (partition.sub_graphs if partition is not None else []), diagnostics

    def partition_with_target_k(self, face_cycle: Any, graph: Any, target_k: int) -> Any:
        return _partition_with_target_k(face_cycle, graph, target_k)

    def can_recurse_partition(self, parent: Any, sub_graphs: list[Any]) -> bool:
        return _can_recurse_partition(parent, sub_graphs)

    def summarize_partition_attempt(
        self,
        target_k: int,
        result: Any,
        can_recurse: bool,
        probes: list[dict[str, Any]],
        accepted: bool,
    ) -> dict[str, Any]:
        return _summarize_partition_attempt(
            target_k=target_k,
            result=result,
            can_recurse=can_recurse,
            probes=probes,
            accepted=accepted,
        )


class TimedPartitionStrategy:
    """Record elapsed time for any mr2s-module DnC partition strategy."""

    def __init__(self, strategy: Any) -> None:
        self.strategy = strategy
        self.last_elapsed_seconds = 0.0

    def run(self, graph: Any) -> EmbeddableGraphPartition:
        start = time.monotonic()
        try:
            return self.strategy.run(graph)
        finally:
            # module-provided strategies do not expose timing, so the wrapper keeps it.
            self.last_elapsed_seconds = time.monotonic() - start


def _find_partition_by_target_k_with_diagnostics(
    mr2s_solver: QuboMR2SSolver,
    face_cycle: Any,
    graph: Any,
) -> tuple[list[Any], dict[str, Any]]:
    return DncPartitionStrategy().find_partition_by_target_k(
        mr2s_solver,
        face_cycle,
        graph,
    )


def _divide_graph_with_diagnostics(solver: DnCMr2sSolver, graph: Any) -> tuple[list[Any], dict[str, Any]]:
    return DncPartitionStrategy().divide(solver, graph)
