"""Random baseline solver for poster results."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from src.commands.poster_results.solvers.base import (
    BaseSolver,
    TrialTimings,
    _mean_finite,
)
from src.commands.poster_results.models import RandomBaselineResult
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts


def _sample_random_orientations(
    graph: nx.Graph,
    max_samples: int,
    seed: int | None = None,
) -> list[nx.DiGraph]:
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    edges = list(graph.edges())
    nodes = list(graph.nodes())
    rng = np.random.default_rng(seed)
    orientations: list[nx.DiGraph] = []

    for _ in range(max_samples):
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


class RandomBaselineSolver(BaseSolver):
    name = "random"
    max_samples: int = 10

    def run(
        self,
        graph: nx.Graph,
        n: int,
        seed: int | None,
    ) -> tuple[RandomBaselineResult, TrialTimings]:
        timings = TrialTimings()
        start = time.monotonic()
        result = self._calculate(graph, n, seed)
        timings.values["random"] = time.monotonic() - start
        return result, timings

    def _calculate(
        self,
        graph: nx.Graph,
        n: int,
        seed: int | None,
    ) -> RandomBaselineResult:
        random_samples = _sample_random_orientations(
            graph, max_samples=self.max_samples, seed=seed
        )
        trial_apsp: list[float] = []
        trial_flow: list[float] = []
        best_edges: list[tuple[int, int]] | None = None
        best_apsp: float = float("inf")

        for orient in random_samples:
            if nx.is_strongly_connected(orient):
                apsp, _ = calculate_apsp_sum_and_nhop_neighbor_counts(orient, hops=[])
                normalized_apsp = apsp / (n * (n - 1))
                trial_apsp.append(normalized_apsp)
                if normalized_apsp < best_apsp:
                    best_apsp = normalized_apsp
                    best_edges = list(orient.edges())
            trial_flow.append(_flow_imbalance_score(orient))

        return RandomBaselineResult(
            apsp=_mean_finite(trial_apsp),
            flow=_mean_finite(trial_flow),
            sample_count=len(random_samples),
            strong_sample_count=len(trial_apsp),
            directed_edges=best_edges,
        )
