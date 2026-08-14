"""``qubo-structure`` subcommand – BQM structure and hop-length ablation.

Builds the MR2S QUBO for planar graphs of increasing size and records the
structural quantities the solver never serialises: the number of quadratic
couplings, the spread of the coefficients, and how both change when the
3-hop term is removed.

Minor embedding is deliberately not attempted here.  A failed
``minorminer`` search costs about 1,000 seconds per instance, which would
dominate the runtime of this command; embedding feasibility is measured by
``poster-results`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

import numpy as np

from src.graph_generator import generate_graph
from src.visualizer import plot_qubo_structure


DEFAULT_SIZES = (100, 200, 300, 400, 500)
DEFAULT_TRIALS = 3

# Arm labels.  ``hop23`` is the configuration the paper proposes; ``hop2``
# drops the 3-hop term so its cost can be attributed.
ARM_HOP23 = "hop23"
ARM_HOP2 = "hop2"


@dataclass(frozen=True)
class StructureRecord:
    """One (size, trial, arm) measurement."""

    size: int
    trial: int
    arm: str
    edges: int
    variables: int
    couplings: int
    couplings_per_variable: float
    coefficient_max: float
    coefficient_min: float
    coefficient_ratio: float
    apsp_sum: float
    flow_score: float
    strong_connect_rate: float
    build_seconds: float
    solve_seconds: float


def _module_version() -> str:
    """Return the installed ``mr2s-module`` version, or ``"unknown"``."""
    import importlib.metadata as metadata

    try:
        return metadata.version("mr2s-module")
    except metadata.PackageNotFoundError:
        return "unknown"


def _domain_graph(num_vertices: int, seed: int | None):
    """Convert a generated planar graph into an ``mr2s_module`` graph."""
    from mr2s_module.domain import Edge, Graph

    nx_graph = generate_graph(num_vertices, connectivity=None, seed=seed)
    edges = [
        Edge(int(min(u, v)), int(max(u, v)), 1, False)
        for u, v in nx_graph.edges()
    ]
    return Graph(edges=edges)


def _build_solver(include_three_hop: bool, num_reads: int):
    """Create a QUBO solver whose hop spec optionally excludes the 3-hop term."""
    from mr2s_module.evaluator import ApspSumRanker
    from mr2s_module.qubo import (
        FlowPolyGenerator,
        NHop,
        NHopPolyGenerator,
        QuboSolver,
        SmallWorldSpec,
    )
    from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver

    hops = [NHop(2, 1)]
    if include_three_hop:
        hops.append(NHop(3, 1))

    return QuboMR2SSolver(
        qubo_solver=QuboSolver.create_sa_solver(
            ranker=ApspSumRanker(), num_reads=num_reads
        ),
        poly_generators=[
            FlowPolyGenerator(),
            NHopPolyGenerator(small_world_spec=SmallWorldSpec(n_hops=hops)),
        ],
    )


def _coefficient_spread(bqm) -> tuple[float, float]:
    """Return (max, min) absolute non-zero coefficient of *bqm*."""
    magnitudes = [
        abs(value)
        for value in list(bqm.linear.values()) + list(bqm.quadratic.values())
        if abs(value) > 1e-12
    ]
    if not magnitudes:
        return 0.0, 0.0
    return max(magnitudes), min(magnitudes)


def _measure(graph, size: int, trial: int, arm: str, num_reads: int) -> StructureRecord:
    """Build and solve one QUBO, returning its structural measurements."""
    solver = _build_solver(include_three_hop=arm == ARM_HOP23, num_reads=num_reads)

    build_started = time.perf_counter()
    bqm = solver.build_bqm(graph)
    build_seconds = time.perf_counter() - build_started

    coefficient_max, coefficient_min = _coefficient_spread(bqm)

    solve_started = time.perf_counter()
    solution = solver.run(graph)
    solve_seconds = time.perf_counter() - solve_started

    variables = len(bqm.variables)
    couplings = len(bqm.quadratic)
    score = solution.score

    return StructureRecord(
        size=size,
        trial=trial,
        arm=arm,
        edges=len(graph.edges),
        variables=variables,
        couplings=couplings,
        couplings_per_variable=couplings / variables if variables else 0.0,
        coefficient_max=coefficient_max,
        coefficient_min=coefficient_min,
        coefficient_ratio=(
            coefficient_max / coefficient_min if coefficient_min else float("inf")
        ),
        apsp_sum=float(score.apsp_sum),
        flow_score=float(score.flow_score),
        strong_connect_rate=float(score.strong_connect_rate),
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
    )


def _mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def summarise(records: list[StructureRecord]) -> dict:
    """Aggregate per-trial records into per-(arm, size) means."""
    sizes = sorted({record.size for record in records})
    summary: dict = {"sizes": sizes, "arms": {}}
    for arm in sorted({record.arm for record in records}):
        fields = (
            "variables",
            "couplings",
            "couplings_per_variable",
            "coefficient_max",
            "coefficient_ratio",
            "apsp_sum",
            "flow_score",
            "strong_connect_rate",
            "solve_seconds",
        )
        summary["arms"][arm] = {
            field: [
                _mean(
                    [
                        float(getattr(record, field))
                        for record in records
                        if record.arm == arm and record.size == size
                    ]
                )
                for size in sizes
            ]
            for field in fields
        }
    return summary


def run(args: argparse.Namespace) -> None:
    """Execute the ``qubo-structure`` subcommand."""
    os.makedirs(args.output_dir, exist_ok=True)
    arms = [ARM_HOP23] if args.no_ablation else [ARM_HOP23, ARM_HOP2]

    records: list[StructureRecord] = []
    for size in args.sizes:
        for trial in range(args.trials):
            seed = None if args.seed is None else args.seed + 100 * trial + size
            graph = _domain_graph(size, seed)
            for arm in arms:
                record = _measure(graph, size, trial, arm, args.num_reads)
                records.append(record)
                print(
                    f"|V|={size} trial={trial} arm={arm} "
                    f"vars={record.variables} couplings={record.couplings} "
                    f"({record.couplings_per_variable:.1f}/var) "
                    f"cmax={record.coefficient_max:.1f} "
                    f"ratio={record.coefficient_ratio:.1f} "
                    f"apsp={record.apsp_sum:.2f} flow={record.flow_score:.1f} "
                    f"sc={record.strong_connect_rate:.2f}",
                    flush=True,
                )

    summary = summarise(records)
    payload = {
        "mr2s_module_version": _module_version(),
        "seed": args.seed,
        "num_reads": args.num_reads,
        "trials": args.trials,
        "summary": summary,
        "records": [asdict(r) for r in records],
    }
    json_path = os.path.join(args.output_dir, "qubo_structure.json")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=1)

    plot_qubo_structure(
        summary,
        save_path=os.path.join(args.output_dir, "qubo_structure.png"),
    )
    print(f"Saved {json_path}")


def register_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the ``qubo-structure`` subcommand to *subparsers*."""
    p = subparsers.add_parser(
        "qubo-structure",
        help="Measure BQM coupling count, coefficient spread, and hop-length ablation.",
        description=(
            "Build the MR2S QUBO for Delaunay planar graphs of increasing size "
            "and record the number of quadratic couplings, the coefficient "
            "spread, and the solution quality with and without the 3-hop term. "
            "Minor embedding is not attempted: a failed minorminer search costs "
            "about 1,000 seconds per instance and would dominate runtime."
        ),
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
        help="Vertex counts to measure (default: 100 200 300 400 500)."
    )
    p.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help="Number of graph instances per size (default: 3)."
    )
    p.add_argument(
        "--num-reads", type=int, default=20,
        help="Simulated annealing reads per QUBO (default: 20)."
    )
    p.add_argument(
        "--no-ablation", action="store_true",
        help="Measure only the proposed 2-hop + 3-hop configuration."
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed. Instance (size, trial) uses seed+100*trial+size."
    )
    p.add_argument(
        "--output-dir", type=str, default="results/qubo_structure",
        help="Directory for the JSON record and plot (default: results/qubo_structure)."
    )
    p.set_defaults(func=run)
