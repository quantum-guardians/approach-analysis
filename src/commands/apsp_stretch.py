"""``apsp-stretch`` subcommand – redraw the APSP figure as a size-invariant ratio.

``poster-results`` reports ``D_avg = APSP_sum / (n*(n-1))``: the mean directed
shortest-path length over ordered pairs.  The axis of ``apsp_reduction.png`` is
labelled "Normalized APSP", but that normalisation is only by the pair count, so
the value still grows with graph size and cannot be compared across sizes.

The paper's size-invariant quantity is the stretch

    F_dist = 1/(n*(n-1)) * sum_{s!=t} D_st / Dbar_st

where ``Dbar_st`` is the shortest-path distance in the *undirected* graph.
Computing it exactly needs the orientation each solver produced, and the poster
caches do not contain one: the cached ``directed_edges`` field holds the
undirected base edge list (every pair is stored as ``(min, max)``), so the
orientation is not recoverable from the stored artifacts.

This command therefore plots the ratio of means

    stretch = D_avg / Dbar_avg

which is size-invariant and uses only what the artifacts do contain: the
published ``D_avg`` per solver plus ``Dbar_avg``, recomputed exactly by
regenerating the Delaunay instances from their seeds.  It is a lower bound
proxy in practice, not ``F_dist`` itself; measured on Robbins-style
orientations of these instances it sits about 4-10% below the exact value,
because pairs that are close in the undirected graph carry the largest stretch
and the ratio of means underweights them.  Recovering the exact ``F_dist``
requires re-running the solvers with the orientation stored.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

from src.commands.face_k_analysis import _generate_delaunay_graph
from src.visualizer import plot_apsp_stretch


DEFAULT_INPUT = "results/poster_results_v2/plotted_data_summary.json"
DEFAULT_OUTPUT_DIR = "results/apsp_stretch"
DEFAULT_TRIALS = 3
DEFAULT_SEED = 42
PLOT_KEY = "apsp_reduction"
METRIC_NOTE = (
    "stretch = D_avg / Dbar_avg (ratio of means). The exact F_dist is the mean "
    "of per-pair ratios and is not recoverable from the poster caches, which "
    "store the undirected edge list rather than each solver's orientation."
)


def _trial_seed(n: int, trial: int, seed: int | None) -> int | None:
    """Instance seed for ``(size, trial)``.

    Mirrors ``src.commands.poster_results.solvers.base._trial_seed``; kept local
    so this command does not import the solver stack (and ``mr2s_module``).
    """
    return (seed + trial * 100 + n) if seed is not None else None


def _mean_undirected_distance(graph: nx.Graph) -> float:
    """Return the mean shortest-path length over ordered pairs of *graph*.

    Edges are unweighted, matching the ``w_ij = 1`` setting of the poster runs.
    """
    n = graph.number_of_nodes()
    if n < 2:
        return float("nan")
    matrix = nx.to_scipy_sparse_array(graph, nodelist=sorted(graph.nodes()), format="csr")
    distances = shortest_path(matrix, method="D", directed=False, unweighted=True)
    off_diagonal = ~np.eye(n, dtype=bool)
    return float(distances[off_diagonal].mean())


def undirected_baselines(
    sizes: list[int],
    trials: int,
    seed: int | None,
) -> dict[int, dict[str, Any]]:
    """Recompute ``Dbar_avg`` for every size by regenerating the instances."""
    baselines: dict[int, dict[str, Any]] = {}
    for n in sizes:
        per_trial = [
            _mean_undirected_distance(_generate_delaunay_graph(n, _trial_seed(n, trial, seed)))
            for trial in range(trials)
        ]
        baselines[n] = {
            "mean": float(np.mean(per_trial)),
            "per_trial": [float(value) for value in per_trial],
            "trial_seeds": [_trial_seed(n, trial, seed) for trial in range(trials)],
        }
    return baselines


def load_apsp_series(path: str) -> tuple[list[int], list[dict[str, Any]]]:
    """Read the plotted ``D_avg`` series from a ``plotted_data_summary.json``."""
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    plot = summary.get("plots", {}).get(PLOT_KEY)
    if plot is None:
        raise ValueError(f"{path} has no plots.{PLOT_KEY} section")
    return list(summary["sizes"]), list(plot["series"])


def stretch_series(
    series: list[dict[str, Any]],
    baselines: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Divide every plotted ``D_avg`` value by the matching ``Dbar_avg``."""
    converted = []
    for entry in series:
        values = []
        for n, value in zip(entry["x"], entry["y"], strict=True):
            baseline = baselines.get(n, {}).get("mean")
            if value is None or baseline is None or not np.isfinite(baseline):
                values.append(None)
            else:
                values.append(float(value) / baseline)
        converted.append({
            "key": entry["key"],
            "display_name": entry["display_name"],
            "color": entry["color"],
            "x": list(entry["x"]),
            "y": values,
            "d_avg": list(entry["y"]),
        })
    return converted


def run(args: argparse.Namespace) -> None:
    """Entry point for the ``apsp-stretch`` subcommand."""
    sizes, series = load_apsp_series(args.input)
    if args.sizes:
        sizes = [n for n in sizes if n in set(args.sizes)]
        series = [
            {
                **entry,
                "x": [n for n in entry["x"] if n in set(sizes)],
                "y": [
                    value
                    for n, value in zip(entry["x"], entry["y"], strict=True)
                    if n in set(sizes)
                ],
            }
            for entry in series
        ]

    baselines = undirected_baselines(sizes, args.trials, args.seed)
    converted = stretch_series(series, baselines)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "apsp_stretch.png")
    plot_apsp_stretch(sizes, converted, save_path=plot_path)

    record = {
        "input": args.input,
        "metric": "stretch_ratio_of_means",
        "metric_note": METRIC_NOTE,
        "seed": args.seed,
        "trials": args.trials,
        "sizes": sizes,
        "undirected_baseline": {str(n): baselines[n] for n in sizes},
        "series": converted,
        "output": os.path.basename(plot_path),
    }
    record_path = os.path.join(args.output_dir, "apsp_stretch.json")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    for n in sizes:
        print(f"|V|={n}: Dbar_avg={baselines[n]['mean']:.4f}")
    for entry in converted:
        formatted = ", ".join(
            "n/a" if value is None else f"{value:.3f}" for value in entry["y"]
        )
        print(f"{entry['display_name']:>6}: {formatted}")
    print(f"Saved {plot_path} and {record_path}")


def register_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the ``apsp-stretch`` subcommand to *subparsers*."""
    p = subparsers.add_parser(
        "apsp-stretch",
        help="Redraw the APSP figure as the size-invariant stretch D_avg / Dbar_avg.",
        description=(
            "Convert the D_avg values plotted by poster-results into a stretch "
            "ratio against the undirected baseline, which is comparable across "
            "graph sizes. The undirected distances are recomputed by "
            "regenerating each Delaunay instance from its seed. This is the "
            "ratio of means, a proxy for the paper's F_dist (mean of per-pair "
            "ratios); the exact value needs the solver orientations, which the "
            "poster caches do not store."
        ),
    )
    p.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help=f"plotted_data_summary.json to convert (default: {DEFAULT_INPUT})."
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=None,
        help="Subset of vertex counts to plot (default: every size in the input)."
    )
    p.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help="Instances per size used for the undirected baseline (default: 3)."
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Base seed of the run being converted. Instance (size, trial) uses "
             "seed+100*trial+size (default: 42, the seed of poster_results_v2)."
    )
    p.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the plot and JSON record (default: {DEFAULT_OUTPUT_DIR})."
    )
    p.set_defaults(func=run)
