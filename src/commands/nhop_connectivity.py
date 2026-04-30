"""``nhop-connectivity`` subcommand – n-hop count vs SC ratio analysis.

Generates multiple random Delaunay planar graphs, directly generates
random orientations, and plots the SC ratio per distinct n-hop
neighbour count value bucket.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

import networkx as nx
import numpy as np

from src.graph_generator import generate_graph
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts
from src.visualizer import plot_nhop_connectivity_comparison


NHOP_CONN_HOPS = (2, 3)

# When the number of distinct n-hop count values (x-axis points) exceeds this
# threshold, the values are grouped into this many equal-width bins so the
# plot remains readable.
_SPECTRUM_BIN_THRESHOLD = 20


def _bin_nhop_buckets(
    bucket_total: dict[int, int],
    bucket_sc: dict[int, int],
    num_bins: int = _SPECTRUM_BIN_THRESHOLD,
) -> tuple[list[float], list[float]]:
    """Return (x_values, sc_ratios) binned into at most *num_bins* buckets.

    When the number of distinct n-hop count values exceeds *num_bins* the
    values are grouped into *num_bins* equal-width bins.  Each non-empty bin
    is represented by its midpoint on the x-axis, and the SC ratio is the
    weighted average (SC count / total count) across all values in the bin.

    If there are at most *num_bins* distinct values the data is returned
    unchanged (except that x-values are cast to ``float`` for type
    consistency).
    """
    sorted_counts = sorted(bucket_total.keys())
    if len(sorted_counts) <= num_bins:
        x = [float(c) for c in sorted_counts]
        y = [bucket_sc.get(c, 0) / bucket_total[c] for c in sorted_counts]
        return x, y

    lo, hi = sorted_counts[0], sorted_counts[-1]
    bin_width = (hi - lo) / num_bins

    bin_total: list[int] = [0] * num_bins
    bin_sc: list[int] = [0] * num_bins

    for c in sorted_counts:
        bin_idx = min(int((c - lo) / bin_width), num_bins - 1)
        bin_total[bin_idx] += bucket_total[c]
        bin_sc[bin_idx] += bucket_sc.get(c, 0)

    x: list[float] = []
    y: list[float] = []
    for b in range(num_bins):
        if bin_total[b] > 0:
            mid = lo + (b + 0.5) * bin_width
            x.append(mid)
            y.append(bin_sc[b] / bin_total[b])
    return x, y


def run(
    num_vertices: int,
    num_graphs: int,
    num_orientations: int,
    seed: int | None,
    output: str | None,
) -> None:
    """Generate Delaunay graphs, generate random orientations, and plot n-hop vs SC ratio.

    For each generated planar graph:

    1. Directly generates *num_orientations* random orientations (bit-masks
       over the edge set). Each sampled orientation is either strongly
       connected or not.
    2. For each sampled orientation computes the 2-hop and 3-hop neighbour
       counts and records whether the orientation is strongly connected.
    3. Groups sampled orientations by their n-hop count and computes the SC
       ratio per group:
       SC ratio = (SC orientations in group) / (all sampled orientations in group)

    When the n-hop spectrum is wide (> 20 distinct values), counts are
    grouped into 20 equal-width bins and the weighted SC ratio is shown per
    bin.

    The result is plotted as a scatter plot with n-hop count on the x-axis
    and SC ratio on the y-axis (one subplot per hop distance).

    Args:
        num_vertices: Number of vertices in each generated Delaunay graph.
        num_graphs: Number of random planar graphs to generate.
        num_orientations: Number of random orientations to generate per graph.
        seed: Base random seed.  Graph *i* uses ``seed + i`` when set.
        output: File path to save the plot.  Auto-generated if ``None``.
    """
    hops = NHOP_CONN_HOPS

    # nhop_bucket_total[hop][count] = sampled orientations with that n-hop count
    # nhop_bucket_sc[hop][count]    = SC orientations with that n-hop count
    nhop_bucket_total: dict[int, dict[int, int]] = {hop: {} for hop in hops}
    nhop_bucket_sc: dict[int, dict[int, int]] = {hop: {} for hop in hops}

    print(
        f"Analysing {num_graphs} Delaunay planar graphs "
        f"({num_vertices} vertices, {num_orientations} sampled orientations each) …"
    )

    for i in range(num_graphs):
        graph_seed = (seed + i) if seed is not None else None
        graph = generate_graph(num_vertices, connectivity=None, seed=graph_seed)

        edges = list(graph.edges())
        nodes = list(graph.nodes())
        edge_count = len(edges)
        rng = np.random.default_rng(graph_seed)
        sample_size = num_orientations

        sc_count = 0
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)

        for bits in rng.integers(0, 2, size=(sample_size, edge_count), dtype=np.int8):
            dg.clear_edges()
            for bit, (u, v) in zip(bits, edges):
                if bit == 0:
                    dg.add_edge(u, v)
                else:
                    dg.add_edge(v, u)

            is_sc = nx.is_strongly_connected(dg)
            _, counts = calculate_apsp_sum_and_nhop_neighbor_counts(dg, hops=hops)

            for hop in hops:
                nhop_count = counts[hop]
                nhop_bucket_total[hop][nhop_count] = nhop_bucket_total[hop].get(nhop_count, 0) + 1
                if is_sc:
                    nhop_bucket_sc[hop][nhop_count] = nhop_bucket_sc[hop].get(nhop_count, 0) + 1

            if is_sc:
                sc_count += 1

        print(
            f"  Graph {i + 1}/{num_graphs}: edges={edge_count}, "
            f"sampled={sample_size}, SC={sc_count}/{sample_size}"
        )

    if not any(nhop_bucket_total[hop] for hop in hops):
        print("No valid graphs to plot.")
        return

    # Convert buckets to (x, y) pairs, binning when the spectrum is wide.
    nhop_x: dict[int, list[float]] = {}
    sc_ratio_y: dict[int, list[float]] = {}
    for hop in hops:
        nhop_x[hop], sc_ratio_y[hop] = _bin_nhop_buckets(
            nhop_bucket_total[hop], nhop_bucket_sc[hop]
        )

    title = (
        f"N-hop Count vs SC Ratio  "
        f"(n={num_vertices} vertices, {num_graphs} Delaunay graphs, "
        f"{num_orientations} orientations sampled)"
    )
    save_path = output or f"nhop_connectivity_v{num_vertices}.png"
    plot_nhop_connectivity_comparison(nhop_x, sc_ratio_y, title=title, save_path=save_path)
    print(f"Plot saved to: {os.path.abspath(save_path)}")


def register_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the ``nhop-connectivity`` subcommand to *subparsers*."""
    p = subparsers.add_parser(
        "nhop-connectivity",
        help="Compare 2-hop / 3-hop counts and SC ratio across multiple planar graphs.",
        description=(
            "Generate multiple random Delaunay planar graphs, directly "
            "generate random orientations for each graph, and plot the SC ratio "
            "(SC orientations / total sampled orientations) per distinct n-hop "
            "neighbour count value — for both 2-hop and 3-hop distances. "
            "When the n-hop spectrum is wide (> 20 distinct values), counts "
            "are grouped into 20 equal-width bins automatically."
        ),
    )
    p.add_argument(
        "--vertices", type=int, default=5,
        help="Number of vertices in each generated Delaunay graph (default: 5)"
    )
    p.add_argument(
        "--num-graphs", type=int, default=20,
        help="Number of Delaunay planar graphs to generate (default: 20)"
    )
    p.add_argument(
        "--num-orientations", type=int, default=200,
        help="Number of random orientations to generate per graph "
             "(default: 200)."
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed. Graph i uses seed+i when set."
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="File path to save the plot (e.g. nhop.png). "
             "If omitted, defaults to nhop_connectivity_v{vertices}.png."
    )
    p.set_defaults(func=_dispatch)


def _dispatch(args: argparse.Namespace) -> None:
    run(
        args.vertices,
        args.num_graphs,
        args.num_orientations,
        args.seed,
        args.output,
    )
