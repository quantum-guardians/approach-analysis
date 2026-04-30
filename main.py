#!/usr/bin/env python3
"""Entry point for the n-hop approach analysis.

Usage examples
--------------
Analyse a single random graph (correlation plots)::

    python main.py analyse

Run with custom parameters::

    python main.py analyse --vertices 5 --connectivity 0.7 --seed 42 --output out.png

Compare n-hop counts and SC ratio across multiple graphs::

    python main.py nhop-connectivity --vertices 5 --num-graphs 20 --num-orientations 300 --seed 0
"""

import argparse
import os
import random
import signal
import time
import types

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

import networkx as nx

from src.graph_generator import generate_graph
from src.case_generator import (
    generate_strongly_connected_orientations,
    sample_strongly_connected_orientations,
)
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts
from src.visualizer import plot_score_correlations, plot_nhop_connectivity_comparison


HOPS = (2, 3, 4)
NHOP_CONN_HOPS = (2, 3)

# Guard against exhaustive enumeration of graphs with too many orientations.
# 2^20 = 1 048 576; graphs beyond this threshold are skipped in nhop-connectivity.
MAX_EXHAUSTIVE_ORIENTATIONS = 1 << 20


def analyse(
    num_vertices: int,
    connectivity: float | None,
    seed: int | None,
    output: str | None,
    workers: int | None,
    chunk_size: int,
    max_samples: int | None = None,
    min_samples: int = 0,
    use_processes: bool = False,
    adaptive_chunk_size: bool = False,
) -> None:
    graph = generate_graph(num_vertices, connectivity, seed=seed)
    connectivity_label = f"{connectivity}" if connectivity is not None else "Delaunay"
    print(
        f"Graph: {num_vertices} vertices, {graph.number_of_edges()} edges "
        f"(connectivity={connectivity_label})"
    )

    apsp_sums: list[float] = []
    nhop_counts: dict[int, list[int]] = {n: [] for n in HOPS}

    n_orientations = 0
    if max_samples is not None:
        orientations_iter = sample_strongly_connected_orientations(
            graph,
            max_samples=max_samples,
            min_samples=min_samples,
            seed=seed,
            num_workers=workers,
            chunk_size=chunk_size,
            use_processes=use_processes,
        )
        msg = f"Sampling up to {max_samples} strongly-connected orientations"
        if min_samples > 0:
            msg += f" (minimum required: {min_samples})"
        print(msg + " …")
    else:
        orientations_iter = generate_strongly_connected_orientations(
            graph, num_workers=workers, chunk_size=chunk_size,
            use_processes=use_processes, adaptive_chunk_size=adaptive_chunk_size,
        )

    # --- SIGINT / Ctrl-C handling ---
    # Register a signal handler so that pressing Ctrl-C sets a flag instead
    # of raising KeyboardInterrupt mid-computation.  The loop checks the flag
    # after each orientation is processed, then falls through to produce a
    # partial chart from whatever data has been collected so far.
    interrupted = False

    def _sigint_handler(sig: int, frame: types.FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        print(
            "\nInterrupted! Generating chart from intermediate results …",
            flush=True,
        )

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)
    start_time = time.monotonic()
    next_report_at = 60.0  # first progress report after 1 minute
    try:
        for orientation in orientations_iter:
            if interrupted:
                break
            n_orientations += 1
            apsp_sum, counts = calculate_apsp_sum_and_nhop_neighbor_counts(
                orientation, hops=HOPS
            )
            apsp_sums.append(apsp_sum)
            for hop in HOPS:
                nhop_counts[hop].append(counts[hop])

            # Periodic progress: print once per minute after the first minute.
            elapsed = time.monotonic() - start_time
            if elapsed >= next_report_at:
                print(
                    f"  [{elapsed / 60:.0f} min elapsed] "
                    f"strongly-connected orientations found so far: {n_orientations}",
                    flush=True,
                )
                next_report_at += 60.0
    except KeyboardInterrupt:
        # Fallback in case the signal handler did not suppress the exception
        # (e.g. when the interrupt arrived while inside a C extension).
        interrupted = True
        print(
            "\nInterrupted! Generating chart from intermediate results …",
            flush=True,
        )
    finally:
        signal.signal(signal.SIGINT, old_handler)

    print(f"Strongly-connected orientations found: {n_orientations}")

    if n_orientations == 0:
        print("No strongly-connected orientations found – nothing to plot.")
        return

    partial_label = " [partial]" if interrupted else ""
    title = (
        f"n={num_vertices} vertices, p={connectivity_label} "
        f"({graph.number_of_edges()} edges, {n_orientations} orientations"
        f"{partial_label})"
    )
    save_path = output or f"result_v{num_vertices}_{connectivity_label}.png"
    plot_score_correlations(
        apsp_sums,
        nhop_counts,
        title=title,
        save_path=save_path,
    )
    print(f"{'Partial ' if interrupted else ''}Plot saved to: {os.path.abspath(save_path)}")


def analyse_nhop_connectivity(
    num_vertices: int,
    num_graphs: int,
    num_orientations: int,
    seed: int | None,
    output: str | None,
) -> None:
    """Generate multiple Delaunay planar graphs and compare n-hop counts with SC ratio.

    For each generated planar graph this function:

    1. Randomly samples *num_orientations* orientation combinations (bit-masks
       over the edge set).  Each sampled orientation is either strongly
       connected or not.
    2. For each sampled orientation computes the 2-hop and 3-hop neighbour
       counts and records whether the orientation is strongly connected.
    3. Groups sampled orientations by their n-hop count and computes the SC
       ratio per group:
       SC ratio = (SC orientations in group) / (all sampled orientations in group)

    The result is plotted as a scatter plot with n-hop count on the x-axis
    and SC ratio on the y-axis (one subplot per hop distance).

    Args:
        num_vertices: Number of vertices in each generated Delaunay graph.
        num_graphs: Number of random planar graphs to generate.
        num_orientations: Number of orientation combinations to sample per graph.
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
        total_orientations = 1 << edge_count

        rng = random.Random(graph_seed)
        sample_size = min(num_orientations, total_orientations)
        sampled_indices = rng.sample(range(total_orientations), sample_size)

        sc_count = 0
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)

        for idx in sampled_indices:
            dg.clear_edges()
            for edge_idx, (u, v) in enumerate(edges):
                if (idx >> edge_idx) & 1:
                    dg.add_edge(v, u)
                else:
                    dg.add_edge(u, v)

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

    # Convert buckets to sorted lists for the scatter plot
    nhop_x: dict[int, list[int]] = {}
    sc_ratio_y: dict[int, list[float]] = {}
    for hop in hops:
        sorted_counts = sorted(nhop_bucket_total[hop].keys())
        nhop_x[hop] = sorted_counts
        sc_ratio_y[hop] = [
            nhop_bucket_sc[hop].get(nhop_count, 0) / nhop_bucket_total[hop][nhop_count]
            for nhop_count in sorted_counts
        ]

    title = (
        f"N-hop Count vs SC Ratio  "
        f"(n={num_vertices} vertices, {num_graphs} Delaunay graphs, "
        f"{num_orientations} orientations sampled)"
    )
    save_path = output or f"nhop_connectivity_v{num_vertices}.png"
    plot_nhop_connectivity_comparison(nhop_x, sc_ratio_y, title=title, save_path=save_path)
    print(f"Plot saved to: {os.path.abspath(save_path)}")


def _add_shared_parallel_args(parser: argparse.ArgumentParser) -> None:
    """Register worker / chunk-size / process arguments shared by both sub-commands."""
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker count for orientation generation "
             "(default: CPU core count)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2048,
        help="Orientation chunk size processed per task (default: 2048)"
    )
    parser.add_argument(
        "--processes", action="store_true", default=False,
        help="Use multiple processes instead of threads for parallel "
             "orientation evaluation. Bypasses the GIL for better CPU-bound "
             "throughput (default: False, i.e. thread-based)."
    )
    parser.add_argument(
        "--adaptive-chunk-size", action="store_true", default=False,
        help="Automatically compute an optimal chunk size based on the total "
             "workload (2^|E|) and number of workers. Overrides --chunk-size "
             "when performing exhaustive enumeration (default: False)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse n-hop approach on random graph orientations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # ------------------------------------------------------------------ #
    # Sub-command: analyse                                                 #
    # ------------------------------------------------------------------ #
    analyse_parser = subparsers.add_parser(
        "analyse",
        help="Analyse n-hop / APSP correlations for a single random graph.",
        description=(
            "Generate a single random graph, enumerate (or sample) its "
            "strongly-connected orientations, and plot the correlation between "
            "APSP sum and each n-hop neighbour count."
        ),
    )
    analyse_parser.add_argument(
        "--vertices", type=int, default=5,
        help="Number of vertices (default: 5)"
    )
    analyse_parser.add_argument(
        "--connectivity", type=float, default=None,
        help="Edge probability 0–1 for Erdős–Rényi model. "
             "If omitted, a Delaunay-based planar graph is generated instead."
    )
    analyse_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    analyse_parser.add_argument(
        "--output", type=str, default=None,
        help="File path to save the plot (e.g. out.png). "
             "If omitted, defaults to result_v{vertices}_c{connectivity}.png."
    )
    analyse_parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Use random sampling instead of exhaustive search. "
             "Yield at most this many strongly-connected orientations "
             "(constant-time regardless of graph size)."
    )
    analyse_parser.add_argument(
        "--min-samples", type=int, default=0,
        help="Minimum number of strongly-connected orientations that must be "
             "found when using --max-samples. Exits with an error if fewer are "
             "found within the attempt budget (default: 0, i.e. no minimum)."
    )
    _add_shared_parallel_args(analyse_parser)

    # ------------------------------------------------------------------ #
    # Sub-command: nhop-connectivity                                       #
    # ------------------------------------------------------------------ #
    nhop_parser = subparsers.add_parser(
        "nhop-connectivity",
        help="Compare 2-hop / 3-hop counts and SC ratio across multiple planar graphs.",
        description=(
            "Generate multiple random Delaunay planar graphs, enumerate all "
            "their orientations, and plot the SC ratio (SC orientations / total "
            "orientations) per distinct n-hop neighbour count value — for both "
            "2-hop and 3-hop distances."
        ),
    )
    nhop_parser.add_argument(
        "--vertices", type=int, default=5,
        help="Number of vertices in each generated Delaunay graph (default: 5)"
    )
    nhop_parser.add_argument(
        "--num-graphs", type=int, default=20,
        help="Number of Delaunay planar graphs to generate (default: 20)"
    )
    nhop_parser.add_argument(
        "--num-orientations", type=int, default=200,
        help="Number of random orientation combinations to sample per graph "
             "(default: 200). Capped at 2^|E| automatically."
    )
    nhop_parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed. Graph i uses seed+i when set."
    )
    nhop_parser.add_argument(
        "--output", type=str, default=None,
        help="File path to save the plot (e.g. nhop.png). "
             "If omitted, defaults to nhop_connectivity_v{vertices}.png."
    )

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #
    args = parser.parse_args()

    if args.command == "analyse":
        analyse(
            args.vertices,
            args.connectivity,
            args.seed,
            args.output,
            args.workers,
            args.chunk_size,
            args.max_samples,
            args.min_samples,
            args.processes,
            args.adaptive_chunk_size,
        )
    elif args.command == "nhop-connectivity":
        analyse_nhop_connectivity(
            args.vertices,
            args.num_graphs,
            args.num_orientations,
            args.seed,
            args.output,
        )


if __name__ == "__main__":
    main()

