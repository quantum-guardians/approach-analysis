#!/usr/bin/env python3
"""Entry point for the n-hop approach analysis.

Usage examples
--------------
Analyse a single random graph (correlation plots)::

    python main.py analyse

Run with custom parameters::

    python main.py analyse --vertices 5 --connectivity 0.7 --seed 42 --output out.png

Compare n-hop counts and SC ratio across multiple graphs::

    python main.py nhop-connectivity --vertices 5 --num-graphs 20 --seed 0

Run nhop-connectivity with a custom connectivity sweep range::

    python main.py nhop-connectivity --vertices 5 --num-graphs 30 \\
        --connectivity-min 0.2 --connectivity-max 0.9 --seed 42 --output nhop.png
"""

import argparse
import os
import signal
import time
import types

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

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
    connectivity_min: float,
    connectivity_max: float,
    seed: int | None,
    output: str | None,
    workers: int | None,
    chunk_size: int,
    use_processes: bool = False,
    adaptive_chunk_size: bool = False,
) -> None:
    """Generate multiple random graphs and compare n-hop counts with SC ratio.

    For each graph in a sweep over edge-probability values (from
    *connectivity_min* to *connectivity_max*), this function:

    1. Generates a random undirected graph.
    2. Exhaustively enumerates all strongly-connected orientations.
    3. Computes the SC ratio (SC orientations / total orientations = 2^|E|).
    4. Averages the 2-hop and 3-hop neighbour counts across SC orientations.

    The collected data is then visualised as a scatter plot with SC ratio on
    the x-axis and average n-hop count on the y-axis (one subplot per hop).

    Args:
        num_vertices: Number of vertices in each generated graph.
        num_graphs: Number of graphs to generate (data points in the plot).
        connectivity_min: Minimum edge probability for the sweep.
        connectivity_max: Maximum edge probability for the sweep.
        seed: Base random seed.  Graph *i* uses seed ``seed + i`` when set.
        output: File path to save the plot.  Auto-generated if ``None``.
        workers: Worker count for orientation enumeration.
        chunk_size: Orientation chunk size per worker task.
        use_processes: Use :class:`~concurrent.futures.ProcessPoolExecutor`
            instead of threads.
        adaptive_chunk_size: Compute chunk size automatically based on workload.
    """
    hops = NHOP_CONN_HOPS

    sc_ratios: list[float] = []
    nhop_avgs: dict[int, list[float]] = {hop: [] for hop in hops}

    if num_graphs == 1:
        connectivity_values = [connectivity_min]
    else:
        step = (connectivity_max - connectivity_min) / (num_graphs - 1)
        connectivity_values = [connectivity_min + i * step for i in range(num_graphs)]

    print(
        f"Analysing {num_graphs} graphs ({num_vertices} vertices, "
        f"connectivity {connectivity_min:.2f}–{connectivity_max:.2f}) …"
    )

    for i, conn in enumerate(connectivity_values):
        graph_seed = (seed + i) if seed is not None else None
        graph = generate_graph(num_vertices, conn, seed=graph_seed)

        edge_count = graph.number_of_edges()
        total_orientations = 1 << edge_count

        if total_orientations > MAX_EXHAUSTIVE_ORIENTATIONS:
            print(
                f"  Graph {i + 1}/{num_graphs}: conn={conn:.2f}, "
                f"edges={edge_count}: too many orientations "
                f"({total_orientations:,}), skipping – "
                f"use fewer vertices or a lower connectivity range."
            )
            continue

        sc_count = 0
        nhop_sums: dict[int, float] = {hop: 0.0 for hop in hops}

        for orientation in generate_strongly_connected_orientations(
            graph,
            num_workers=workers,
            chunk_size=chunk_size,
            use_processes=use_processes,
            adaptive_chunk_size=adaptive_chunk_size,
        ):
            sc_count += 1
            _, counts = calculate_apsp_sum_and_nhop_neighbor_counts(
                orientation, hops=hops
            )
            for hop in hops:
                nhop_sums[hop] += counts[hop]

        sc_ratio = sc_count / total_orientations if total_orientations > 0 else 0.0
        sc_ratios.append(sc_ratio)
        for hop in hops:
            nhop_avgs[hop].append(nhop_sums[hop] / sc_count if sc_count > 0 else 0.0)

        print(
            f"  Graph {i + 1}/{num_graphs}: conn={conn:.2f}, "
            f"edges={edge_count}, SC={sc_count}/{total_orientations} "
            f"(ratio={sc_ratio:.3f})"
        )

    if not sc_ratios:
        print("No valid graphs to plot.")
        return

    title = (
        f"N-hop vs SC-ratio  (n={num_vertices} vertices, "
        f"{len(sc_ratios)} graphs)"
    )
    save_path = output or f"nhop_connectivity_v{num_vertices}.png"
    plot_nhop_connectivity_comparison(
        sc_ratios, nhop_avgs, title=title, save_path=save_path
    )
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
        help="Compare 2-hop / 3-hop counts and SC ratio across multiple graphs.",
        description=(
            "Sweep the edge-probability parameter over a range, generate one "
            "random graph per step, exhaustively enumerate its strongly-connected "
            "orientations, and plot average 2-hop and 3-hop neighbour counts "
            "against the SC ratio (strongly-connected / total orientations)."
        ),
    )
    nhop_parser.add_argument(
        "--vertices", type=int, default=5,
        help="Number of vertices in each generated graph (default: 5)"
    )
    nhop_parser.add_argument(
        "--num-graphs", type=int, default=20,
        help="Number of graphs to generate across the connectivity sweep "
             "(default: 20)"
    )
    nhop_parser.add_argument(
        "--connectivity-min", type=float, default=0.3,
        help="Minimum edge probability for the sweep (default: 0.3)"
    )
    nhop_parser.add_argument(
        "--connectivity-max", type=float, default=0.9,
        help="Maximum edge probability for the sweep (default: 0.9)"
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
    _add_shared_parallel_args(nhop_parser)

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
            args.connectivity_min,
            args.connectivity_max,
            args.seed,
            args.output,
            args.workers,
            args.chunk_size,
            args.processes,
            args.adaptive_chunk_size,
        )


if __name__ == "__main__":
    main()

