"""``analyse`` subcommand – n-hop / APSP correlation analysis.

Generates a single random graph, enumerates (or samples) its
strongly-connected orientations, and plots the correlation between the
APSP sum and each n-hop neighbour count.
"""

from __future__ import annotations

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
from src.visualizer import plot_score_correlations


HOPS = (2, 3, 4)


def run(
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
    """Generate a graph, find SC orientations, and plot APSP vs n-hop counts."""
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


def _add_shared_parallel_args(parser: argparse.ArgumentParser) -> None:
    """Register worker / chunk-size / process arguments."""
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


def register_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the ``analyse`` subcommand to *subparsers*."""
    p = subparsers.add_parser(
        "analyse",
        help="Analyse n-hop / APSP correlations for a single random graph.",
        description=(
            "Generate a single random graph, enumerate (or sample) its "
            "strongly-connected orientations, and plot the correlation between "
            "APSP sum and each n-hop neighbour count."
        ),
    )
    p.add_argument(
        "--vertices", type=int, default=5,
        help="Number of vertices (default: 5)"
    )
    p.add_argument(
        "--connectivity", type=float, default=None,
        help="Edge probability 0–1 for Erdős–Rényi model. "
             "If omitted, a Delaunay-based planar graph is generated instead."
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="File path to save the plot (e.g. out.png). "
             "If omitted, defaults to result_v{vertices}_c{connectivity}.png."
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Use random sampling instead of exhaustive search. "
             "Yield at most this many strongly-connected orientations "
             "(constant-time regardless of graph size)."
    )
    p.add_argument(
        "--min-samples", type=int, default=0,
        help="Minimum number of strongly-connected orientations that must be "
             "found when using --max-samples. Exits with an error if fewer are "
             "found within the attempt budget (default: 0, i.e. no minimum)."
    )
    _add_shared_parallel_args(p)
    p.set_defaults(func=_dispatch)


def _dispatch(args: argparse.Namespace) -> None:
    run(
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
