"""Plotting helpers for the ``poster-results`` command."""

from __future__ import annotations

import os
from typing import Any

from src.visualizer import (
    plot_apsp_reduction,
    plot_flow_stability,
    plot_preprocessing_scalability,
    plot_spent_time,
)


def _plot_results(results: dict[str, Any], output_dir: str) -> None:
    sizes = results["sizes"]
    plot_apsp_reduction(
        sizes,
        results["random"]["apsp"],
        results["raw_sa"]["apsp"],
        results["global"]["apsp"],
        results["mr2s"]["apsp"],
        mr2s_variants=results.get("mr2s_variants"),
        dnc_strategies=results.get("dnc_strategies"),
        save_path=os.path.join(output_dir, "apsp_reduction.png"),
    )
    plot_flow_stability(
        sizes,
        results["random"]["flow"],
        results["raw_sa"]["flow"],
        results["global"]["flow"],
        results["mr2s"]["flow"],
        mr2s_variants=results.get("mr2s_variants"),
        dnc_strategies=results.get("dnc_strategies"),
        save_path=os.path.join(output_dir, "flow_stability.png"),
    )
    plot_preprocessing_scalability(
        sizes,
        results["global"]["qubo_vars"],
        results["mr2s"]["qubo_vars"],
        results["global"]["subgraph_size"],
        results["mr2s"]["subgraph_size"],
        global_physical=results["global"].get("phys_total"),
        clustered_physical_total=results["mr2s"].get("phys_total"),
        clustered_physical_max=results["mr2s"].get("phys_max"),
        clustered_physical_mean=results["mr2s"].get("phys_mean"),
        clustered_physical_min=results["mr2s"].get("phys_min"),
        mr2s_variants=results.get("mr2s_variants"),
        dnc_strategies=results.get("dnc_strategies"),
        save_path=os.path.join(output_dir, "scalability.png"),
    )
    if "timings" in results:
        timings = results["timings"]
        plot_spent_time(
            sizes,
            timings.get("graph", []),
            timings.get("raw_sa", []),
            timings.get("global_solve", []),
            timings.get("global_embed", []),
            timings.get("clustered_solve", []),
            timings.get("clustered_embed", []),
            timings.get("random", []),
            mr2s_variant_timings=timings,
            dnc_timings=timings,
            save_path=os.path.join(output_dir, "spent_time.png"),
        )
