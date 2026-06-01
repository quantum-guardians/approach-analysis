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


CLUSTER_DNC_STRATEGIES = ("embedding_aware",)
BASELINE_MR2S_VARIANTS = ("robbin_mr2s", "iterated_local_search_mr2s")


def _select_sections(results: dict[str, Any], section: str, names: tuple[str, ...]) -> dict[str, Any]:
    source = results.get(section, {})
    return {
        name: source[name]
        for name in names
        if name in source
    }


def _cluster_qvar_stats(section: dict[str, Any]) -> dict[str, Any]:
    result = dict(section)
    qvars_mean = []
    qvars_min = []
    for size_partitions in section.get("partition", []):
        trial_means = []
        trial_mins = []
        for partition in size_partitions:
            qvars = [
                probe["qvars"]
                for probe in partition.get("selected_probes", [])
                if isinstance(probe.get("qvars"), (int, float)) and probe["qvars"] > 0
            ]
            if qvars:
                trial_means.append(sum(qvars) / len(qvars))
                trial_mins.append(min(qvars))
        qvars_mean.append(sum(trial_means) / len(trial_means) if trial_means else float("nan"))
        qvars_min.append(sum(trial_mins) / len(trial_mins) if trial_mins else float("nan"))
    result["qvars_mean"] = qvars_mean
    result["qvars_min"] = qvars_min
    return result


def _plot_results(results: dict[str, Any], output_dir: str) -> None:
    sizes = results["sizes"]
    # Cluster MR2S is the QA-backed embedding-aware DnC series.
    mr2s_variants = _select_sections(results, "mr2s_variants", BASELINE_MR2S_VARIANTS)
    dnc_strategies = _select_sections(results, "dnc_strategies", CLUSTER_DNC_STRATEGIES)
    if "embedding_aware" in dnc_strategies:
        dnc_strategies["embedding_aware"] = _cluster_qvar_stats(dnc_strategies["embedding_aware"])
    clustered = dnc_strategies.get("embedding_aware", results["mr2s"])
    plot_apsp_reduction(
        sizes,
        results["random"]["apsp"],
        results["raw_sa"]["apsp"],
        [],
        clustered["apsp"],
        mr2s_variants=mr2s_variants,
        dnc_strategies=dnc_strategies,
        save_path=os.path.join(output_dir, "apsp_reduction.png"),
    )
    plot_flow_stability(
        sizes,
        results["random"]["flow"],
        results["raw_sa"]["flow"],
        [],
        clustered["flow"],
        mr2s_variants=mr2s_variants,
        dnc_strategies=dnc_strategies,
        save_path=os.path.join(output_dir, "flow_stability.png"),
    )
    plot_preprocessing_scalability(
        sizes,
        results["global"]["qubo_vars"],
        clustered["qubo_vars"],
        results["global"]["subgraph_size"],
        clustered["subgraph_size"],
        global_physical=results["global"].get("phys_total"),
        clustered_physical_total=clustered.get("phys_total"),
        clustered_physical_max=clustered.get("phys_max"),
        clustered_physical_mean=clustered.get("phys_mean"),
        clustered_physical_min=clustered.get("phys_min"),
        mr2s_variants=mr2s_variants,
        dnc_strategies=dnc_strategies,
        save_path=os.path.join(output_dir, "scalability.png"),
    )
    if "timings" in results:
        timings = results["timings"]
        plot_spent_time(
            sizes,
            timings.get("graph", []),
            timings.get("raw_sa", []),
            [],
            [],
            timings.get("clustered_solve", []),
            timings.get("clustered_embed", []),
            timings.get("random", []),
            mr2s_variant_timings={
                key: value
                for key, value in timings.items()
                if key.startswith(BASELINE_MR2S_VARIANTS)
            },
            dnc_timings={
                key: value
                for key, value in timings.items()
                if key.startswith("dnc_embedding_aware")
            },
            save_path=os.path.join(output_dir, "spent_time.png"),
        )
