"""Plotting helpers for the ``poster-results`` command."""

from __future__ import annotations

import json
import math
import os
from typing import Any

from src.visualizer import (
    BQM_DISPLAY_NAMES,
    PUBLICATION_COLORS,
    SOLVER_DISPLAY_NAMES,
    plot_apsp_reduction,
    plot_flow_stability,
    plot_preprocessing_scalability,
    plot_spent_time,
)


CLUSTER_DNC_STRATEGIES = ("embedding_aware",)
BASELINE_MR2S_VARIANTS = ("robbin_mr2s", "iterated_local_search_mr2s")
SERIES_COLORS = {
    "raw_sa": PUBLICATION_COLORS["traditional_raw"],
    "robbin_mr2s": PUBLICATION_COLORS["traditional_robbin"],
    "iterated_local_search_mr2s": PUBLICATION_COLORS["traditional_ils"],
    "embedding_aware": PUBLICATION_COLORS["ours"],
    "global": PUBLICATION_COLORS["global"],
    "embedding_aware_sum": PUBLICATION_COLORS["ours"],
    "embedding_aware_max": PUBLICATION_COLORS["ours_light"],
    "embedding_aware_avg": "#D98D89",
    "embedding_aware_min": "#F0B7B2",
}


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


def _aligned_values(sizes: list[int], values: list[float] | None) -> list[float | None]:
    aligned = list(values or [])
    if len(aligned) < len(sizes):
        aligned.extend([float("nan")] * (len(sizes) - len(aligned)))
    return [_json_number(value) for value in aligned[:len(sizes)]]


def _sum_series(sizes: list[int], first: list[float] | None, second: list[float] | None) -> list[float | None]:
    first_values = _aligned_values(sizes, first)
    second_values = _aligned_values(sizes, second)
    totals = []
    for lhs, rhs in zip(first_values, second_values, strict=True):
        if lhs is not None and rhs is not None:
            totals.append(lhs + rhs)
        elif lhs is not None:
            totals.append(lhs)
        elif rhs is not None:
            totals.append(rhs)
        else:
            totals.append(None)
    return totals


def _json_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _series_summary(
    sizes: list[int],
    key: str,
    display_name: str,
    color: str,
    values: list[float] | list[float | None] | None,
) -> dict[str, Any]:
    return {
        "key": key,
        "display_name": display_name,
        "color": color,
        "x": sizes,
        "y": values if values and len(values) == len(sizes) and any(value is None for value in values) else _aligned_values(sizes, values),
    }


def _plot_data_summary(
    sizes: list[int],
    results: dict[str, Any],
    clustered: dict[str, Any],
    mr2s_variants: dict[str, Any],
) -> dict[str, Any]:
    raw_sa = results.get("raw_sa", {})
    global_result = results.get("global", {})
    timings = results.get("timings", {})

    apsp_series = [
        _series_summary(sizes, "raw_sa", SOLVER_DISPLAY_NAMES["raw_sa"], SERIES_COLORS["raw_sa"], raw_sa.get("apsp")),
    ]
    flow_series = [
        _series_summary(sizes, "raw_sa", SOLVER_DISPLAY_NAMES["raw_sa"], SERIES_COLORS["raw_sa"], raw_sa.get("flow")),
    ]
    for name in BASELINE_MR2S_VARIANTS:
        if name not in mr2s_variants:
            continue
        apsp_series.append(
            _series_summary(
                sizes,
                name,
                SOLVER_DISPLAY_NAMES[name],
                SERIES_COLORS[name],
                mr2s_variants[name].get("apsp"),
            )
        )
        flow_series.append(
            _series_summary(
                sizes,
                name,
                SOLVER_DISPLAY_NAMES[name],
                SERIES_COLORS[name],
                mr2s_variants[name].get("flow"),
            )
        )
    apsp_series.append(
        _series_summary(
            sizes,
            "embedding_aware",
            SOLVER_DISPLAY_NAMES["embedding_aware"],
            SERIES_COLORS["embedding_aware"],
            clustered.get("apsp"),
        )
    )
    flow_series.append(
        _series_summary(
            sizes,
            "embedding_aware",
            SOLVER_DISPLAY_NAMES["embedding_aware"],
            SERIES_COLORS["embedding_aware"],
            clustered.get("flow"),
        )
    )

    scalability_series = [
        _series_summary(
            sizes,
            "global",
            BQM_DISPLAY_NAMES["global"],
            SERIES_COLORS["global"],
            global_result.get("qubo_vars"),
        ),
        _series_summary(
            sizes,
            "embedding_aware_sum",
            BQM_DISPLAY_NAMES["embedding_aware_sum"],
            SERIES_COLORS["embedding_aware_sum"],
            clustered.get("qubo_vars"),
        ),
        _series_summary(
            sizes,
            "embedding_aware_max",
            BQM_DISPLAY_NAMES["embedding_aware_max"],
            SERIES_COLORS["embedding_aware_max"],
            clustered.get("subgraph_size"),
        ),
        _series_summary(
            sizes,
            "embedding_aware_avg",
            BQM_DISPLAY_NAMES["embedding_aware_avg"],
            SERIES_COLORS["embedding_aware_avg"],
            clustered.get("qvars_mean"),
        ),
        _series_summary(
            sizes,
            "embedding_aware_min",
            BQM_DISPLAY_NAMES["embedding_aware_min"],
            SERIES_COLORS["embedding_aware_min"],
            clustered.get("qvars_min"),
        ),
    ]

    spent_time_series = [
        _series_summary(sizes, "raw_sa", SOLVER_DISPLAY_NAMES["raw_sa"], SERIES_COLORS["raw_sa"], timings.get("raw_sa")),
    ]
    for name in BASELINE_MR2S_VARIANTS:
        key = f"{name}_solve"
        if key in timings:
            spent_time_series.append(
                _series_summary(sizes, name, SOLVER_DISPLAY_NAMES[name], SERIES_COLORS[name], timings.get(key))
            )
    spent_time_series.append(
        _series_summary(
            sizes,
            "embedding_aware",
            SOLVER_DISPLAY_NAMES["embedding_aware"],
            SERIES_COLORS["embedding_aware"],
            _sum_series(
                sizes,
                timings.get("dnc_embedding_aware_solve", timings.get("clustered_solve")),
                timings.get("dnc_embedding_aware_embed", timings.get("clustered_embed")),
            ),
        )
    )

    return {
        "sizes": sizes,
        "plots": {
            "apsp_reduction": {
                "output": "apsp_reduction.png",
                "y_label": "Normalized APSP (lower is better)",
                "series": apsp_series,
            },
            "flow_stability": {
                "output": "flow_stability.png",
                "y_label": "Flow imbalance score (lower is better)",
                "series": flow_series,
            },
            "scalability": {
                "output": "scalability.png",
                "y_label": "BQM binary variable count",
                "series": scalability_series,
            },
            "spent_time": {
                "output": "spent_time.png",
                "y_label": "Mean runtime (seconds)",
                "series": spent_time_series,
            },
        },
    }


def _plot_results(results: dict[str, Any], output_dir: str) -> None:
    sizes = results["sizes"]
    # Cluster MR2S is the QA-backed embedding-aware DnC series.
    mr2s_variants = _select_sections(results, "mr2s_variants", BASELINE_MR2S_VARIANTS)
    dnc_strategies = _select_sections(results, "dnc_strategies", CLUSTER_DNC_STRATEGIES)
    if "embedding_aware" in dnc_strategies:
        dnc_strategies["embedding_aware"] = _cluster_qvar_stats(dnc_strategies["embedding_aware"])
    clustered = dnc_strategies.get("embedding_aware", results["mr2s"])
    summary = _plot_data_summary(sizes, results, clustered, mr2s_variants)
    with open(os.path.join(output_dir, "plotted_data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_apsp_reduction(
        sizes,
        results["random"]["apsp"],
        results["raw_sa"]["apsp"],
        results["global"]["apsp"],
        clustered["apsp"],
        mr2s_variants=mr2s_variants,
        dnc_strategies=dnc_strategies,
        save_path=os.path.join(output_dir, "apsp_reduction.png"),
    )
    plot_flow_stability(
        sizes,
        results["random"]["flow"],
        results["raw_sa"]["flow"],
        results["global"]["flow"],
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
            timings.get("global_solve", []),
            timings.get("global_embed", []),
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
