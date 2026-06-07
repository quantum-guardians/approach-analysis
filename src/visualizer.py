"""Visualization module.

Produces scatter-plot matrices (pair plots) that illustrate the correlation
between APSP sum and n-hop neighbour counts across all strongly-connected
orientations of a graph, as well as plots comparing n-hop counts and
strongly-connected orientation ratios across multiple graphs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PUBLICATION_COLORS = {
    "ours": "#C43C39",
    "ours_light": "#E06B65",
    "traditional_raw": "#2F6CA3",
    "traditional_robbin": "#5B8CC0",
    "traditional_ils": "#1B4F72",
    "global": "#4B5563",
    "global_light": "#8A95A3",
}
PUBLICATION_MARKERS = {
    "raw_sa": "o",
    "robbin_mr2s": "s",
    "iterated_local_search_mr2s": "^",
    "embedding_aware": "D",
    "global": "P",
}
SOLVER_DISPLAY_NAMES = {
    "raw_sa": "SA",
    "robbin_mr2s": "Robbin",
    "iterated_local_search_mr2s": "ILS",
    "embedding_aware": "Ours",
    "global": "Global MR2S Solver",
}
BQM_DISPLAY_NAMES = {
    "global": "Mono",
    "embedding_aware_sum": "Cluster sum",
    "embedding_aware_max": "Cluster max",
    "embedding_aware_avg": "Cluster avg",
    "embedding_aware_min": "Cluster min",
}
DNC_STRATEGY_STYLES = {
    "poster": ("D-", PUBLICATION_COLORS["ours_light"], "DnC poster"),
    "embedding_aware": ("D-", PUBLICATION_COLORS["ours"], SOLVER_DISPLAY_NAMES["embedding_aware"]),
    "degeneracy_pruning": ("v-", "purple", "DnC degeneracy-pruning"),
}
MR2S_VARIANT_STYLES = {
    "robbin_mr2s": ("s-", PUBLICATION_COLORS["traditional_robbin"], SOLVER_DISPLAY_NAMES["robbin_mr2s"]),
    "iterated_local_search_mr2s": ("^-", PUBLICATION_COLORS["traditional_ils"], SOLVER_DISPLAY_NAMES["iterated_local_search_mr2s"]),
}


def _aligned_values(sizes: Sequence[int], values: Sequence[float] | None) -> list[float]:
    """Return a y-series that is safe to plot against ``sizes``."""
    aligned = list(values or [])
    target_len = len(sizes)
    if len(aligned) < target_len:
        aligned.extend([float("nan")] * (target_len - len(aligned)))
    return aligned[:target_len]


def _finite_values(series: Sequence[dict[str, Any]]) -> list[float]:
    values = []
    for item in series:
        for value in item["y"]:
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
    return values


def _publication_style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.tick_params(axis="both", labelsize=10, colors="#374151", width=0.8)
    ax.grid(True, color="#D1D5DB", linestyle="-", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)


def _axis_break_limits(values: Sequence[float]) -> tuple[float, float, float, float] | None:
    finite = sorted(v for v in values if np.isfinite(v))
    if len(finite) < 4:
        return None
    gaps = [
        (
            finite[index + 1] / max(finite[index], 1e-12),
            finite[index + 1] - finite[index],
            index,
        )
        for index in range(len(finite) - 1)
    ]
    ratio, absolute_gap, index = max(gaps, key=lambda item: (item[0], item[1]))
    data_span = finite[-1] - finite[0]
    if ratio < 1.35 and absolute_gap < data_span * 0.22:
        return None
    lower_top = finite[index] + max(abs(finite[index]) * 0.08, absolute_gap * 0.08)
    upper_bottom = finite[index + 1] - max(abs(finite[index + 1]) * 0.06, absolute_gap * 0.08)
    upper_top = finite[-1] * 1.08
    lower_bottom = 0.0 if finite[0] >= 0 else finite[0] * 1.08
    if upper_bottom <= lower_top:
        return None
    return lower_bottom, lower_top, upper_bottom, upper_top


def _sum_aligned_values(
    sizes: Sequence[int],
    first: Sequence[float] | None,
    second: Sequence[float] | None,
) -> list[float]:
    first_values = _aligned_values(sizes, first)
    second_values = _aligned_values(sizes, second)
    totals = []
    for lhs, rhs in zip(first_values, second_values, strict=True):
        if np.isfinite(lhs) and np.isfinite(rhs):
            totals.append(lhs + rhs)
        elif np.isfinite(lhs):
            totals.append(lhs)
        elif np.isfinite(rhs):
            totals.append(rhs)
        else:
            totals.append(float("nan"))
    return totals


def _draw_break_marks(lower_ax: plt.Axes, upper_ax: plt.Axes) -> None:
    kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=9, linestyle="none",
                  color="#4B5563", mec="#4B5563", mew=0.9, clip_on=False)
    upper_ax.plot([0, 1], [0, 0], transform=upper_ax.transAxes, **kwargs)
    lower_ax.plot([0, 1], [1, 1], transform=lower_ax.transAxes, **kwargs)


def _plot_series(
    ax: plt.Axes,
    sizes: Sequence[int],
    series: Sequence[dict[str, Any]],
) -> None:
    for item in series:
        ax.plot(
            sizes,
            item["y"],
            label=item["label"],
            color=item["color"],
            marker=item.get("marker", "o"),
            linestyle=item.get("linestyle", "-"),
            linewidth=3.0,
            markersize=7.0,
            markeredgewidth=1.0,
            markeredgecolor="white",
            alpha=item.get("alpha", 0.95),
        )


def _publication_line_figure(
    sizes: Sequence[int],
    series: Sequence[dict[str, Any]],
    *,
    ylabel: str,
    title: str,
    save_path: str | None,
    use_axis_break: bool = True,
    yscale: str = "linear",
) -> plt.Figure:
    values = _finite_values(series)
    limits = _axis_break_limits(values) if use_axis_break else None

    if limits is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        axes = [ax]
    else:
        fig, (upper_ax, lower_ax) = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.2),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 2.8], "hspace": 0.06},
        )
        axes = [upper_ax, lower_ax]

    for ax in axes:
        _plot_series(ax, sizes, series)
        ax.set_yscale(yscale)
        _publication_style_axes(ax)

    if limits is not None:
        lower_bottom, lower_top, upper_bottom, upper_top = limits
        upper_ax, lower_ax = axes
        upper_ax.set_ylim(upper_bottom, upper_top)
        lower_ax.set_ylim(lower_bottom, lower_top)
        upper_ax.spines["bottom"].set_visible(False)
        lower_ax.spines["top"].set_visible(False)
        upper_ax.tick_params(labelbottom=False, bottom=False)
        _draw_break_marks(lower_ax, upper_ax)
        ax = lower_ax
    else:
        ax = axes[0]

    axes[0].set_title(title, fontsize=13, fontweight="semibold", pad=10, color="#111827")
    ax.set_xlabel("Graph size (vertices)", fontsize=11, color="#111827")
    fig.supylabel(ylabel, fontsize=11, color="#111827")
    axes[0].legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#E5E7EB",
        fontsize=12,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def _solver_series(
    sizes: Sequence[int],
    key: str,
    label: str,
    color: str,
    values: Sequence[float] | None,
    marker: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "color": color,
        "marker": marker,
        "y": _aligned_values(sizes, values),
    }


def plot_score_correlations(
    apsp_sums: Sequence[float],
    nhop_counts: dict[int, Sequence[int]],
    title: str = "Score Correlations",
    save_path: str | None = None,
) -> plt.Figure:
    """Draw scatter plots correlating APSP sum with each n-hop neighbour count.

    One subplot is created per hop value.  The x-axis shows the APSP sum and
    the y-axis shows the number of node pairs at that hop distance.

    Args:
        apsp_sums: APSP sum for each graph orientation.
        nhop_counts: Mapping from hop distance to a sequence of neighbour
            counts, one per orientation (same order as *apsp_sums*).
        title: Super-title displayed above the figure.
        save_path: If provided, the figure is saved to this file path instead
            of being displayed interactively.

    Returns:
        The :class:`matplotlib.figure.Figure` that was created.
    """
    hops = sorted(nhop_counts.keys())
    n_plots = len(hops)

    fig = plt.figure(figsize=(5 * n_plots, 4))
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(1, n_plots, figure=fig)

    for idx, hop in enumerate(hops):
        ax = fig.add_subplot(gs[0, idx])
        ax.scatter(apsp_sums, nhop_counts[hop], alpha=0.6, edgecolors="none", s=20)
        ax.set_xlabel("APSP sum")
        ax.set_ylabel(f"{hop}-hop neighbour count")
        ax.set_title(f"APSP vs {hop}-hop")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


def plot_nhop_connectivity_comparison(
    nhop_counts: dict[int, Sequence[int | float]],
    sc_ratios: dict[int, Sequence[float]],
    title: str = "N-hop Count vs SC Ratio",
    save_path: str | None = None,
) -> plt.Figure:
    """Draw scatter plots comparing n-hop neighbour counts and SC ratio.

    One subplot is created per hop value.  The x-axis shows distinct n-hop
    neighbour count values and the y-axis shows the SC ratio for orientations
    that have that n-hop count.

    The SC ratio for a given n-hop count value *k* is defined as:
    (number of orientations with n-hop count = k that are strongly connected) /
    (total number of orientations with n-hop count = k).

    Each data point on the scatter plot represents a distinct n-hop count
    value, aggregated across all generated graphs and their orientations.

    Args:
        nhop_counts: Mapping from hop distance to a sequence of distinct n-hop
            count values (x-axis), one entry per distinct bucket.
        sc_ratios: Mapping from hop distance to the corresponding SC ratios
            (y-axis), one entry per distinct n-hop count bucket (same order as
            *nhop_counts*).
        title: Super-title displayed above the figure.
        save_path: If provided, the figure is saved to this file path instead
            of being displayed interactively.

    Returns:
        The :class:`matplotlib.figure.Figure` that was created.
    """
    hops = sorted(nhop_counts.keys())
    n_plots = len(hops)

    fig = plt.figure(figsize=(5 * n_plots, 4))
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(1, n_plots, figure=fig)

    for idx, hop in enumerate(hops):
        ax = fig.add_subplot(gs[0, idx])
        ax.scatter(nhop_counts[hop], sc_ratios[hop], alpha=0.7, edgecolors="none", s=40)
        ax.set_xlabel(f"{hop}-hop neighbour count")
        ax.set_ylabel("SC ratio (strongly-connected / total)")
        ax.set_title(f"{hop}-hop count vs SC ratio")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


def plot_face_k_analysis(
    results: dict[str, Any],
    graph_sizes: list[int],
    removal_pcts: list[float],
    target_ks: list[int],
    title: str = "Face-k Analysis",
    save_path: str | None = None,
) -> plt.Figure:
    """Draw trend plots for the face-k analysis.

    Creates a 2x2 panel figure:

    * **Top-left**: SC ratio vs ``target_k`` for each graph size
      (at ``removal_pct = 0``).
    * **Top-right**: SC ratio vs ``target_k`` for each removal percentage
      (at the median graph size).
    * **Bottom-left**: Mean normalised APSP vs ``target_k`` for each graph size
      (at ``removal_pct = 0``).
    * **Bottom-right**: Mean normalised APSP vs ``target_k`` for each removal
      percentage (at the median graph size).

    Args:
        results: Nested dict ``results[n_str][pct_str][k_str]`` as produced by
            :func:`src.commands.face_k_analysis.run`.
        graph_sizes: List of vertex counts swept in the experiment.
        removal_pcts: List of edge-removal fractions swept.
        target_ks: List of ``target_k`` values swept.
        title: Super-title for the figure.
        save_path: File path to save the figure; displayed interactively when
            ``None``.

    Returns:
        The :class:`matplotlib.figure.Figure` that was created.
    """
    ref_pct = removal_pcts[0]
    ref_n = graph_sizes[len(graph_sizes) // 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    k_arr = np.array(target_ks, dtype=float)

    # Top-left: SC ratio vs k, varying graph size (fixed removal_pct)
    ax = axes[0, 0]
    for n in graph_sizes:
        sc_vals = [
            results.get(str(n), {}).get(str(ref_pct), {}).get(str(k), {}).get(
                "sc_ratio", float("nan")
            )
            for k in target_ks
        ]
        ax.plot(k_arr, sc_vals, marker="o", label=f"n={n}")
    ax.set_xlabel("target k")
    ax.set_ylabel("SC ratio")
    ax.set_title(f"SC ratio vs k  (removal={ref_pct:.0%})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Top-right: SC ratio vs k, varying removal_pct (fixed graph size)
    ax = axes[0, 1]
    for pct in removal_pcts:
        sc_vals = [
            results.get(str(ref_n), {}).get(str(pct), {}).get(str(k), {}).get(
                "sc_ratio", float("nan")
            )
            for k in target_ks
        ]
        ax.plot(k_arr, sc_vals, marker="s", label=f"removal={pct:.0%}")
    ax.set_xlabel("target k")
    ax.set_ylabel("SC ratio")
    ax.set_title(f"SC ratio vs k  (n={ref_n})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Bottom-left: mean APSP vs k, varying graph size (fixed removal_pct)
    ax = axes[1, 0]
    for n in graph_sizes:
        apsp_vals = [
            results.get(str(n), {}).get(str(ref_pct), {}).get(str(k), {}).get(
                "mean_apsp", float("nan")
            )
            for k in target_ks
        ]
        ax.plot(k_arr, apsp_vals, marker="o", label=f"n={n}")
    ax.set_xlabel("target k")
    ax.set_ylabel("mean normalised APSP")
    ax.set_title(f"APSP vs k  (removal={ref_pct:.0%})")
    ax.legend(fontsize=8)

    # Bottom-right: mean APSP vs k, varying removal_pct (fixed graph size)
    ax = axes[1, 1]
    for pct in removal_pcts:
        apsp_vals = [
            results.get(str(ref_n), {}).get(str(pct), {}).get(str(k), {}).get(
                "mean_apsp", float("nan")
            )
            for k in target_ks
        ]
        ax.plot(k_arr, apsp_vals, marker="s", label=f"removal={pct:.0%}")
    ax.set_xlabel("target k")
    ax.set_ylabel("mean normalised APSP")
    ax.set_title(f"APSP vs k  (n={ref_n})")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


def plot_optimal_k_fit_evidence(
    optimal: dict[tuple[int, float], int],
    graph_sizes: list[int],
    removal_pcts: list[float],
    predicted: dict[tuple[int, float], int],
    title: str = "Optimal target-k Fit Evidence",
    save_path: str | None = None,
) -> plt.Figure:
    """Visualise observed vs predicted optimal-k values and fit error.

    Creates a 2x2 panel figure:

    * observed optimal ``k`` heatmap
    * predicted optimal ``k`` heatmap
    * absolute error heatmap
    * observed-vs-predicted scatter with identity line
    """
    observed_grid = np.array(
        [[optimal[(n, pct)] for pct in removal_pcts] for n in graph_sizes],
        dtype=float,
    )
    predicted_grid = np.array(
        [[predicted[(n, pct)] for pct in removal_pcts] for n in graph_sizes],
        dtype=float,
    )
    error_grid = np.abs(predicted_grid - observed_grid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    def _heatmap(ax: plt.Axes, data: np.ndarray, cmap: str, panel_title: str) -> None:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xlabel("edge removal ratio")
        ax.set_ylabel("vertex count")
        ax.set_xticks(range(len(removal_pcts)))
        ax.set_xticklabels([f"{pct:.0%}" for pct in removal_pcts])
        ax.set_yticks(range(len(graph_sizes)))
        ax.set_yticklabels([str(n) for n in graph_sizes])
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                ax.text(col, row, f"{int(round(data[row, col]))}", ha="center", va="center")
        fig.colorbar(im, ax=ax, shrink=0.85)

    _heatmap(axes[0, 0], observed_grid, "Blues", "Observed optimal k")
    _heatmap(axes[0, 1], predicted_grid, "Greens", "Formula-predicted optimal k")
    _heatmap(axes[1, 0], error_grid, "Oranges", "Absolute prediction error")

    ax = axes[1, 1]
    observed_vals = observed_grid.ravel()
    predicted_vals = predicted_grid.ravel()
    ax.scatter(observed_vals, predicted_vals, s=50, alpha=0.8)
    lower = min(observed_vals.min(initial=0), predicted_vals.min(initial=0))
    upper = max(observed_vals.max(initial=1), predicted_vals.max(initial=1))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


def plot_apsp_reduction(
    sizes: list[int],
    random_apsp: list[float],
    raw_sa_apsp: list[float],
    global_apsp: list[float],
    clustered_apsp: list[float],
    mr2s_variants: dict[str, dict] | None = None,
    dnc_strategies: dict[str, dict] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot normalized APSP for poster/paper comparison; lower is better."""
    del random_apsp, global_apsp
    series = [
        _solver_series(
            sizes,
            "raw_sa",
            SOLVER_DISPLAY_NAMES["raw_sa"],
            PUBLICATION_COLORS["traditional_raw"],
            raw_sa_apsp,
            PUBLICATION_MARKERS["raw_sa"],
        )
    ]
    for name in ("robbin_mr2s", "iterated_local_search_mr2s"):
        if name in (mr2s_variants or {}):
            _, color, label = MR2S_VARIANT_STYLES[name]
            series.append(
                _solver_series(
                    sizes,
                    name,
                    label,
                    color,
                    mr2s_variants[name].get("apsp", []),
                    PUBLICATION_MARKERS[name],
                )
            )
    cluster_section = (dnc_strategies or {}).get("embedding_aware", {"apsp": clustered_apsp})
    series.append(
        _solver_series(
            sizes,
            "embedding_aware",
            SOLVER_DISPLAY_NAMES["embedding_aware"],
            PUBLICATION_COLORS["ours"],
            cluster_section.get("apsp", clustered_apsp),
            PUBLICATION_MARKERS["embedding_aware"],
        )
    )
    return _publication_line_figure(
        sizes,
        series,
        ylabel="Normalized APSP (lower is better)",
        title="APSP Objective by Solver",
        save_path=save_path,
    )


def plot_flow_stability(
    sizes: list[int],
    random_flow: list[float],
    raw_sa_flow: list[float],
    global_flow: list[float],
    clustered_flow: list[float],
    mr2s_variants: dict[str, dict] | None = None,
    dnc_strategies: dict[str, dict] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot flow imbalance for poster/paper comparison; lower is better."""
    del random_flow, global_flow
    series = [
        _solver_series(
            sizes,
            "raw_sa",
            SOLVER_DISPLAY_NAMES["raw_sa"],
            PUBLICATION_COLORS["traditional_raw"],
            raw_sa_flow,
            PUBLICATION_MARKERS["raw_sa"],
        )
    ]
    for name in ("robbin_mr2s", "iterated_local_search_mr2s"):
        if name in (mr2s_variants or {}):
            _, color, label = MR2S_VARIANT_STYLES[name]
            series.append(
                _solver_series(
                    sizes,
                    name,
                    label,
                    color,
                    mr2s_variants[name].get("flow", []),
                    PUBLICATION_MARKERS[name],
                )
            )
    cluster_section = (dnc_strategies or {}).get("embedding_aware", {"flow": clustered_flow})
    series.append(
        _solver_series(
            sizes,
            "embedding_aware",
            SOLVER_DISPLAY_NAMES["embedding_aware"],
            PUBLICATION_COLORS["ours"],
            cluster_section.get("flow", clustered_flow),
            PUBLICATION_MARKERS["embedding_aware"],
        )
    )
    return _publication_line_figure(
        sizes,
        series,
        ylabel="Flow imbalance score (lower is better)",
        title="Flow Stability by Solver",
        save_path=save_path,
    )


def plot_preprocessing_scalability(
    sizes: list[int],
    global_vars: list[float],
    clustered_vars: list[float],
    global_sg: list[float],
    clustered_sg: list[float],
    global_physical: list[float] | None = None,
    clustered_physical_total: list[float] | None = None,
    clustered_physical_max: list[float] | None = None,
    clustered_physical_mean: list[float] | None = None,
    clustered_physical_min: list[float] | None = None,
    mr2s_variants: dict[str, dict] | None = None,
    dnc_strategies: dict[str, dict] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot one BQM binary-variable scalability comparison."""
    del global_sg, global_physical, clustered_physical_total, mr2s_variants
    cluster_section = (dnc_strategies or {}).get(
        "embedding_aware",
        {
            "qubo_vars": clustered_vars,
            "subgraph_size": clustered_sg,
            "qvars_mean": clustered_physical_mean,
            "qvars_min": clustered_physical_min,
        },
    )
    series = [
        _solver_series(
            sizes,
            "global",
            BQM_DISPLAY_NAMES["global"],
            PUBLICATION_COLORS["global"],
            global_vars,
            PUBLICATION_MARKERS["global"],
        ),
        _solver_series(
            sizes,
            "embedding_aware_sum",
            BQM_DISPLAY_NAMES["embedding_aware_sum"],
            PUBLICATION_COLORS["ours"],
            cluster_section.get("qubo_vars", clustered_vars),
            "D",
        ),
        _solver_series(
            sizes,
            "embedding_aware_max",
            BQM_DISPLAY_NAMES["embedding_aware_max"],
            PUBLICATION_COLORS["ours_light"],
            cluster_section.get("subgraph_size", clustered_sg),
            "^",
        ),
        _solver_series(
            sizes,
            "embedding_aware_avg",
            BQM_DISPLAY_NAMES["embedding_aware_avg"],
            "#D98D89",
            cluster_section.get("qvars_mean", clustered_physical_mean),
            "o",
        ),
        _solver_series(
            sizes,
            "embedding_aware_min",
            BQM_DISPLAY_NAMES["embedding_aware_min"],
            "#F0B7B2",
            cluster_section.get("qvars_min", clustered_physical_min),
            "v",
        ),
    ]
    return _publication_line_figure(
        sizes,
        series,
        ylabel="BQM binary variable count",
        title="BQM Size Scaling",
        save_path=save_path,
        use_axis_break=False,
    )


def plot_spent_time(
    sizes: list[int],
    graph_time: list[float],
    raw_sa_time: list[float],
    global_solve_time: list[float],
    global_embed_time: list[float],
    clustered_solve_time: list[float],
    clustered_embed_time: list[float],
    random_time: list[float],
    mr2s_variant_timings: dict[str, list[float]] | None = None,
    dnc_timings: dict[str, list[float]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot solver runtime for poster/paper comparison."""
    del graph_time, global_solve_time, global_embed_time, random_time
    series = [
        _solver_series(
            sizes,
            "raw_sa",
            SOLVER_DISPLAY_NAMES["raw_sa"],
            PUBLICATION_COLORS["traditional_raw"],
            raw_sa_time,
            PUBLICATION_MARKERS["raw_sa"],
        )
    ]
    for name in ("robbin_mr2s", "iterated_local_search_mr2s"):
        values = (mr2s_variant_timings or {}).get(f"{name}_solve", [])
        if values:
            _, color, label = MR2S_VARIANT_STYLES[name]
            series.append(
                _solver_series(
                    sizes,
                    name,
                    label,
                    color,
                    values,
                    PUBLICATION_MARKERS[name],
                )
            )
    cluster_values = (dnc_timings or {}).get("dnc_embedding_aware_solve", clustered_solve_time)
    cluster_embed_values = (dnc_timings or {}).get("dnc_embedding_aware_embed", clustered_embed_time)
    series.append(
        _solver_series(
            sizes,
            "embedding_aware",
            SOLVER_DISPLAY_NAMES["embedding_aware"],
            PUBLICATION_COLORS["ours"],
            _sum_aligned_values(sizes, cluster_values, cluster_embed_values),
            PUBLICATION_MARKERS["embedding_aware"],
        )
    )
    return _publication_line_figure(
        sizes,
        series,
        ylabel="Mean runtime (seconds)",
        title="Solver Runtime by Graph Size",
        save_path=save_path,
        use_axis_break=False,
    )
