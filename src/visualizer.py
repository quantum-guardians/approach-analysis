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

DNC_STRATEGY_STYLES = {
    "poster": ("D-", "red", "DnC poster"),
    "embedding_aware": ("^-", "green", "DnC embedding-aware"),
    "degeneracy_pruning": ("v-", "purple", "DnC degeneracy-pruning"),
}
MR2S_VARIANT_STYLES = {
    "robbin_mr2s": ("P-", "brown", "Robbin + MR2S"),
    "iterated_local_search_mr2s": ("*-", "teal", "ILS + MR2S"),
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
    """Plot APSP reduction comparing Random, Raw SA, Global (QUBO), and Clustered (DnC)."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(sizes, random_apsp, "o-", label="Random Orientation", color="gray", alpha=0.6)
    plt.plot(sizes, raw_sa_apsp, "x--", label="Raw SA (SAMR2SSolver)", color="orange")
    plt.plot(sizes, global_apsp, "s-", label="Global (QuboMR2SSolver)", color="blue")
    for name, section in (mr2s_variants or {}).items():
        marker, color, label = MR2S_VARIANT_STYLES.get(name, ("P-", None, name.replace("_", " ")))
        plt.plot(sizes, section.get("apsp", []), marker, label=label, color=color, linewidth=2)
    strategies = dnc_strategies or {"poster": {"apsp": clustered_apsp}}
    for name, section in strategies.items():
        marker, color, label = DNC_STRATEGY_STYLES.get(name, ("D-", None, f"DnC {name}"))
        plt.plot(sizes, section.get("apsp", []), marker, label=label, color=color, linewidth=2)
    plt.xlabel("Graph Size (Number of Vertices)", fontsize=12)
    plt.ylabel("Normalized APSP Sum", fontsize=12)
    plt.title("APSP Reduction: Comparison of Solvers", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


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
    """Plot Flow Stability comparing Random, Raw SA, Global (QUBO), and Clustered (DnC)."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(sizes, random_flow, "o-", label="Random Orientation", color="gray", alpha=0.6)
    plt.plot(sizes, raw_sa_flow, "x--", label="Raw SA (SAMR2SSolver)", color="orange")
    plt.plot(sizes, global_flow, "s-", label="Global (QuboMR2SSolver)", color="blue")
    for name, section in (mr2s_variants or {}).items():
        marker, color, label = MR2S_VARIANT_STYLES.get(name, ("P-", None, name.replace("_", " ")))
        plt.plot(sizes, section.get("flow", []), marker, label=label, color=color, linewidth=2)
    strategies = dnc_strategies or {"poster": {"flow": clustered_flow}}
    for name, section in strategies.items():
        marker, color, label = DNC_STRATEGY_STYLES.get(name, ("D-", None, f"DnC {name}"))
        plt.plot(sizes, section.get("flow", []), marker, label=label, color=color, linewidth=2)
    plt.xlabel("Graph Size (Number of Vertices)", fontsize=12)
    plt.ylabel("Flow Imbalance Score Σ(In-Out)²", fontsize=12)
    plt.title("Flow Stability: Comparison of Solvers", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


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
    """Plot Preprocessing Scalability (Global QUBO vs Clustered DnC)."""
    n_panels = 3 if global_physical is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # 1. QUBO Variables
    ax1 = axes[0]
    ax1.plot(sizes, global_vars, "s-", label="Global (Full Graph)", color="blue")
    for name, section in (mr2s_variants or {}).items():
        marker, color, label = MR2S_VARIANT_STYLES.get(name, ("P-", None, name.replace("_", " ")))
        ax1.plot(sizes, section.get("qubo_vars", []), marker, label=label, color=color, linewidth=2)
    strategies = dnc_strategies or {"poster": {"qubo_vars": clustered_vars, "subgraph_size": clustered_sg}}
    for name, section in strategies.items():
        marker, color, label = DNC_STRATEGY_STYLES.get(name, ("D-", None, f"DnC {name}"))
        ax1.plot(sizes, section.get("qubo_vars", []), marker, label=label, color=color, linewidth=2)
    ax1.set_xlabel("Graph Size (Vertices)", fontsize=12)
    ax1.set_ylabel("Number of QUBO Variables", fontsize=12)
    ax1.set_title("QUBO Complexity Reduction", fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 2. Subgraph Size
    ax2 = axes[1]
    ax2.plot(sizes, global_sg, "s-", label="Global (Full Graph)", color="blue")
    for name, section in (mr2s_variants or {}).items():
        marker, color, label = MR2S_VARIANT_STYLES.get(name, ("P-", None, name.replace("_", " ")))
        ax2.plot(sizes, section.get("subgraph_size", []), marker, label=label, color=color, linewidth=2)
    for name, section in strategies.items():
        marker, color, label = DNC_STRATEGY_STYLES.get(name, ("D-", None, f"DnC {name}"))
        ax2.plot(sizes, section.get("subgraph_size", []), marker, label=label, color=color, linewidth=2)
    ax2.set_xlabel("Graph Size (Vertices)", fontsize=12)
    ax2.set_ylabel("Max QUBO Variables in One Subgraph", fontsize=12)
    ax2.set_title("Subgraph Size Reduction", fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # 3. Physical Qubits (Detailed)
    if global_physical is not None and clustered_physical_total is not None:
        ax3 = axes[2]
        ax3.plot(sizes, global_physical, "s-", label="Global Total", color="blue", alpha=0.8)
        for name, section in (mr2s_variants or {}).items():
            marker, color, label = MR2S_VARIANT_STYLES.get(name, ("P-", None, name.replace("_", " ")))
            ax3.plot(sizes, section.get("phys_total", []), marker, label=f"{label} total", color=color, linewidth=2)
        for name, section in strategies.items():
            marker, color, label = DNC_STRATEGY_STYLES.get(name, ("D-", None, f"DnC {name}"))
            ax3.plot(sizes, section.get("phys_total", []), marker, label=f"{label} total", color=color, linewidth=2)

        if clustered_physical_max:
            ax3.plot(sizes, clustered_physical_max, "^--", label="Clustered Max", color="orange")
        if clustered_physical_mean:
            ax3.plot(sizes, clustered_physical_mean, "x--", label="Clustered Mean", color="green")
        if clustered_physical_min:
            ax3.plot(sizes, clustered_physical_min, "v--", label="Clustered Min", color="purple")

        ax3.set_xlabel("Graph Size (Vertices)", fontsize=12)
        ax3.set_ylabel("Physical Qubit Estimate (Pegasus)", fontsize=12)
        ax3.set_title("Physical Resource Scalability", fontsize=14)
        ax3.legend(fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


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
    """Plot mean spent time by graph size for each poster-results stage."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(sizes, graph_time, "o-", label="Graph generation", color="black", alpha=0.7)
    plt.plot(sizes, raw_sa_time, "x--", label="Raw SA", color="orange")
    plt.plot(sizes, global_solve_time, "s-", label="Global solve", color="blue")
    for name, (marker, color, label) in MR2S_VARIANT_STYLES.items():
        values = (mr2s_variant_timings or {}).get(f"{name}_solve", [])
        if values:
            plt.plot(sizes, values, marker, label=f"{label} solve", color=color, linewidth=2)
    timings = dnc_timings or {"dnc_poster_solve": clustered_solve_time}
    for name, (marker, color, label) in DNC_STRATEGY_STYLES.items():
        values = timings.get(f"dnc_{name}_solve", [])
        if values:
            plt.plot(sizes, values, marker, label=f"{label} solve", color=color, linewidth=2)
    plt.plot(sizes, random_time, "o--", label="Random baseline", color="gray", alpha=0.7)
    plt.xlabel("Graph Size (Vertices)", fontsize=12)
    plt.ylabel("Mean Spent Time (seconds)", fontsize=12)
    plt.title("Spent Time by Stage", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
