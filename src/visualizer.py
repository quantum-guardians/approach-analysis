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
    ax.set_xlabel("Observed optimal k")
    ax.set_ylabel("Predicted optimal k")
    ax.set_title("Observed vs predicted")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig
