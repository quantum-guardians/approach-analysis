"""Visualization module.

Produces scatter-plot matrices (pair plots) that illustrate the correlation
between APSP sum and n-hop neighbour counts across all strongly-connected
orientations of a graph, as well as plots comparing n-hop counts and
strongly-connected orientation ratios across multiple graphs.
"""

from __future__ import annotations

from typing import Sequence

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
    sc_ratios: Sequence[float],
    nhop_avgs: dict[int, Sequence[float]],
    title: str = "N-hop Count vs Connectivity",
    save_path: str | None = None,
) -> plt.Figure:
    """Draw scatter plots comparing avg n-hop neighbour counts and SC ratio.

    One subplot is created per hop value.  The x-axis shows the
    strongly-connected orientation ratio (SC orientations / total orientations)
    and the y-axis shows the average n-hop neighbour count across all
    strongly-connected orientations for that graph.

    Each data point represents one undirected graph instance.

    Args:
        sc_ratios: Strongly-connected orientation ratio for each graph
            (SC count / total orientations).
        nhop_avgs: Mapping from hop distance to a list of average n-hop
            neighbour counts, one per graph (same order as *sc_ratios*).
        title: Super-title displayed above the figure.
        save_path: If provided, the figure is saved to this file path instead
            of being displayed interactively.

    Returns:
        The :class:`matplotlib.figure.Figure` that was created.
    """
    hops = sorted(nhop_avgs.keys())
    n_plots = len(hops)

    fig = plt.figure(figsize=(5 * n_plots, 4))
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(1, n_plots, figure=fig)

    for idx, hop in enumerate(hops):
        ax = fig.add_subplot(gs[0, idx])
        ax.scatter(sc_ratios, nhop_avgs[hop], alpha=0.7, edgecolors="none", s=40)
        ax.set_xlabel("SC ratio (strongly-connected / total)")
        ax.set_ylabel(f"avg {hop}-hop neighbour count")
        ax.set_title(f"{hop}-hop vs SC ratio")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig
