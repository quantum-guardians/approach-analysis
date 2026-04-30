"""Tests for the visualizer module."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.visualizer import plot_score_correlations, plot_nhop_connectivity_comparison


class TestPlotScoreCorrelations:
    def test_returns_figure(self):
        fig = plot_score_correlations(
            apsp_sums=[9.0, 10.0, 11.0],
            nhop_counts={2: [3, 2, 1], 3: [0, 1, 2], 4: [0, 0, 1]},
            title="Test",
            save_path=None,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_subplot_count_matches_hops(self):
        nhop_counts = {2: [1, 2], 3: [3, 4]}
        fig = plot_score_correlations(
            apsp_sums=[5.0, 6.0],
            nhop_counts=nhop_counts,
            title="Test",
            save_path=None,
        )
        assert len(fig.axes) == len(nhop_counts)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "out.png"
        fig = plot_score_correlations(
            apsp_sums=[9.0],
            nhop_counts={2: [3]},
            title="Test",
            save_path=str(out),
        )
        assert out.exists()
        plt.close(fig)

    def test_single_hop(self):
        fig = plot_score_correlations(
            apsp_sums=[1.0, 2.0],
            nhop_counts={2: [4, 5]},
            title="Single hop",
            save_path=None,
        )
        assert len(fig.axes) == 1
        plt.close(fig)


class TestPlotNhopConnectivityComparison:
    def test_returns_figure(self):
        fig = plot_nhop_connectivity_comparison(
            nhop_counts={2: [1, 2, 3], 3: [0, 1, 2]},
            sc_ratios={2: [0.1, 0.3, 0.5], 3: [0.0, 0.2, 0.4]},
            title="Test",
            save_path=None,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_subplot_count_matches_hops(self):
        nhop_counts = {2: [1, 2], 3: [0, 1]}
        sc_ratios = {2: [0.2, 0.4], 3: [0.0, 0.3]}
        fig = plot_nhop_connectivity_comparison(
            nhop_counts=nhop_counts,
            sc_ratios=sc_ratios,
            title="Test",
            save_path=None,
        )
        assert len(fig.axes) == len(nhop_counts)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "nhop.png"
        fig = plot_nhop_connectivity_comparison(
            nhop_counts={2: [1, 2], 3: [0, 1]},
            sc_ratios={2: [0.25, 0.5], 3: [0.0, 0.1]},
            title="Test",
            save_path=str(out),
        )
        assert out.exists()
        plt.close(fig)

    def test_single_hop(self):
        fig = plot_nhop_connectivity_comparison(
            nhop_counts={2: [0, 1, 2]},
            sc_ratios={2: [0.0, 0.5, 1.0]},
            title="Single hop",
            save_path=None,
        )
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_axis_labels(self):
        fig = plot_nhop_connectivity_comparison(
            nhop_counts={2: [1, 2]},
            sc_ratios={2: [0.3, 0.6]},
            title="Labels test",
            save_path=None,
        )
        ax = fig.axes[0]
        assert "2-hop" in ax.get_xlabel()
        assert "SC ratio" in ax.get_ylabel()
        plt.close(fig)

    def test_empty_data(self):
        """Should not raise even with empty sequences."""
        fig = plot_nhop_connectivity_comparison(
            nhop_counts={2: [], 3: []},
            sc_ratios={2: [], 3: []},
            title="Empty",
            save_path=None,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
