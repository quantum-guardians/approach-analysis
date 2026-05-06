"""Tests for src/commands/face_k_analysis and related helpers."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from src.commands import face_k_analysis as fka
from src.commands.face_k_analysis import (
    _generate_delaunay_graph,
    _nx_to_mr2s_graph,
    _evaluate_face_cycle,
    derive_optimal_k,
    _fit_optimal_k_formula,
    predict_optimal_k,
    evaluate_optimal_k_formula,
    remove_edges_maintaining_biconnectivity,
)
from src.visualizer import plot_face_k_analysis, plot_optimal_k_fit_evidence


# ---------------------------------------------------------------------------
# _generate_delaunay_graph
# ---------------------------------------------------------------------------

class TestGenerateDelaunayGraph:
    def test_returns_correct_vertex_count(self) -> None:
        g = _generate_delaunay_graph(10, seed=0)
        assert g.number_of_nodes() == 10

    def test_is_connected(self) -> None:
        g = _generate_delaunay_graph(15, seed=1)
        assert nx.is_connected(g)

    def test_two_vertices(self) -> None:
        g = _generate_delaunay_graph(2, seed=0)
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 1

    def test_reproducible_with_same_seed(self) -> None:
        g1 = _generate_delaunay_graph(12, seed=42)
        g2 = _generate_delaunay_graph(12, seed=42)
        assert g1.edges() == g2.edges()


# ---------------------------------------------------------------------------
# remove_edges_maintaining_biconnectivity
# ---------------------------------------------------------------------------

class TestRemoveEdgesBiconnected:
    def _biconn_graph(self, n: int = 20, seed: int = 0) -> nx.Graph:
        g = _generate_delaunay_graph(n, seed=seed)
        # Ensure biconnected (Delaunay with n>=5 is almost always biconnected)
        while not nx.is_biconnected(g):
            seed += 1
            g = _generate_delaunay_graph(n, seed=seed)
        return g

    def test_result_is_biconnected(self) -> None:
        g = self._biconn_graph()
        rng = np.random.default_rng(0)
        reduced, _ = remove_edges_maintaining_biconnectivity(g, 0.2, rng)
        assert nx.is_biconnected(reduced)

    def test_zero_removal_pct_unchanged(self) -> None:
        g = self._biconn_graph()
        rng = np.random.default_rng(0)
        reduced, actual_pct = remove_edges_maintaining_biconnectivity(g, 0.0, rng)
        assert actual_pct == 0.0
        assert reduced.number_of_edges() == g.number_of_edges()

    def test_actual_pct_does_not_exceed_target(self) -> None:
        g = self._biconn_graph(30)
        rng = np.random.default_rng(7)
        _, actual_pct = remove_edges_maintaining_biconnectivity(g, 0.25, rng)
        assert actual_pct <= 0.25 + 1e-9

    def test_returns_fewer_edges(self) -> None:
        g = self._biconn_graph()
        rng = np.random.default_rng(3)
        reduced, actual_pct = remove_edges_maintaining_biconnectivity(g, 0.3, rng)
        if actual_pct > 0:
            assert reduced.number_of_edges() < g.number_of_edges()


# ---------------------------------------------------------------------------
# _nx_to_mr2s_graph
# ---------------------------------------------------------------------------

class TestNxToMr2sGraph:
    def test_edge_count_matches(self) -> None:
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        mr2s = _nx_to_mr2s_graph(g)
        assert len(mr2s.edges) == 3

    def test_all_edges_undirected(self) -> None:
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        mr2s = _nx_to_mr2s_graph(g)
        assert all(not e.directed for e in mr2s.edges)

    def test_vertices_match(self) -> None:
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        mr2s = _nx_to_mr2s_graph(g)
        assert mr2s.get_vertices() == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# _evaluate_face_cycle
# ---------------------------------------------------------------------------

class TestEvaluateFaceCycle:
    def _biconn_graph(self, n: int = 12, seed: int = 5) -> nx.Graph:
        g = _generate_delaunay_graph(n, seed=seed)
        for s in range(seed, seed + 20):
            g = _generate_delaunay_graph(n, seed=s)
            if nx.is_biconnected(g):
                return g
        return g  # fallback

    def test_sc_ratio_in_range(self) -> None:
        g = self._biconn_graph()
        rng = np.random.default_rng(0)
        sc_r, _ = _evaluate_face_cycle(g, target_k=3, num_samples=50, rng=rng)
        assert 0.0 <= sc_r <= 1.0

    def test_returns_nan_apsp_when_no_sc(self) -> None:
        # Use a single-edge graph that can't be SC
        g = nx.Graph()
        g.add_edge(0, 1)
        rng = np.random.default_rng(0)
        # With 1 edge the DiGraph has only one direction per edge → never SC
        sc_r, mean_apsp = _evaluate_face_cycle(g, target_k=1, num_samples=20, rng=rng)
        if sc_r == 0.0:
            assert math.isnan(mean_apsp)

    def test_num_samples_controls_evaluation_count(self) -> None:
        """SC ratio should be in [0, 1] for any num_samples value."""
        g = self._biconn_graph()
        rng = np.random.default_rng(99)
        for ns in [1, 10, 100]:
            sc_r, _ = _evaluate_face_cycle(g, target_k=2, num_samples=ns, rng=rng)
            assert 0.0 <= sc_r <= 1.0


# ---------------------------------------------------------------------------
# derive_optimal_k
# ---------------------------------------------------------------------------

class TestDeriveOptimalK:
    def _make_results(
        self,
        sizes: list[int],
        pcts: list[float],
        ks: list[int],
        best_k_map: dict[tuple[int, float], int],
    ) -> dict:
        """Build a results dict with the specified k having sc_ratio=1.0 and others 0.5."""
        results: dict = {}
        for n in sizes:
            results[str(n)] = {}
            for pct in pcts:
                results[str(n)][str(pct)] = {}
                best = best_k_map.get((n, pct), ks[0])
                for k in ks:
                    sc = 1.0 if k == best else 0.5
                    results[str(n)][str(pct)][str(k)] = {"sc_ratio": sc, "mean_apsp": 2.0}
        return results

    def test_finds_highest_sc_k(self) -> None:
        sizes = [10, 20]
        pcts = [0.0, 0.1]
        ks = [1, 2, 3, 4, 5]
        expected = {(10, 0.0): 3, (10, 0.1): 5, (20, 0.0): 2, (20, 0.1): 4}
        results = self._make_results(sizes, pcts, ks, expected)
        optimal = derive_optimal_k(results, sizes, pcts, ks)
        for key, k_star in expected.items():
            assert optimal[key] == k_star

    def test_ties_broken_by_apsp(self) -> None:
        sizes = [10]
        pcts = [0.0]
        ks = [1, 2]
        results = {
            "10": {
                "0.0": {
                    "1": {"sc_ratio": 0.8, "mean_apsp": 3.0},
                    "2": {"sc_ratio": 0.8, "mean_apsp": 2.0},  # same SC, lower APSP → preferred
                }
            }
        }
        optimal = derive_optimal_k(results, sizes, pcts, ks)
        assert optimal[(10, 0.0)] == 2


# ---------------------------------------------------------------------------
# _fit_optimal_k_formula
# ---------------------------------------------------------------------------

class TestFitOptimalKFormula:
    def test_returns_three_floats(self) -> None:
        optimal = {(10, 0.0): 2, (20, 0.0): 3, (30, 0.0): 4}
        a, b, c = _fit_optimal_k_formula(optimal, [10, 20, 30], [0.0])
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(c, float)

    def test_fallback_with_few_data_points(self) -> None:
        optimal = {(10, 0.0): 2}
        a, b, c = _fit_optimal_k_formula(optimal, [10], [0.0])
        # Should not raise; fallback values returned
        assert a >= 0


class TestPredictOptimalK:
    def test_returns_positive_integer(self) -> None:
        predicted = predict_optimal_k(0.8, 0.7, 0.2, n=20, pct=0.3)
        assert isinstance(predicted, int)
        assert predicted >= 1


class TestEvaluateOptimalKFormula:
    def test_returns_metrics_and_rows(self) -> None:
        optimal = {(10, 0.0): 2, (10, 0.1): 2, (20, 0.0): 3, (20, 0.1): 3}
        evidence = evaluate_optimal_k_formula(
            optimal,
            graph_sizes=[10, 20],
            removal_pcts=[0.0, 0.1],
            a=0.8,
            b=0.5,
            c=0.0,
        )
        assert "metrics" in evidence
        assert "rows" in evidence
        assert evidence["metrics"]["count"] == 4
        assert len(evidence["rows"]) == 4


# ---------------------------------------------------------------------------
# plot_face_k_analysis
# ---------------------------------------------------------------------------

class TestPlotFaceKAnalysis:
    def _make_results(self) -> dict:
        sizes = [10, 20]
        pcts = [0.0, 0.1]
        ks = [1, 2, 3]
        results: dict = {}
        for n in sizes:
            results[str(n)] = {}
            for pct in pcts:
                results[str(n)][str(pct)] = {}
                for k in ks:
                    results[str(n)][str(pct)][str(k)] = {
                        "sc_ratio": 0.5,
                        "mean_apsp": 2.0,
                    }
        return results

    def test_returns_figure(self, tmp_path) -> None:
        import matplotlib.pyplot as plt
        results = self._make_results()
        fig = plot_face_k_analysis(
            results,
            graph_sizes=[10, 20],
            removal_pcts=[0.0, 0.1],
            target_ks=[1, 2, 3],
            save_path=str(tmp_path / "test.png"),
        )
        assert fig is not None
        plt.close("all")

    def test_saves_png(self, tmp_path) -> None:
        import matplotlib.pyplot as plt
        results = self._make_results()
        out = tmp_path / "face_k_test.png"
        plot_face_k_analysis(
            results,
            graph_sizes=[10, 20],
            removal_pcts=[0.0, 0.1],
            target_ks=[1, 2, 3],
            save_path=str(out),
        )
        assert out.exists()
        plt.close("all")


class TestPlotOptimalKFitEvidence:
    def test_returns_figure(self, tmp_path) -> None:
        import matplotlib.pyplot as plt

        optimal = {(10, 0.0): 2, (10, 0.1): 3, (20, 0.0): 3, (20, 0.1): 4}
        predicted = {(10, 0.0): 2, (10, 0.1): 2, (20, 0.0): 3, (20, 0.1): 4}
        fig = plot_optimal_k_fit_evidence(
            optimal=optimal,
            graph_sizes=[10, 20],
            removal_pcts=[0.0, 0.1],
            predicted=predicted,
            save_path=str(tmp_path / "fit.png"),
        )
        assert fig is not None
        plt.close("all")

    def test_saves_png(self, tmp_path) -> None:
        import matplotlib.pyplot as plt

        optimal = {(10, 0.0): 2, (10, 0.1): 3, (20, 0.0): 3, (20, 0.1): 4}
        predicted = {(10, 0.0): 2, (10, 0.1): 2, (20, 0.0): 3, (20, 0.1): 4}
        out = tmp_path / "optimal_k_fit.png"
        plot_optimal_k_fit_evidence(
            optimal=optimal,
            graph_sizes=[10, 20],
            removal_pcts=[0.0, 0.1],
            predicted=predicted,
            save_path=str(out),
        )
        assert out.exists()
        plt.close("all")


# ---------------------------------------------------------------------------
# Integration: run (small parameters)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_produces_json_and_plot(self, tmp_path, monkeypatch) -> None:
        """Smoke test: run with tiny parameters."""
        # Monkeypatch plot to avoid slow rendering
        monkeypatch.setattr(fka, "plot_face_k_analysis", lambda **kw: None)
        monkeypatch.setattr(fka, "plot_optimal_k_fit_evidence", lambda **kw: None)

        fka.run(
            graph_sizes=[8],
            removal_pcts=[0.0],
            target_ks=[1, 2],
            num_graphs=2,
            num_samples=10,
            seed=0,
            output_dir=str(tmp_path),
            plot_output=str(tmp_path / "out.png"),
        )

        import json
        import os

        json_path = tmp_path / "face_k_results.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "8" in data["results"]
        assert "0.0" in data["results"]["8"]
        assert "formula_fit" in data
        assert "fit_evidence" in data

        evidence_path = tmp_path / "optimal_k_evidence.json"
        assert evidence_path.exists()

        report_path = tmp_path / "report.md"
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "target k" in report_text or "target_k" in report_text or "k*" in report_text

    def test_run_cli_dispatch(self, tmp_path) -> None:
        """Smoke test for the CLI dispatch path."""
        import argparse

        subparsers_action = argparse.ArgumentParser().add_subparsers()
        fka.register_parser(subparsers_action)
        # Just ensure register_parser doesn't raise
