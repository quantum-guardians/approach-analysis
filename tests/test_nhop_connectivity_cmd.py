"""Tests for src/commands/nhop_connectivity helper utilities."""

import networkx as nx
import pytest

from src.commands import nhop_connectivity
from src.commands.nhop_connectivity import _bin_nhop_buckets


class TestBinNhopBuckets:
    """Tests for the _bin_nhop_buckets helper."""

    def _make_buckets(self, values: list[int]) -> tuple[dict[int, int], dict[int, int]]:
        """Build total/sc dicts where every value is SC."""
        return {v: 1 for v in values}, {v: 1 for v in values}

    def test_no_binning_when_below_threshold(self) -> None:
        total = {1: 10, 2: 20, 3: 5}
        sc = {1: 5, 2: 10}
        x, y = _bin_nhop_buckets(total, sc)
        assert x == [1.0, 2.0, 3.0]
        assert y == pytest.approx([0.5, 0.5, 0.0])

    def test_binning_when_above_threshold(self) -> None:
        # 200 distinct integer values → should be binned into 20 buckets
        total = {i: 1 for i in range(200)}
        sc = {i: 1 for i in range(0, 200, 2)}  # even values are SC
        x, y = _bin_nhop_buckets(total, sc)
        assert len(x) == 20
        assert len(y) == 20
        # Every bin should have exactly 10 values (5 SC, 5 non-SC) → ratio = 0.5
        assert all(abs(r - 0.5) < 1e-9 for r in y)

    def test_x_values_are_bin_midpoints(self) -> None:
        # 200 values from 0 to 199: bin_width = (199-0)/20 = 9.95
        # midpoint of first bin  = 0 + 0.5 * 9.95 = 4.975
        # midpoint of second bin = 0 + 1.5 * 9.95 = 14.925
        total = {i: 1 for i in range(200)}
        sc = {}
        x, y = _bin_nhop_buckets(total, sc)
        bin_width = 199 / 20
        assert x[0] == pytest.approx(0 + 0.5 * bin_width)
        assert x[1] == pytest.approx(0 + 1.5 * bin_width)

    def test_empty_bins_are_omitted(self) -> None:
        # Only 3 values but ask for 20 bins → only 3 non-empty bins
        total = {0: 1, 50: 1, 100: 1}
        sc = {0: 1, 50: 1, 100: 1}
        x, y = _bin_nhop_buckets(total, sc, num_bins=20)
        assert len(x) == 3
        assert all(r == pytest.approx(1.0) for r in y)

    def test_single_value_below_threshold(self) -> None:
        total = {42: 7}
        sc = {42: 3}
        x, y = _bin_nhop_buckets(total, sc)
        assert x == [42.0]
        assert y == pytest.approx([3 / 7])

    def test_sc_count_defaults_to_zero(self) -> None:
        total = {1: 5, 2: 5, 3: 5}
        sc: dict[int, int] = {}
        x, y = _bin_nhop_buckets(total, sc)
        assert y == pytest.approx([0.0, 0.0, 0.0])

    def test_exactly_at_threshold_no_binning(self) -> None:
        total = {i: 1 for i in range(20)}
        sc = {i: 1 for i in range(20)}
        x, y = _bin_nhop_buckets(total, sc)
        # Exactly 20 distinct values → no binning
        assert len(x) == 20
        assert all(r == pytest.approx(1.0) for r in y)

    def test_one_more_than_threshold_triggers_binning(self) -> None:
        total = {i: 1 for i in range(21)}
        sc = {}
        x, y = _bin_nhop_buckets(total, sc)
        # 21 distinct values → binned into 20 buckets
        assert len(x) <= 20


class TestRun:
    def test_run_generates_requested_number_of_orientations(self, monkeypatch) -> None:
        graph = nx.Graph()
        graph.add_edge(0, 1)

        calc_calls = 0

        def fake_generate_graph(
            num_vertices: int, connectivity: float | None, seed: int | None = None
        ) -> nx.Graph:
            return graph

        def fake_calculate_apsp_sum_and_nhop_neighbor_counts(
            dg: nx.DiGraph, hops: tuple[int, ...]
        ) -> tuple[float, dict[int, int]]:
            nonlocal calc_calls
            calc_calls += 1
            return 0.0, {hop: 0 for hop in hops}

        monkeypatch.setattr(nhop_connectivity, "generate_graph", fake_generate_graph)
        monkeypatch.setattr(
            nhop_connectivity,
            "calculate_apsp_sum_and_nhop_neighbor_counts",
            fake_calculate_apsp_sum_and_nhop_neighbor_counts,
        )
        monkeypatch.setattr(
            nhop_connectivity,
            "plot_nhop_connectivity_comparison",
            lambda *args, **kwargs: None,
        )

        nhop_connectivity.run(
            num_vertices=2,
            num_graphs=1,
            num_orientations=5,
            seed=0,
            output=None,
        )

        assert calc_calls == 5
