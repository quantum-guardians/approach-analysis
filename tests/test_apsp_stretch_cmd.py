"""Tests for the ``apsp-stretch`` subcommand."""

import argparse
import json
import math
import os

import networkx as nx
import pytest

from src.commands.apsp_stretch import (
    _mean_undirected_distance,
    _trial_seed,
    load_apsp_series,
    run,
    stretch_series,
    undirected_baselines,
)


def _summary_payload() -> dict:
    return {
        "sizes": [10, 20],
        "plots": {
            "apsp_reduction": {
                "output": "apsp_reduction.png",
                "y_label": "Normalized APSP (lower is better)",
                "series": [
                    {
                        "key": "raw_sa",
                        "display_name": "SA",
                        "color": "#2F6CA3",
                        "x": [10, 20],
                        "y": [4.0, None],
                    },
                    {
                        "key": "embedding_aware",
                        "display_name": "Ours",
                        "color": "#C43C39",
                        "x": [10, 20],
                        "y": [2.0, 6.0],
                    },
                ],
            }
        },
    }


def _write_summary(tmp_path) -> str:
    path = os.path.join(tmp_path, "plotted_data_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_summary_payload(), handle)
    return path


class TestTrialSeed:
    """The instance seed must match the poster-results convention."""

    def test_offsets_by_trial_and_size(self):
        assert _trial_seed(100, 0, 42) == 142
        assert _trial_seed(100, 2, 42) == 342
        assert _trial_seed(500, 1, 42) == 642

    def test_none_seed_stays_none(self):
        assert _trial_seed(100, 1, None) is None


class TestMeanUndirectedDistance:
    """Mean shortest-path length over ordered pairs of an undirected graph."""

    def test_path_graph(self):
        # P3: distances 1,1,2 in each direction -> mean 4/3.
        assert _mean_undirected_distance(nx.path_graph(3)) == pytest.approx(4 / 3)

    def test_complete_graph_is_one(self):
        assert _mean_undirected_distance(nx.complete_graph(5)) == pytest.approx(1.0)

    def test_single_node_is_nan(self):
        assert math.isnan(_mean_undirected_distance(nx.empty_graph(1)))


class TestUndirectedBaselines:
    """Baselines are reproducible from the seed and grow with graph size."""

    def test_deterministic_for_same_seed(self):
        first = undirected_baselines([20], trials=2, seed=42)
        second = undirected_baselines([20], trials=2, seed=42)
        assert first == second
        assert first[20]["trial_seeds"] == [62, 162]
        assert len(first[20]["per_trial"]) == 2

    def test_mean_matches_per_trial(self):
        baselines = undirected_baselines([20], trials=3, seed=7)
        per_trial = baselines[20]["per_trial"]
        assert baselines[20]["mean"] == pytest.approx(sum(per_trial) / len(per_trial))

    def test_larger_graphs_have_longer_distances(self):
        baselines = undirected_baselines([20, 80], trials=1, seed=42)
        assert baselines[80]["mean"] > baselines[20]["mean"]


class TestStretchSeries:
    """D_avg is divided by the matching undirected baseline."""

    def test_divides_by_baseline(self):
        baselines = {10: {"mean": 2.0}, 20: {"mean": 3.0}}
        converted = stretch_series(_summary_payload()["plots"]["apsp_reduction"]["series"], baselines)
        assert converted[0]["y"] == [2.0, None]
        assert converted[1]["y"] == [1.0, 2.0]

    def test_keeps_source_d_avg(self):
        baselines = {10: {"mean": 2.0}, 20: {"mean": 3.0}}
        converted = stretch_series(_summary_payload()["plots"]["apsp_reduction"]["series"], baselines)
        assert converted[1]["d_avg"] == [2.0, 6.0]

    def test_missing_baseline_yields_none(self):
        converted = stretch_series(_summary_payload()["plots"]["apsp_reduction"]["series"], {})
        assert converted[1]["y"] == [None, None]


class TestLoadApspSeries:
    """The input record must expose the apsp_reduction plot."""

    def test_reads_sizes_and_series(self, tmp_path):
        sizes, series = load_apsp_series(_write_summary(str(tmp_path)))
        assert sizes == [10, 20]
        assert [entry["key"] for entry in series] == ["raw_sa", "embedding_aware"]

    def test_rejects_record_without_plot(self, tmp_path):
        path = os.path.join(str(tmp_path), "empty.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"sizes": [10], "plots": {}}, handle)
        with pytest.raises(ValueError):
            load_apsp_series(path)


class TestRun:
    """End-to-end: the command writes a figure and a machine-readable record."""

    def test_writes_plot_and_record(self, tmp_path):
        output_dir = os.path.join(str(tmp_path), "out")
        args = argparse.Namespace(
            input=_write_summary(str(tmp_path)),
            sizes=None,
            trials=2,
            seed=42,
            output_dir=output_dir,
        )
        run(args)

        assert os.path.exists(os.path.join(output_dir, "apsp_stretch.png"))
        with open(os.path.join(output_dir, "apsp_stretch.json"), encoding="utf-8") as handle:
            record = json.load(handle)
        assert record["sizes"] == [10, 20]
        assert record["seed"] == 42
        assert set(record["undirected_baseline"]) == {"10", "20"}
        ours = next(entry for entry in record["series"] if entry["key"] == "embedding_aware")
        baseline = record["undirected_baseline"]["10"]["mean"]
        assert ours["y"][0] == pytest.approx(2.0 / baseline)

    def test_size_filter(self, tmp_path):
        output_dir = os.path.join(str(tmp_path), "filtered")
        args = argparse.Namespace(
            input=_write_summary(str(tmp_path)),
            sizes=[20],
            trials=1,
            seed=42,
            output_dir=output_dir,
        )
        run(args)

        with open(os.path.join(output_dir, "apsp_stretch.json"), encoding="utf-8") as handle:
            record = json.load(handle)
        assert record["sizes"] == [20]
        assert all(len(entry["y"]) == 1 for entry in record["series"])
