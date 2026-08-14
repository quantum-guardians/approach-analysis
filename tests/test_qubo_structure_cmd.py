"""Tests for src/commands/qubo_structure helper utilities."""

import math

import pytest

from src.commands.qubo_structure import (
    ARM_HOP2,
    ARM_HOP23,
    StructureRecord,
    _coefficient_spread,
    _mean,
    summarise,
)


class _FakeBqm:
    """Minimal stand-in exposing the linear/quadratic mappings we read."""

    def __init__(self, linear: dict, quadratic: dict) -> None:
        self.linear = linear
        self.quadratic = quadratic


def _record(size: int, trial: int, arm: str, **overrides) -> StructureRecord:
    """Build a StructureRecord with sensible defaults for the fields under test."""
    defaults = dict(
        size=size,
        trial=trial,
        arm=arm,
        edges=10,
        variables=10,
        couplings=20,
        couplings_per_variable=2.0,
        coefficient_max=8.0,
        coefficient_min=2.0,
        coefficient_ratio=4.0,
        apsp_sum=100.0,
        flow_score=5.0,
        strong_connect_rate=1.0,
        build_seconds=0.1,
        solve_seconds=0.2,
    )
    defaults.update(overrides)
    return StructureRecord(**defaults)


class TestCoefficientSpread:
    """Tests for the _coefficient_spread helper."""

    def test_returns_max_and_min_absolute_value(self) -> None:
        bqm = _FakeBqm({"a": -5.0, "b": 2.0}, {("a", "b"): 9.0})
        assert _coefficient_spread(bqm) == (9.0, 2.0)

    def test_ignores_numerically_zero_coefficients(self) -> None:
        bqm = _FakeBqm({"a": 1e-15}, {("a", "b"): 3.0})
        assert _coefficient_spread(bqm) == (3.0, 3.0)

    def test_returns_zeros_when_no_coefficients(self) -> None:
        assert _coefficient_spread(_FakeBqm({}, {})) == (0.0, 0.0)


class TestMean:
    """Tests for the _mean helper."""

    def test_ignores_non_finite_values(self) -> None:
        assert _mean([1.0, float("inf"), 3.0]) == pytest.approx(2.0)

    def test_returns_nan_when_all_values_non_finite(self) -> None:
        assert math.isnan(_mean([float("inf"), float("nan")]))


class TestSummarise:
    """Tests for the summarise aggregator."""

    def test_groups_by_arm_and_size(self) -> None:
        records = [
            _record(100, 0, ARM_HOP23, couplings_per_variable=10.0),
            _record(100, 1, ARM_HOP23, couplings_per_variable=12.0),
            _record(100, 0, ARM_HOP2, couplings_per_variable=4.0),
            _record(200, 0, ARM_HOP23, couplings_per_variable=14.0),
            _record(200, 0, ARM_HOP2, couplings_per_variable=5.0),
        ]
        summary = summarise(records)

        assert summary["sizes"] == [100, 200]
        assert set(summary["arms"]) == {ARM_HOP23, ARM_HOP2}
        assert summary["arms"][ARM_HOP23]["couplings_per_variable"] == pytest.approx(
            [11.0, 14.0]
        )
        assert summary["arms"][ARM_HOP2]["couplings_per_variable"] == pytest.approx(
            [4.0, 5.0]
        )

    def test_non_strongly_connected_trials_do_not_break_aggregation(self) -> None:
        records = [
            _record(100, 0, ARM_HOP23, apsp_sum=float("inf")),
            _record(100, 1, ARM_HOP23, apsp_sum=200.0),
        ]
        summary = summarise(records)
        assert summary["arms"][ARM_HOP23]["apsp_sum"] == pytest.approx([200.0])
