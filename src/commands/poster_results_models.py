"""Typed result models for the ``poster-results`` experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BasicSolverResult:
    apsp: float
    flow: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BasicSolverResult":
        return cls(apsp=data["apsp"], flow=data["flow"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "apsp": self.apsp,
            "flow": self.flow,
        }


@dataclass
class GlobalSolverResult(BasicSolverResult):
    qvars: float
    subgraph_size: float
    physical_total: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalSolverResult":
        return cls(
            apsp=data["apsp"],
            flow=data["flow"],
            qvars=data["qvars"],
            subgraph_size=data["sg"],
            physical_total=data["pt"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "apsp": self.apsp,
            "flow": self.flow,
            "qvars": self.qvars,
            "sg": self.subgraph_size,
            "pt": self.physical_total,
        }


@dataclass
class Mr2sSolverResult(BasicSolverResult):
    qvars: float
    subgraph_size: float
    phys_total: float
    phys_max: float
    phys_mean: float
    phys_min: float
    partition: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mr2sSolverResult":
        return cls(
            apsp=data["apsp"],
            flow=data["flow"],
            qvars=data["qvars"],
            subgraph_size=data["sg"],
            phys_total=data["phys_total"],
            phys_max=data["phys_max"],
            phys_mean=data["phys_mean"],
            phys_min=data["phys_min"],
            partition=data.get("partition", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "apsp": self.apsp,
            "flow": self.flow,
            "qvars": self.qvars,
            "sg": self.subgraph_size,
            "phys_total": self.phys_total,
            "phys_max": self.phys_max,
            "phys_mean": self.phys_mean,
            "phys_min": self.phys_min,
            "partition": self.partition,
        }


@dataclass
class RandomBaselineResult(BasicSolverResult):
    sample_count: int | None = None
    strong_sample_count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RandomBaselineResult":
        return cls(
            apsp=data["apsp"],
            flow=data["flow"],
            sample_count=data.get("sample_count"),
            strong_sample_count=data.get("strong_sample_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "apsp": self.apsp,
            "flow": self.flow,
        }
        if self.sample_count is not None:
            data["sample_count"] = self.sample_count
        if self.strong_sample_count is not None:
            data["strong_sample_count"] = self.strong_sample_count
        return data

    def normalized(self) -> "RandomBaselineResult":
        # legacy cache는 sample이 없을 때 0점처럼 저장된 경우가 있어 unavailable로 보정한다.
        missing_legacy_sample = (
            self.sample_count is None
            and self.apsp == 0
            and self.flow == 0
        )
        if self.sample_count == 0 or missing_legacy_sample:
            return RandomBaselineResult(
                apsp=float("nan"),
                flow=float("nan"),
                sample_count=0,
                strong_sample_count=self.strong_sample_count,
            )
        return self


@dataclass
class TrialTimings:
    values: dict[str, float | bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialTimings":
        return cls(values=dict(data))

    def mark_cache_hit(self, cache_hit: bool) -> None:
        self.values["cache_hit"] = cache_hit

    @property
    def cache_hit(self) -> bool:
        return bool(self.values.get("cache_hit"))

    def to_dict(self) -> dict[str, float | bool]:
        return dict(self.values)


@dataclass
class PosterTrialResult:
    raw_sa: BasicSolverResult
    global_result: GlobalSolverResult
    mr2s: Mr2sSolverResult
    random: RandomBaselineResult
    timings: TrialTimings
    mr2s_variants: dict[str, GlobalSolverResult] = field(default_factory=dict)
    dnc_strategies: dict[str, Mr2sSolverResult] = field(default_factory=dict)
    cache_hit: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PosterTrialResult":
        timings = TrialTimings.from_dict(data.get("timings", {}))
        cache_hit = bool(data.get("cache_hit", timings.cache_hit))
        timings.mark_cache_hit(cache_hit)
        mr2s = Mr2sSolverResult.from_dict(data["mr2s"])
        dnc_strategies = {
            name: Mr2sSolverResult.from_dict(result)
            for name, result in data.get("dnc_strategies", {}).items()
        }
        mr2s_variants = {
            name: GlobalSolverResult.from_dict(result)
            for name, result in data.get("mr2s_variants", {}).items()
        }
        if not dnc_strategies:
            # 기존 cache/result에는 strategy별 결과가 없으므로 poster 전략으로 보정한다.
            dnc_strategies["poster"] = mr2s
        return cls(
            raw_sa=BasicSolverResult.from_dict(data["raw_sa"]),
            global_result=GlobalSolverResult.from_dict(data["global"]),
            mr2s=mr2s,
            random=RandomBaselineResult.from_dict(data["random"]),
            timings=timings,
            mr2s_variants=mr2s_variants,
            dnc_strategies=dnc_strategies,
            cache_hit=cache_hit,
        )

    def mark_cache_hit(self, cache_hit: bool) -> None:
        self.cache_hit = cache_hit
        self.timings.mark_cache_hit(cache_hit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_sa": self.raw_sa.to_dict(),
            "global": self.global_result.to_dict(),
            "mr2s": self.mr2s.to_dict(),
            "mr2s_variants": {
                name: result.to_dict()
                for name, result in self.mr2s_variants.items()
            },
            "dnc_strategies": {
                name: result.to_dict()
                for name, result in self.dnc_strategies.items()
            },
            "random": self.random.to_dict(),
            "timings": self.timings.to_dict(),
            "cache_hit": self.cache_hit,
        }


@dataclass
class Mr2sTrialResult:
    mr2s: Mr2sSolverResult
    timings: TrialTimings
    cache_hit: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mr2sTrialResult":
        timings = TrialTimings.from_dict(data.get("timings", {}))
        cache_hit = bool(data.get("cache_hit", timings.cache_hit))
        timings.mark_cache_hit(cache_hit)
        return cls(
            mr2s=Mr2sSolverResult.from_dict(data["mr2s"]),
            timings=timings,
            cache_hit=cache_hit,
        )

    def mark_cache_hit(self, cache_hit: bool) -> None:
        self.cache_hit = cache_hit
        self.timings.mark_cache_hit(cache_hit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mr2s": self.mr2s.to_dict(),
            "timings": self.timings.to_dict(),
            "cache_hit": self.cache_hit,
        }


TrialPayload = PosterTrialResult | Mr2sTrialResult


@dataclass
class BasicResultSeries:
    apsp: list[float] = field(default_factory=list)
    flow: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "apsp": self.apsp,
            "flow": self.flow,
        }


@dataclass
class GlobalResultSeries(BasicResultSeries):
    qubo_vars: list[float] = field(default_factory=list)
    subgraph_size: list[float] = field(default_factory=list)
    phys_total: list[float] = field(default_factory=list)
    phys_max: list[float] = field(default_factory=list)
    phys_mean: list[float] = field(default_factory=list)
    phys_min: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "apsp": self.apsp,
            "flow": self.flow,
            "qubo_vars": self.qubo_vars,
            "subgraph_size": self.subgraph_size,
            "phys_total": self.phys_total,
            "phys_max": self.phys_max,
            "phys_mean": self.phys_mean,
            "phys_min": self.phys_min,
        }


@dataclass
class Mr2sResultSeries(GlobalResultSeries):
    partition: list[list[dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["partition"] = self.partition
        return data


@dataclass
class TimingResultSeries:
    graph: list[float] = field(default_factory=list)
    raw_sa: list[float] = field(default_factory=list)
    global_solve: list[float] = field(default_factory=list)
    global_embed: list[float] = field(default_factory=list)
    clustered_solve: list[float] = field(default_factory=list)
    clustered_embed: list[float] = field(default_factory=list)
    dnc_poster_solve: list[float] = field(default_factory=list)
    dnc_poster_embed: list[float] = field(default_factory=list)
    dnc_embedding_aware_solve: list[float] = field(default_factory=list)
    dnc_embedding_aware_embed: list[float] = field(default_factory=list)
    dnc_degeneracy_pruning_solve: list[float] = field(default_factory=list)
    dnc_degeneracy_pruning_embed: list[float] = field(default_factory=list)
    robbin_mr2s_solve: list[float] = field(default_factory=list)
    robbin_mr2s_embed: list[float] = field(default_factory=list)
    iterated_local_search_mr2s_solve: list[float] = field(default_factory=list)
    iterated_local_search_mr2s_embed: list[float] = field(default_factory=list)
    random: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "graph": self.graph,
            "raw_sa": self.raw_sa,
            "global_solve": self.global_solve,
            "global_embed": self.global_embed,
            "clustered_solve": self.clustered_solve,
            "clustered_embed": self.clustered_embed,
            "dnc_poster_solve": self.dnc_poster_solve,
            "dnc_poster_embed": self.dnc_poster_embed,
            "dnc_embedding_aware_solve": self.dnc_embedding_aware_solve,
            "dnc_embedding_aware_embed": self.dnc_embedding_aware_embed,
            "dnc_degeneracy_pruning_solve": self.dnc_degeneracy_pruning_solve,
            "dnc_degeneracy_pruning_embed": self.dnc_degeneracy_pruning_embed,
            "robbin_mr2s_solve": self.robbin_mr2s_solve,
            "robbin_mr2s_embed": self.robbin_mr2s_embed,
            "iterated_local_search_mr2s_solve": self.iterated_local_search_mr2s_solve,
            "iterated_local_search_mr2s_embed": self.iterated_local_search_mr2s_embed,
            "random": self.random,
        }


@dataclass
class PosterResultsSummary:
    sizes: list[int]
    mr2s: Mr2sResultSeries = field(default_factory=Mr2sResultSeries)
    global_result: GlobalResultSeries = field(default_factory=GlobalResultSeries)
    raw_sa: BasicResultSeries = field(default_factory=BasicResultSeries)
    random: BasicResultSeries = field(default_factory=BasicResultSeries)
    timings: TimingResultSeries = field(default_factory=TimingResultSeries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sizes": self.sizes,
            "mr2s": self.mr2s.to_dict(),
            "global": self.global_result.to_dict(),
            "raw_sa": self.raw_sa.to_dict(),
            "random": self.random.to_dict(),
            "timings": self.timings.to_dict(),
        }


def coerce_trial_payload(data: TrialPayload | dict[str, Any]) -> TrialPayload:
    if isinstance(data, (PosterTrialResult, Mr2sTrialResult)):
        return data
    if "raw_sa" in data:
        return PosterTrialResult.from_dict(data)
    return Mr2sTrialResult.from_dict(data)


def payload_to_dict(data: TrialPayload | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, (PosterTrialResult, Mr2sTrialResult)):
        return data.to_dict()
    return data
