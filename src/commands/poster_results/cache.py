"""Cache helpers for the ``poster-results`` command."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

from src.cache import SimpleCache
from src.commands.poster_results.models import (
    BasicSolverResult,
    GlobalSolverResult,
    Mr2sSolverResult,
    Mr2sTrialResult,
    PosterTrialResult,
    RandomBaselineResult,
    TrialTimings,
)
from src.commands.poster_results.solvers.dnc_strategy import DNC_STRATEGIES
from src.commands.poster_results.solvers.mr2s_variant import MR2S_VARIANTS

TrialTask = tuple[int, int, int | None]
CachedTrialTask = tuple[int, int, int | None, str | None]
TrialResult = TypeVar("TrialResult")
AlgorithmResult = BasicSolverResult | GlobalSolverResult | Mr2sSolverResult | RandomBaselineResult
AlgorithmRunner = Callable[[int, int, int | None, str], tuple[AlgorithmResult, TrialTimings]]

FULL_TRIAL_ALGORITHMS = (
    "raw_sa",
    "global",
    "random",
    *MR2S_VARIANTS,
    *DNC_STRATEGIES,
)


class CachedTrialWorker:
    def __init__(
        self,
        worker: Callable[[TrialTask], tuple[int, int, Any]],
        cache_key: Callable[[int, int, int | None], str],
        from_dict: Callable[[dict[str, Any]], Any],
        coerce_result: Callable[[Any], Any],
    ) -> None:
        self.worker = worker
        self.cache_key = cache_key
        self.from_dict = from_dict
        self.coerce_result = coerce_result

    def __call__(self, task: CachedTrialTask) -> tuple[int, int, Any]:
        n, trial, seed, cache_dir = task
        if cache_dir is None:
            n, trial, result = self.worker((n, trial, seed))
            result = self.coerce_result(result)
            result.mark_cache_hit(False)
            return n, trial, result

        cache = SimpleCache(cache_dir)
        cache_key = self.cache_key(n, trial, seed)
        cached_result = cache.get(cache_key)
        if isinstance(cached_result, dict):
            result = self.from_dict(cached_result)
            result.mark_cache_hit(True)
            return n, trial, result

        n, trial, result = self.worker((n, trial, seed))
        result = self.coerce_result(result)
        result.mark_cache_hit(False)
        cache.set(cache_key, result.to_dict())
        return n, trial, result


def cached_trial_result(
    cache_key: Callable[[int, int, int | None], str],
    from_dict: Callable[[dict[str, Any]], TrialResult],
    coerce_result: Callable[[Any], TrialResult],
) -> Callable[
    [Callable[[TrialTask], tuple[int, int, Any]]],
    Callable[[CachedTrialTask], tuple[int, int, TrialResult]],
]:
    def decorator(
        worker: Callable[[TrialTask], tuple[int, int, Any]],
    ) -> Callable[[CachedTrialTask], tuple[int, int, TrialResult]]:
        return CachedTrialWorker(
            worker=worker,
            cache_key=cache_key,
            from_dict=from_dict,
            coerce_result=coerce_result,
        )

    return decorator


def _algorithm_cache(cache_dir: str, algorithm: str) -> SimpleCache:
    return SimpleCache(os.path.join(cache_dir, algorithm))


def _algorithm_result_from_dict(algorithm: str, data: dict[str, Any]) -> AlgorithmResult:
    if algorithm == "raw_sa":
        return BasicSolverResult.from_dict(data)
    if algorithm == "global" or algorithm in MR2S_VARIANTS:
        return GlobalSolverResult.from_dict(data)
    if algorithm == "random":
        return RandomBaselineResult.from_dict(data)
    if algorithm in DNC_STRATEGIES or algorithm == "mr2s":
        return Mr2sSolverResult.from_dict(data)
    raise ValueError(f"Unknown poster algorithm: {algorithm}")


def _timing_keys_for_algorithm(algorithm: str) -> tuple[str, ...]:
    if algorithm == "raw_sa":
        return ("graph", "raw_sa")
    if algorithm == "global":
        return ("graph", "global_solve", "global_embed")
    if algorithm == "random":
        return ("graph", "random")
    if algorithm in MR2S_VARIANTS:
        return ("graph", f"{algorithm}_solve", f"{algorithm}_embed")
    if algorithm == "poster":
        return ("graph", "clustered_solve", "clustered_embed", "dnc_poster_solve", "dnc_poster_embed")
    if algorithm in DNC_STRATEGIES:
        return ("graph", f"dnc_{algorithm}_solve", f"dnc_{algorithm}_embed")
    return ("graph",)


def _filter_timings(timings: TrialTimings, algorithm: str) -> TrialTimings:
    keys = _timing_keys_for_algorithm(algorithm)
    return TrialTimings({
        key: value
        for key, value in timings.to_dict().items()
        if key in keys
    })


def _algorithm_payload(
    algorithm: str,
    result: AlgorithmResult,
    timings: TrialTimings,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "values": result.to_dict(),
        "timings": timings.to_dict(),
    }


def _algorithm_from_payload(payload: dict[str, Any]) -> tuple[AlgorithmResult, TrialTimings]:
    algorithm = payload["algorithm"]
    return (
        _algorithm_result_from_dict(algorithm, payload["values"]),
        TrialTimings.from_dict(payload.get("timings", {})),
    )


def _extract_legacy_algorithm(
    legacy_result: PosterTrialResult,
    algorithm: str,
) -> tuple[AlgorithmResult, TrialTimings] | None:
    # 기존 full-trial cache를 solver별 cache로 옮길 때 필요한 항목만 꺼낸다.
    if algorithm == "raw_sa":
        result: AlgorithmResult = legacy_result.raw_sa
    elif algorithm == "global":
        result = legacy_result.global_result
    elif algorithm == "random":
        result = legacy_result.random
    elif algorithm in MR2S_VARIANTS:
        result = legacy_result.mr2s_variants.get(algorithm)
    elif algorithm in DNC_STRATEGIES:
        result = legacy_result.dnc_strategies.get(algorithm)
        if result is None and algorithm == "poster":
            result = legacy_result.mr2s
    else:
        raise ValueError(f"Unknown poster algorithm: {algorithm}")

    if result is None:
        return None
    return result, _filter_timings(legacy_result.timings, algorithm)


def _merge_algorithm_timings(timing_items: list[TrialTimings], cache_hit: bool) -> TrialTimings:
    merged: dict[str, float | bool] = {}
    for timings in timing_items:
        for key, value in timings.to_dict().items():
            if key == "cache_hit":
                continue
            # graph 생성 시간은 solver마다 중복 기록될 수 있으므로 trial 대표값 하나만 남긴다.
            if key == "graph" and "graph" in merged:
                continue
            merged[key] = value
    result = TrialTimings(merged)
    result.mark_cache_hit(cache_hit)
    return result


def _missing_mr2s_result() -> Mr2sSolverResult:
    return Mr2sSolverResult(
        apsp=float("nan"),
        flow=float("nan"),
        qvars=float("nan"),
        subgraph_size=float("nan"),
        phys_total=float("nan"),
        phys_max=float("nan"),
        phys_mean=float("nan"),
        phys_min=float("nan"),
    )


def _build_trial_result(
    algorithm_results: dict[str, AlgorithmResult],
    timing_items: list[TrialTimings],
    cache_hit: bool,
) -> PosterTrialResult:
    dnc_results = {
        name: result
        for name, result in algorithm_results.items()
        if name in DNC_STRATEGIES and isinstance(result, Mr2sSolverResult)
    }
    mr2s_result = dnc_results.get("poster", _missing_mr2s_result())
    result = PosterTrialResult(
        raw_sa=algorithm_results["raw_sa"],
        global_result=algorithm_results["global"],
        mr2s=mr2s_result,
        random=algorithm_results["random"],
        timings=_merge_algorithm_timings(timing_items, cache_hit),
        mr2s_variants={
            name: result
            for name, result in algorithm_results.items()
            if name in MR2S_VARIANTS and isinstance(result, GlobalSolverResult)
        },
        dnc_strategies=dnc_results,
        cache_hit=cache_hit,
    )
    result.mark_cache_hit(cache_hit)
    return result


class CachedSplitTrialWorker:
    """Run and cache each poster solver independently for one graph trial."""

    def __init__(
        self,
        full_worker: Callable[[TrialTask], tuple[int, int, Any]],
        algorithm_runner: AlgorithmRunner,
        algorithm_cache_key: Callable[[int, int, int | None, str], str],
        legacy_trial_cache_key: Callable[[int, int, int | None], str],
        full_from_dict: Callable[[dict[str, Any]], PosterTrialResult],
        coerce_full_result: Callable[[Any], PosterTrialResult],
        algorithms: tuple[str, ...] = FULL_TRIAL_ALGORITHMS,
    ) -> None:
        self.full_worker = full_worker
        self.algorithm_runner = algorithm_runner
        self.algorithm_cache_key = algorithm_cache_key
        self.legacy_trial_cache_key = legacy_trial_cache_key
        self.full_from_dict = full_from_dict
        self.coerce_full_result = coerce_full_result
        self.algorithms = algorithms

    def __call__(self, task: CachedTrialTask) -> tuple[int, int, PosterTrialResult]:
        n, trial, seed, cache_dir = task
        if cache_dir is None:
            # cache를 끈 경우에는 기존 monolithic trial 실행 경로를 유지한다.
            n, trial, result = self.full_worker((n, trial, seed))
            result = self.coerce_full_result(result)
            result.mark_cache_hit(False)
            return n, trial, result

        legacy_cache = SimpleCache(cache_dir)
        legacy_result: PosterTrialResult | None = None
        algorithm_results: dict[str, AlgorithmResult] = {}
        timing_items: list[TrialTimings] = []
        cache_hits: list[bool] = []

        for algorithm in self.algorithms:
            # solver별 cache hit/miss를 독립적으로 판단해 새 solver만 추가 실행할 수 있게 한다.
            result, timings, cache_hit = self._load_or_run_algorithm(
                n=n,
                trial=trial,
                seed=seed,
                cache_dir=cache_dir,
                legacy_cache=legacy_cache,
                legacy_result=legacy_result,
                algorithm=algorithm,
            )
            if legacy_result is None:
                legacy_result = self._load_legacy_result(legacy_cache, n, trial, seed)
            algorithm_results[algorithm] = result
            timing_items.append(timings)
            cache_hits.append(cache_hit)

        return n, trial, _build_trial_result(
            algorithm_results,
            timing_items,
            cache_hit=all(cache_hits),
        )

    def _load_or_run_algorithm(
        self,
        n: int,
        trial: int,
        seed: int | None,
        cache_dir: str,
        legacy_cache: SimpleCache,
        legacy_result: PosterTrialResult | None,
        algorithm: str,
    ) -> tuple[AlgorithmResult, TrialTimings, bool]:
        cache = _algorithm_cache(cache_dir, algorithm)
        key = self.algorithm_cache_key(n, trial, seed, algorithm)
        cached_payload = cache.get(key)
        if isinstance(cached_payload, dict):
            result, timings = _algorithm_from_payload(cached_payload)
            return result, timings, True

        if legacy_result is None:
            legacy_result = self._load_legacy_result(legacy_cache, n, trial, seed)
        if legacy_result is not None:
            extracted = _extract_legacy_algorithm(legacy_result, algorithm)
            if extracted is not None:
                result, timings = extracted
                # migration은 lazy하게 수행한다: 요청된 trial/solver만 새 위치에 저장한다.
                cache.set(key, _algorithm_payload(algorithm, result, timings))
                return result, timings, True

        result, timings = self.algorithm_runner(n, trial, seed, algorithm)
        cache.set(key, _algorithm_payload(algorithm, result, timings))
        return result, timings, False

    def _load_legacy_result(
        self,
        cache: SimpleCache,
        n: int,
        trial: int,
        seed: int | None,
    ) -> PosterTrialResult | None:
        legacy_payload = cache.get(self.legacy_trial_cache_key(n, trial, seed))
        if not isinstance(legacy_payload, dict):
            return None
        return self.full_from_dict(legacy_payload)


class CachedPosterAlgorithmWorker:
    """Cache a single poster algorithm, used by MR2S-only mode."""

    def __init__(
        self,
        algorithm: str,
        fallback_worker: Callable[[TrialTask], tuple[int, int, Any]],
        algorithm_runner: AlgorithmRunner,
        algorithm_cache_key: Callable[[int, int, int | None, str], str],
        legacy_mr2s_cache_key: Callable[[int, int, int | None], str],
        legacy_trial_cache_key: Callable[[int, int, int | None], str],
        mr2s_from_dict: Callable[[dict[str, Any]], Mr2sTrialResult],
        full_from_dict: Callable[[dict[str, Any]], PosterTrialResult],
        coerce_mr2s_result: Callable[[Any], Mr2sTrialResult],
    ) -> None:
        self.algorithm = algorithm
        self.fallback_worker = fallback_worker
        self.algorithm_runner = algorithm_runner
        self.algorithm_cache_key = algorithm_cache_key
        self.legacy_mr2s_cache_key = legacy_mr2s_cache_key
        self.legacy_trial_cache_key = legacy_trial_cache_key
        self.mr2s_from_dict = mr2s_from_dict
        self.full_from_dict = full_from_dict
        self.coerce_mr2s_result = coerce_mr2s_result

    def __call__(self, task: CachedTrialTask) -> tuple[int, int, Mr2sTrialResult]:
        n, trial, seed, cache_dir = task
        if cache_dir is None:
            n, trial, result = self.fallback_worker((n, trial, seed))
            result = self.coerce_mr2s_result(result)
            result.mark_cache_hit(False)
            return n, trial, result

        cache = _algorithm_cache(cache_dir, self.algorithm)
        key = self.algorithm_cache_key(n, trial, seed, self.algorithm)
        cached_payload = cache.get(key)
        if isinstance(cached_payload, dict):
            result, timings = _algorithm_from_payload(cached_payload)
            return n, trial, self._mr2s_trial(result, timings, cache_hit=True)

        legacy_cache = SimpleCache(cache_dir)
        legacy_mr2s = legacy_cache.get(self.legacy_mr2s_cache_key(n, trial, seed))
        if isinstance(legacy_mr2s, dict):
            result = self.mr2s_from_dict(legacy_mr2s)
            cache.set(key, _algorithm_payload(self.algorithm, result.mr2s, result.timings))
            result.mark_cache_hit(True)
            return n, trial, result

        legacy_full = legacy_cache.get(self.legacy_trial_cache_key(n, trial, seed))
        if isinstance(legacy_full, dict):
            extracted = _extract_legacy_algorithm(self.full_from_dict(legacy_full), self.algorithm)
            if extracted is not None:
                result, timings = extracted
                cache.set(key, _algorithm_payload(self.algorithm, result, timings))
                return n, trial, self._mr2s_trial(result, timings, cache_hit=True)

        result, timings = self.algorithm_runner(n, trial, seed, self.algorithm)
        cache.set(key, _algorithm_payload(self.algorithm, result, timings))
        return n, trial, self._mr2s_trial(result, timings, cache_hit=False)

    def _mr2s_trial(
        self,
        result: AlgorithmResult,
        timings: TrialTimings,
        cache_hit: bool,
    ) -> Mr2sTrialResult:
        if not isinstance(result, Mr2sSolverResult):
            raise TypeError(f"{self.algorithm} did not return an MR2S solver result")
        trial_result = Mr2sTrialResult(mr2s=result, timings=timings, cache_hit=cache_hit)
        trial_result.mark_cache_hit(cache_hit)
        return trial_result
