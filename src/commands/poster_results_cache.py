"""Cache decorators for the ``poster-results`` command."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from src.cache import SimpleCache

TrialTask = tuple[int, int, int | None]
CachedTrialTask = tuple[int, int, int | None, str | None]
TrialResult = TypeVar("TrialResult")


class CachedTrialWorker:
    """Pickle-safe callable wrapper for cached trial execution."""

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
            # 캐시를 끈 실행도 progress/JSON 스키마를 동일하게 유지한다.
            n, trial, result = self.worker((n, trial, seed))
            result = self.coerce_result(result)
            result.mark_cache_hit(False)
            return n, trial, result

        cache = SimpleCache(cache_dir)
        cache_key = self.cache_key(n, trial, seed)
        cached_result = cache.get(cache_key)
        if isinstance(cached_result, dict):
            # cache hit 표시는 timing에도 전파되어 progress 출력이 짧게 끝난다.
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
    """Wrap a trial worker with disk cache read/write behavior."""

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
