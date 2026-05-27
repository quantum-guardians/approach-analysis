"""Shared schema helpers for poster batch tasks and payloads."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Iterable

import numpy as np

from src.cache import generate_cache_key

POSTER_BATCH_SCHEMA_VERSION = 1
POSTER_RESULTS_PROBLEM = "poster-results"
POSTER_BATCH_ALGORITHMS = ("raw_sa", "global", "mr2s", "random")
DEFAULT_QUEUE = "poster-results"
DEFAULT_VISIBILITY_TIMEOUT = 5


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def task_payload(task: dict[str, Any]) -> str:
    return json.dumps(task, sort_keys=True, default=json_default, allow_nan=True)


def decode_task(payload: bytes | str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def normalise_s3_prefix(prefix: str) -> str:
    return prefix.strip("/")


def stable_task_id(n: int, trial: int, seed: int | None, algorithm: str) -> str:
    task_key = generate_cache_key(
        "poster-batch-task",
        version=POSTER_BATCH_SCHEMA_VERSION,
        problem=POSTER_RESULTS_PROBLEM,
        n=n,
        trial=trial,
        seed=seed,
        algorithm=algorithm,
    )
    return hashlib.sha1(task_key.encode("utf-8")).hexdigest()


def build_task(
    n: int,
    trial: int,
    seed: int | None,
    algorithm: str,
    s3_prefix: str,
    max_attempts: int,
) -> dict[str, Any]:
    if algorithm not in POSTER_BATCH_ALGORITHMS:
        raise ValueError(f"Unknown poster algorithm: {algorithm}")

    task_id = stable_task_id(n, trial, seed, algorithm)
    prefix = normalise_s3_prefix(s3_prefix)
    result_key = (
        f"{prefix}/chunks/problem={POSTER_RESULTS_PROBLEM}/algorithm={algorithm}/n={n}/"
        f"trial={trial}/seed={seed if seed is not None else 'none'}/{task_id}.json"
    )
    return {
        "schema_version": POSTER_BATCH_SCHEMA_VERSION,
        "problem": POSTER_RESULTS_PROBLEM,
        "task_type": POSTER_RESULTS_PROBLEM,
        "task_id": task_id,
        "n": n,
        "trial": trial,
        "seed": seed,
        "algorithm": algorithm,
        "attempt": 0,
        "max_attempts": max_attempts,
        "s3_key": result_key,
        "created_at": time.time(),
    }


def build_tasks(
    sizes: Iterable[int],
    num_graphs: int,
    seed: int | None,
    algorithms: Iterable[str],
    s3_prefix: str,
    max_attempts: int,
) -> list[dict[str, Any]]:
    return [
        build_task(n, trial, seed, algorithm, s3_prefix, max_attempts)
        for n in sizes
        for trial in range(num_graphs)
        for algorithm in algorithms
    ]
