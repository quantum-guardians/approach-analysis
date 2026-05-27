"""Task routing and worker execution for poster batch jobs."""

from __future__ import annotations

import time
from typing import Any

from src.commands.poster_batch_queue import Queue
from src.commands.poster_batch_schema import (
    DEFAULT_VISIBILITY_TIMEOUT,
    POSTER_BATCH_SCHEMA_VERSION,
    POSTER_RESULTS_PROBLEM,
)
from src.commands.poster_batch_store import Store
from src.commands.poster_results_solvers import _run_poster_algorithm


def enqueue_tasks(queue: Queue, tasks: list[dict[str, Any]]) -> int:
    return queue.enqueue(tasks)


def write_result(store: Store, task: dict[str, Any], result: dict[str, Any]) -> None:
    store.put_json(task["s3_key"], {
        "schema_version": POSTER_BATCH_SCHEMA_VERSION,
        "task": task,
        "result": result,
        "completed_at": time.time(),
    })


def to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def run_poster_results_task(task: dict[str, Any]) -> dict[str, Any]:
    result, timings = _run_poster_algorithm(
        int(task["n"]),
        int(task["trial"]),
        task.get("seed"),
        task["algorithm"],
    )
    return {
        "problem": POSTER_RESULTS_PROBLEM,
        "algorithm": task["algorithm"],
        "n": task["n"],
        "trial": task["trial"],
        "seed": task.get("seed"),
        "values": to_plain_dict(result),
        "timings": to_plain_dict(timings),
    }


TASK_ROUTERS = {
    POSTER_RESULTS_PROBLEM: run_poster_results_task,
}


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    problem = task.get("problem") or task.get("task_type")
    if problem not in TASK_ROUTERS:
        raise ValueError(f"Unknown batch task problem: {problem}")
    return TASK_ROUTERS[problem](task)


def run_worker(
    queue: Queue,
    store: Store,
    max_tasks: int | None = None,
    block_timeout: int = DEFAULT_VISIBILITY_TIMEOUT,
) -> int:
    def handle_task(task: dict[str, Any]) -> dict[str, Any]:
        result = run_task(task)
        write_result(store, task, result)
        return {"s3_key": task["s3_key"]}

    return queue.subscribe(handle_task, max_tasks=max_tasks, block_timeout=block_timeout)
