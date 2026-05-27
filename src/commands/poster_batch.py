"""AWS Batch/Redis/S3 workflow for distributed poster-result computation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from src.commands import poster_results
from src.commands.poster_results_runner import PosterResultsAggregator
from src.commands.poster_results_solvers import _run_poster_algorithm
from src.cache import generate_cache_key

POSTER_BATCH_SCHEMA_VERSION = 1
POSTER_RESULTS_PROBLEM = "poster-results"
POSTER_BATCH_ALGORITHMS = ("raw_sa", "global", "mr2s", "random")
DEFAULT_QUEUE = "poster-results"
DEFAULT_VISIBILITY_TIMEOUT = 5

TaskHandler = Callable[[dict[str, Any]], dict[str, Any] | None]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_task_id(n: int, trial: int, seed: int | None, algorithm: str) -> str:
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


def _normalise_s3_prefix(prefix: str) -> str:
    return prefix.strip("/")


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

    task_id = _stable_task_id(n, trial, seed, algorithm)
    prefix = _normalise_s3_prefix(s3_prefix)
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


def _task_payload(task: dict[str, Any]) -> str:
    return json.dumps(task, sort_keys=True, default=_json_default, allow_nan=True)


def _decode_task(payload: bytes | str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def _redis_status_key(queue_name: str) -> str:
    return f"{queue_name}:status"


class Queue(Protocol):
    def enqueue(self, tasks: list[dict[str, Any]]) -> int:
        """Add tasks to the queue."""

    def subscribe(
        self,
        handler: TaskHandler,
        max_tasks: int | None = None,
        block_timeout: int = DEFAULT_VISIBILITY_TIMEOUT,
    ) -> int:
        """Poll tasks and call *handler* for each task."""


class Store(Protocol):
    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        """Store a JSON payload by key."""

    def iter_json(self, prefix: str) -> Iterable[dict[str, Any]]:
        """Yield JSON payloads under prefix."""


class RedisTaskQueue:
    def __init__(self, redis_client: Any, queue_name: str):
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.status_key = _redis_status_key(queue_name)

    def enqueue(self, tasks: list[dict[str, Any]]) -> int:
        if not tasks:
            return 0

        payloads = [_task_payload(task) for task in tasks]
        self.redis_client.rpush(self.queue_name, *payloads)
        for task in tasks:
            self._set_status(task, {"state": "queued", "task": task})
        return len(tasks)

    def subscribe(
        self,
        handler: TaskHandler,
        max_tasks: int | None = None,
        block_timeout: int = DEFAULT_VISIBILITY_TIMEOUT,
    ) -> int:
        processed = 0

        while max_tasks is None or processed < max_tasks:
            popped = self.redis_client.blpop(self.queue_name, timeout=block_timeout)
            if popped is None:
                break

            _, payload = popped
            task = _decode_task(payload)
            task["attempt"] = int(task.get("attempt", 0)) + 1
            self._set_status(
                task,
                {"state": "running", "task": task, "started_at": time.time()},
            )

            try:
                status = handler(task) or {}
                self._set_status(
                    task,
                    {
                        "state": "succeeded",
                        "task": task,
                        "completed_at": time.time(),
                        **status,
                    },
                )
                processed += 1
            except Exception as exc:
                self._set_status(
                    task,
                    {
                        "state": "failed",
                        "task": task,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at": time.time(),
                    },
                )
                if task["attempt"] < int(task.get("max_attempts", 1)):
                    self.redis_client.rpush(self.queue_name, _task_payload(task))
                else:
                    processed += 1

        return processed

    def _set_status(self, task: dict[str, Any], status: dict[str, Any]) -> None:
        self.redis_client.hset(
            self.status_key,
            task["task_id"],
            _task_payload(status),
        )


class S3Store:
    def __init__(self, s3_client: Any, bucket: str):
        self.s3_client = s3_client
        self.bucket = bucket

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(
                payload,
                indent=2,
                default=_json_default,
                allow_nan=True,
            ).encode("utf-8"),
            ContentType="application/json",
        )

    def iter_json(self, prefix: str) -> Iterable[dict[str, Any]]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".json"):
                    continue
                obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                yield json.loads(obj["Body"].read().decode("utf-8"))


def get_redis_client(redis_url: str | None = None) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("Install redis package to use poster-batch Redis queue.") from exc

    url = redis_url or os.environ.get("POSTER_REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url)


def get_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 package to use poster-batch S3 storage.") from exc

    return boto3.client("s3")


def enqueue_tasks(queue: Queue, tasks: list[dict[str, Any]]) -> int:
    return queue.enqueue(tasks)


def _write_result(store: Store, task: dict[str, Any], result: dict[str, Any]) -> None:
    store.put_json(task["s3_key"], {
        "schema_version": POSTER_BATCH_SCHEMA_VERSION,
        "task": task,
        "result": result,
        "completed_at": time.time(),
    })


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def _run_poster_results_task(task: dict[str, Any]) -> dict[str, Any]:
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
        "values": _to_plain_dict(result),
        "timings": _to_plain_dict(timings),
    }


TASK_ROUTERS = {
    POSTER_RESULTS_PROBLEM: _run_poster_results_task,
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
        _write_result(store, task, result)
        return {"s3_key": task["s3_key"]}

    return queue.subscribe(handle_task, max_tasks=max_tasks, block_timeout=block_timeout)


def _iter_store_json_objects(store: Store, prefix: str) -> Iterable[dict[str, Any]]:
    yield from store.iter_json(
        f"{_normalise_s3_prefix(prefix)}/chunks/problem={POSTER_RESULTS_PROBLEM}/"
    )


def collect_s3_trial_results(
    store: Store,
    prefix: str,
    sizes: list[int],
    num_graphs: int,
    algorithms: Iterable[str],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    algorithms = tuple(algorithms)
    by_trial: dict[tuple[int, int], dict[str, Any]] = {}
    seen: set[tuple[int, int, str]] = set()

    for payload in _iter_store_json_objects(store, prefix):
        result = payload["result"]
        problem = result.get("problem") or payload.get("task", {}).get("problem")
        if problem != POSTER_RESULTS_PROBLEM:
            continue
        n = int(result["n"])
        trial = int(result["trial"])
        algorithm = result["algorithm"]
        if n not in sizes or trial >= num_graphs or algorithm not in algorithms:
            continue
        by_trial.setdefault((n, trial), {})[algorithm] = result["values"]
        seen.add((n, trial, algorithm))

    missing = [
        {"n": n, "trial": trial, "algorithm": algorithm}
        for n in sizes
        for trial in range(num_graphs)
        for algorithm in algorithms
        if (n, trial, algorithm) not in seen
    ]

    trial_results: dict[int, list[dict[str, Any]]] = {n: [] for n in sizes}
    for n in sizes:
        for trial in range(num_graphs):
            row = by_trial.get((n, trial), {})
            if all(algorithm in row for algorithm in POSTER_BATCH_ALGORITHMS):
                trial_results[n].append(
                    {
                        "raw_sa": row["raw_sa"],
                        "global": row["global"],
                        "mr2s": row["mr2s"],
                        "random": row["random"],
                    }
                )

    return trial_results, missing


def collect_and_plot(
    store: Store,
    prefix: str,
    sizes: list[int],
    num_graphs: int,
    output_dir: str,
    allow_missing: bool = False,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    trial_results, missing = collect_s3_trial_results(
        store,
        prefix,
        sizes,
        num_graphs,
        POSTER_BATCH_ALGORITHMS,
    )
    if missing:
        missing_path = os.path.join(output_dir, "missing_tasks.json")
        with open(missing_path, "w") as f:
            json.dump(missing, f, indent=2)
        if not allow_missing:
            raise RuntimeError(f"Missing {len(missing)} poster batch chunk(s); see {missing_path}")

    results = PosterResultsAggregator().aggregate_full(sizes, trial_results)
    with open(os.path.join(output_dir, "poster_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default, allow_nan=True)
    poster_results._plot_results(results, output_dir)
    return results


def _dispatch_enqueue(args: argparse.Namespace) -> None:
    redis_client = get_redis_client(args.redis_url)
    queue = RedisTaskQueue(redis_client, args.queue)
    tasks = build_tasks(
        args.sizes,
        args.num_graphs,
        args.seed,
        args.algorithms,
        args.s3_prefix,
        args.max_attempts,
    )
    count = enqueue_tasks(queue, tasks)
    print(f"Queued {count} poster batch task(s) into {args.queue}.")


def _dispatch_worker(args: argparse.Namespace) -> None:
    redis_client = get_redis_client(args.redis_url)
    s3_client = get_s3_client()
    bucket = args.s3_bucket or os.environ["POSTER_S3_BUCKET"]
    queue = RedisTaskQueue(redis_client, args.queue)
    store = S3Store(s3_client, bucket)
    processed = run_worker(
        queue,
        store,
        args.max_tasks,
        args.block_timeout,
    )
    print(f"Processed {processed} poster batch task(s).")


def _dispatch_collect(args: argparse.Namespace) -> None:
    s3_client = get_s3_client()
    bucket = args.s3_bucket or os.environ["POSTER_S3_BUCKET"]
    store = S3Store(s3_client, bucket)
    collect_and_plot(
        store,
        args.s3_prefix,
        args.sizes,
        args.num_graphs,
        args.output_dir,
        args.allow_missing,
    )
    print(f"Wrote poster batch visualization to {args.output_dir}.")


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("poster-batch", help="Run poster results via Redis, AWS Batch, and S3.")
    mode = p.add_subparsers(dest="mode", required=True)

    enqueue = mode.add_parser("enqueue", help="Create poster tasks in Redis.")
    enqueue.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    enqueue.add_argument("--num-graphs", type=int, default=5)
    enqueue.add_argument("--seed", type=int, default=42)
    enqueue.add_argument("--algorithms", choices=POSTER_BATCH_ALGORITHMS, nargs="+", default=list(POSTER_BATCH_ALGORITHMS))
    enqueue.add_argument("--redis-url", type=str, default=None)
    enqueue.add_argument("--queue", type=str, default=os.environ.get("POSTER_BATCH_QUEUE", DEFAULT_QUEUE))
    enqueue.add_argument("--s3-prefix", type=str, default=os.environ.get("POSTER_S3_PREFIX", "poster-batch"))
    enqueue.add_argument("--max-attempts", type=int, default=3)
    enqueue.set_defaults(func=_dispatch_enqueue)

    worker = mode.add_parser("worker", help="Run one AWS Batch worker process.")
    worker.add_argument("--redis-url", type=str, default=None)
    worker.add_argument("--queue", type=str, default=os.environ.get("POSTER_BATCH_QUEUE", DEFAULT_QUEUE))
    worker.add_argument("--s3-bucket", type=str, default=os.environ.get("POSTER_S3_BUCKET"))
    worker.add_argument("--max-tasks", type=int, default=None)
    worker.add_argument("--block-timeout", type=int, default=DEFAULT_VISIBILITY_TIMEOUT)
    worker.set_defaults(func=_dispatch_worker)

    collect = mode.add_parser("collect", help="Collect S3 chunks and generate poster plots.")
    collect.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    collect.add_argument("--num-graphs", type=int, default=5)
    collect.add_argument("--s3-bucket", type=str, default=os.environ.get("POSTER_S3_BUCKET"))
    collect.add_argument("--s3-prefix", type=str, default=os.environ.get("POSTER_S3_PREFIX", "poster-batch"))
    collect.add_argument("--output-dir", type=str, default="results/poster_batch")
    collect.add_argument("--allow-missing", action="store_true")
    collect.set_defaults(func=_dispatch_collect)
