"""``poster-batch`` CLI for Redis/AWS Batch/S3 poster-result computation."""

from __future__ import annotations

import argparse
import os

from src.commands.poster_batch.collect import (
    collect_and_plot,
    collect_s3_trial_results,
)
from src.commands.poster_batch.queue import (
    Queue,
    RedisTaskQueue,
    get_redis_client,
)
from src.commands.poster_batch.schema import (
    DEFAULT_QUEUE,
    DEFAULT_VISIBILITY_TIMEOUT,
    POSTER_BATCH_ALGORITHMS,
    POSTER_BATCH_SCHEMA_VERSION,
    POSTER_RESULTS_PROBLEM,
    build_task,
    build_tasks,
)
from src.commands.poster_batch.store import (
    S3Store,
    Store,
    get_s3_client,
)
from src.commands.poster_batch.tasks import (
    TASK_ROUTERS,
    enqueue_tasks,
    run_task,
    run_worker,
    write_result as _write_result,
)


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
        args.algorithms,
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
    collect.add_argument("--algorithms", choices=POSTER_BATCH_ALGORITHMS, nargs="+", default=None)
    collect.set_defaults(func=_dispatch_collect)
