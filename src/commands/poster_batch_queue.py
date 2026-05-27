"""Queue abstractions for distributed batch tasks."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Callable, Protocol

from src.commands.poster_batch_schema import (
    DEFAULT_VISIBILITY_TIMEOUT,
    decode_task,
    task_payload,
)

TaskHandler = Callable[[dict[str, Any]], dict[str, Any] | None]


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


def redis_status_key(queue_name: str) -> str:
    return f"{queue_name}:status"


class RedisTaskQueue:
    def __init__(self, redis_client: Any, queue_name: str):
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.status_key = redis_status_key(queue_name)

    def enqueue(self, tasks: list[dict[str, Any]]) -> int:
        if not tasks:
            return 0

        payloads = [task_payload(task) for task in tasks]
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
            task = decode_task(payload)
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
                    self.redis_client.rpush(self.queue_name, task_payload(task))
                else:
                    processed += 1

        return processed

    def _set_status(self, task: dict[str, Any], status: dict[str, Any]) -> None:
        self.redis_client.hset(
            self.status_key,
            task["task_id"],
            task_payload(status),
        )


def get_redis_client(redis_url: str | None = None) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("Install redis package to use poster-batch Redis queue.") from exc

    url = redis_url or os.environ.get("POSTER_REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url)
