"""Tests for the poster-batch command helpers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.commands import poster_batch as pb


class FakeRedis:
    def __init__(self) -> None:
        self.queues: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    def rpush(self, name: str, *values: str) -> int:
        self.queues.setdefault(name, []).extend(values)
        return len(self.queues[name])

    def blpop(self, name: str, timeout: int = 0) -> tuple[str, str] | None:
        del timeout
        queue = self.queues.setdefault(name, [])
        if not queue:
            return None
        return name, queue.pop(0)

    def hset(self, name: str, key: str, value: str) -> int:
        self.hashes.setdefault(name, {})[key] = value
        return 1


class FakeBody:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakePaginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        del Bucket
        return [
            {
                "Contents": [
                    {"Key": key}
                    for key in sorted(self.objects)
                    if key.startswith(Prefix)
                ]
            }
        ]


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        del Bucket, ContentType
        self.objects[Key] = Body

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)

    def get_object(self, Bucket: str, Key: str) -> dict[str, FakeBody]:
        del Bucket
        return {"Body": FakeBody(self.objects[Key])}


def _fake_result(task: dict[str, Any]) -> dict[str, Any]:
    algorithm = task["algorithm"]
    values = {
        "raw_sa": {"apsp": 1.0, "flow": 2.0},
        "global": {"apsp": 3.0, "flow": 4.0, "qvars": 5.0, "sg": 6.0, "pt": 7.0},
        "mr2s": {
            "apsp": 8.0,
            "flow": 9.0,
            "qvars": 10.0,
            "sg": 11.0,
            "phys_total": 12.0,
            "phys_max": 13.0,
            "phys_mean": 14.0,
            "phys_min": 15.0,
            "partition": {"selected_reason": "test"},
        },
        "random": {"apsp": 16.0, "flow": 17.0, "sample_count": 1},
    }[algorithm]
    return {
        "algorithm": algorithm,
        "n": task["n"],
        "trial": task["trial"],
        "seed": task["seed"],
        "values": values,
        "timings": {algorithm: 0.0},
    }


def test_build_tasks_splits_each_trial_by_algorithm() -> None:
    tasks = pb.build_tasks(
        sizes=[8, 10],
        num_graphs=2,
        seed=42,
        algorithms=["raw_sa", "mr2s"],
        s3_prefix="poster/run",
        max_attempts=3,
    )

    assert len(tasks) == 8
    assert {task["algorithm"] for task in tasks} == {"raw_sa", "mr2s"}
    assert all(task["s3_key"].startswith("poster/run/chunks/") for task in tasks)
    assert len({task["task_id"] for task in tasks}) == len(tasks)


def test_worker_processes_task_and_writes_s3(monkeypatch) -> None:
    redis = FakeRedis()
    s3 = FakeS3()
    task = pb.build_task(8, 0, 42, "raw_sa", "poster/run", max_attempts=1)
    pb.enqueue_tasks(redis, "queue", [task])
    monkeypatch.setattr(pb, "run_task", _fake_result)

    processed = pb.run_worker(redis, s3, "queue", "bucket", max_tasks=1, block_timeout=0)

    assert processed == 1
    assert task["s3_key"] in s3.objects
    status = json.loads(redis.hashes["queue:status"][task["task_id"]])
    assert status["state"] == "succeeded"


def test_worker_requeues_failed_task_until_max_attempts(monkeypatch) -> None:
    redis = FakeRedis()
    s3 = FakeS3()
    task = pb.build_task(8, 0, 42, "raw_sa", "poster/run", max_attempts=2)
    pb.enqueue_tasks(redis, "queue", [task])

    calls = 0

    def flaky_run_task(task: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return _fake_result(task)

    monkeypatch.setattr(pb, "run_task", flaky_run_task)

    assert pb.run_worker(redis, s3, "queue", "bucket", max_tasks=1, block_timeout=0) == 1
    assert calls == 2
    assert task["s3_key"] in s3.objects


def test_collect_s3_results_aggregates_complete_chunks(tmp_path, monkeypatch) -> None:
    s3 = FakeS3()
    monkeypatch.setattr(pb.poster_results, "_plot_results", lambda results, output_dir: None)
    tasks = pb.build_tasks(
        sizes=[8],
        num_graphs=1,
        seed=42,
        algorithms=pb.POSTER_BATCH_ALGORITHMS,
        s3_prefix="poster/run",
        max_attempts=1,
    )
    for task in tasks:
        pb._write_result_to_s3(s3, "bucket", task, _fake_result(task))

    results = pb.collect_and_plot(
        s3,
        "bucket",
        "poster/run",
        sizes=[8],
        num_graphs=1,
        output_dir=str(tmp_path),
        allow_missing=False,
    )

    assert results["raw_sa"]["apsp"] == [1.0]
    assert results["global"]["qubo_vars"] == [5.0]
    assert results["mr2s"]["phys_total"] == [12.0]
    assert results["random"]["flow"] == [17.0]


def test_collect_s3_results_reports_missing_chunks() -> None:
    s3 = FakeS3()
    task = pb.build_task(8, 0, 42, "raw_sa", "poster/run", max_attempts=1)
    pb._write_result_to_s3(s3, "bucket", task, _fake_result(task))

    _, missing = pb.collect_s3_trial_results(
        s3,
        "bucket",
        "poster/run",
        sizes=[8],
        num_graphs=1,
        algorithms=pb.POSTER_BATCH_ALGORITHMS,
    )

    assert {item["algorithm"] for item in missing} == {"global", "mr2s", "random"}


def test_collect_and_plot_raises_when_chunks_are_missing(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Missing 4 poster batch chunk"):
        pb.collect_and_plot(
            FakeS3(),
            "bucket",
            "poster/run",
            sizes=[8],
            num_graphs=1,
            output_dir=str(tmp_path),
        )
    assert (tmp_path / "missing_tasks.json").exists()
