"""S3/store collection and plotting for poster batch chunks."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from src.commands import poster_results
from src.commands.poster_batch.schema import (
    POSTER_BATCH_ALGORITHMS,
    POSTER_RESULTS_PROBLEM,
    json_default,
    normalise_s3_prefix,
)
from src.commands.poster_batch.store import Store
from src.commands.poster_results_runner import PosterResultsAggregator


def iter_store_json_objects(store: Store, prefix: str) -> Iterable[dict[str, Any]]:
    yield from store.iter_json(
        f"{normalise_s3_prefix(prefix)}/chunks/problem={POSTER_RESULTS_PROBLEM}/"
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

    for payload in iter_store_json_objects(store, prefix):
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
        json.dump(results, f, indent=2, default=json_default, allow_nan=True)
    poster_results._plot_results(results, output_dir)
    return results
