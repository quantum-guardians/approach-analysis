"""Runners for the ``poster-results`` experiment."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from src.commands.poster_results.models import (
    Mr2sTrialResult,
    PosterResultsSummary,
    PosterTrialResult,
    TrialPayload,
    coerce_trial_payload,
)
from src.commands.poster_results.solvers.base import _mean_finite
from src.commands.poster_results.solvers.mr2s_variant import MR2S_VARIANTS
from src.commands.poster_results.solvers.dnc_strategy import DNC_STRATEGIES
from src.logging_config import configure_worker_logging

TrialResult = tuple[int, int, TrialPayload | dict[str, Any]]
TrialTask = tuple[int, int, int | None, str | None]
TrialWorker = Callable[[TrialTask], TrialResult]
ProgressPrinter = Callable[[int, int, int, int, dict[str, float]], None]
Plotter = Callable[[dict[str, Any], str], None]

RESULT_SERIES_KEYS = {
    "mr2s": [
        "apsp", "flow", "qubo_vars", "subgraph_size",
        "phys_total", "phys_max", "phys_mean", "phys_min", "partition",
    ],
    "global": [
        "apsp", "flow", "qubo_vars", "subgraph_size",
        "phys_total", "phys_max", "phys_mean", "phys_min",
    ],
    "raw_sa": ["apsp", "flow"],
    "random": ["apsp", "flow"],
    "timings": [
        "graph", "raw_sa", "global_solve", "global_embed",
        "clustered_solve", "clustered_embed", "random",
        "dnc_poster_solve", "dnc_poster_embed",
        "dnc_embedding_aware_solve", "dnc_embedding_aware_embed",
        "dnc_degeneracy_pruning_solve", "dnc_degeneracy_pruning_embed",
        "robbin_mr2s_solve", "robbin_mr2s_embed",
        "iterated_local_search_mr2s_solve", "iterated_local_search_mr2s_embed",
    ],
}

FULL_TIMING_KEYS = [
    "graph", "raw_sa", "global_solve", "global_embed",
    "clustered_solve", "clustered_embed", "random",
    "dnc_poster_solve", "dnc_poster_embed",
    "dnc_embedding_aware_solve", "dnc_embedding_aware_embed",
    "dnc_degeneracy_pruning_solve", "dnc_degeneracy_pruning_embed",
    "robbin_mr2s_solve", "robbin_mr2s_embed",
    "iterated_local_search_mr2s_solve", "iterated_local_search_mr2s_embed",
]
MR2S_TIMING_KEYS = ["graph", "clustered_solve", "clustered_embed"]
REQUIRED_RESULT_SERIES_KEYS = {
    "mr2s": [
        "apsp", "flow", "qubo_vars", "subgraph_size",
        "phys_total", "phys_max", "phys_mean", "phys_min", "partition",
    ],
    "global": ["apsp", "flow", "qubo_vars", "subgraph_size", "phys_total"],
    "raw_sa": ["apsp", "flow"],
    "random": ["apsp", "flow"],
    "timings": FULL_TIMING_KEYS,
}
DEFAULT_GRAPH_REMOVAL_PCT = 0.0


def _normalize_random_baseline(result: Any) -> Any:
    from src.commands.poster_results.models import RandomBaselineResult
    if isinstance(result, RandomBaselineResult):
        return result.normalized()

    sample_count = result.get("sample_count")
    missing_legacy_sample = sample_count is None and result.get("apsp") == 0 and result.get("flow") == 0
    if sample_count == 0 or missing_legacy_sample:
        normalized = dict(result)
        normalized["apsp"] = float("nan")
        normalized["flow"] = float("nan")
        normalized["sample_count"] = 0
        return normalized
    return result


def _coerce_full_result(result: PosterTrialResult | dict[str, Any]) -> PosterTrialResult:
    if isinstance(result, PosterTrialResult):
        return result
    return PosterTrialResult.from_dict(result)


def _coerce_mr2s_result(result: Mr2sTrialResult | dict[str, Any]) -> Mr2sTrialResult:
    if isinstance(result, Mr2sTrialResult):
        return result
    return Mr2sTrialResult.from_dict(result)


@dataclass(frozen=True)
class PosterRunConfig:
    sizes: list[int]
    num_graphs: int
    seed: int | None
    output_dir: str
    num_workers: int | None = None
    cache_dir: str | None = None
    use_cache: bool = True
    refresh_algorithms: tuple[str, ...] = ()
    graph_removal_pct: float = DEFAULT_GRAPH_REMOVAL_PCT

    def resolved_cache_dir(self, default_name: str) -> str | None:
        if not self.use_cache:
            return None
        if self.cache_dir is not None:
            return self.cache_dir
        return os.path.join(self.output_dir, default_name)

    def with_sizes(self, sizes: list[int]) -> "PosterRunConfig":
        return PosterRunConfig(
            sizes=sizes,
            num_graphs=self.num_graphs,
            seed=self.seed,
            output_dir=self.output_dir,
            num_workers=self.num_workers,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            refresh_algorithms=self.refresh_algorithms,
            graph_removal_pct=self.graph_removal_pct,
        )


class TrialScheduler:
    def effective_workers(self, total_tasks: int, requested_workers: int | None) -> int:
        if requested_workers is not None:
            return requested_workers
        return min(total_tasks, max((os.cpu_count() or 2) - 1, 1))

    def iter_completed(
        self,
        worker: TrialWorker,
        tasks: list[TrialTask],
        num_workers: int,
    ) -> Any:
        if num_workers == 0:
            for task in tasks:
                yield worker(task)
            return

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=self._process_pool_context(),
            initializer=configure_worker_logging,
        ) as executor:
            futures = [executor.submit(worker, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                yield future.result()

    def _process_pool_context(self) -> multiprocessing.context.BaseContext:
        if sys.platform == "darwin":
            return multiprocessing.get_context("spawn")
        if "fork" in multiprocessing.get_all_start_methods():
            return multiprocessing.get_context("fork")
        return multiprocessing.get_context()


class PosterResultsAggregator:
    def empty_results(self, sizes: list[int]) -> dict[str, Any]:
        return PosterResultsSummary(sizes=sizes).to_dict()

    def aggregate_full(
        self,
        sizes: list[int],
        trial_results: dict[int, list[PosterTrialResult | dict[str, Any]]],
    ) -> dict[str, Any]:
        summary = PosterResultsSummary(sizes=sizes)
        dnc_strategy_summary = {
            name: {
                "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
                "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": [],
                "partition": [],
            }
            for name in DNC_STRATEGIES
        }
        mr2s_variant_summary = {
            name: {
                "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
                "phys_total": [],
            }
            for name in MR2S_VARIANTS
        }

        for n in sizes:
            a_rsa, f_rsa = [], []
            a_glb, f_glb, v_glb, s_glb, p_glb = [], [], [], [], []
            a_cls, f_cls, v_cls, s_cls = [], [], [], []
            pt_cls, pmax_cls, pmean_cls, pmin_cls = [], [], [], []
            a_rnd, f_rnd = [], []
            timing_values = {key: [] for key in FULL_TIMING_KEYS}
            partitions = []
            strategy_values = {
                name: {
                    "apsp": [], "flow": [], "qvars": [], "subgraph_size": [],
                    "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": [],
                    "partition": [],
                }
                for name in DNC_STRATEGIES
            }
            variant_values = {
                name: {
                    "apsp": [], "flow": [], "qvars": [], "subgraph_size": [],
                    "phys_total": [],
                }
                for name in MR2S_VARIANTS
            }

            print(f"\n>>> Aggregating size n={n}")

            for raw_result in trial_results[n]:
                result = _coerce_full_result(raw_result)
                res_rnd = _normalize_random_baseline(result.random)
                a_rsa.append(result.raw_sa.apsp)
                f_rsa.append(result.raw_sa.flow)
                a_glb.append(result.global_result.apsp)
                f_glb.append(result.global_result.flow)
                v_glb.append(result.global_result.qvars)
                s_glb.append(result.global_result.subgraph_size)
                p_glb.append(result.global_result.physical_total)
                a_cls.append(result.mr2s.apsp)
                f_cls.append(result.mr2s.flow)
                v_cls.append(result.mr2s.qvars)
                s_cls.append(result.mr2s.subgraph_size)
                pt_cls.append(result.mr2s.phys_total)
                pmax_cls.append(result.mr2s.phys_max)
                pmean_cls.append(result.mr2s.phys_mean)
                pmin_cls.append(result.mr2s.phys_min)
                partitions.append(result.mr2s.partition)
                for strategy_name, strategy_result in result.dnc_strategies.items():
                    if strategy_name not in strategy_values:
                        continue
                    strategy_values[strategy_name]["apsp"].append(strategy_result.apsp)
                    strategy_values[strategy_name]["flow"].append(strategy_result.flow)
                    strategy_values[strategy_name]["qvars"].append(strategy_result.qvars)
                    strategy_values[strategy_name]["subgraph_size"].append(strategy_result.subgraph_size)
                    strategy_values[strategy_name]["phys_total"].append(strategy_result.phys_total)
                    strategy_values[strategy_name]["phys_max"].append(strategy_result.phys_max)
                    strategy_values[strategy_name]["phys_mean"].append(strategy_result.phys_mean)
                    strategy_values[strategy_name]["phys_min"].append(strategy_result.phys_min)
                    strategy_values[strategy_name]["partition"].append(strategy_result.partition)
                for variant_name, variant_result in result.mr2s_variants.items():
                    if variant_name not in variant_values:
                        continue
                    variant_values[variant_name]["apsp"].append(variant_result.apsp)
                    variant_values[variant_name]["flow"].append(variant_result.flow)
                    variant_values[variant_name]["qvars"].append(variant_result.qvars)
                    variant_values[variant_name]["subgraph_size"].append(variant_result.subgraph_size)
                    variant_values[variant_name]["phys_total"].append(variant_result.physical_total)
                a_rnd.append(res_rnd.apsp)
                f_rnd.append(res_rnd.flow)
                self._collect_timings(result.timings.to_dict(), timing_values)

            summary.raw_sa.apsp.append(_mean_finite(a_rsa))
            summary.raw_sa.flow.append(_mean_finite(f_rsa))
            summary.global_result.apsp.append(_mean_finite(a_glb))
            summary.global_result.flow.append(_mean_finite(f_glb))
            summary.global_result.qubo_vars.append(_mean_finite(v_glb))
            summary.global_result.subgraph_size.append(_mean_finite(s_glb))
            summary.global_result.phys_total.append(_mean_finite(p_glb))
            summary.mr2s.apsp.append(_mean_finite(a_cls))
            summary.mr2s.flow.append(_mean_finite(f_cls))
            summary.mr2s.qubo_vars.append(_mean_finite(v_cls))
            summary.mr2s.subgraph_size.append(_mean_finite(s_cls))
            summary.mr2s.phys_total.append(_mean_finite(pt_cls))
            summary.mr2s.phys_max.append(_mean_finite(pmax_cls))
            summary.mr2s.phys_mean.append(_mean_finite(pmean_cls))
            summary.mr2s.phys_min.append(_mean_finite(pmin_cls))
            summary.mr2s.partition.append(partitions)
            for strategy_name, values in strategy_values.items():
                target = dnc_strategy_summary[strategy_name]
                target["apsp"].append(_mean_finite(values["apsp"]))
                target["flow"].append(_mean_finite(values["flow"]))
                target["qubo_vars"].append(_mean_finite(values["qvars"]))
                target["subgraph_size"].append(_mean_finite(values["subgraph_size"]))
                target["phys_total"].append(_mean_finite(values["phys_total"]))
                target["phys_max"].append(_mean_finite(values["phys_max"]))
                target["phys_mean"].append(_mean_finite(values["phys_mean"]))
                target["phys_min"].append(_mean_finite(values["phys_min"]))
                target["partition"].append(values["partition"])
            for variant_name, values in variant_values.items():
                target = mr2s_variant_summary[variant_name]
                target["apsp"].append(_mean_finite(values["apsp"]))
                target["flow"].append(_mean_finite(values["flow"]))
                target["qubo_vars"].append(_mean_finite(values["qvars"]))
                target["subgraph_size"].append(_mean_finite(values["subgraph_size"]))
                target["phys_total"].append(_mean_finite(values["phys_total"]))
            summary.random.apsp.append(_mean_finite(a_rnd))
            summary.random.flow.append(_mean_finite(f_rnd))
            self._append_timings(summary.timings, timing_values)

        data = summary.to_dict()
        data["dnc_strategies"] = dnc_strategy_summary
        data["mr2s_variants"] = mr2s_variant_summary
        return data

    def merge_full_results(
        self,
        existing: dict[str, Any] | None,
        computed: dict[str, Any],
    ) -> dict[str, Any]:
        if existing is None:
            return computed
        merged = _merge_results_by_size(existing, computed, replace_sections=set(RESULT_SERIES_KEYS))
        merged = _merge_nested_results_by_size(
            merged, existing, computed, "dnc_strategies", replace_existing=True
        )
        return _merge_nested_results_by_size(
            merged, existing, computed, "mr2s_variants", replace_existing=True
        )

    def merge_mr2s_only(
        self,
        results: dict[str, Any],
        trial_results: dict[int, list[Mr2sTrialResult | dict[str, Any]]],
    ) -> dict[str, Any]:
        requested_sizes = set(trial_results)
        missing_sizes = [size for size in requested_sizes if size not in results.get("sizes", [])]
        if missing_sizes:
            raise ValueError(
                "MR2S-only mode can only update sizes already present in existing results: "
                f"{sorted(missing_sizes)}"
            )

        partial_results = PosterResultsSummary(sizes=list(trial_results)).to_dict()
        mr2s_summary = PosterResultsSummary(sizes=list(trial_results)).mr2s

        for n in trial_results:
            a_cls, f_cls, v_cls, s_cls = [], [], [], []
            pt_cls, pmax_cls, pmean_cls, pmin_cls = [], [], [], []
            timing_values = {key: [] for key in MR2S_TIMING_KEYS}
            partitions = []

            print(f"\n>>> Aggregating MR2S-only size n={n}")

            for raw_result in trial_results[n]:
                result = _coerce_mr2s_result(raw_result)
                res_cls = result.mr2s
                a_cls.append(res_cls.apsp)
                f_cls.append(res_cls.flow)
                v_cls.append(res_cls.qvars)
                s_cls.append(res_cls.subgraph_size)
                pt_cls.append(res_cls.phys_total)
                pmax_cls.append(res_cls.phys_max)
                pmean_cls.append(res_cls.phys_mean)
                pmin_cls.append(res_cls.phys_min)
                partitions.append(res_cls.partition)
                self._collect_timings(result.timings.to_dict(), timing_values)

            mr2s_summary.apsp.append(_mean_finite(a_cls))
            mr2s_summary.flow.append(_mean_finite(f_cls))
            mr2s_summary.qubo_vars.append(_mean_finite(v_cls))
            mr2s_summary.subgraph_size.append(_mean_finite(s_cls))
            mr2s_summary.phys_total.append(_mean_finite(pt_cls))
            mr2s_summary.phys_max.append(_mean_finite(pmax_cls))
            mr2s_summary.phys_mean.append(_mean_finite(pmean_cls))
            mr2s_summary.phys_min.append(_mean_finite(pmin_cls))
            mr2s_summary.partition.append(partitions)
            self._ensure_timing_section(partial_results)
            self._append_timings_to_dict(partial_results["timings"], timing_values)

        partial_results["mr2s"] = mr2s_summary.to_dict()
        merged = _merge_results_by_size(results, partial_results, replace_sections={"mr2s"})
        return _merge_timing_keys_by_size(
            merged,
            partial_results,
            replace_keys=set(MR2S_TIMING_KEYS),
        )

    def _collect_timings(
        self,
        timings: dict[str, Any],
        timing_values: dict[str, list[float]],
    ) -> None:
        for key in timing_values:
            value = timings.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timing_values[key].append(float(value))

    def _append_timings(
        self,
        timing_series: Any,
        timing_values: dict[str, list[float]],
    ) -> None:
        for key, values in timing_values.items():
            getattr(timing_series, key).append(_mean_finite(values))

    def _ensure_timing_section(self, results: dict[str, Any]) -> None:
        results.setdefault("timings", {key: [] for key in RESULT_SERIES_KEYS["timings"]})

    def _append_timings_to_dict(
        self,
        timing_series: dict[str, list[float]],
        timing_values: dict[str, list[float]],
    ) -> None:
        for key, values in timing_values.items():
            timing_series.setdefault(key, []).append(_mean_finite(values))


def _merge_results_by_size(
    existing: dict[str, Any],
    updates: dict[str, Any],
    replace_sections: set[str],
) -> dict[str, Any]:
    merged = copy.deepcopy(existing)
    existing_sizes = list(existing.get("sizes", []))
    update_sizes = list(updates.get("sizes", []))
    output_sizes = existing_sizes + [size for size in update_sizes if size not in existing_sizes]
    merged["sizes"] = output_sizes

    existing_size_set = set(existing_sizes)
    update_size_set = set(update_sizes)

    for section, keys in RESULT_SERIES_KEYS.items():
        merged.setdefault(section, {})
        for key in keys:
            existing_by_size = _series_by_size(existing, section, key)
            update_by_size = _series_by_size(updates, section, key)
            values = []
            has_value = bool(existing_by_size or update_by_size)

            for size in output_sizes:
                use_update = size in update_size_set and (
                    section in replace_sections or size not in existing_size_set
                )

                if use_update and size in update_by_size:
                    value = update_by_size[size]
                elif size in existing_by_size:
                    value = existing_by_size[size]
                elif size in update_by_size:
                    value = update_by_size[size]
                elif has_value:
                    value = _missing_series_value(key)
                else:
                    continue

                values.append(value)

            if has_value:
                merged[section][key] = values
            elif key in merged[section]:
                merged[section][key] = []

    return merged


def _merge_timing_keys_by_size(
    existing: dict[str, Any],
    updates: dict[str, Any],
    replace_keys: set[str],
) -> dict[str, Any]:
    merged = copy.deepcopy(existing)
    merged.setdefault("timings", {})
    existing_sizes = list(existing.get("sizes", []))
    update_sizes = list(updates.get("sizes", []))
    output_sizes = list(merged.get("sizes", existing_sizes))
    update_size_set = set(update_sizes)

    for key in RESULT_SERIES_KEYS["timings"]:
        existing_by_size = _series_by_size(existing, "timings", key)
        update_by_size = _series_by_size(updates, "timings", key)
        values = []
        has_value = bool(existing_by_size or update_by_size)

        for size in output_sizes:
            if key in replace_keys and size in update_size_set and size in update_by_size:
                values.append(update_by_size[size])
            elif size in existing_by_size:
                values.append(existing_by_size[size])
            elif size in update_by_size:
                values.append(update_by_size[size])
            elif has_value:
                values.append(_missing_series_value(key))

        if has_value:
            merged["timings"][key] = values

    return merged


def _merge_nested_results_by_size(
    merged: dict[str, Any],
    existing: dict[str, Any],
    updates: dict[str, Any],
    section: str,
    replace_existing: bool,
) -> dict[str, Any]:
    output_sizes = list(merged.get("sizes", []))
    update_sizes = set(updates.get("sizes", []))
    existing_sizes = set(existing.get("sizes", []))
    merged.setdefault(section, {})

    item_names = set(existing.get(section, {})) | set(updates.get(section, {}))
    for item_name in item_names:
        merged[section].setdefault(item_name, {})
        keys = set(existing.get(section, {}).get(item_name, {}))
        keys |= set(updates.get(section, {}).get(item_name, {}))

        for key in keys:
            existing_by_size = _nested_series_by_size(existing, section, item_name, key)
            update_by_size = _nested_series_by_size(updates, section, item_name, key)
            values = []
            has_value = bool(existing_by_size or update_by_size)
            for size in output_sizes:
                use_update = size in update_sizes and (
                    replace_existing or size not in existing_sizes
                )
                if use_update and size in update_by_size:
                    values.append(update_by_size[size])
                elif size in existing_by_size:
                    values.append(existing_by_size[size])
                elif size in update_by_size:
                    values.append(update_by_size[size])
                elif has_value:
                    values.append(_missing_series_value(key))
            if has_value:
                merged[section][item_name][key] = values
    return merged


def _missing_series_value(key: str) -> Any:
    if key == "partition":
        return []
    return float("nan")


def _nested_series_by_size(
    results: dict[str, Any],
    section: str,
    item_name: str,
    key: str,
) -> dict[int, Any]:
    sizes = results.get("sizes", [])
    values = results.get(section, {}).get(item_name, {}).get(key, [])
    return {
        size: values[index]
        for index, size in enumerate(sizes)
        if index < len(values)
    }


def _series_by_size(results: dict[str, Any], section: str, key: str) -> dict[int, Any]:
    sizes = results.get("sizes", [])
    values = results.get(section, {}).get(key, [])
    return {
        size: values[index]
        for index, size in enumerate(sizes)
        if index < len(values)
    }


def _has_series_value(results: dict[str, Any], section: str, key: str, size: int) -> bool:
    return size in _series_by_size(results, section, key)


def _has_nested_series_value(
    results: dict[str, Any],
    section: str,
    item_name: str,
    key: str,
    size: int,
) -> bool:
    return size in _nested_series_by_size(results, section, item_name, key)


def _has_current_result_schema(
    results: dict[str, Any],
    size: int,
    graph_removal_pct: float,
) -> bool:
    # 새 solver/metric이 추가되면 기존 poster_results.json만 보고 size 전체를 건너뛰지 않는다.
    current_pct = results.get("graph_removal_pct")
    if current_pct is None or abs(float(current_pct) - graph_removal_pct) > 1e-9:
        return False

    for section, keys in REQUIRED_RESULT_SERIES_KEYS.items():
        for key in keys:
            if not _has_series_value(results, section, key, size):
                return False

    for strategy_name in DNC_STRATEGIES:
        for key in (
            "apsp", "flow", "qubo_vars", "subgraph_size",
            "phys_total", "phys_max", "phys_mean", "phys_min", "partition",
        ):
            if not _has_nested_series_value(results, "dnc_strategies", strategy_name, key, size):
                return False

    for variant_name in MR2S_VARIANTS:
        for key in ("apsp", "flow", "qubo_vars", "subgraph_size", "phys_total"):
            if not _has_nested_series_value(results, "mr2s_variants", variant_name, key, size):
                return False

    return True


class PosterResultsRunner:
    cache_name = "poster_trial_cache"

    def __init__(
        self,
        config: PosterRunConfig,
        worker: TrialWorker,
        progress_printer: ProgressPrinter,
        plotter: Plotter,
        scheduler: TrialScheduler | None = None,
        aggregator: PosterResultsAggregator | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.progress_printer = progress_printer
        self.plotter = plotter
        self.scheduler = scheduler or TrialScheduler()
        self.aggregator = aggregator or PosterResultsAggregator()

    def run(self) -> dict[str, Any]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        existing_results = self._load_existing_results()
        run_sizes = self._sizes_to_compute(existing_results)
        cache_dir = self.config.resolved_cache_dir(self.cache_name)

        tasks = self._build_tasks(cache_dir, run_sizes)
        total_tasks = len(tasks)
        workers = self.scheduler.effective_workers(total_tasks, self.config.num_workers)

        print(
            f"Starting poster results: {len(run_sizes)} new size(s), "
            f"{self.config.num_graphs} graph(s) each, {workers} worker process(es)."
        )
        if existing_results is not None:
            reused_sizes = [
                size
                for size in self.config.sizes
                if size in existing_results.get("sizes", [])
                and _has_current_result_schema(existing_results, size, self.config.graph_removal_pct)
            ]
            if reused_sizes:
                print(f"Reusing existing poster results for size(s): {reused_sizes}")
        if cache_dir is not None:
            print(f"Using poster trial cache: {cache_dir}")

        if tasks:
            trial_results = self._collect_trials(tasks, workers, total_tasks, run_sizes)
            computed_results = self.aggregator.aggregate_full(run_sizes, trial_results)
            results = self.aggregator.merge_full_results(existing_results, computed_results)
        elif existing_results is not None:
            results = existing_results
        else:
            results = self.aggregator.empty_results(self.config.sizes)

        results["graph_removal_pct"] = self.config.graph_removal_pct
        self._write_results(results)
        self.plotter(results, self.config.output_dir)
        return results

    def _load_existing_results(self) -> dict[str, Any] | None:
        if not self.config.use_cache:
            return None
        results_path = os.path.join(self.config.output_dir, "poster_results.json")
        if not os.path.exists(results_path):
            return None
        with open(results_path) as f:
            return json.load(f)

    def _sizes_to_compute(self, existing_results: dict[str, Any] | None) -> list[int]:
        if self.config.refresh_algorithms:
            return self.config.sizes
        if existing_results is None:
            return self.config.sizes
        existing_sizes = set(existing_results.get("sizes", []))
        return [
            size
            for size in self.config.sizes
            if size not in existing_sizes
            or not _has_current_result_schema(existing_results, size, self.config.graph_removal_pct)
        ]

    def _build_tasks(self, cache_dir: str | None, sizes: list[int]) -> list[TrialTask]:
        return [
            (n, trial, self.config.seed, cache_dir)
            for n in sizes
            for trial in range(self.config.num_graphs)
        ]

    def _collect_trials(
        self,
        tasks: list[TrialTask],
        workers: int,
        total_tasks: int,
        sizes: list[int],
    ) -> dict[int, list[TrialPayload]]:
        trial_results: dict[int, list[TrialPayload]] = {
            n: [] for n in sizes
        }
        for index, (n, trial, raw_result) in enumerate(
            self.scheduler.iter_completed(self.worker, tasks, workers),
            start=1,
        ):
            result = coerce_trial_payload(raw_result)
            self.progress_printer(index, total_tasks, n, trial, result.timings.to_dict())
            trial_results[n].append(result)
        return trial_results

    def _write_results(self, results: dict[str, Any]) -> None:
        with open(os.path.join(self.config.output_dir, "poster_results.json"), "w") as f:
            json.dump(results, f, indent=2)


class Mr2sOnlyPosterResultsRunner(PosterResultsRunner):
    cache_name = "poster_mr2s_trial_cache"

    def __init__(
        self,
        config: PosterRunConfig,
        worker: TrialWorker,
        progress_printer: ProgressPrinter,
        plotter: Plotter,
        source_results_path: str | None = None,
        scheduler: TrialScheduler | None = None,
        aggregator: PosterResultsAggregator | None = None,
    ) -> None:
        super().__init__(
            config=config,
            worker=worker,
            progress_printer=progress_printer,
            plotter=plotter,
            scheduler=scheduler,
            aggregator=aggregator,
        )
        self.source_results_path = source_results_path

    def run(self) -> dict[str, Any]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        source_results_path = self._source_results_path()
        if not os.path.exists(source_results_path):
            raise FileNotFoundError(
                f"MR2S-only mode needs existing results to merge: {source_results_path}"
            )

        with open(source_results_path) as f:
            results = json.load(f)
        current_pct = results.get("graph_removal_pct")
        if current_pct is None or abs(float(current_pct) - self.config.graph_removal_pct) > 1e-9:
            raise ValueError(
                "MR2S-only mode needs existing results generated with "
                f"graph_removal_pct={self.config.graph_removal_pct}"
            )

        cache_dir = self.config.resolved_cache_dir(self.cache_name)
        tasks = self._build_tasks(cache_dir, self.config.sizes)
        total_tasks = len(tasks)
        workers = self.scheduler.effective_workers(total_tasks, self.config.num_workers)

        print(
            f"Starting MR2S-only poster results: {len(self.config.sizes)} size(s), "
            f"{self.config.num_graphs} graph(s) each, {workers} worker process(es)."
        )
        print(f"Merging source results from: {source_results_path}")
        if cache_dir is not None:
            print(f"Using MR2S-only trial cache: {cache_dir}")

        trial_results = self._collect_trials(tasks, workers, total_tasks, self.config.sizes)
        merged = self.aggregator.merge_mr2s_only(results, trial_results)
        merged["graph_removal_pct"] = self.config.graph_removal_pct
        self._write_results(merged)
        self.plotter(merged, self.config.output_dir)
        return merged

    def _source_results_path(self) -> str:
        if self.source_results_path is not None:
            return self.source_results_path
        return os.path.join(self.config.output_dir, "poster_results.json")
