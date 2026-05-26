"""Plotting helpers for the ``poster-results`` command."""

from __future__ import annotations

import os

from src.visualizer import (
    plot_apsp_reduction,
    plot_flow_stability,
    plot_preprocessing_scalability,
    plot_spent_time,
)


def _plot_results(results: dict, output_dir: str) -> None:
    sizes = results["sizes"]
    # solver별 normalized APSP를 한 그래프에 모아 poster용 비교 그림을 만든다.
    plot_apsp_reduction(
        sizes,
        results["random"]["apsp"],
        results["raw_sa"]["apsp"],
        results["global"]["apsp"],
        results["mr2s"]["apsp"],
        mr2s_variants=results.get("mr2s_variants"),
        dnc_strategies=results.get("dnc_strategies"),
        save_path=os.path.join(output_dir, "apsp_reduction.png"),
    )
    # 방향 선택이 각 vertex의 in/out 균형을 얼마나 안정적으로 유지하는지 비교한다.
    plot_flow_stability(
        sizes,
        results["random"]["flow"],
        results["raw_sa"]["flow"],
        results["global"]["flow"],
        results["mr2s"]["flow"],
        mr2s_variants=results.get("mr2s_variants"),
        dnc_strategies=results.get("dnc_strategies"),
        save_path=os.path.join(output_dir, "flow_stability.png"),
    )
    # global QUBO와 clustered QUBO의 변수 수 및 physical qubit 추정치를 비교한다.
    plot_preprocessing_scalability(
        sizes,
        results["global"]["qubo_vars"],
        results["mr2s"]["qubo_vars"],
        results["global"]["subgraph_size"],
        results["mr2s"]["subgraph_size"],
        global_physical=results["global"].get("phys_total"),
        clustered_physical_total=results["mr2s"].get("phys_total"),
        clustered_physical_max=results["mr2s"].get("phys_max"),
        clustered_physical_mean=results["mr2s"].get("phys_mean"),
        clustered_physical_min=results["mr2s"].get("phys_min"),
        mr2s_variants=results.get("mr2s_variants"),
        dnc_strategies=results.get("dnc_strategies"),
        save_path=os.path.join(output_dir, "scalability.png"),
    )
    if "timings" in results:
        # trial별 timing 평균을 size 축으로 모아 stage별 실행 시간을 비교한다.
        timings = results["timings"]
        plot_spent_time(
            sizes,
            timings.get("graph", []),
            timings.get("raw_sa", []),
            timings.get("global_solve", []),
            timings.get("global_embed", []),
            timings.get("clustered_solve", []),
            timings.get("clustered_embed", []),
            timings.get("random", []),
            mr2s_variant_timings=timings,
            dnc_timings=timings,
            save_path=os.path.join(output_dir, "spent_time.png"),
        )
