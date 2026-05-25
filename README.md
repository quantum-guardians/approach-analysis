# n-hop-approach-analysis

Analysing the n-hop approach on random directed graphs.

## 기능 (Features)

| 모듈 | 설명 |
|---|---|
| `src/graph_generator.py` | 정점 수와 연결성 확률로 무방향 랜덤 그래프 생성 |
| `src/case_generator.py` | 전수 열거(`generate_strongly_connected_orientations`) 및 무작위 샘플링(`sample_strongly_connected_orientations`) 기반으로 강연결(strongly-connected) 방향 그래프 산출 |
| `src/score_calculator.py` | NumPy 기반 APSP 합계 및 n-hop 이웃 수(n=2,3,4) 계산 |
| `src/visualizer.py` | APSP 점수 간 상관관계 산점도, n-hop 수 / 연결성 비교 그래프, face-k 및 poster 결과 시각화 |
| `src/commands/face_k_analysis.py` | `mr2s-module`의 `FaceCycle`을 활용한 최적 face-cycle target k 분석 |
| `src/commands/poster_results.py` | MR2S poster용 solver 비교 실험 실행 |
| `src/commands/poster_results_solvers.py` | Raw SA, Global QUBO, DnC MR2S, random baseline trial 실행 |
| `src/commands/poster_results_partition_strategy.py` | DnC partition strategy별 실행 및 진단 |
| `src/commands/poster_results_runner.py` | poster-results 집계, cache 재사용, 병렬 실행 |

## 설치 (Installation)

```bash
pip install -r requirements.txt
```

## 사용법 (Usage)

`main.py`는 `analyse`, `nhop-connectivity`, `face-k-analysis`, `poster-results` 서브 커맨드를 제공합니다.

### `analyse` – 단일 그래프의 APSP·n-hop 상관관계 분석

```bash
# 기본 실행 (5 정점, Delaunay 그래프)
python main.py analyse

# 파라미터 지정
python main.py analyse --vertices 5 --connectivity 0.7 --seed 42 --output result.png

# 멀티스레드/청크 사이즈 지정
python main.py analyse --vertices 6 --connectivity 0.7 --workers 8 --chunk-size 4096 --output result.png

# 무작위 샘플링 (정점 수가 커도 일정한 시간 소요)
python main.py analyse --vertices 10 --connectivity 0.5 --seed 42 --max-samples 500 --output result.png

# 멀티프로세스 병렬 실행 (GIL 우회로 CPU 바운드 처리량 향상)
python main.py analyse --vertices 10 --connectivity 0.5 --seed 42 --max-samples 500 --workers 8 --processes --output result.png
```

#### `analyse` CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--vertices` | 5 | 정점 수 |
| `--connectivity` | None | 간선 존재 확률 (0–1). 미지정 시 Delaunay 기반 평면 그래프 생성 |
| `--seed` | None | 재현성을 위한 랜덤 시드 |
| `--output` | `result_v{N}_{p}.png` | 저장할 이미지 경로 |
| `--workers` | CPU 코어 수 | 방향 조합 탐색용 워커 수 |
| `--chunk-size` | 2048 | 워커 작업 단위 방향 조합 수 |
| `--max-samples` | None | 무작위 샘플링 모드: 최대 N개의 강연결 방향 조합을 샘플링 |
| `--min-samples` | 0 | `--max-samples` 사용 시 최소 필요 강연결 방향 조합 수 |
| `--processes` | False | 스레드 대신 프로세스를 사용한 병렬 실행 |
| `--adaptive-chunk-size` | False | 전체 작업량과 워커 수에 따라 청크 사이즈 자동 계산 |

---

### `nhop-connectivity` – 2-hop·3-hop 수와 강연결 비율 비교

여러 Delaunay 평면 그래프를 생성하고, 각 그래프에서 `--num-orientations`개의 방향 조합을 무작위로 샘플링합니다.
각 방향 조합에 대해 2-hop / 3-hop 이웃 수를 계산하고, 동일한 n-hop 값을 가진 방향 조합들 중
**강연결(SC)인 비율**을 y축에, **n-hop 이웃 수**를 x축에 표시하는 산점도를 그립니다.

> **연결성 비율 정의**: n-hop 이웃 수가 `k`인 샘플 방향 조합 중 강연결인 방향 조합의 비율
> = (n-hop 수가 k이고 강연결인 샘플 방향 조합 수) / (n-hop 수가 k인 전체 샘플 방향 조합 수)

```bash
# 기본 실행 (5 정점, 20개 Delaunay 그래프, 그래프당 200개 방향 조합 샘플)
python main.py nhop-connectivity

# 파라미터 지정
python main.py nhop-connectivity --vertices 5 --num-graphs 30 --num-orientations 500 --seed 42 --output nhop.png
```

#### `nhop-connectivity` CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--vertices` | 5 | 각 Delaunay 그래프의 정점 수 |
| `--num-graphs` | 20 | 생성할 그래프 수 |
| `--num-orientations` | 200 | 그래프당 무작위로 샘플링할 방향 조합 수 (2^&#124;E&#124; 이하로 자동 제한) |
| `--seed` | None | 기본 랜덤 시드. 그래프 i는 seed+i 사용 |
| `--output` | `nhop_connectivity_v{N}.png` | 저장할 이미지 경로 |

---

### `face-k-analysis` – 최적 FaceCycle target k 분석

달로네 평면 그래프에서 쌍연결성(biconnectivity)을 유지하면서 간선을 제거한 뒤
`mr2s-module`의 `FaceCycle(target_k)`를 적용합니다.  세 가지 변수:
**그래프 크기**, **간선 제거 비율**, **face cluster 수(target k)**에 대해
강연결 비율(SC ratio)과 정규화 APSP 평균을 계산하고 추이 그래프를 생성합니다.
결과 데이터는 JSON으로 저장되며 최적 k 공식을 담은 Markdown 보고서도 함께 생성됩니다.

```bash
# 기본 실행 (정점 10/20/30, 제거 비율 0~30%, k=1..10)
python main.py face-k-analysis

# 파라미터 지정
python main.py face-k-analysis \
    --sizes 10 20 30 \
    --removal-pcts 0.0 0.1 0.2 0.3 \
    --target-ks 1 2 3 4 5 6 7 8 9 10 \
    --num-graphs 10 \
    --num-samples 200 \
    --seed 42 \
    --output-dir results/face_k_analysis
```

#### `face-k-analysis` CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--sizes` | 10 20 30 | 탐색할 그래프 정점 수 목록 |
| `--removal-pcts` | 0.0 0.1 0.2 0.3 | 간선 제거 비율 목록 (각 0–1 범위) |
| `--target-ks` | 1..10 | FaceCycle target_k 후보 목록 |
| `--num-graphs` | 10 | 조합별 독립 그래프 생성 수 |
| `--num-samples` | 200 | 그래프당 무작위 방향 샘플 수 |
| `--seed` | None | 재현성을 위한 기본 랜덤 시드 |
| `--output-dir` | `results/face_k_analysis` | 결과 JSON·플롯·보고서 저장 디렉토리 |
| `--output` | `<output-dir>/face_k_analysis.png` | 플롯 파일 경로 재정의 |

결과 파일:
- `face_k_results.json` – 전체 수치 결과
- `face_k_analysis.png` – SC 비율 / APSP 추이 2×2 그래프
- `report.md` – 실험 요약 및 경험적 최적 k 공식 보고서

---

### `poster-results` – MR2S poster solver 비교

Raw SA, Global QUBO, MR2S 변형 solver, DnC MR2S, random baseline을 같은 Delaunay 그래프에서 비교합니다.
DnC MR2S는 `mr2s-module==0.0.8`의 `graph_partition_strategy` hook을 사용하며,
현재 결과에는 다음 MR2S 변형과 DnC partition strategy가 같은 solver 비교 그래프 안에 함께 표시됩니다.

| MR2S variant | 설명 |
|---|---|
| `robbin_mr2s` | `Robbin` edge orienter로 초기 방향을 만든 뒤 MR2S QUBO solver 실행 |
| `iterated_local_search_mr2s` | `IteratedLocalSearch` edge orienter로 초기 방향을 만든 뒤 MR2S QUBO solver 실행 |

| DnC strategy | 설명 |
|---|---|
| `poster` | poster 실험용 embedding 진단과 target_k 이진 탐색 strategy |
| `embedding_aware` | `mr2s-module`의 `EmbeddingAwareFaceCyclePartitionStrategy` |
| `degeneracy_pruning` | `mr2s-module`의 `DegeneracyPruningFaceCyclePartitionStrategy` |

```bash
# 현재 poster 결과 재현: size 5, 10, 20 / size당 5개 graph
python main.py poster-results \
    --sizes 5 10 20 \
    --output-dir results/poster \
    --no-cache

# cache를 사용해 누락된 size만 계산하고 기존 poster_results.json과 병합
python main.py poster-results --sizes 5 10 20 --output-dir results/poster

# 기존 결과의 MR2S/DnC 결과만 다시 계산해 병합
python main.py poster-results \
    --sizes 5 10 20 \
    --output-dir results/poster \
    --mr2s-only
```

#### `poster-results` CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--sizes` | 5 10 20 30 | 실험할 그래프 정점 수 목록 |
| `--num-graphs` | 5 | size별 독립 그래프 생성 수 |
| `--seed` | None | 재현성을 위한 기본 랜덤 시드 |
| `--output-dir` | `results/poster` | 결과 JSON·플롯·cache 저장 디렉토리 |
| `--num-workers` | CPU 기반 자동값 | trial 병렬 실행 process 수. 0이면 순차 실행 |
| `--cache-dir` | `<output-dir>/poster_trial_cache` | trial cache 디렉토리 |
| `--no-cache` | False | trial cache 읽기/쓰기를 끄고 재계산 |
| `--mr2s-only` | False | 기존 결과를 유지하고 MR2S/DnC 결과만 다시 계산 |
| `--source-results` | `<output-dir>/poster_results.json` | `--mr2s-only` 병합 기준 결과 파일 |

poster 결과의 기본 저장 경로는 `results/poster/poster_results.json`이며, 위 재현 명령은 size 축 `[5, 10, 20]`을 사용합니다.
`spent_time.png`는 embed/probe 시간을 제외한 solve time만 solver 라인으로 표시합니다.
embed timing은 plot에는 넣지 않지만 JSON의 `timings` 섹션에는 보존합니다.

결과 파일:
- `poster_results.json` – solver별 평균 점수, MR2S 변형, DnC strategy별 결과, timing, partition 진단
- `apsp_reduction.png` – Random / Raw SA / Global / MR2S 변형 / DnC strategy별 normalized APSP 비교
- `flow_stability.png` – Random / Raw SA / Global / MR2S 변형 / DnC strategy별 flow imbalance 비교
- `scalability.png` – Global QUBO와 DnC strategy별 QUBO 변수·subgraph·physical qubit 비교
- `spent_time.png` – graph generation, Raw SA, Global solve, MR2S 변형, DnC strategy별 solve time, random baseline 비교

## 테스트 (Tests)

```bash
python -m pytest tests/ -v
```

## 프로젝트 구조 (Structure)

```
n-hop-approach-analysis/
├── main.py               # 실행 진입점 (analyse / nhop-connectivity / face-k-analysis / poster-results)
├── requirements.txt
├── src/
│   ├── graph_generator.py
│   ├── case_generator.py
│   ├── score_calculator.py
│   ├── visualizer.py
│   └── commands/
│       ├── analyse.py
│       ├── nhop_connectivity.py
│       ├── face_k_analysis.py
│       ├── poster_results.py
│       ├── poster_results_models.py
│       ├── poster_results_plotting.py
│       ├── poster_results_runner.py
│       ├── poster_results_solvers.py
│       └── poster_results_partition_strategy.py
├── results/
│   ├── face_k_analysis/
│       ├── face_k_results.json
│       ├── face_k_analysis.png
│       └── report.md
│   └── poster/
│       ├── poster_results.json
│       ├── apsp_reduction.png
│       ├── flow_stability.png
│       ├── scalability.png
│       └── spent_time.png
└── tests/
    ├── test_graph_generator.py
    ├── test_case_generator.py
    ├── test_score_calculator.py
    ├── test_visualizer.py
    ├── test_nhop_connectivity_cmd.py
    ├── test_face_k_analysis_cmd.py
    └── test_poster_results_cmd.py
```
