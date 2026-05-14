# n-hop-approach-analysis

Analysing the n-hop approach on random directed graphs.

## 기능 (Features)

| 모듈 | 설명 |
|---|---|
| `src/graph_generator.py` | 정점 수와 연결성 확률로 무방향 랜덤 그래프 생성 |
| `src/case_generator.py` | 전수 열거(`generate_strongly_connected_orientations`) 및 무작위 샘플링(`sample_strongly_connected_orientations`) 기반으로 강연결(strongly-connected) 방향 그래프 산출 |
| `src/score_calculator.py` | NumPy 기반 APSP 합계 및 n-hop 이웃 수(n=2,3,4) 계산 |
| `src/visualizer.py` | APSP 점수 간 상관관계 산점도, n-hop 수 / 연결성 비교 그래프, face-k 분석 시각화 |
| `src/commands/face_k_analysis.py` | `mr2s-module`의 `FaceCycle`을 활용한 최적 face-cycle target k 분석 |
| `src/commands/poster_results.py` | MR2S poster용 Raw SA / Global QUBO / DnC MR2S / Random baseline 비교 데이터와 플롯 생성 |

## 설치 (Installation)

```bash
pip install -r requirements.txt
```

## 사용법 (Usage)

`main.py`는 네 개의 서브 커맨드를 제공합니다.

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
| `--num-workers` | min(조합 수, CPU 코어 수) | 조합별 trial 실행용 병렬 워커 수 |

결과 파일:
- `face_k_results.json` – 전체 수치 결과
- `face_k_analysis.png` – SC 비율 / APSP 추이 2×2 그래프
- `report.md` – 실험 요약 및 경험적 최적 k 공식 보고서

---

### `poster-results` – MR2S poster 비교 데이터 생성

Raw SA(`SAMR2SSolver`), Global QUBO(`QuboMR2SSolver`), DnC MR2S(`DnCMr2sSolver`),
Random baseline을 같은 Delaunay 그래프 trial에서 비교합니다. 결과는 JSON으로 저장되고,
APSP reduction, flow stability, preprocessing scalability 플롯을 함께 생성합니다.

Random baseline은 임의 방향 조합을 샘플링합니다. Flow 점수는 강연결 여부와 관계없이 모든 샘플에서 계산하고,
APSP는 강연결 샘플에서만 계산합니다. 강연결 샘플이 없으면 random APSP는 NaN으로 남깁니다.

```bash
# 기본 실행 (정점 100/200/300/400/500, 크기별 5개 그래프)
python main.py poster-results

# 파라미터 지정
python main.py poster-results \
    --sizes 100 200 300 \
    --num-graphs 3 \
    --seed 42 \
    --output-dir results/poster

# DnC MR2S 결과만 다시 계산해서 기존 poster_results.json에 병합
python main.py poster-results \
    --mr2s-only \
    --output-dir results/poster \
    --num-graphs 5 \
    --seed 42

# Random baseline만 다시 계산해서 기존 poster_results.json에 병합
python main.py poster-results \
    --random-only \
    --output-dir results/poster \
    --num-graphs 5 \
    --seed 42
```

#### `poster-results` CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--sizes` | 100 200 300 400 500 | 탐색할 그래프 정점 수 목록. `--random-only`에서 생략하면 기존 `poster_results.json`의 `sizes` 사용 |
| `--num-graphs` | 5 | 크기별 독립 그래프 trial 수 |
| `--seed` | 42 | 재현성을 위한 기본 랜덤 시드. trial seed는 `seed + trial * 100 + n` |
| `--output-dir` | `results/poster` | 결과 JSON·플롯 저장 디렉토리 |
| `--cache-dir` | `<output-dir>/poster_trial_cache` | trial cache 디렉토리. 모드별 기본값은 full run / MR2S-only / random-only마다 다름 |
| `--no-cache` | False | trial cache 읽기·쓰기 비활성화 |
| `--num-workers` | min(trial 수, CPU 코어 수 - 1) | trial 실행용 병렬 worker process 수. `0`이면 순차 실행 |
| `--mr2s-only` | False | DnC MR2S 결과만 재계산해 기존 `poster_results.json`에 병합 |
| `--random-only` | False | Random baseline만 재계산해 기존 `poster_results.json`에 병합 |
| `--source-results` | `<output-dir>/poster_results.json` | `--mr2s-only` 또는 `--random-only` 병합에 사용할 기존 결과 JSON |

결과 파일:
- `poster_results.json` – size별 solver 비교 수치
- `apsp_reduction.png` – Random / Raw SA / Global / DnC MR2S 정규화 APSP 비교
- `flow_stability.png` – solver별 flow imbalance 비교
- `scalability.png` – QUBO 변수 수, subgraph 크기, physical qubit 추정치 비교

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
│       └── poster_results.py
├── results/
│   └── face_k_analysis/
│       ├── face_k_results.json
│       ├── face_k_analysis.png
│       └── report.md
└── tests/
    ├── test_graph_generator.py
    ├── test_case_generator.py
    ├── test_score_calculator.py
    ├── test_visualizer.py
    ├── test_nhop_connectivity_cmd.py
    ├── test_face_k_analysis_cmd.py
    └── test_poster_results_cmd.py
```
