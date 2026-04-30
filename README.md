# n-hop-approach-analysis

Analysing the n-hop approach on random directed graphs.

## 기능 (Features)

| 모듈 | 설명 |
|---|---|
| `src/graph_generator.py` | 정점 수와 연결성 확률로 무방향 랜덤 그래프 생성 |
| `src/case_generator.py` | 전수 열거(`generate_strongly_connected_orientations`) 및 무작위 샘플링(`sample_strongly_connected_orientations`) 기반으로 강연결(strongly-connected) 방향 그래프 산출 |
| `src/score_calculator.py` | NumPy 기반 APSP 합계 및 n-hop 이웃 수(n=2,3,4) 계산 |
| `src/visualizer.py` | APSP 점수 간 상관관계 산점도 및 n-hop 수 / 연결성 비교 그래프 시각화 |

## 설치 (Installation)

```bash
pip install -r requirements.txt
```

## 사용법 (Usage)

`main.py`는 두 개의 서브 커맨드를 제공합니다.

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

## 테스트 (Tests)

```bash
python -m pytest tests/ -v
```

## 프로젝트 구조 (Structure)

```
n-hop-approach-analysis/
├── main.py               # 실행 진입점 (analyse / nhop-connectivity 서브 커맨드)
├── requirements.txt
├── src/
│   ├── graph_generator.py
│   ├── case_generator.py
│   ├── score_calculator.py
│   └── visualizer.py
└── tests/
    ├── test_graph_generator.py
    ├── test_case_generator.py
    ├── test_score_calculator.py
    └── test_visualizer.py
```

