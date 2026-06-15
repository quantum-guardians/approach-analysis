# Project Context

## Overview

- Product: `mr2s` approach-analysis CLI
- Primary users: research and engineering users running graph experiments and poster analyses
- Core domain: random graph generation, orientation sampling, APSP scoring, face-k analysis, poster solver benchmarking
- Runtime environment: Python CLI in a local virtual environment

## Architecture

- Entry points: `main.py`
- Main modules: `src/graph_generator.py`, `src/case_generator.py`, `src/score_calculator.py`, `src/visualizer.py`, `src/commands/`
- Dependency direction: `main.py` → `src/commands/*` → shared graph/scoring/plotting helpers
- External systems: optional Redis, AWS Batch, and S3 for `poster-batch`
- Persistent data: `results/`

## Commands

| Purpose | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Run locally | `python main.py analyse` |
| Format | Not configured |
| Lint | Not configured |
| Type-check | Not configured |
| Unit tests | `python -m pytest tests/ -v` |
| Integration tests | `python -m pytest tests/ -v` |
| Build | Not configured |

## Constraints

- Supported platforms: macOS and Linux environments with Python available; `poster-batch` also needs Redis/AWS access when used
- Compatibility requirements: `mr2s-module==0.1.5`, `networkx`, `numpy`, `scipy`, `matplotlib`
- Performance constraints: keep graph experiments deterministic with explicit seeds when reproducibility matters
- Security or privacy requirements: do not commit local results or cache outputs unless they are intentional fixtures

## Ownership

- Maintainers: TODO
- Sensitive modules: `src/commands/poster_batch/`, cache paths under `results/`, generated experiment artifacts
- Changes requiring explicit review: behavior changes in command runners, cache logic, and result aggregation
