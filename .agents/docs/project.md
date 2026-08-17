# Project Context

Fill this document during project initialization. Agents must verify commands against repository configuration before running them.

## Overview

- Product: n-hop-approach-analysis — an experiment CLI that measures the n-hop
  approach on random graphs and produces the figures used in the MR2S paper and
  poster.
- Repository: https://github.com/quantum-guardians/approach-analysis
- Primary users: the MR2S authors running and reproducing experiments.
- Core domain: random graph generation, strongly-connected orientation sampling,
  APSP and n-hop scoring, and solver comparison.
- Runtime environment: Python 3.11 in a local `.venv`; the `poster-batch`
  workers also run as containers (`Dockerfile.worker`,
  `Dockerfile.task-maker`).

## Architecture

- Entry points: `main.py` registers the subcommands `analyse`,
  `nhop-connectivity`, `face-k-analysis`, `poster-results`, and `poster-batch`.
- Main modules: `src/graph_generator.py` (graph generation),
  `src/case_generator.py` (orientation enumeration and sampling),
  `src/score_calculator.py` (NumPy APSP and n-hop counts),
  `src/visualizer.py` (plots), `src/cache.py`, `src/logging_config.py`, and one
  handler per experiment under `src/commands/`.
- Dependency direction: command handlers depend on the `src/` helpers; helpers
  do not import commands. Every experiment is a subcommand, never a root-level
  script.
- External systems: `mr2s-module` (pinned), and for `poster-batch` a Redis queue
  and an S3 bucket configured through `.env` (see `.env.example`).
- Persistent data: figures and machine-readable records under
  `results/<subcommand-name>/`.

## Commands

| Purpose | Command |
|---|---|
| Install dependencies | `python -m venv .venv && .venv/bin/pip install -r requirements.txt` |
| Run locally | `python main.py analyse` (see `README.md` for every subcommand) |
| Format | TODO — none configured |
| Lint | TODO — none configured |
| Type-check | TODO — none configured |
| Unit tests | `python -m pytest tests/ -v` |
| Integration tests | The long experiment runs themselves, e.g. `python main.py face-k-analysis --output-dir results/face_k_analysis` |
| Build | Not applicable; containers via `docker build -f Dockerfile.worker .` |

## Constraints

- Supported platforms: Linux and macOS with Python 3.11; containers use
  `python:3.11-slim` with `MPLBACKEND=Agg`.
- Compatibility requirements: `mr2s-module` is pinned to `0.1.4` in
  `requirements.txt`. Keep it pinned — `poster-results` imports modules that
  were removed after that version, and solver defaults changed between
  versions, so results are not comparable across them. Any experiment feeding a
  paper must record the resolved `mr2s-module` version in its artifact.
- Performance constraints: minor-embedding searches cost roughly 1,000 seconds
  per failed attempt; long runs belong in the background with output redirected
  to a log, never on an interactive path.
- Security or privacy requirements: AWS and Redis settings come from the
  environment; keep credentials out of source and out of `results/`.

## Ownership

- Maintainers: Yunseong <me@yunseong.dev>
- Sensitive modules: `requirements.txt` (the `mr2s-module` pin),
  `src/score_calculator.py`, `src/commands/poster_results/`
- Changes requiring explicit review: the `mr2s-module` pin, scoring definitions,
  and any change to a figure that is already published in the paper or poster.
