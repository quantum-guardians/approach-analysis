# Project Agent Instructions

## Scope and Precedence

This file is the repository-level entrypoint for coding agents.

Read `.agents/docs/project.md` before non-trivial work. Repository-specific commands,
constraints, and narrower instructions take precedence over these template defaults.

## Project Workflow

For non-trivial work, follow:

- `.agents/docs/workflow.md`
- `.agents/docs/testing.md`

For tracked Git work, follow:

- `.agents/docs/issue.md`
- `.agents/docs/branch.md`
- `.agents/docs/commit.md`
- `.agents/docs/pull-request.md`

Use project-local skills when installed and applicable. Skill instructions define
their own triggers, formats, and output paths.

## Project Structure & Module Organization

`main.py` is the CLI entry point and registers the `analyse`, `nhop-connectivity`,
`face-k-analysis`, `poster-results`, `poster-batch`, and `qubo-structure`
subcommands. Core logic lives in `src/`: graph generation in `graph_generator.py`, orientation sampling
in `case_generator.py`, scoring in `score_calculator.py`, plotting in
`visualizer.py`, and command handlers in `src/commands/`. Tests mirror this
layout under `tests/` with files such as `tests/test_graph_generator.py`.
Generated plots and analysis artifacts belong in `results/`; avoid committing
ad hoc local outputs unless they are intentional fixtures or published examples.
Give each experiment its own `results/<subcommand-name>/` directory and write a
machine-readable record next to the plot, so a figure can always be traced back
to the numbers that produced it. Do not add one-off scripts at the repository
root; every experiment is a subcommand under `src/commands/` so that it is
discoverable, documented by `--help`, and testable.

## Build, Test, and Development Commands

Create a virtual environment at `.venv` and install dependencies with
`python -m venv .venv && .venv/bin/pip install -r requirements.txt`.

`mr2s-module` is pinned to an exact version in `requirements.txt`. Keep it
pinned: `poster-results` imports modules that were removed after that version,
so the CLI does not import against `mr2s-module` at HEAD. Any experiment whose
output is used in a paper must record the resolved `mr2s-module` version in its
artifact, because solver defaults have changed between versions and results are
not comparable across them.

Run the full test suite with `python -m pytest tests/ -v`.
Run the default single-graph analysis with `python main.py analyse`.
Run connectivity sampling with `python main.py nhop-connectivity --vertices 5 --num-graphs 20 --num-orientations 200`.
Run the larger face-cycle experiment with `python main.py face-k-analysis --output-dir results/face_k_analysis`.
Run the poster solver comparison with `python main.py poster-results --sizes 5 10 20 --output-dir results/poster --no-cache`.
Run the QUBO structure and hop-length ablation with `python main.py qubo-structure --sizes 100 200 --trials 1 --seed 42`.
Use `--seed <int>` for any experiment that should be reproducible. Long
experiments belong in the background with their output redirected to a log;
minor-embedding searches in particular cost about 1,000 seconds per failed
attempt and should never be placed on an interactive path.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, `snake_case` for functions
and modules, `PascalCase` for test classes, and explicit type hints where practical.
Keep modules focused and prefer small helpers over large command functions.
Preserve the current pattern of concise docstrings, standard-library imports first,
then third-party imports, then local imports. No formatter or linter is configured
in the repository today, so match the surrounding file style closely when editing.

## Testing Guidelines

Add or update pytest coverage for every behavior change. Name new tests
`test_<behavior>.py` or extend the existing module-specific test file.
Prefer deterministic tests with fixed seeds, especially for graph generation
and sampling logic. When changing CLI behavior, add assertions in the
corresponding command test such as `tests/test_nhop_connectivity_cmd.py`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative Conventional Commit prefixes such as `feat:`
and occasional focused refactor commits. Keep commit subjects concise and specific,
for example `feat: add adaptive chunk sizing`. Pull requests should explain the
user-visible or research-impacting change, list validation performed, and include
sample output paths or plots when a visualization changes.
Use branch names in the `<tag>/<issue num>` format, for example `feat/30` or
`fix/31`.
