# Contributing to MiniRSA Breaker

Thanks for using automation to sharpen this teaching tool. Follow the steps below to keep AI-assisted changes safe and consistent.

## 1. Before You Start

- Install dev extras once: `pip install -e .[dev,gui]`.
- Install and enable git hooks: `pip install pre-commit` then `pre-commit install`. Run `pre-commit run --all-files` after large refactors.
- Skim the [AI-Assisted Quality Playbook](docs/ai-assisted-quality-playbook.md) for the full layering philosophy.
- For fully automated checks (including when using Codex), run `make codex-pipeline` to execute hooks, quality gates, and mutation testing with a JSON summary.
- Create a dedicated feature branch for each change. Avoid committing AI-generated patches directly to `main`.

## 2. Local Quality Gates

Use the `Makefile` wrappers to stay aligned with CI:

```bash
make quality        # Ruff + Black + mypy + bandit + pip-audit + pytest (not gui)
make gui            # Qt smoke tests (needs PyQt6; use xvfb-run on Linux)
make mutation       # Coverage + mutmut (override scope with MUTATE_PATHS=...)
make mutation-clean # Wipe cached mutmut/Hypothesis artefacts if they clutter diffs
```

Property-based scenarios live in `tests/property/`; add new strategies whenever AI introduces edge cases. For GUI work, add tests under `tests/gui/` so `make gui` continues to pass.

## 3. Submitting Changes

- Use the pull request checklist (see `.github/pull_request_template.md`) to confirm hooks, `make quality`, targeted tests, and mutation scope have all been run.
- Run `make quality` and address any failures.
- If you touched `src/mini_rsa`, run `make mutation MUTATE_PATHS=src/mini_rsa/<file>.py`.
- Commit with meaningful messages and include test notes (e.g., “make quality”).
- Open a pull request and ensure GitHub Actions is green (fast gates + GUI smoke). Nightly mutation runs will comment if coverage regressions slip through.

Questions or proposals? Open a discussion or PR referencing the section of the playbook you’d like to evolve.
