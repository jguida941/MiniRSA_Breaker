# Codex Integration Runbook (Docker-Free)

This guide shows how to run the full MiniRSA Breaker quality pipeline without containers. It works both for humans and for Codex sessions executing in a clean checkout.

## 1. Prerequisites

1. Install Python 3.11 (matching the CI matrix baseline). Optional: install 3.10 and 3.12 if you plan to run multi-version checks locally.
2. Install tooling helpers once:
   ```bash
   pip install --upgrade pip
   pip install uv  # optional but fast, https://github.com/astral-sh/uv
   pip install pre-commit nox
   ```
3. Clone the repository and create a virtual environment (either directly or via `uv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e .[dev,gui]
   pre-commit install
   ```

Codex sessions should mimic the same steps: create/activate a virtualenv, install the project with extras, then run the automation script described below.

## 2. Single Entry Point: `make codex-pipeline`

The Makefile exposes an orchestration target that runs the lint/type/security/test stack and produces a JSON summary for Codex or other automation:

```bash
make codex-pipeline            # full gate + mutation (scoped if GIT changes)
make codex-pipeline SKIP_MUTATION=1  # skip the mutation phase
make codex-pipeline MUTATE_PATHS=src/mini_rsa/core.py  # limit mutation scope
```

Internally the target executes:

1. `pre-commit run --all-files` (Ruff, Ruff-format, Black, mypy, Bandit, optional pip-audit, gitleaks).
2. `make quality` (lint → type → security → import smoke → pytest with coverage gate).
3. Optional mutation pass via `make mutation` scoped to changed files.
4. Writes the final status to `codex-report.json`.

Codex can parse `codex-report.json` to decide whether to continue prompting for fixes.

## 3. Orchestration Script

The actual work is orchestrated by `scripts/run_quality.py` (see below). It can be invoked directly for fine-grained control:

```bash
python scripts/run_quality.py \
    --skip-mutation \
    --mutation-paths src/mini_rsa/core.py \
    --output codex-report.json
```

Key behaviours:

- Detects staged/modified files to auto-scope mutation.
- Executes commands sequentially and surfaces the first failure with context.
- Summarises results to stdout and to the JSON file (status + failing stage + command output paths).

## 4. CI Usage (Docker-Free)

GitHub Actions call the same script inside the hosted runners:

- `fast-gates` job runs `python scripts/run_quality.py --skip-mutation`.
- `mutation` job runs `python scripts/run_quality.py --mutation-paths <diff>` and uploads `codex-report.json`.
- Nightly mutation job runs the script without `--skip-mutation`.

Because the workflow mirrors local usage, environment drift stays minimal.

## 5. Local Multi-Version Testing (Optional)

Use `nox` to execute the same gate under multiple Python versions (matching CI):

```bash
nox -s tests-3.10
nox -s tests-3.11
nox -s tests-3.12
```

Nox sessions reuse the Makefile targets internally, so they respect the same thresholds.

## 6. Troubleshooting Checklist

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `PyQt6` import fails | Missing system Qt libs on Linux | `sudo apt-get install libegl1 libgl1` |
| Mutation step times out | Default scope large | Re-run with `make codex-pipeline MUTATE_PATHS=src/mini_rsa/core.py` |
| `pre-commit` hook fails | Format or lint drift | Run `pre-commit run --all-files --hook-stage manual` to auto-fix |
| `pip-audit` fails offline | No internet | Run script with `--skip-pip-audit` (see below) |

## 7. Codex Session Template

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -e .[dev,gui]`
3. `pip install pre-commit`
4. `python scripts/run_quality.py --output codex-report.json`
5. Parse `codex-report.json` and re-prompt if `"status": "failure"` with the specific failing stage.

## 8. Optional Flags

`run_quality.py` supports:

- `--skip-hooks` to bypass `pre-commit` (not recommended except during bootstrap).
- `--skip-pip-audit` to avoid network calls.
- `--skip-mutation` when quick feedback is needed (default for fast CI jobs).

These map to environment variables so you can set `SKIP_MUTATION=1` ahead of calling `make codex-pipeline`.

---

With these pieces in place, both humans and Codex get a deterministic, Docker-free pathway to run the entire quality pipeline and act on the results. Keep this document updated as the tooling evolves.
