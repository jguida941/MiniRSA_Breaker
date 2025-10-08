# AI-Assisted Quality Playbook

This playbook turns MiniRSA Breaker into a safe harness for AI-generated changes. Follow it to layer automated feedback around every edit so you can ship quickly without losing trust in the math engine or GUI.

## 0. Environment Primer

- Install tooling once: `pip install -e .[dev,gui]`.
- Enable `pre-commit` hooks (`pip install pre-commit && pre-commit install`) so Ruff/Black/mypy/Bandit run automatically before each commit.
- IDE helpers: enable Ruff extension, turn on “format on save”, map `pytest` to your unit-test shortcut.
- Branch discipline: create a fresh feature branch for every AI experiment; never push AI code directly to `main`.
- For fully automated (Codex/CI) runs without Docker, follow the companion runbook in `docs/codex-integration.md` or invoke `make codex-pipeline`.

## 1. Static Gate — Fail Fast

Purpose: catch type, style, and obvious security regressions before runtime.

| Check | Command | Notes |
| --- | --- | --- |
| Ruff lint & format | `ruff check . --fix` | Runs with `pyproject.toml` profile (`E,F,I,B,UP,N,C4,SIM,S` minus `S311`). Re-run after AI code drops in large files. |
| Type safety | `mypy src/mini_rsa tests rsa.py` | Strict on the pure core (`mini_rsa`). PyQt bits are whitelisted via overrides; keep the ignore list short. |
| Security lint | `bandit -r src/mini_rsa rsa.py` | Flag risky crypto/UI patterns (hard-coded secrets, unsafe eval, etc.). |
| Import health | `pytest tests/test_imports.py` | Add/maintain a smoke test that imports every public module to surface circular imports early. |

Automation hooks:

1. Pre-commit (optional): tie `ruff` and `mypy` into a Git hook to block quick mistakes.
2. `make quality` (local wrapper) chains lint → type → security → import smoke → pytest with coverage.
3. CI fast gate (`fast-gates` job) runs the same stack (Ruff, Black, mypy, pytest, import smoke, bandit, pip-audit). Locally, `pre-commit run --all-files` mirrors the lint/type/security steps.

### Tool Reference

- **Ruff (`ruff check . --fix`)** — lints with pyflakes/pycodestyle/isort/F401 equivalents, catches unused variables, import cycles, shadowed names, and can auto-fix simple style regressions.
- **Black (`black --check .`)** — enforces consistent formatting; keeps diffs focused on behaviour when AI rewrites blocks.
- **Mypy (`mypy src/mini_rsa tests rsa.py`)** — strict types on the math core; surfaces mismatched return types, accidental `Any`, or missing branches that AI might introduce.
- **Bandit (`bandit -r src/mini_rsa rsa.py`)** — flags security smells (weak randomness, unsafe eval/exec, shell injections) often missed in generated patches.
- **pip-audit (`pip-audit -r requirements.txt`)** — checks dependencies for known CVEs.
- **Import smoke (`pytest tests/unit/test_imports.py`)** — ensures public modules still import cleanly; catches circular imports and optional-dependency slips early.

## 2. Test Gate — Prove Behaviour

Purpose: ensure features still work and guard against silent math errors introduced by generated patches.

- **Unit tests:** `make test`. Targets 90%+ coverage (configured in coverage report). Focus new assertions around RSA primitives, encoding tables, and GUI adapters.
- **Property-based tests:** `pytest tests/property/test_encrypt_decrypt_property.py` (Hypothesis). Covers round-trip encryption, non-coprime exponents, ciphertext tokenisation, and modulus guards.
- **Golden examples:** store canonical plaintext/ciphertext pairs in `tests/data/` and validate via approval tests when UI copy changes.
- **GUI smoke:** `pytest -m gui` (run under `xvfb-run` in headless CI) to verify signals, thread wiring, and tab layout.
- **Mutation scope:** `make mutation` (overridable via `MUTATE_PATHS=...`). Use it on PRs that touch the core; nightly job runs full mutation suite.

Workflow checklist whenever AI edits core logic:

1. Re-run targeted pytest selection: `pytest tests/test_rsa_invariants.py`.
2. Expand to full suite: `pytest -q`.
3. When confident, trigger mutation surface: `make mutation MUTATE_PATHS=src/mini_rsa/core.py`.

## 3. Runtime Guard — Catch What Tests Miss

Purpose: stop invalid states at runtime and make issues observable.

- **Boundary assertions:** Keep `ValueError` checks in `encrypt_block`/`decrypt_block` to reject non-integer or out-of-range symbols; add unit tests when AI injects new code paths (`make test` exercises these).
- **Input hygiene tests:** `tests/unit/test_core_unit.py::test_encrypt_block_*` and Hypothesis guards ensure message/cipher blocks stay below the modulus and raise Fast when AI-generated code bypasses mapping logic.
- **Logging:** Use structured logging (INFO for UI actions, DEBUG for math) to trace AI regressions during manual runs.
- **Telemetry:** Record mutation score trend (store `mutation_results.txt` artifact in CI nightly job) and investigate drops >5%.

## 4. What Still Slips Through

Even with the layered net, some classes of defects demand specialised checks. Keep these on your radar:

- **Spec misunderstandings:** Tests that assert the wrong requirement let bugs ship. Counter with executable specifications (acceptance tests, BDD) and reviewer sign-off on the test diff.
- **Environment drift:** Prod-only configs, locale/timezone differences, or container vs. host permissions show up only in integration/e2e runs. Use docker-compose smoke suites and run the same images locally and in CI.
- **Concurrency & longevity:** Race conditions, deadlocks, and resource leaks require stress or long-run harnesses. Add soak tests or run pytest under `--maxfail=1` while looping for 10k iterations.
- **Edge data & fuzz:** Very large inputs, NaNs, Unicode edge cases, and malformed ciphertext need fuzz/property coverage. Keep extending Hypothesis strategies and add Atheris/python-afl when parsers grow complex.
- **Third-party drift:** Dependencies and external APIs evolve. Pin with lockfiles, run `pip-audit`, and monitor changelog feeds; Dependabot can open upgrade PRs that must pass the same gates.
- **Security & performance regressions:** Static analysis catches obvious issues, but add `pytest-benchmark` thresholds and optional prowler/zap scans for deeper coverage.

Rule of thumb: aim for a high mutation kill rate, strong property suites, and prod-like integration tests. That makes serious escapes rare, while acknowledging they remain possible.

### Confidence Stack for AI-Augmented Delivery

1. **Freeze the spec in code:** Lead with acceptance/BDD tests, Hypothesis properties, differential checks, and golden artefacts so requirements stay executable.
2. **Let automation scaffold safely:** Prompt AI with failing tests, land the patch on a branch, and let CI (ruff, mypy `--strict`, pytest, mutation, coverage deltas, bandit, pip-audit, perf smoke) decide.
3. **Tight feedback loop:** On failures, feed the concrete error back to the model instead of asking for “improvements.” Keep humans curating specs and reviews.
4. **Integration & prod parity:** Run docker-compose integration suites, Qt smoke tests under `xvfb`, and matrix CI (Py3.10/3.11/3.12, Ubuntu/macOS). Stage/canary deployments with structured logs and metrics.
5. **Measure & adapt:** Track DORA-style metrics (lead time, deployments, change-fail rate, MTTR) to decide where additional automation or human review is needed.

## 5. Continuous Integration Layout

| Job | Trigger | Scope |
| --- | --- | --- |
| `fast-gates` (`.github/workflows/ci.yml`) | PR + push | Checkout → `pip install -e .[dev]` → run Ruff lint/format, Black check, mypy (core + tests), import smoke, pytest (non-GUI) with coverage gate, bandit, and pip-audit. |
| `gui-smoke` (`ci.yml`) | PR + push | Installs GUI extras, runs `xvfb-run -a make gui`. |
| `mutation` (`ci.yml`) | PR + push | Diffs PR vs. base, runs `make mutation` with scoped `MUTATE_PATHS`, uploads mutmut artefacts, enforces survivor threshold. |
| `mutation-nightly` (`.github/workflows/mutation-nightly.yml`) | Scheduled + manual dispatch | Runs full `make mutation` across `src/mini_rsa`, uploads nightly report. |

Add another workflow (`release.yml`) if you need full GUI smoke + mutation before tagging releases.

## 6. Day-to-Day AI Workflow

1. **Frame the request:** prompt the model with failing tests and expectations; prefer patch-level prompts over “write module” blasts.
2. **Drop the patch:** let AI write into a temp branch or scratch buffer; commit after you’ve run gates.
3. **Static gate:** run `make quality` (Ruff + Black + mypy + bandit + pip-audit + import smoke + pytest). Fix findings immediately—AI often misses typing nuances.
4. **Focused tests:** run unit/property tests that touch edited files (`pytest tests/test_*.py -k <module>`).
5. **Full sweep:** run `make test` (or `pytest -q --cov=mini_rsa --cov-report=term-missing`). Investigate coverage dips.
6. **Mutation (targeted):** if AI modified math in `src/mini_rsa/core`, run `make mutation MUTATE_PATHS=src/mini_rsa/core.py`.
7. **Review:** sanity-check the diff yourself; drop TODOs or docstrings to explain non-obvious logic; fill out the PR checklist to confirm hooks/tests/mutation were executed.
8. **Push & watch CI:** ensure GitHub Actions greenlights PR; fix red builds before merging.

## 7. Escalation Paths

- When AI adds new dependencies, update `pyproject.toml` and regenerate lockfiles before running gates.
- If mutation runtime exceeds 20 minutes, narrow scope with `make mutation MUTATE_PATHS=src/mini_rsa/<module>.py`.
- Use `make mutation-clean` to wipe cached Hypothesis/mutmut artefacts when they drown diffs.
- For GUI regressions, capture screen recordings (Qt built-in) and attach to PR for manual verification.
- For prod readiness, layer on acceptance tests, dockerised integration suites, playwright/pytest-qt e2e flows, and performance smoke alerts.

## 8. Backlog & Extensions

- Completed: Hypothesis strategies now cover non-coprime exponents, oversized cipher blocks, and malformed cipher tokens.
- Next: integrate `pytest-randomly` seed storage to reproduce AI-induced flaky tests.
- Completed: `make quality` wrapper chains Ruff → mypy → pytest → bandit + pip-audit + import smoke.
- Next: explore dependency scanning (`pip-audit`) in a weekly cron workflow.

---

**TL;DR:** keep editing speed high by running the static gate in <1 minute, targeted tests in <5, and mutation nightly. The layered “bug net” gives you confidence that AI-generated patches won’t break the cryptography engine or teaching UI.
