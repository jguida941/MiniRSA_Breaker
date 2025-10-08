from __future__ import annotations

import nox

TARGETS = ("src", "tests", "rsa.py")

nox.options.sessions = ("lint", "tests-3.11")
nox.options.reuse_existing_virtualenvs = True


@nox.session(name="lint")
def lint_session(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="tests-3.11", python="3.11")
def tests_311_session(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run(
        "pytest", "-m", "not gui", "--randomly-seed=1", "--cov=mini_rsa", "--cov-report=xml"
    )


@nox.session(name="tests-3.10", python="3.10")
def tests_310_session(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "not gui", "--randomly-seed=1")


@nox.session(name="tests-3.12", python="3.12")
def tests_312_session(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "not gui", "--randomly-seed=1")


@nox.session(name="mutation", python="3.11")
def mutation_session(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("python", "scripts/run_quality.py", "--skip-hooks", "--skip-pip-audit")
