#!/usr/bin/env python
"""Orchestrate the MiniRSA Breaker quality pipeline without Docker.

- Runs pre-commit hooks (lint, format, type, security) unless skipped.
- Executes the Makefile quality target (ruff → mypy → pytest → bandit → pip-audit).
- Optionally executes mutation testing with automatic scope detection.
- Emits a machine-readable JSON summary for Codex or other automation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "codex_logs"
DEFAULT_REPORT = ROOT / "codex-report.json"
DEFAULT_MUTATION_FALLBACK = "src/mini_rsa/core.py"


@dataclass
class Stage:
    name: str
    command: List[str]
    env: Dict[str, str]
    skip: bool = False
    description: str = ""


@dataclass
class StageResult:
    name: str
    status: str
    command: List[str]
    returncode: int
    log_path: Optional[Path]
    duration_s: float


def run_command(cmd: List[str], *, env: Optional[Dict[str, str]] = None, log_path: Path) -> subprocess.CompletedProcess[str]:
    combined_env = os.environ.copy()
    if env:
        combined_env.update(env)

    process = subprocess.run(
        cmd,
        cwd=ROOT,
        env=combined_env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(process.stdout)
        if process.stderr:
            log_file.write("\n=== STDERR ===\n")
            log_file.write(process.stderr)
    sys.stdout.write(process.stdout)
    if process.stderr:
        sys.stderr.write(process.stderr)
    return process


def detect_mutation_paths(explicit: Optional[Iterable[str]] = None) -> str:
    if explicit:
        cleaned = [p for p in explicit if p]
        return " ".join(cleaned) if cleaned else DEFAULT_MUTATION_FALLBACK

    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        diff_files = {line.strip() for line in diff.stdout.splitlines() if line.strip()}

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        diff_files.update(line.strip() for line in untracked.stdout.splitlines() if line.strip())
    except (FileNotFoundError, OSError):
        diff_files = set()

    candidates = sorted(
        {path for path in diff_files if path.startswith("src/mini_rsa/") and path.endswith(".py")}
    )
    if candidates:
        return " ".join(candidates)
    return DEFAULT_MUTATION_FALLBACK


def build_stage_list(
    *,
    skip_hooks: bool,
    skip_mutation: bool,
    skip_pip_audit: bool,
    mutation_paths: Optional[str],
) -> List[Stage]:
    stages: List[Stage] = []

    if not skip_hooks:
        stages.append(
            Stage(
                name="pre-commit",
                command=["pre-commit", "run", "--all-files"],
                env={},
                description="Run lint/format/type/security hooks via pre-commit",
            )
        )

    quality_env = {}
    if skip_pip_audit:
        quality_env["SKIP_PIP_AUDIT"] = "1"
    stages.append(
        Stage(
            name="quality",
            command=["make", "quality"],
            env=quality_env,
            description="Run lint, type, security, import smoke, and pytest with coverage",
        )
    )

    if not skip_mutation:
        mut_paths = detect_mutation_paths(mutation_paths.split() if mutation_paths else None)
        mutation_env = {"MUTATE_PATHS": mut_paths}
        stages.append(
            Stage(
                name="mutation",
                command=["make", "mutation"],
                env=mutation_env,
                description=f"Run coverage + mutation testing on {mut_paths}",
            )
        )

    return stages


def stage_runner(stages: List[Stage], log_dir: Path) -> List[StageResult]:
    results: List[StageResult] = []
    for index, stage in enumerate(stages, start=1):
        log_path = log_dir / f"{index:02d}-{stage.name}.log"
        start_time = datetime.now(timezone.utc)
        completed = run_command(stage.command, env=stage.env, log_path=log_path)
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        status = "success" if completed.returncode == 0 else "failure"
        results.append(
            StageResult(
                name=stage.name,
                status=status,
                command=stage.command,
                returncode=completed.returncode,
                log_path=log_path,
                duration_s=duration,
            )
        )
        if status == "failure":
            break
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniRSA Breaker quality pipeline")
    parser.add_argument("--output", default=str(DEFAULT_REPORT), help="Path to write JSON summary")
    parser.add_argument(
        "--skip-hooks",
        action="store_true",
        default=os.getenv("SKIP_HOOKS", "").lower() in {"1", "true", "yes"},
        help="Skip pre-commit hooks",
    )
    parser.add_argument(
        "--skip-mutation",
        action="store_true",
        default=os.getenv("SKIP_MUTATION", "").lower() in {"1", "true", "yes"},
        help="Skip mutation testing stage",
    )
    parser.add_argument(
        "--skip-pip-audit",
        action="store_true",
        default=os.getenv("SKIP_PIP_AUDIT", "").lower() in {"1", "true", "yes"},
        help="Skip pip-audit during the quality stage",
    )
    parser.add_argument(
        "--mutation-paths",
        default=os.getenv("MUTATE_PATHS", ""),
        help="Whitespace-separated list of files/directories to mutate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.output).resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / timestamp
    stages = build_stage_list(
        skip_hooks=args.skip_hooks,
        skip_mutation=args.skip_mutation,
        skip_pip_audit=args.skip_pip_audit,
        mutation_paths=args.mutation_paths or None,
    )

    results = stage_runner(stages, log_dir)

    overall_status = "success" if results and results[-1].status == "success" else "failure"
    report = {
        "status": overall_status,
        "timestamp": timestamp,
        "stages": [
            {
                "name": result.name,
                "status": result.status,
                "command": result.command,
                "returncode": result.returncode,
                "log_path": str(result.log_path.relative_to(ROOT)) if result.log_path else None,
                "duration_seconds": result.duration_s,
            }
            for result in results
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSummary written to {report_path}")
    if overall_status != "success":
        failed_stage = next((r for r in results if r.status == "failure"), None)
        if failed_stage:
            print(f"Pipeline failed at stage '{failed_stage.name}'. See {failed_stage.log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
