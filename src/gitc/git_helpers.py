"""Utilities for interacting with Git from gitc."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """Raised when a git operation fails."""


def run_git(args: list[str], *, cwd: Path | None = None, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    cmd = ["git", *args]
    cwd = cwd or get_git_root()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=check,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as exc:
        raise GitError(exc.stderr or exc.stdout or str(exc)) from exc
    except FileNotFoundError as exc:  # pragma: no cover - git missing
        raise GitError("git executable not found.") from exc
    return result


def get_git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(result.stdout.strip())


def get_cached_diff(*, context: int | None = None) -> str:
    args = ["diff", "--cached"]
    if context is not None:
        args.append(f"-U{int(context)}")
    result = run_git(args, capture_output=True)
    return result.stdout


def stage_paths(paths: list[str]) -> None:
    if not paths:
        return
    run_git(["add", "--verbose", "--", *paths], check=True)


def stage_tracked_paths(paths: list[str]) -> None:
    if not paths:
        return
    run_git(["add", "-u", "--", *paths], check=True)


def stage_all_tracked(repo_root: Path) -> None:
    run_git(["add", "-u", str(repo_root)], check=True)


def staged_files() -> list[str]:
    result = run_git(["diff", "--name-only", "--cached"], capture_output=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def unstage_all() -> None:
    try:
        run_git(["rev-parse", "--verify", "HEAD"], check=True)
    except GitError:
        run_git(["rm", "-r", "--cached", "."], check=False)
        return
    run_git(["reset", "-q"], check=False)


def run_pre_commit(repo_root: Path) -> None:
    config = repo_root / ".pre-commit-config.yaml"
    if not config.exists():
        return
    if not staged_files():
        return
    if shutil.which("pre-commit") is None:
        raise GitError(
            "pre-commit not found but .pre-commit-config.yaml exists. "
            "Install via 'pip install pre-commit' (or pipx)."
        )
    try:
        subprocess.run(
            ["pre-commit", "run"],
            cwd=str(repo_root),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GitError("pre-commit hooks failed.") from exc
