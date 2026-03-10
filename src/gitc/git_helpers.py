"""Utilities for interacting with Git from gitc."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class GitError(RuntimeError):
    """Raised when a git operation fails."""


def _git_error_message(exc: subprocess.CalledProcessError) -> str:
    message = exc.stderr or exc.stdout or str(exc)
    return message.strip() or str(exc)


def run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
    echo_output: bool = False,
) -> subprocess.CompletedProcess:
    cmd = ["git", *args]
    try:
        cwd = cwd or get_git_root()
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=check,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GitError(_git_error_message(exc)) from exc
    except FileNotFoundError as exc:  # pragma: no cover - git missing
        raise GitError("git executable not found.") from exc
    if echo_output:
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    return result


def get_git_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GitError(_git_error_message(exc)) from exc
    except FileNotFoundError as exc:  # pragma: no cover - git missing
        raise GitError("git executable not found.") from exc
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
    run_git(["add", "--", *paths], cwd=Path.cwd(), check=True)


def stage_tracked_paths(paths: list[str]) -> None:
    if not paths:
        return
    run_git(["add", "-u", "--", *paths], cwd=Path.cwd(), check=True)


def stage_all_tracked(repo_root: Path) -> None:
    run_git(["add", "-u", str(repo_root)], cwd=repo_root, check=True)


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
