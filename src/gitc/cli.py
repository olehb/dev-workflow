"""Command-line entry point for gitc."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from itertools import cycle
from typing import Sequence

from .git_helpers import (
    GitError,
    get_cached_diff,
    get_git_root,
    run_git,
    run_pre_commit,
    stage_all_tracked,
    stage_paths,
    stage_tracked_paths,
    staged_files,
    unstage_all,
)
from .openai_client import OpenAIError, OpenAIResponse, call_openai
from .prompts import COMMIT_PROMPT, REVIEW_PROMPT


RESET = "\033[0m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
COMMIT_REASONING_EFFORT = "minimal"
COMMIT_RETRY_MAX_TOKENS = 4096


def _print(msg: str, *, stream) -> None:
    stream.write(msg + ("\n" if not msg.endswith("\n") else ""))


def _apply_color(text: str, color: str, stream=None) -> str:
    stream = stream or sys.stdout
    if hasattr(stream, "isatty") and stream.isatty():
        return f"{color}{text}{RESET}"
    return text


def _yellow(text: str, *, stream=None) -> str:
    return _apply_color(text, YELLOW, stream=stream)


def _green(text: str, *, stream=None) -> str:
    return _apply_color(text, GREEN, stream=stream)


class ThinkingIndicator:
    """Display a spinner with elapsed time while waiting for the model."""

    def __init__(self, stream, label: str = "Thinking...", interval: float = 0.1) -> None:
        self.stream = stream
        self.label = label
        self.interval = interval
        self.enabled = hasattr(stream, "isatty") and stream.isatty()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._stopped = False
        self._elapsed = 0.0
        self._width = 0

    def start(self) -> None:
        if self._stopped or self._thread is not None:
            return
        self._start_time = time.perf_counter()
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._stopped:
            return self._elapsed
        self._elapsed = time.perf_counter() - self._start_time
        if self.enabled and self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            clear = "\r" + " " * self._width + "\r"
            self.stream.write(clear)
            self.stream.flush()
        self._stopped = True
        return self._elapsed

    def _run(self) -> None:
        frames = cycle("|/-\\")
        while not self._stop_event.is_set():
            frame = next(frames)
            elapsed = time.perf_counter() - self._start_time
            text = f"\r{frame} {self.label} {elapsed:.1f}s"
            self._width = max(self._width, max(len(text) - 1, 0))
            self.stream.write(text)
            self.stream.flush()
            if self._stop_event.wait(self.interval):
                break


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def _format_model_info(model: str, reasoning_effort: str | None, diff: str) -> str:
    size = _format_bytes(len(diff.encode("utf-8")))
    info_parts = [f"Model: {model}", f"diff size: {size}"]
    if reasoning_effort:
        info_parts.append(f"effort: {reasoning_effort}")
    return ", ".join(info_parts)


def _retry_commit_max_tokens(max_tokens: int) -> int | None:
    if max_tokens >= COMMIT_RETRY_MAX_TOKENS:
        return None
    return min(COMMIT_RETRY_MAX_TOKENS, max(max_tokens * 4, 2048))


def _generate_commit_message(prompt: str, diff: str, model: str, max_tokens: int) -> OpenAIResponse:
    try:
        return call_openai(
            prompt=prompt,
            diff=diff,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=COMMIT_REASONING_EFFORT,
        )
    except OpenAIError as exc:
        retry_max_tokens = _retry_commit_max_tokens(max_tokens)
        if exc.reason != "max_output_tokens" or retry_max_tokens is None:
            raise
        _print(
            _yellow(
                f"Retrying commit generation with a larger output budget ({retry_max_tokens} tokens).",
                stream=sys.stdout,
            ),
            stream=sys.stdout,
        )
        return call_openai(
            prompt=prompt,
            diff=diff,
            model=model,
            max_tokens=retry_max_tokens,
            reasoning_effort=COMMIT_REASONING_EFFORT,
        )


def commit_command(args: argparse.Namespace) -> int:
    repo_root = get_git_root()
    prompt = os.environ.get("GITC_COMMIT_PROMPT", COMMIT_PROMPT)
    model = args.model or os.environ.get("GITC_COMMIT_MODEL", "gpt-5-mini")

    preexisting_diff = get_cached_diff()
    we_staged = False

    if preexisting_diff.strip():
        if args.all or args.update or args.paths:
            _print("Refusing to stage paths because staged changes already exist.", stream=sys.stderr)
            return 1

    if args.all:
        stage_all_tracked(repo_root)
        we_staged = True
    elif args.update:
        if args.paths:
            stage_tracked_paths(args.paths)
        else:
            stage_all_tracked(repo_root)
        we_staged = True
    if args.paths and not args.update:
        stage_paths(args.paths)
        we_staged = True

    diff = get_cached_diff()
    if not diff.strip():
        _print("No staged changes found.", stream=sys.stderr)
        return 0

    try:
        run_pre_commit(repo_root)
    except GitError as exc:
        if we_staged:
            unstage_all()
        _print(str(exc), stream=sys.stderr)
        return 1

    diff = get_cached_diff()
    if not diff.strip():
        _print("No staged changes remain after running pre-commit.", stream=sys.stderr)
        return 0

    model_info = _format_model_info(model, COMMIT_REASONING_EFFORT, diff)
    _print(_yellow(model_info, stream=sys.stdout), stream=sys.stdout)
    spinner = ThinkingIndicator(sys.stdout, label="Generating commit message...")
    spinner.start()
    response = None
    openai_error: OpenAIError | None = None
    elapsed = 0.0
    try:
        response = _generate_commit_message(
            prompt=prompt,
            diff=diff,
            model=model,
            max_tokens=args.max_tokens,
        )
    except OpenAIError as exc:
        openai_error = exc
    finally:
        elapsed = spinner.stop()

    if openai_error is not None:
        if we_staged:
            unstage_all()
        _print(f"OpenAI error: {openai_error}", stream=sys.stderr)
        return 1

    _print(_green(f"✓ Model responded in {elapsed:.1f}s", stream=sys.stdout), stream=sys.stdout)

    commit_message = response.text.strip() if response else ""
    if not commit_message:
        if we_staged:
            unstage_all()
        _print("OpenAI returned an empty commit message.", stream=sys.stderr)
        return 1

    try:
        proc = subprocess.run(
            ["git", "commit", "--no-verify", "-F", "-"],
            input=commit_message,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if we_staged:
            unstage_all()
        _print(exc.stderr or "git commit failed.", stream=sys.stderr)
        return 1

    _print("Committed.", stream=sys.stdout)

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            answer = input("Do you want to run 'git push'? [y/N]: ").strip()
        except EOFError:
            answer = ""
        if answer.lower().startswith("y"):
            try:
                run_git(["push"])
            except GitError as exc:
                _print(str(exc), stream=sys.stderr)
                return 1

    return 0


def review_command(args: argparse.Namespace) -> int:
    prompt = os.environ.get("GITC_REVIEW_PROMPT", REVIEW_PROMPT)
    if args.prompt:
        prompt = prompt + "\n\n" + args.prompt

    model = args.model or os.environ.get("GITC_REVIEW_MODEL", "gpt-5")
    reasoning_effort = args.effort or os.environ.get("GITC_REASONING_EFFORT", "low")

    if reasoning_effort not in {"", "minimal", "low", "medium", "high"}:
        _print(
            f"Invalid reasoning effort '{reasoning_effort}'. Expected minimal, low, medium, or high.",
            stream=sys.stderr,
        )
        return 1

    diff = get_cached_diff(context=args.context)
    if not diff.strip():
        _print("No staged changes to review.", stream=sys.stderr)
        return 0

    model_info = _format_model_info(model, reasoning_effort or None, diff)
    _print(_yellow(model_info, stream=sys.stdout), stream=sys.stdout)
    spinner = ThinkingIndicator(sys.stdout, label="Reviewing changes...")
    spinner.start()
    response = None
    openai_error: OpenAIError | None = None
    elapsed = 0.0
    try:
        response = call_openai(
            prompt=prompt,
            diff=diff,
            model=model,
            max_tokens=args.max_tokens,
            reasoning_effort=reasoning_effort or None,
        )
    except OpenAIError as exc:
        openai_error = exc
    finally:
        elapsed = spinner.stop()

    if openai_error is not None:
        _print(f"OpenAI error: {openai_error}", stream=sys.stderr)
        return 1

    _print(_green(f"✓ Model responded in {elapsed:.1f}s", stream=sys.stdout), stream=sys.stdout)

    text = response.text.strip() if response else ""
    if text == "LGTM":
        _print("✨ LGTM!", stream=sys.stdout)
    else:
        _print(text, stream=sys.stdout)
    return 0


def staged_command(_: argparse.Namespace) -> int:
    files = staged_files()
    if not files:
        _print("No staged files.", stream=sys.stdout)
        return 0
    for path in files:
        _print(path, stream=sys.stdout)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gitc", description="Git commit/review helper powered by GPT-5.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commit_parser = subparsers.add_parser("commit", help="Generate a commit message and create a commit.")
    stage_group = commit_parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Stage all tracked changes before committing (git add -u).",
    )
    stage_group.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Stage only tracked changes in the given paths before committing (git add -u -- <paths>).",
    )
    commit_parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to stage before committing.",
    )
    commit_parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("GITC_COMMIT_MAX_TOKENS", "512")),
        help="Maximum number of tokens to request for the commit message.",
    )
    commit_parser.add_argument(
        "--model",
        help="Override the OpenAI model to use for commit messages (default: gpt-5-mini).",
    )
    commit_parser.set_defaults(func=commit_command)

    review_parser = subparsers.add_parser("review", help="Request an AI review of the staged diff.")
    review_parser.add_argument(
        "--effort",
        choices=["minimal", "low", "medium", "high"],
        default=os.environ.get("GITC_REASONING_EFFORT", "low"),
        help="Reasoning effort to request from GPT-5.",
    )
    review_parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("GITC_REVIEW_MAX_TOKENS", "4000")),
        help="Maximum number of tokens to request for the review.",
    )
    review_parser.add_argument(
        "--context",
        type=int,
        default=20,
        help="Number of context lines in the diff (git diff -U value).",
    )
    review_parser.add_argument(
        "--prompt",
        help="Additional instructions appended to the review prompt.",
    )
    review_parser.add_argument(
        "--model",
        help="Override the OpenAI model to use for reviews (default: gpt-5).",
    )
    review_parser.set_defaults(func=review_command)

    staged_parser = subparsers.add_parser("staged", help="List paths currently staged (debug helper).")
    staged_parser.set_defaults(func=staged_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
