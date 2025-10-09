"""Command-line entry point for gitc."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

from .git_helpers import (
    GitError,
    get_cached_diff,
    get_git_root,
    run_git,
    run_pre_commit,
    stage_all_tracked,
    stage_paths,
    staged_files,
    unstage_all,
)
from .openai_client import OpenAIError, call_openai
from .prompts import COMMIT_PROMPT, REVIEW_PROMPT


def _print(msg: str, *, stream) -> None:
    stream.write(msg + ("\n" if not msg.endswith("\n") else ""))


def commit_command(args: argparse.Namespace) -> int:
    repo_root = get_git_root()
    prompt = os.environ.get("GITC_COMMIT_PROMPT", COMMIT_PROMPT)
    model = args.model or os.environ.get("GITC_COMMIT_MODEL", "gpt-5-mini")

    preexisting_diff = get_cached_diff()
    we_staged = False

    if preexisting_diff.strip():
        if args.all or args.paths:
            _print("Refusing to stage paths because staged changes already exist.", stream=sys.stderr)
            return 1

    if args.all:
        stage_all_tracked(repo_root)
        we_staged = True
    if args.paths:
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

    try:
        response = call_openai(
            prompt=prompt,
            diff=diff,
            model=model,
            max_tokens=args.max_tokens,
        )
    except OpenAIError as exc:
        if we_staged:
            unstage_all()
        _print(f"OpenAI error: {exc}", stream=sys.stderr)
        return 1

    commit_message = response.text.strip()
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

    try:
        response = call_openai(
            prompt=prompt,
            diff=diff,
            model=model,
            max_tokens=args.max_tokens,
            reasoning_effort=reasoning_effort or None,
        )
    except OpenAIError as exc:
        _print(f"OpenAI error: {exc}", stream=sys.stderr)
        return 1

    text = response.text.strip()
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
    commit_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Stage all tracked changes before committing (git add -u).",
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
