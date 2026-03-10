from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from unittest.mock import patch

from gitc.cli import _generate_commit_message, _retry_commit_max_tokens, commit_command, main
from gitc.openai_client import OpenAIError, OpenAIResponse


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        text=True,
        capture_output=True,
    )


def _init_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "a@b.c")
    _git(repo, "config", "user.name", "t")
    for rel_path, contents in (files or {"a.txt": "one\n"}).items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        _git(repo, "add", rel_path)
    _git(repo, "commit", "-m", "init")
    return repo


def test_retry_commit_max_tokens_scales_and_caps() -> None:
    assert _retry_commit_max_tokens(512) == 2048
    assert _retry_commit_max_tokens(1024) == 4096
    assert _retry_commit_max_tokens(4096) is None


def test_generate_commit_message_retries_on_output_budget_exhaustion() -> None:
    first_error = OpenAIError(
        "OpenAI API returned no text output (status=incomplete; reason=max_output_tokens).",
        status="incomplete",
        reason="max_output_tokens",
    )
    final_response = OpenAIResponse(text="test commit message", raw={})

    with patch("gitc.cli.call_openai", side_effect=[first_error, final_response]) as mocked_call:
        response = _generate_commit_message(
            prompt="prompt",
            diff="diff",
            model="gpt-5-mini",
            max_tokens=512,
        )

    assert response.text == "test commit message"
    assert mocked_call.call_count == 2
    assert mocked_call.call_args_list[0].kwargs["max_tokens"] == 512
    assert mocked_call.call_args_list[1].kwargs["max_tokens"] == 2048
    assert mocked_call.call_args_list[0].kwargs["reasoning_effort"] == "minimal"
    assert mocked_call.call_args_list[1].kwargs["reasoning_effort"] == "minimal"


def test_generate_commit_message_does_not_retry_other_errors() -> None:
    error = OpenAIError("boom", status="failed", reason="server_error")

    with patch("gitc.cli.call_openai", side_effect=error) as mocked_call:
        try:
            _generate_commit_message(
                prompt="prompt",
                diff="diff",
                model="gpt-5-mini",
                max_tokens=512,
            )
        except OpenAIError as exc:
            assert exc is error
        else:
            raise AssertionError("expected OpenAIError")

    assert mocked_call.call_count == 1


def test_main_commit_uses_caller_relative_paths(monkeypatch, tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, {"sub/a.txt": "one\n"})
    (repo / "sub" / "a.txt").write_text("two\n")
    monkeypatch.chdir(repo / "sub")

    with patch("gitc.cli.call_openai", return_value=OpenAIResponse(text="subdir commit", raw={})):
        assert main(["commit", "a.txt"]) == 0

    assert _git(repo, "log", "-1", "--pretty=%s").stdout.strip() == "subdir commit"
    assert _git(repo, "show", "--name-only", "--pretty=", "HEAD").stdout.strip() == "sub/a.txt"


def test_main_commit_honors_commit_msg_hooks(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("two\n")
    hook = repo / ".git" / "hooks" / "commit-msg"
    hook.write_text("#!/bin/sh\necho blocked >&2\nexit 1\n")
    hook.chmod(0o755)
    monkeypatch.chdir(repo)

    with patch("gitc.cli.call_openai", return_value=OpenAIResponse(text="hooked commit", raw={})):
        assert main(["commit", "a.txt"]) == 1

    captured = capsys.readouterr()
    assert "blocked" in captured.err
    assert _git(repo, "log", "-1", "--pretty=%s").stdout.strip() == "init"
    assert _git(repo, "diff", "--name-only", "--cached").stdout.strip() == "a.txt"


def test_main_reports_clean_error_outside_repo(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["staged"]) == 1

    captured = capsys.readouterr()
    assert "not a git repository" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_main_reports_clean_error_for_invalid_path(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(repo)

    assert main(["commit", "does-not-exist"]) == 1

    captured = capsys.readouterr()
    assert "pathspec 'does-not-exist'" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_commit_command_does_not_print_head_sha_when_unstaging(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = _init_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("two\n")
    monkeypatch.chdir(repo)
    args = argparse.Namespace(all=False, update=False, paths=["a.txt"], max_tokens=512, model="gpt-5-mini")

    with patch("gitc.cli.call_openai", side_effect=OpenAIError("boom")):
        assert commit_command(args) == 1

    captured = capsys.readouterr()
    assert head not in captured.out
    assert head not in captured.err
    assert _git(repo, "diff", "--name-only", "--cached").stdout.strip() == ""
