from __future__ import annotations

from unittest.mock import patch

from gitc.cli import _generate_commit_message, _retry_commit_max_tokens
from gitc.openai_client import OpenAIError, OpenAIResponse


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
