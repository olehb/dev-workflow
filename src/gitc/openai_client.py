"""Minimal OpenAI Responses API client for gitc."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI, OpenAIError as SDKOpenAIError


class OpenAIError(RuntimeError):
    """Raised when the OpenAI API reports an error."""

    def __init__(
        self,
        message: str,
        *,
        status: str | None = None,
        reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


@dataclass
class OpenAIResponse:
    text: str
    raw: dict


def _normalise_base_url(api_base: str | None) -> str | None:
    if not api_base:
        return None
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _build_client(api_key: str, api_base: str | None) -> OpenAI:
    kwargs = {"api_key": api_key}
    normalised = _normalise_base_url(api_base)
    if normalised:
        kwargs["base_url"] = normalised
    return OpenAI(**kwargs)


def _extract_text_from_output(raw: Dict[str, Any]) -> str:
    output = raw.get("output")
    if not isinstance(output, list):
        return ""
    fragments: List[str] = []
    for item in output:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if chunk.get("type") == "output_text":
                text = chunk.get("text")
                if isinstance(text, str):
                    fragments.append(text)
    return "\n".join(fragments).strip()


def call_openai(
    *,
    prompt: str,
    diff: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> OpenAIResponse:
    """Send a prompt+diff to the OpenAI Responses API and return the text output."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIError("OPENAI_API_KEY environment variable is not set.")

    api_base = os.environ.get("OPENAI_API_BASE")
    client = _build_client(api_key, api_base)

    combined_input = f"{prompt}\n\n{diff}"
    kwargs: dict[str, object] = {
        "model": model,
        "input": combined_input,
        "max_output_tokens": int(max_tokens),
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        response = client.responses.create(**kwargs)
    except SDKOpenAIError as exc:
        raise OpenAIError(str(exc)) from exc

    raw = response.model_dump()
    text = (response.output_text or "").strip()
    if not text:
        text = _extract_text_from_output(raw)
    if not text:
        status = raw.get("status")
        reason = raw.get("incomplete_details", {}).get("reason")
        prompt_tokens = raw.get("usage", {}).get("input_tokens")
        completion_tokens = raw.get("usage", {}).get("output_tokens")
        details = []
        if status:
            details.append(f"status={status}")
        if reason:
            details.append(f"reason={reason}")
        if prompt_tokens is not None and completion_tokens is not None:
            details.append(f"usage={prompt_tokens}/{completion_tokens} tokens")
        joined = "; ".join(details) if details else "no extra details"
        raise OpenAIError(
            f"OpenAI API returned no text output ({joined}).",
            status=status,
            reason=reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return OpenAIResponse(text=text, raw=raw)
