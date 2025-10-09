"""Minimal OpenAI Responses API client for gitc."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class OpenAIError(RuntimeError):
    """Raised when the OpenAI API reports an error."""


@dataclass
class OpenAIResponse:
    text: str
    raw: dict


def _build_request(api_base: str, api_key: str, payload: dict) -> urllib.request.Request:
    data = json.dumps(payload).encode("utf-8")
    url = api_base.rstrip("/") + "/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return urllib.request.Request(url=url, data=data, headers=headers, method="POST")


def _parse_text_from_response(data: dict) -> str:
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    fragments: list[str] = []
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

    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com")

    combined_input = f"{prompt}\n\n{diff}"
    payload: dict[str, object] = {
        "model": model,
        "input": combined_input,
        "max_output_tokens": int(max_tokens),
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    request = _build_request(api_base, api_key, payload)
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(error_text)
        except json.JSONDecodeError:
            raise OpenAIError(f"OpenAI API error (HTTP {exc.code}): {error_text}") from exc
        message = error_payload.get("error", {}).get("message") or error_payload.get("message")
        status = error_payload.get("status")
        details = error_payload.get("incomplete_details", {}).get("reason")
        error_bits = [f"HTTP {exc.code}"]
        if status:
            error_bits.append(f"status={status}")
        if details:
            error_bits.append(f"reason={details}")
        if message:
            error_bits.append(message)
        raise OpenAIError(" - ".join(error_bits)) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network errors hard to simulate
        raise OpenAIError(f"Failed to reach OpenAI API: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenAIError("Failed to decode OpenAI response as JSON.") from exc

    text = _parse_text_from_response(data)
    if not text:
        message = data.get("error", {}).get("message") or data.get("message")
        status = data.get("status")
        details = data.get("incomplete_details", {}).get("reason")
        error_bits = []
        if status:
            error_bits.append(f"status={status}")
        if details:
            error_bits.append(f"reason={details}")
        if message:
            error_bits.append(message)
        joined = " - ".join(error_bits) if error_bits else "unknown"
        raise OpenAIError(f"OpenAI API returned no text output ({joined}).")

    return OpenAIResponse(text=text, raw=data)
