"""Prompt templates used by gitc."""

COMMIT_PROMPT = (
    "Summarize the following code changes as a concise commit message. "
    "Aim for a single sentence under 50 words."
)

REVIEW_PROMPT = (
    "Review the following code changes and report only substantive issues "
    "(correctness, security, performance, reliability). For each bug, start a new "
    "line with one of: '### 🐞 bug:' (low/medium), '### 🐞🐞 bug:' (high), or "
    "'### 🐞🐞🐞 bug:' (critical). Describe impact, evidence (include file and line "
    "if possible), and a concrete fix. If no issues, reply with 'LGTM'."
)
