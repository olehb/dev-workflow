# gitc

Python implementation of the `gitc` helper that drafts commit messages and code reviews with GPT-5.

## Installation

```bash
pip install -e .
```

## Usage

- `gitc commit [-a|--all] [paths ...]` – stages optionally, generates a commit message, and creates a commit.
- `gitc review [--effort {minimal,low,medium,high}] [--max-tokens N]` – requests an AI review of the staged diff.

Set `OPENAI_API_KEY` in your environment before running commands.
