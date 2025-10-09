"""Git helper that drafts commits and reviews using GPT-5."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gitc")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
