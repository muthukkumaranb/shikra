"""Trivial smoke test to ensure CI can import the package and run pytest."""

import shikra


def test_package_importable() -> None:
    """The shikra package must import without errors."""
    assert shikra.__version__ == "0.1.0"


def test_placeholder() -> None:
    """Placeholder: CI must have at least one passing test."""
    assert 1 + 1 == 2
