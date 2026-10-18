"""Tests for scripts/deck_figures.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
FIGURES = ["lever_map.png", "policies.png", "indigenise.png", "rul_band.png"]


def _load():
    spec = importlib.util.spec_from_file_location(
        "deck_figures", ROOT / "scripts" / "deck_figures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deck_figures_written(tmp_path: Path):
    _load().main(tmp_path)
    for name in FIGURES:
        path = tmp_path / name
        assert path.exists(), name
        with Image.open(path) as im:
            assert im.width >= 3000, (name, im.size)
    manifest = pd.read_csv(tmp_path / "figures_manifest.csv")
    assert list(manifest["figure"]) == FIGURES


def test_rul_band_never_reads_rul_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load()
    real = pd.read_parquet
    calls: list[dict] = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        df = real(*args, **kwargs)
        assert "rul_true" not in df.columns
        return df

    monkeypatch.setattr(mod.pd, "read_parquet", spy)
    mod.rul_band(tmp_path, mod._wing())
    assert calls, "rul_band did not read the replay parquet"
    for kw in calls:
        assert kw.get("columns") is not None, "rul_band must select columns explicitly"
        assert "rul_true" not in kw["columns"]
