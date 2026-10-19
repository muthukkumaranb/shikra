"""Smoke tests for the add-ons: Health Replay page and Commander utilisation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.lib.utilisation import POLICIES, utilisation_table

ROOT = Path(__file__).resolve().parent.parent
REPLAY_PAGE = ROOT / "app" / "pages" / "7_Health_Replay.py"
UTIL_LIB = ROOT / "app" / "lib" / "utilisation.py"


def test_health_replay_renders_without_exception():
    at = AppTest.from_file(str(REPLAY_PAGE), default_timeout=60).run(timeout=60)
    assert len(at.exception) == 0, [e.value for e in at.exception]
    # Drag the slider to the last cycle: the removal alert must be shown.
    at.slider(key="hr_cycle").set_value(at.slider(key="hr_cycle").max).run(timeout=60)
    assert len(at.exception) == 0, [e.value for e in at.exception]
    assert any("REMOVAL ADVISED" in m.value for m in at.markdown)


def test_health_replay_calls_theme_apply():
    src = REPLAY_PAGE.read_text(encoding="utf-8")
    assert re.search(r"\btheme\.apply\(", src)
    assert src.index("theme.apply(") > src.index("st.set_page_config(")


def test_addon_sources_do_not_mention_hidden_column():
    for path in (REPLAY_PAGE, UTIL_LIB):
        assert "rul_true" not in path.read_text(encoding="utf-8"), path.name


def test_utilisation_matches_policy_summary_csv():
    summary = pd.read_csv(ROOT / "results" / "policy_summary.csv")
    util = utilisation_table(summary)
    assert util["policy"].tolist() == POLICIES
    assert len(util) == 4
    for row in util.itertuples():
        for metric, prefix in (("sorties_flown", "sorties"), ("flying_task_pct", "task")):
            src = summary[(summary["policy"] == row.policy) & (summary["metric"] == metric)].iloc[0]
            assert getattr(row, f"{prefix}_mean") == src["mean"]
            assert getattr(row, f"{prefix}_ci_lo") == src["ci_lo"]
            assert getattr(row, f"{prefix}_ci_hi") == src["ci_hi"]
