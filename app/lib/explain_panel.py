"""'Why is this engine flagged?' panel for the SHIKRA app.

Reads results/explain_contrib.csv (written by ``make explain``) and shows the
top-5 feature contributions to one replay engine's pessimistic (q10)
remaining-life estimate. Shows q10 and the prediction parts only; ground
truth is never read or displayed.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.lib.data import load_replay_rul

ROOT = Path(__file__).parent.parent.parent
CONTRIB_PATH = ROOT / "results" / "explain_contrib.csv"
GLOBAL_PATH = ROOT / "results" / "explain_global.csv"

TITLE = "Why is this engine flagged?"
CAPTION = (
    "Feature contributions to the pessimistic (q10) remaining-life estimate, from the "
    "model's own tree structure (LightGBM SHAP). Explains the model, not the physics. "
    "NASA C-MAPSS replay engine; synthetic wing."
)
# Okabe-Ito colour-blind-safe pair.
UP, DOWN = "#0072B2", "#D55E00"
UP_LABEL, DOWN_LABEL = "pushes q10 up (more life left)", "pushes q10 down (less life left)"


@st.cache_data
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _contrib() -> pd.DataFrame | None:
    if not CONTRIB_PATH.exists():
        return None
    return _load_csv(str(CONTRIB_PATH), CONTRIB_PATH.stat().st_mtime)


def _render_global() -> None:
    """Fallback when per-engine explanations are missing."""
    if not GLOBAL_PATH.exists():
        st.info("Per-engine explanations not generated yet (`make explain`).")
        return
    g = _load_csv(str(GLOBAL_PATH), GLOBAL_PATH.stat().st_mtime)
    st.caption("Per-engine explanations unavailable; showing global feature importance (gain).")
    st.altair_chart(
        alt.Chart(g.head(10))
        .mark_bar(color=UP)
        .encode(
            x=alt.X(f"{g.columns[-1]}:Q", title="Importance (gain)"),
            y=alt.Y(f"{g.columns[0]}:N", sort="-x", title=None),
        ),
        use_container_width=True,
    )


def render_why_flagged(subset: str, unit: int, cycle: int, r_star: float) -> None:
    """Render the panel for one replay engine row (subset, unit, cycle)."""
    st.markdown(f"### {TITLE}")
    df = _contrib()
    if df is None:
        _render_global()
        return
    rows = df[(df["subset"] == subset) & (df["unit"] == int(unit)) & (df["cycle"] == int(cycle))]
    if rows.empty:
        st.info(f"No explanation stored for {subset} unit {unit} at cycle {cycle}.")
        return

    replay = load_replay_rul()
    hit = replay[
        (replay["subset"] == subset) & (replay["unit"] == int(unit)) & (replay["cycle"] == int(cycle))
    ]
    q10 = float(hit["q10"].iloc[0]) if not hit.empty else float(rows["contribution"].sum())

    top = rows[rows["rank"].between(1, 5)].copy()
    top["direction"] = top["contribution"].map(lambda v: UP_LABEL if v >= 0 else DOWN_LABEL)
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution to q10 (cycles)"),
            y=alt.Y(
                "sensor_label:N",
                sort=alt.SortField("rank"),
                title=None,
                axis=alt.Axis(labelLimit=0, labelOverlap=False, labelFontSize=13),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=[UP_LABEL, DOWN_LABEL], range=[UP, DOWN]),
                legend=alt.Legend(title=None, orient="bottom", labelLimit=0, labelFontSize=13),
            ),
            tooltip=["sensor_label", "feature", alt.Tooltip("contribution:Q", format="+.2f")],
        )
        .properties(height=260, title=f"{subset} unit {unit}, cycle {cycle}: top 5 features")
    )
    st.altair_chart(chart, use_container_width=True)

    verdict = "REMOVAL DUE" if q10 <= r_star else "OK"
    st.markdown(f"**q10 = {q10:.1f} cycles vs R\\* = {r_star:g} → {verdict}**")
    st.caption(CAPTION)
    with st.expander("How the parts add up to q10"):
        st.dataframe(
            rows[["rank", "sensor_label", "contribution"]].rename(
                columns={"sensor_label": "part", "contribution": "cycles"}
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_for_tail(snap: dict | None, tail_idx: int, r_star: float) -> None:
    """Hook helper: pick an engine on the selected tail and render its panel."""
    if not snap:
        return
    engines = [e for e in snap.get("engines", []) if e.get("tail") == tail_idx]
    if not engines:
        return
    engines.sort(key=lambda e: float(e["q10"]))  # most at-risk engine is the default
    labels = [f"Engine pos {e['pos'] % 2}: {e['subset']} unit {e['unit']}" for e in engines]
    choice = st.selectbox("Engine to explain", labels, index=0, key="why_flagged_engine")
    eng = engines[labels.index(choice)]
    render_why_flagged(eng["subset"], int(eng["unit"]), int(eng["cycle"]), r_star)
