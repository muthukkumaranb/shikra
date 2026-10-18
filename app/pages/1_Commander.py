"""SHIKRA — Commander's View.

Lever map (interactive Altair heatmap), supply-shock ranking,
and readiness forecast panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib.data import load_config, load_heatmap, load_supply_shock  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Commander",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
st.title("Commander's View")
st.caption("Synthetic wing · advisory — all recommendations require officer sign-off")

# ── Load data ────────────────────────────────────────────────────────────────
heatmap_raw = load_heatmap()
config = load_config()
shock_df = load_supply_shock()

pipeline_mult = config.get("pipeline_mult", 1.7)

# Format heatmap units in pp
heatmap_df = heatmap_raw.copy()
heatmap_df["gain_pp"] = heatmap_df["gain_vs_reactive_mean"] * 100
heatmap_df["gain_ci_lo_pp"] = heatmap_df["gain_ci_lo"] * 100
heatmap_df["gain_ci_hi_pp"] = heatmap_df["gain_ci_hi"] * 100
heatmap_df["gain_label"] = heatmap_df["gain_pp"].apply(lambda v: f"{v:+.1f} pp")
heatmap_df["serviceability_pct"] = heatmap_df["serviceability_mean"] * 100
heatmap_df["serviceability_ci_lo_pct"] = heatmap_df["serviceability_ci_lo"] * 100
heatmap_df["serviceability_ci_hi_pct"] = heatmap_df["serviceability_ci_hi"] * 100
heatmap_df["serviceability_label"] = heatmap_df["serviceability_pct"].apply(lambda v: f"{v:.1f}%")

# ── Lever Map ────────────────────────────────────────────────────────────────
st.markdown("### Lever Map")

# Toggle between gain vs reactive and absolute serviceability
show_absolute = st.toggle("Show absolute serviceability", value=False)

if show_absolute:
    color_col = "serviceability_pct"
    color_title = "Serviceability (%)"
    text_col = "serviceability_label"
    tooltip_cols = [
        alt.Tooltip("pipeline_mult:Q", title="Pipeline speed"),
        alt.Tooltip("quality_label:N", title="Prediction quality"),
        alt.Tooltip("serviceability_label:N", title="Serviceability"),
        alt.Tooltip("serviceability_ci_lo_pct:Q", title="CI low (%)", format=".1f"),
        alt.Tooltip("serviceability_ci_hi_pct:Q", title="CI high (%)", format=".1f"),
    ]
else:
    color_col = "gain_pp"
    color_title = "Gain vs reactive (pp)"
    text_col = "gain_label"
    tooltip_cols = [
        alt.Tooltip("pipeline_mult:Q", title="Pipeline speed"),
        alt.Tooltip("quality_label:N", title="Prediction quality"),
        alt.Tooltip("gain_label:N", title="Gain vs reactive"),
        alt.Tooltip("gain_ci_lo_pp:Q", title="CI low (pp)", format="+.1f"),
        alt.Tooltip("gain_ci_hi_pp:Q", title="CI high (pp)", format="+.1f"),
    ]

# Base heatmap with explicit x-axis title
base = alt.Chart(heatmap_df).encode(
    x=alt.X(
        "pipeline_mult:O",
        title="repair & supply time (× today’s baseline; lower = faster)",
        sort="ascending",
    ),
    y=alt.Y("quality_label:N", title="Prediction Quality", sort="ascending"),
)

heatmap_chart = base.mark_rect(stroke="white", strokeWidth=1).encode(
    color=alt.Color(
        f"{color_col}:Q",
        title=color_title,
        scale=alt.Scale(scheme="viridis"),
    ),
    tooltip=tooltip_cols,
)

# Text labels on cells in pp
text_chart = base.mark_text(fontSize=11, fontWeight="bold").encode(
    text=alt.Text(f"{text_col}:N"),
    color=alt.condition(
        alt.datum[color_col] > heatmap_df[color_col].median(),
        alt.value("black"),
        alt.value("white"),
    ),
)

# Mark today's calibrated point
today_df = heatmap_df[
    (heatmap_df["pipeline_mult"] == pipeline_mult) & (heatmap_df["quality"] == 1.0)
]
if not today_df.empty:
    today_mark = (
        alt.Chart(today_df)
        .mark_rect(stroke="red", strokeWidth=3, filled=False)
        .encode(
            x=alt.X("pipeline_mult:O", sort="ascending"),
            y=alt.Y("quality_label:N", sort="ascending"),
        )
    )
    chart = (heatmap_chart + text_chart + today_mark).properties(
        width=500, height=350, title="Lever Map — prediction quality × pipeline speed"
    )
else:
    chart = (heatmap_chart + text_chart).properties(
        width=500, height=350, title="Lever Map — prediction quality × pipeline speed"
    )

st.altair_chart(chart, use_container_width=True)
st.caption(
    f"Red outline marks today's calibrated point (pipeline_mult={pipeline_mult}×, "
    f"SHIKRA quality=1). Synthetic wing."
)

st.divider()

# ── Supply-Shock Ranking ─────────────────────────────────────────────────────
st.markdown("### Supply-Shock Ranking")
st.caption(
    "180-day foreign-supply halt, at 1.0× pipeline speed, synthetic parameters; "
    "the method is the claim."
)

# Bar chart: loss over the halt with CI bars
shock_items = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")
bar_chart = (
    alt.Chart(shock_items)
    .mark_bar(color="#3b82f6")
    .encode(
        x=alt.X("loss_pp_shock_window:Q", title="Loss over halt (pp)"),
        y=alt.Y("lru_id:N", title="LRU", sort="-x"),
        tooltip=[
            alt.Tooltip("lru_id:N", title="LRU"),
            alt.Tooltip("loss_pp_shock_window:Q", title="Loss (pp)", format=".1f"),
            alt.Tooltip("ci_lo:Q", title="CI low", format=".1f"),
            alt.Tooltip("ci_hi:Q", title="CI high", format=".1f"),
            alt.Tooltip("source:N", title="Source"),
        ],
    )
)

error_bars = (
    alt.Chart(shock_items)
    .mark_errorbar(color="white")
    .encode(
        x=alt.X("ci_lo:Q"),
        x2=alt.X2("ci_hi:Q"),
        y=alt.Y("lru_id:N", sort="-x"),
    )
)

st.altair_chart(
    (bar_chart + error_bars).properties(width=500, height=300),
    use_container_width=True,
)

# Table: peak extra grounded and recovery days
st.markdown("**Detailed supply-shock metrics:**")
table_df = shock_df[[
    "lru_id", "source", "loss_pp_shock_window", "peak_extra_grounded",
    "recovery_days", "rank"
]].copy()
table_df.columns = ["LRU", "Source", "Loss (pp)", "Peak extra grounded", "Recovery days", "Rank"]
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.divider()

# ── Readiness Forecast Panel ─────────────────────────────────────────────────
st.markdown("### Readiness Forecast")

try:
    from shikra.strategy.readiness import forecast  # noqa: F401
    has_readiness = True
except (ImportError, AttributeError):
    has_readiness = False

if has_readiness:
    # After merge of T1.7: call forecast and show P(>=N MC) with bands
    st.markdown("#### Squadron Readiness Forecast (D+1 ... D+7)")
    # Compute and display real numbers once available
else:
    st.markdown(
        '<div style="background: #374151; border-radius: 8px; padding: 1rem; '
        'color: #9ca3af; text-align: center;">'
        "Readiness forecast: available after merge of T1.7"
        "</div>",
        unsafe_allow_html=True,
    )
