"""SHIKRA — Home page.

Landing page: what SHIKRA is, the three findings (generated from data),
and the reproduce command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path for imports
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib.data import load_heatmap, load_supply_shock  # noqa: E402
from app.lib.state import get_sim_result, get_wing_state_on_day  # noqa: E402
from app.lib.visuals import render_flight_line_html  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"

st.set_page_config(
    page_title="SHIKRA — Advisory Decision Support",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)

st.sidebar.caption("SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")

# ── Pre-warm Simulation Cache ──────────────────────────────────────────────
with st.spinner("Pre-warming wing simulation cache..."):
    _ = get_sim_result("shikra_v2", seed=1)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="font-size:3rem; margin-bottom:0.2rem; letter-spacing: 2px;">SHIKRA</h1>
        <p style="font-size:1.3rem; color:#888; font-style:italic;">
            "Predict the snag, place the spare, fly the sortie."
        </p>
        <p style="font-size:0.85rem; color:#64748b; margin-top: -0.5rem;">
            Simulation cache pre-warmed for instant offline page access.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Flight-Line View (Opening Shot) ─────────────────────────────────────────
st.markdown("### Fleet Flight-Line View")
st.caption(
    "Operational status across all 54 aircraft on the selected simulation day. "
    "Synthetic wing."
)

col_slider, col_summary = st.columns([2, 3])
with col_slider:
    flight_day = st.slider(
        "Simulation Day (Measured Window: 365–729)",
        min_value=365,
        max_value=729,
        value=485,
        step=1,
        help="Day 485 = Day 120 of the measured window (post-warmup)",
    )

wing_state = get_wing_state_on_day(flight_day, policy="shikra_v2", seed=1)

with col_summary:
    mc_total = wing_state["mc_count"]
    pct_total = (mc_total / 54.0) * 100.0
    st.markdown(
        f'<div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; margin-top: 0.4rem;">'
        f'<span style="color: #94a3b8; font-size: 0.85rem;">Day {flight_day} Fleet Serviceability:</span> '
        f'<span style="font-size: 1.15rem; font-weight: 700; color: #4ade80;">{mc_total} / 54 MC</span> '
        f'<span style="font-size: 0.9rem; color: #38bdf8;">({pct_total:.1f}%)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

flight_line_html = render_flight_line_html(
    wing_state["tail_statuses"], wing_state["sqn_mc"]
)
st.markdown(flight_line_html, unsafe_allow_html=True)
st.caption("synthetic wing")

st.divider()

# ── Three Findings (from data, reusing readme_table.py logic) ──────────────
# Finding 1: The pipeline lever (from heatmap.csv)
heatmap = load_heatmap()
# At the wing's calibrated pipeline_mult=1.7, quality=1 (SHIKRA)
row_17 = heatmap[(heatmap["pipeline_mult"] == 1.7) & (heatmap["quality"] == 1.0)]
if not row_17.empty:
    gain_pp = row_17.iloc[0]["gain_vs_reactive_mean"] * 100
    svc_mean = row_17.iloc[0]["serviceability_mean"] * 100
    reactive_svc = row_17.iloc[0]["reactive_serviceability_mean"] * 100
    finding1 = (
        f"📌 **The pipeline is the lever:** at 1.7× pipeline speed, "
        f"SHIKRA serviceability is {svc_mean:.1f}% vs reactive {reactive_svc:.1f}% "
        f"(gain {gain_pp:+.1f} pp). Prediction warning time is not the binding constraint "
        f"— the engine-spare pipeline is."
    )
else:
    finding1 = "📌 **The pipeline is the lever:** (data unavailable for 1.7× pipeline speed)."

# Finding 2: Calibrated prediction's share of the perfect-prediction gain
# Compare SHIKRA vs Oracle at 0.5× pipeline speed (where there IS a gap)
row_05_shikra = heatmap[(heatmap["pipeline_mult"] == 0.5) & (heatmap["quality"] == 1.0)]
row_05_oracle = heatmap[(heatmap["pipeline_mult"] == 0.5) & (heatmap["quality"] == 0.0)]
if not row_05_shikra.empty and not row_05_oracle.empty:
    shikra_gain = row_05_shikra.iloc[0]["gain_vs_reactive_mean"] * 100
    oracle_gain = row_05_oracle.iloc[0]["gain_vs_reactive_mean"] * 100
    if oracle_gain > 0:
        share = shikra_gain / oracle_gain * 100
        finding2 = (
            f"📌 **Calibrated prediction captures {share:.0f}% of the perfect-prediction gain** "
            f"at 0.5× pipeline speed ({shikra_gain:.1f} pp vs oracle {oracle_gain:.1f} pp)."
        )
    else:
        finding2 = (
            "📌 **Calibrated prediction:** perfect prediction provides no significant gain "
            "at the current pipeline speed."
        )
else:
    finding2 = "📌 **Calibrated prediction share:** (data unavailable)."

# Finding 3: The top indigenisation item (from supply_shock.csv)
shock = load_supply_shock()
top_lrus = shock[shock["lru_id"] != "all_foreign"].sort_values("rank")
if not top_lrus.empty:
    top1 = top_lrus.iloc[0]
    finding3 = (
        f"📌 **Top indigenisation priority: {top1['lru_id']}** — "
        f"a 180-day foreign-supply halt causes {top1['loss_pp_shock_window']:.1f} pp loss "
        f"(peak {top1['peak_extra_grounded']:.0f} extra grounded, "
        f"recovery {top1['recovery_days']:.0f} days). Synthetic parameters; the method is the claim."
    )
else:
    finding3 = "📌 **Top indigenisation item:** (data unavailable)."

st.markdown("### Three Key Findings")
st.info(finding1)
st.info(finding2)
st.info(finding3)

st.divider()

# ── Advisory box ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px;
                padding: 1.2rem 1.5rem; margin: 1rem 0; color: #e2e8f0;">
        <strong>Advisory</strong> · offline · synthetic wing + NASA C-MAPSS ·
        reproduce with <code>make reproduce-sim</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── What is SHIKRA? ────────────────────────────────────────────────────────
st.markdown("### What is SHIKRA?")
st.markdown(
    """
    SHIKRA is an **advisory** decision-support system for an air wing's
    maintenance planning. It predicts engine remaining useful life (RUL)
    using calibrated quantile models trained on NASA C-MAPSS data, then
    simulates a synthetic 54-aircraft wing to evaluate maintenance policies.

    **Pages in this dashboard:**
    - **Commander** — lever map, supply-shock ranking, readiness forecast
    - **Engineering Officer** — engine health table, RUL bands, sign-off queue
    - **Logistics** — stock view, supply-shock exposure, spares allocation
    - **Technician** — snag list for the selected day
    - **Scenario** — inject-failure walkthrough (for demonstration)
    - **Evidence** — gates, policy table, model card, reproducibility

    Every number shown is read from data files or computed live.
    All recommendations require explicit officer sign-off.
    """
)
