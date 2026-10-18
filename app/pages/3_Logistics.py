"""SHIKRA — Logistics View.

Stock view (engines and 10 LRUs), supply-shock exposure per LRU,
and VARI-METRIC allocation panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib.data import load_config, load_spares_alloc, load_supply_shock  # noqa: E402

st.set_page_config(page_title="SHIKRA — Logistics", page_icon="🦅", layout="wide")
st.title("📦 Logistics View")
st.caption("Synthetic wing · advisory — all recommendations require officer sign-off")

config = load_config()
shock_df = load_supply_shock()

# ── Stock View ───────────────────────────────────────────────────────────────
st.markdown("### Stock View")
st.caption(
    "Base stock per squadron and depot stock for engines and LRUs. "
    "Values from wing.yaml configuration. Synthetic wing."
)

# Engine stock summary
eng_cfg = config.get("spares", {}).get("engines", {})
eng_base = eng_cfg.get("base_per_sqn", 1)
eng_depot = eng_cfg.get("depot", 3)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Engine Base Stock (S1)", f"{eng_base}")
col2.metric("Engine Base Stock (S2)", f"{eng_base}")
col3.metric("Engine Base Stock (S3)", f"{eng_base}")
col4.metric("Engine Depot Stock", f"{eng_depot}")

st.markdown("#### LRU Inventory & Characteristics")
lrus_cfg = config.get("lrus", [])
stock_rows = []
for lru in lrus_cfg:
    lid = lru.get("id", "")
    desc = lru.get("description", lid)
    src = lru.get("source", "domestic")
    cost = lru.get("cost_lakh", 0)
    tat = lru.get("repair_tat_days", 0)
    lead = lru.get("lead_time_days", 0)
    s_base = lru.get("stock_base", 0)
    s_depot = lru.get("stock_depot", 0)
    s_total = s_base * 3 + s_depot

    stock_rows.append({
        "LRU ID": lid,
        "Description": desc,
        "Source": src.upper(),
        "Cost (₹ Lakh)": cost,
        "Repair TAT (days)": tat,
        "Lead Time (days)": lead,
        "Base Stock (per Sqn)": s_base,
        "Depot Stock": s_depot,
        "Total Wing Stock": s_total,
    })

stock_table = pd.DataFrame(stock_rows)
st.dataframe(stock_table, use_container_width=True, hide_index=True)

st.divider()

# ── Supply-Shock Exposure per LRU ───────────────────────────────────────────
st.markdown("### Supply-Shock Exposure per LRU")
st.caption(
    "180-day foreign-supply halt, at 1.0× pipeline speed, synthetic parameters; "
    "the method is the claim."
)

lru_shock = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")

# Bar chart of loss pp per LRU
bar_chart = (
    alt.Chart(lru_shock)
    .mark_bar(color="#f97316")
    .encode(
        x=alt.X("loss_pp_shock_window:Q", title="Loss over shock window (pp)"),
        y=alt.Y("lru_id:N", title="LRU", sort="-x"),
        tooltip=[
            alt.Tooltip("lru_id:N", title="LRU"),
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("loss_pp_shock_window:Q", title="Loss (pp)", format=".1f"),
            alt.Tooltip("peak_extra_grounded:Q", title="Peak extra grounded", format=".1f"),
            alt.Tooltip("recovery_days:Q", title="Recovery days", format=".1f"),
        ],
    )
)

st.altair_chart(bar_chart.properties(height=280), use_container_width=True)

# Detailed LRU shock metrics table
display_shock = lru_shock[[
    "rank", "lru_id", "source", "loss_pp_shock_window", "peak_extra_grounded",
    "recovery_days", "failures_in_shock_window_mean", "stock_base_total", "stock_depot"
]].copy()
display_shock.columns = [
    "Rank", "LRU", "Source", "Loss (pp)", "Peak Extra Grounded",
    "Recovery Days", "Failures in Shock", "Total Base Stock", "Depot Stock"
]
st.dataframe(display_shock, use_container_width=True, hide_index=True)

st.divider()

# ── VARI-METRIC Allocation ──────────────────────────────────────────────────
st.markdown("### VARI-METRIC Spares Allocation")

spares_alloc = load_spares_alloc()
if spares_alloc is not None and not spares_alloc.empty:
    st.info("VARI-METRIC spares allocation model loaded.")
    st.dataframe(spares_alloc, use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="background: #374151; border-radius: 8px; padding: 1rem; '
        'color: #9ca3af; text-align: center;">'
        "Spares optimiser: available after merge of T1.6"
        "</div>",
        unsafe_allow_html=True,
    )
