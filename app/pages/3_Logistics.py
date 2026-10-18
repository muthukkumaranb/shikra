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

from app.lib.data import (  # noqa: E402
    load_config,
    load_spares_alloc,
    load_spares_compare,
    load_supply_shock,
)

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Logistics",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
st.title("Logistics View")
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
spares_compare = load_spares_compare()

if (
    spares_alloc is not None
    and not spares_alloc.empty
    and spares_compare is not None
    and not spares_compare.empty
):
    available_mults = sorted(float(m) for m in spares_alloc["pipeline_mult"].unique())

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        sel_mult = st.radio(
            "Pipeline Resupply Speed Multiplier",
            options=available_mults,
            index=available_mults.index(1.0) if 1.0 in available_mults else 0,
            format_func=lambda x: f"{x}× speed ({'Baseline' if x == 1.0 else ('Fast' if x < 1.0 else 'Strained')})",
            horizontal=True,
        )

    alloc_sub = spares_alloc[spares_alloc["pipeline_mult"] == sel_mult].copy()

    # Paired bar charts: Base per squadron and Depot stock
    st.markdown("#### Current vs VARI-METRIC Stock per Part")

    base_df = alloc_sub[["lru_id", "current_base_per_sqn", "vm_base_per_sqn"]].melt(
        id_vars=["lru_id"],
        value_vars=["current_base_per_sqn", "vm_base_per_sqn"],
        var_name="Allocation",
        value_name="Stock",
    )
    base_df["Allocation"] = base_df["Allocation"].map({
        "current_base_per_sqn": "Current",
        "vm_base_per_sqn": "VARI-METRIC",
    })

    depot_df = alloc_sub[["lru_id", "current_depot", "vm_depot"]].melt(
        id_vars=["lru_id"],
        value_vars=["current_depot", "vm_depot"],
        var_name="Allocation",
        value_name="Stock",
    )
    depot_df["Allocation"] = depot_df["Allocation"].map({
        "current_depot": "Current",
        "vm_depot": "VARI-METRIC",
    })

    color_scale = alt.Scale(
        domain=["Current", "VARI-METRIC"], range=["#64748b", "#0ea5e9"]
    )

    chart_base = (
        alt.Chart(base_df)
        .mark_bar()
        .encode(
            x=alt.X("lru_id:N", title="Part (LRU)", sort=None),
            xOffset=alt.XOffset("Allocation:N"),
            y=alt.Y("Stock:Q", title="Base Stock per Squadron (units)"),
            color=alt.Color("Allocation:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("lru_id:N", title="LRU"),
                alt.Tooltip("Allocation:N", title="Allocation"),
                alt.Tooltip("Stock:Q", title="Base Units / Sqn"),
            ],
        )
        .properties(
            title=f"Base Stock per Squadron ({sel_mult}× Pipeline)", height=280
        )
    )

    chart_depot = (
        alt.Chart(depot_df)
        .mark_bar()
        .encode(
            x=alt.X("lru_id:N", title="Part (LRU)", sort=None),
            xOffset=alt.XOffset("Allocation:N"),
            y=alt.Y("Stock:Q", title="Depot Stock (units)"),
            color=alt.Color("Allocation:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("lru_id:N", title="LRU"),
                alt.Tooltip("Allocation:N", title="Allocation"),
                alt.Tooltip("Stock:Q", title="Depot Units"),
            ],
        )
        .properties(
            title=f"Depot Stock ({sel_mult}× Pipeline)", height=280
        )
    )

    c_b, c_d = st.columns(2)
    with c_b:
        st.altair_chart(chart_base, use_container_width=True)
    with c_d:
        st.altair_chart(chart_depot, use_container_width=True)

    # Result Card from results/spares_compare.csv across each pipeline speed
    st.markdown("#### Serviceability & NMCS Bottleneck Comparison")

    cmp_shikra = spares_compare[spares_compare["policy"] == "shikra_v2"].copy()

    cols = st.columns(len(available_mults))
    for i, mult in enumerate(available_mults):
        curr_row = cmp_shikra[
            (cmp_shikra["pipeline_mult"] == mult)
            & (cmp_shikra["stock"] == "current")
        ].iloc[0]
        vm_row = cmp_shikra[
            (cmp_shikra["pipeline_mult"] == mult)
            & (cmp_shikra["stock"] == "vari_metric")
        ].iloc[0]

        diff_pp = float(vm_row["diff_vs_current_pp"])
        diff_lo = float(vm_row["diff_ci_lo"])
        diff_hi = float(vm_row["diff_ci_hi"])

        vm_serv = float(vm_row["serviceability_mean"]) * 100
        curr_serv = float(curr_row["serviceability_mean"]) * 100

        lru_nmcs = float(vm_row["nmcs_lru_sim"])
        curr_lru = float(curr_row["nmcs_lru_sim"])
        eng_nmcs = float(vm_row["nmcs_engine_sim"])
        curr_eng = float(curr_row["nmcs_engine_sim"])
        tot_nmcs = float(vm_row["nmcs_mean"])

        is_active = mult == sel_mult
        border_color = "#0ea5e9" if is_active else "#334155"
        bg_color = "#1e293b" if is_active else "#0f172a"

        with cols[i]:
            card_html = (
                f'<div style="background: {bg_color}; border: 1px solid {border_color}; '
                f'border-radius: 8px; padding: 1rem; color: #f8fafc; font-family: monospace;">'
                f'<div style="font-weight: 700; font-size: 1.05rem; color: #38bdf8; margin-bottom: 0.5rem;">'
                f'{mult}× Pipeline Speed{" (Selected)" if is_active else ""}</div>'
                f'<div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.25rem;">Serviceability Gain</div>'
                f'<div style="font-size: 1.3rem; font-weight: 700; color: #4ade80; margin-bottom: 0.25rem;">'
                f'+{diff_pp:.2f} pp</div>'
                f'<div style="font-size: 0.75rem; color: #cbd5e1; margin-bottom: 0.75rem;">'
                f'95% CI: [{diff_lo:+.2f}, {diff_hi:+.2f}] pp<br>'
                f'Rate: {curr_serv:.1f}% → {vm_serv:.1f}%</div>'
                f'<div style="border-top: 1px solid #334155; padding-top: 0.5rem; font-size: 0.8rem;">'
                f'<div style="color: #94a3b8; margin-bottom: 0.25rem;">NMCS Split (Tails):</div>'
                f'<div>LRU-caused: <b style="color: #f1f5f9;">{lru_nmcs:.2f}</b> <span style="color: #94a3b8;">(was {curr_lru:.2f})</span></div>'
                f'<div>Engine-caused: <b style="color: #f1f5f9;">{eng_nmcs:.2f}</b> <span style="color: #94a3b8;">(was {curr_eng:.2f})</span></div>'
                f'<div>Total NMCS: <b style="color: #38bdf8;">{tot_nmcs:.2f}</b></div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # Caption with exact required text and dynamically generated numbers from CSV
    sub_curr = cmp_shikra[
        (cmp_shikra["pipeline_mult"] == sel_mult)
        & (cmp_shikra["stock"] == "current")
    ].iloc[0]
    sub_vm = cmp_shikra[
        (cmp_shikra["pipeline_mult"] == sel_mult)
        & (cmp_shikra["stock"] == "vari_metric")
    ].iloc[0]
    budget_lakh = float(sub_vm["budget_lakh"])
    eng_share = float(sub_vm["nmcs_engine_sim"] / sub_vm["nmcs_mean"] * 100.0)
    lru_diff = float(sub_curr["nmcs_lru_sim"] - sub_vm["nmcs_lru_sim"])
    eng_diff = float(sub_vm["nmcs_engine_sim"] - sub_curr["nmcs_engine_sim"])

    st.caption(
        f"same budget (₹{budget_lakh:,.0f} Lakh); synthetic wing; the engine queue is the binding constraint "
        f"({eng_share:.1f}% of NMCS is engine-caused at {sel_mult}× pipeline: {float(sub_vm['nmcs_engine_sim']):.2f} engine vs "
        f"{float(sub_vm['nmcs_lru_sim']):.2f} LRU tails; LRU queue cut by {lru_diff:.2f} shifts bottleneck to engines by +{eng_diff:.2f})."
    )
else:
    st.markdown(
        '<div style="background: #374151; border-radius: 8px; padding: 1rem; '
        'color: #9ca3af; text-align: center;">'
        "Spares optimiser: available after merge of T1.6"
        "</div>",
        unsafe_allow_html=True,
    )

