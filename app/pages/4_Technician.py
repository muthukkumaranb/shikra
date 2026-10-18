"""SHIKRA — Technician's View.

Snag list for the selected day: tails that are non-MC with reason,
squadron, and expected return day.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib.data import load_config  # noqa: E402
from app.lib.state import get_sim_result  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Technician",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
st.title("Technician's View — Snag List")
st.caption("Synthetic wing · advisory — all recommendations require officer sign-off")
st.sidebar.caption("SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")

config = load_config()
warmup = config.get("warmup_days", 365)

# Simulation data
result = get_sim_result("shikra_v2", seed=1)
daily = result.daily
events = result.events
tail_mc_arr = result.tail_mc
tail_status_list = result.tail_status

total_days = len(daily)
measure_start = warmup
measure_end = total_days - 1
default_day = min(warmup + 120, measure_end)

selected_day = st.slider(
    "Select day",
    min_value=measure_start,
    max_value=measure_end,
    value=default_day,
    help="Day 485 = 365 warm-up + 120 measured",
)

st.divider()

# Calculate squadron MC counts on selected_day
sqn_mc_counts = {}
for s in range(3):
    sqn_mc = int(sum(tail_mc_arr[selected_day][s * 18 : (s + 1) * 18]))
    sqn_mc_counts[f"S{s + 1}"] = sqn_mc

# Squadron summary metrics
cols = st.columns(3)
for idx, (sqn_name, mc_cnt) in enumerate(sqn_mc_counts.items()):
    cols[idx].metric(f"{sqn_name} Readiness", f"{mc_cnt} / 18 MC", delta=f"{18 - mc_cnt} Snags", delta_color="inverse")

st.markdown("### Active Groundings & Snags")
st.caption("Sorted by squadron with fewest aircraft MC, then by soonest expected return. Synthetic wing.")

# Identify non-MC tails
snags = []
for tail in range(54):
    if not tail_mc_arr[selected_day][tail]:
        sqn_idx = tail // 18
        sqn_label = f"S{sqn_idx + 1}"
        tail_label = f"T{tail + 1:02d}"
        status = tail_status_list[selected_day][tail]

        # Find relevant engine event for this tail
        tail_events = events[
            (events["pos"] // 2 == tail) & (events["day"] <= selected_day)
        ].sort_values("day")

        reason = "Awaiting maintenance slot or spare parts"
        expected_return = selected_day + 2  # default estimate

        if not tail_events.empty:
            last_ev = tail_events.iloc[-1]
            ev_day = int(last_ev["day"])
            ev_type = last_ev["type"]

            if status == "NMCM":
                change_days = 2 if ev_type == "planned" else 5
                reason = f"Engine change in progress ({ev_type})"
                expected_return = max(selected_day + 1, ev_day + change_days)
            elif status == "NMCS":
                reason = "Waiting for spare engine from depot"
                expected_return = max(selected_day + 1, ev_day + 5)
        else:
            if status == "NMCM":
                reason = "LRU maintenance / replacement in progress"
                expected_return = selected_day + 1
            elif status == "NMCS":
                reason = "Waiting for LRU spare transfer"
                expected_return = selected_day + 3

        snags.append({
            "Tail": tail_label,
            "Squadron": sqn_label,
            "Status": status,
            "Reason": reason,
            "Expected Return Day": expected_return,
            "_sqn_mc": sqn_mc_counts[sqn_label],
            "_tail_num": tail,
        })

if snags:
    snag_df = pd.DataFrame(snags)
    # Sort by: 1) Squadron with fewest MC aircraft (ascending _sqn_mc),
    # 2) Soonest expected return (ascending Expected Return Day)
    snag_df = snag_df.sort_values(
        ["_sqn_mc", "Expected Return Day", "Squadron", "Tail"]
    ).drop(columns=["_sqn_mc", "_tail_num"])

    # Display large, readable cards or table
    for _, row in snag_df.iterrows():
        status_color = "#ef4444" if row["Status"] == "NMCS" else "#f59e0b"
        st.markdown(
            f"""
            <div style="background: #1e293b; border-left: 6px solid {status_color};
                        border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 1.6rem; font-weight: bold; color: #f8fafc;">
                        {row['Tail']} ({row['Squadron']})
                    </span>
                    <span style="font-size: 1.1rem; font-weight: 600; color: {status_color};
                                 background: rgba(255,255,255,0.08); padding: 0.2rem 0.6rem; border-radius: 4px;">
                        {row['Status']}
                    </span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 1.15rem; color: #cbd5e1;">
                    <strong>Reason:</strong> {row['Reason']}
                </div>
                <div style="margin-top: 0.3rem; font-size: 1.05rem; color: #94a3b8;">
                    <strong>Expected Return:</strong> Day {row['Expected Return Day']}
                    (in {row['Expected Return Day'] - selected_day} days)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Tabular Summary")
    st.dataframe(snag_df, use_container_width=True, hide_index=True)
else:
    st.success(f"All 54 aircraft are Mission Capable (MC) on day {selected_day}! No snags reported.")
