"""SHIKRA — Engineering Officer's View.

Engine health table, RUL band chart, and sign-off queue.
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

from app.lib.audit import append as audit_append  # noqa: E402
from app.lib.audit import get_recent as audit_recent  # noqa: E402
from app.lib.data import load_config, load_params, load_replay_rul  # noqa: E402
from app.lib.state import get_sim_result  # noqa: E402
from app.lib.visuals import render_aircraft_detail_svg  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Engineering Officer",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
st.title("Engineering Officer's View")
st.caption("Synthetic wing · advisory — all recommendations require officer sign-off")

# ── Load data ────────────────────────────────────────────────────────────────
config = load_config()
params = load_params()
R_star = params["R_star"]
warmup = config.get("warmup_days", 365)

# Simulation result (cached)
result = get_sim_result("shikra_v2", seed=1)
daily = result.daily
events = result.events
tail_mc_arr = result.tail_mc
tail_status_list = result.tail_status

total_days = len(daily)
measure_start = warmup
measure_end = total_days - 1

# Day selector — over the measured window
default_day = min(warmup + 120, measure_end)
selected_day = st.slider(
    "Select day",
    min_value=measure_start,
    max_value=measure_end,
    value=default_day,
    help="Day 485 = 365 warm-up + 120 measured",
)

st.divider()

# ── Aircraft Visual Detail ──────────────────────────────────────────────────
st.markdown("### Aircraft Detail")

tail_options = [f"T{i:02d}" for i in range(1, 55)]
c_sel, _ = st.columns([1, 2])
with c_sel:
    selected_tail = st.selectbox("Select Aircraft Tail", tail_options, index=18)  # default T19

tail_idx = int(selected_tail[1:]) - 1
day_status = tail_status_list[selected_day]
tail_st = day_status[tail_idx]

# Check if readiness snapshot_at is available
snapshot_fn = None
try:
    import shikra.strategy.readiness as rd

    if hasattr(rd, "snapshot_at") and hasattr(rd, "forecast"):
        snapshot_fn = rd.snapshot_at
except (ImportError, AttributeError):
    pass

c_img, c_meta = st.columns([1, 1])

if snapshot_fn is None:
    # Before T1.7 merge: show tail-level status only, captioned "engine detail after T1.7 merge"
    with c_img:
        st.caption("engine detail after T1.7 merge")
        svg_detail = render_aircraft_detail_svg(
            selected_tail, tail_st, width=380, height=480
        )
        st.markdown(svg_detail, unsafe_allow_html=True)
    with c_meta:
        st.markdown(f"#### Tail **{selected_tail}** Overview")
        st.markdown(f"- **Assigned Squadron:** S{(tail_idx // 18) + 1}")
        st.markdown(f"- **Aircraft Status:** `{tail_st}` on Day {selected_day}")
        st.markdown(f"- **Removal Threshold R*:** `{R_star}` cycles")
        st.markdown(
            '<div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; color: #94a3b8; margin-top: 1rem;">'
            "<b>Subsystem Health:</b> Live per-engine RUL quantiles (q10/q50/q90) and individual LRU component markers "
            "are available after T1.7 merge."
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Engine Health Table ──────────────────────────────────────────────────────
st.markdown("### Engine Health Table")
st.caption("Engine RUL table (q10/q50/q90) available after T1.7 merge")

# Derive engine state from the simulation
# We re-run to get the detailed engine state at the selected day
# For efficiency, we derive what we can from the events and daily data
replay_rul = load_replay_rul()

# Build engine health table from events up to selected_day
# Show which engines had events and their status
day_mc = tail_mc_arr[selected_day]
day_status = tail_status_list[selected_day]

# Build a table of all tails
health_rows = []
for tail in range(54):
    sqn = tail // 18
    sqn_label = f"S{sqn + 1}"
    tail_label = f"T{tail + 1:02d}"
    status = day_status[tail]

    for eng_pos in range(2):
        pos = tail * 2 + eng_pos

        # Find most recent events for this position up to selected_day
        pos_events = events[
            (events["pos"] == pos) & (events["day"] <= selected_day)
        ].sort_values("day")

        if not pos_events.empty:
            last_event = pos_events.iloc[-1]
            last_event_day = int(last_event["day"])
            last_event_type = last_event["type"]
            cycles_at_event = int(last_event.get("cycle", 0))
            # Estimate current cycles: rough estimate based on days since event
            days_since = selected_day - last_event_day
            est_cycles = max(0, int(days_since * 0.8))  # rough: sortie_demand * MC rate
        else:
            last_event_day = 0
            last_event_type = "none"
            est_cycles = int(selected_day * 0.5)  # rough initial estimate

        # Determine removal status
        removal_note = ""
        if last_event_type == "planned" and last_event_day == selected_day:
            removal_note = "In maintenance (planned)"
        elif last_event_type == "unscheduled" and last_event_day == selected_day:
            removal_note = "In maintenance (unscheduled)"

        health_rows.append({
            "Tail": tail_label,
            "Squadron": sqn_label,
            "Engine Position": eng_pos,
            "Status": status,
            "Last Event Day": str(last_event_day) if not pos_events.empty else "—",
            "Last Event Type": last_event_type if not pos_events.empty else "—",
            "Note": removal_note,
        })

health_df = pd.DataFrame(health_rows)
# Show non-MC aircraft first
health_df["_sort"] = health_df["Status"].map({"MC": 1, "NMCM": 0, "NMCS": 0})
health_df = health_df.sort_values(["_sort", "Squadron", "Tail"]).drop(columns=["_sort"])

st.dataframe(health_df, use_container_width=True, hide_index=True, height=400)

st.markdown(f"**R* (removal threshold):** {R_star} cycles")
st.markdown("**Note:** `rul_true` is never shown — only predictive quantiles (q10/q50/q90).")

st.divider()

# ── RUL Band Chart ───────────────────────────────────────────────────────────
st.markdown("### RUL Band Chart")
st.caption("Predictive quantile bands from replay_rul.parquet. Synthetic wing.")

# Let user pick a replay unit to visualize
units = replay_rul[["subset", "unit"]].drop_duplicates().sort_values(["subset", "unit"])
unit_options = [f"{row['subset']} / unit {row['unit']}" for _, row in units.iterrows()]
selected_unit_str = st.selectbox("Select replay unit", unit_options, index=0)

if selected_unit_str:
    parts = selected_unit_str.split(" / unit ")
    sel_subset = parts[0]
    sel_unit = int(parts[1])

    unit_data = replay_rul[
        (replay_rul["subset"] == sel_subset) & (replay_rul["unit"] == sel_unit)
    ].sort_values("cycle")

    if not unit_data.empty:
        # Melt for Altair: q10, q50, q90 (predictive only — never rul_true)
        band_data = unit_data[["cycle", "q10", "q50", "q90"]].copy()

        # Area chart: q10 to q90 band
        area = (
            alt.Chart(band_data)
            .mark_area(opacity=0.3, color="#3b82f6")
            .encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y("q10:Q", title="RUL (cycles)"),
                y2="q90:Q",
            )
        )

        # q50 line
        median_line = (
            alt.Chart(band_data)
            .mark_line(color="#3b82f6", strokeWidth=2)
            .encode(
                x="cycle:Q",
                y="q50:Q",
            )
        )

        # R* horizontal line
        r_star_df = pd.DataFrame({"y": [R_star]})
        r_star_line = (
            alt.Chart(r_star_df)
            .mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2)
            .encode(y="y:Q")
        )

        rul_chart = (area + median_line + r_star_line).properties(
            width=700,
            height=350,
            title=f"RUL Bands — {sel_subset} unit {sel_unit} (q10/q50/q90, R*={R_star})",
        )
        st.altair_chart(rul_chart, use_container_width=True)
    else:
        st.warning("No data for selected unit.")

st.divider()

# ── Sign-Off Queue ───────────────────────────────────────────────────────────
st.markdown("### Sign-Off Queue")
st.caption("Each recommendation requires explicit officer sign-off. Append-only audit log.")

# Show recent events that may need sign-off
recent_events = events[events["day"] >= selected_day - 7].sort_values("day", ascending=False)

if not recent_events.empty:
    for idx, row in recent_events.head(10).iterrows():
        tail_num = row["pos"] // 2 + 1
        eng_pos = row["pos"] % 2
        rec_id = f"EV-{int(row['day']):04d}-P{int(row['pos']):03d}"

        with st.container():
            cols = st.columns([3, 1, 1, 2])
            cols[0].markdown(
                f"**T{tail_num:02d}** eng {eng_pos} — {row['type']} removal "
                f"(day {int(row['day'])}, cycle {int(row.get('cycle', 0))})"
            )

            if cols[1].button("✅ Approve", key=f"approve_{rec_id}"):
                audit_append(
                    role="Engineering Officer",
                    action=f"{row['type']}_removal",
                    target=f"T{tail_num:02d}/eng{eng_pos}",
                    recommendation_id=rec_id,
                    decision="approved",
                )
                st.toast(f"Approved {rec_id}")
                st.rerun()

            if cols[2].button("⏸️ Defer", key=f"defer_{rec_id}"):
                audit_append(
                    role="Engineering Officer",
                    action=f"{row['type']}_removal",
                    target=f"T{tail_num:02d}/eng{eng_pos}",
                    recommendation_id=rec_id,
                    decision="deferred",
                )
                st.toast(f"Deferred {rec_id}")
                st.rerun()
else:
    st.info("No recent events requiring sign-off near the selected day.")

st.divider()

# ── Audit Log ────────────────────────────────────────────────────────────────
st.markdown("### Recent Audit Log")
recent = audit_recent(20)
if recent:
    audit_df = pd.DataFrame(recent)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
else:
    st.info("No audit entries yet.")
