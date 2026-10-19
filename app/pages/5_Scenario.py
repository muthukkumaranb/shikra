"""SHIKRA — Failure Injection Walkthrough Scenario.

Interactive stepper demonstrating deterministic degradation injection,
RUL band crossing, removal probability surge, pre-positioning transfer,
hangar plan update, and Engineering Officer sign-off.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import theme  # noqa: E402
from app.lib.audit import append as audit_append  # noqa: E402
from app.lib.audit import get_recent as audit_recent  # noqa: E402
from app.lib.data import load_config, load_params  # noqa: E402
from app.lib.readiness import (  # noqa: E402
    get_forecast,
    get_readiness_module,
    get_snapshot,
    load_readiness_calib_coverage,
)
from app.lib.state import get_deterministic_injection_trajectory, get_sim_result  # noqa: E402
from shikra.spares.demand import removal_prob  # noqa: E402

start_time = time.perf_counter()

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Scenario Walkthrough",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("scenario")
theme.header(
    "Scenario · failure-injection walkthrough",
    "What happens, step by step, when one engine starts to degrade faster than expected?",
    "Deterministic demonstration of predictive advisory workflow: injection, RUL band crossing, "
    "removal-probability rise, pre-positioning and hangar plan, then officer sign-off. Synthetic wing.",
    "artifacts/replay_rul.parquet · artifacts/policy_params.json · config/wing.yaml · "
    "readiness snapshot day 485 (shikra_v2, seed 1) · app_data audit log",
)

config = load_config()
params = load_params()
R_star = int(params["R_star"])
pm = config.get("pipeline_mult", 1.7)
depot_to_base_days = config.get("spares", {}).get("engines", {}).get("depot_to_base_days", 3)
planned_days = config.get("maintenance", {}).get("planned_change_days", 2)
sortie_p = config.get("ops", {}).get("sortie_demand", 0.8)

# Fixed planning horizon in cycles for lead-time aware pre-positioning:
h_horizon = (depot_to_base_days + planned_days) * pm * sortie_p

# ── Simulation Baseline (Day 485, Seed 1, Policy shikra_v2) ──────────────────
DAY = 485
sim_res = get_sim_result("shikra_v2", seed=1)
tail_mc = sim_res.tail_mc[DAY]
theme.status_strip(strip, day=DAY, mc=int(tail_mc.sum()))

# Determine first MC tail in S2 (tails 18 to 35)
s2_mc_tails = [t for t in range(18, 36) if tail_mc[t]]
default_tail_idx = s2_mc_tails[0] if s2_mc_tails else 18
default_tail_str = f"T{default_tail_idx + 1:02d}"

# Initialize session state for stepper
if "scenario_step" not in st.session_state:
    st.session_state.scenario_step = 1
if "injected" not in st.session_state:
    st.session_state.injected = False
if "selected_tail" not in st.session_state:
    st.session_state.selected_tail = default_tail_str
if "selected_eng" not in st.session_state:
    st.session_state.selected_eng = 0

# Stepper Navigation
step_names = [
    "1. Baseline Wing State",
    "2. Failure Injection",
    "3. RUL Band & Threshold Crossing",
    "4. Removal Probability Surge",
    "5. Pre-positioning & Hangar Plan",
    "6. EO Sign-Off",
]

_cur = st.session_state.scenario_step
theme.html(
    '<div class="sk-steps">'
    + "".join(
        f'<div class="sk-step{" sk-on" if _cur == i else (" sk-past" if i < _cur else "")}">{name}</div>'
        for i, name in enumerate(step_names, 1)
    )
    + "</div>"
)

# ── Deterministic Replay Unit Selection Rule ──────────────────────────────────
# Deterministic injection rule:
# Replay units are filtered from replay_rul.parquet where predictive q10 is between
# (R_star + 15) and (R_star + 30). Rows are sorted by (subset, unit, cycle).
# The first matching unit and starting cycle are chosen deterministically.
inj_subset, inj_unit, inj_cycle, inj_traj = get_deterministic_injection_trajectory(R_star)

# ── Step 1: Baseline Wing State ─────────────────────────────────────────────
if st.session_state.scenario_step == 1:
    st.markdown("### Step 1: Baseline Wing State (Day 485)")
    st.caption("Day 485 = 365 warm-up + 120 measured days. Seed 1, Policy shikra_v2. Synthetic wing.")

    mc_s1 = int(sum(tail_mc[0:18]))
    mc_s2 = int(sum(tail_mc[18:36]))
    mc_s3 = int(sum(tail_mc[36:54]))
    total_mc = mc_s1 + mc_s2 + mc_s3

    theme.kpi_row([
        theme.kpi("Squadron 1 (S1)", f"{mc_s1}", unit="/ 18 MC", extra=theme.bar(mc_s1 / 18 * 100)),
        theme.kpi("Squadron 2 (S2)", f"{mc_s2}", unit="/ 18 MC", extra=theme.bar(mc_s2 / 18 * 100)),
        theme.kpi("Squadron 3 (S3)", f"{mc_s3}", unit="/ 18 MC", extra=theme.bar(mc_s3 / 18 * 100)),
        theme.kpi("Total Fleet MC", f"{total_mc}", unit="/ 54 MC", tone="blue", sub=f"day {DAY}"),
    ])

    st.markdown("#### Select Aircraft and Engine for Injection")
    tail_options = [f"T{i + 1:02d}" for i in range(18, 36) if tail_mc[i]]
    st.session_state.selected_tail = st.selectbox(
        "Target Tail (Squadron 2 MC aircraft)",
        tail_options,
        index=0 if default_tail_str in tail_options else 0,
    )
    st.session_state.selected_eng = st.radio("Engine Position", [0, 1], horizontal=True)

    if st.button("🚀 Inject Degradation", type="primary"):
        st.session_state.injected = True
        st.session_state.scenario_step = 2
        st.rerun()

# ── Step 2: Degradation Injection ───────────────────────────────────────────
elif st.session_state.scenario_step == 2:
    st.markdown("### Step 2: Anomaly Injected into Telemetry")
    st.caption("Engine telemetry updated with accelerated degradation profile. Synthetic wing.")

    st.info(
        f"**Target:** {st.session_state.selected_tail} (S2), Engine Position {st.session_state.selected_eng}\n\n"
        f"**Trajectory Injected:** NASA C-MAPSS `{inj_subset}` unit `{inj_unit}` at cycle `{inj_cycle}`.\n\n"
        f"**Initial State at Day 485:** q10 = {inj_traj.iloc[0]['q10']:.1f} cycles "
        f"(within target window 15–30 cycles above R*={R_star})."
    )

    st.markdown(
        """
        ```python
        # Deterministic injection rule:
        # Replay units are filtered from replay_rul.parquet where predictive q10 is between
        # (R_star + 15) and (R_star + 30). Rows are sorted by (subset, unit, cycle).
        # The first matching unit and starting cycle are chosen deterministically.
        ```
        """
    )

    col_nav1, col_nav2 = st.columns([1, 1])
    if col_nav1.button("⬅️ Previous"):
        st.session_state.scenario_step = 1
        st.rerun()
    if col_nav2.button("➡️ Next: View RUL Band Crossing", type="primary"):
        st.session_state.scenario_step = 3
        st.rerun()

# ── Step 3: RUL Band & Threshold Crossing ────────────────────────────────────
elif st.session_state.scenario_step == 3:
    st.markdown(f"### Step 3: RUL Quantile Bands — {st.session_state.selected_tail} Engine {st.session_state.selected_eng}")
    st.caption(f"Predictive q10 crosses R*={R_star} threshold as operational sorties accumulate. Synthetic wing.")

    # Slice trajectory for the next 40 cycles
    plot_df = inj_traj.head(40).copy()
    plot_df["day_offset"] = ((plot_df["cycle"] - inj_cycle) / sortie_p).astype(int)
    plot_df["sim_day"] = DAY + plot_df["day_offset"]

    # Crossing point
    cross_df = plot_df[plot_df["q10"] <= R_star]
    cross_cycle = int(cross_df.iloc[0]["cycle"]) if not cross_df.empty else "N/A"
    cross_day = int(cross_df.iloc[0]["sim_day"]) if not cross_df.empty else "N/A"

    if cross_day != "N/A":
        st.warning(f"⚠️ **R* Threshold Alert:** q10 crosses R* ({R_star} cycles) at Day **{cross_day}** (Cycle {cross_cycle}). Removal due!")

    # Altair chart: q10 to q90 band + median line + R* rule
    band = (
        alt.Chart(plot_df)
        .mark_area(opacity=0.25, color="#38bdf8")
        .encode(
            x=alt.X("sim_day:Q", title="Simulation Day"),
            y=alt.Y("q10:Q", title="RUL (cycles)"),
            y2="q90:Q",
        )
    )
    median = (
        alt.Chart(plot_df)
        .mark_line(color="#0284c7", strokeWidth=2)
        .encode(x="sim_day:Q", y="q50:Q", tooltip=[
            alt.Tooltip("sim_day:Q", title="Day"),
            alt.Tooltip("cycle:Q", title="Engine Cycle"),
            alt.Tooltip("q10:Q", title="q10 RUL", format=".1f"),
            alt.Tooltip("q50:Q", title="q50 (median)", format=".1f"),
            alt.Tooltip("q90:Q", title="q90 RUL", format=".1f"),
        ])
    )
    r_rule = (
        alt.Chart(pd.DataFrame({"y": [R_star]}))
        .mark_rule(color="#ef4444", strokeDash=[4, 4], strokeWidth=2)
        .encode(y="y:Q")
    )

    rul_chart = (band + median + r_rule).properties(height=320)
    st.altair_chart(rul_chart, use_container_width=True)
    st.caption(f"Red dashed line: R* = {R_star} cycles. Only calibrated predictive quantiles are displayed.")

    col_nav1, col_nav2 = st.columns([1, 1])
    if col_nav1.button("⬅️ Previous"):
        st.session_state.scenario_step = 2
        st.rerun()
    if col_nav2.button("➡️ Next: Removal Probability Surge", type="primary"):
        st.session_state.scenario_step = 4
        st.rerun()

# ── Step 4: Removal Probability Surge ────────────────────────────────────────
elif st.session_state.scenario_step == 4:
    st.markdown("### Step 4: Removal Probability Calculation")
    st.caption(
        f"Calculated via shikra.spares.demand.removal_prob over planning horizon h={h_horizon:.1f} cycles. "
        "Synthetic wing."
    )

    plot_df = inj_traj.head(40).copy()
    plot_df["day_offset"] = ((plot_df["cycle"] - inj_cycle) / sortie_p).astype(int)
    plot_df["sim_day"] = DAY + plot_df["day_offset"]
    plot_df["removal_prob"] = [
        removal_prob(row["q10"], row["q50"], row["q90"], h_horizon)
        for _, row in plot_df.iterrows()
    ]

    prob_chart = (
        alt.Chart(plot_df)
        .mark_line(color="#f97316", strokeWidth=3)
        .encode(
            x=alt.X("sim_day:Q", title="Simulation Day"),
            y=alt.Y("removal_prob:Q", title="P(Removal within horizon h)", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("sim_day:Q", title="Day"),
                alt.Tooltip("removal_prob:Q", title="Removal Probability", format=".2%"),
            ],
        )
        .properties(height=280)
    )

    st.altair_chart(prob_chart, use_container_width=True)

    curr_p = plot_df.iloc[0]["removal_prob"]
    future_p = plot_df.iloc[-1]["removal_prob"]
    st.markdown(
        f"**Expected demand surge:** Removal probability increases from **{curr_p:.1%}** at Day {DAY} "
        f"to **{future_p:.1%}**, exceeding squadron on-hand spare threshold. "
        f"A pre-positioning transfer is recommended for officer sign-off."
    )

    col_nav1, col_nav2 = st.columns([1, 1])
    if col_nav1.button("⬅️ Previous"):
        st.session_state.scenario_step = 3
        st.rerun()
    if col_nav2.button("➡️ Next: Pre-Positioning & Hangar Plan", type="primary"):
        st.session_state.scenario_step = 5
        st.rerun()

# ── Step 5: Pre-Positioning & Hangar Plan ────────────────────────────────────
elif st.session_state.scenario_step == 5:
    st.markdown("### Step 5: Pre-Positioning Request & Hangar Slot Allocation")
    st.caption("Lead-time aware spare dispatch and hangar scheduling. Synthetic wing.")

    transit_days = int(depot_to_base_days * pm)
    arrival_day = DAY + transit_days

    # Check readiness module
    readiness_mod = get_readiness_module()
    snap = get_snapshot("shikra_v2", seed=1, day=DAY) if readiness_mod else None

    if snap is not None:
        depot_engines = int(snap.get("stock", {}).get("engines", {}).get("depot", 0))
        if depot_engines > 0:
            transfer_status_html = '<span style="color:#22c55e; font-weight:bold;">dispatched; in transit</span>'
            transfer_note = f"Depot stock: {depot_engines} available on shelf"
        else:
            transfer_status_html = '<span style="color:#ef4444; font-weight:bold;">no depot engine available; pending repair completion</span>'
            transfer_note = "no depot engine available (0 on shelf; depot repair pipeline active)"
    else:
        transfer_status_html = '<span style="color:#f59e0b; font-weight:bold;">request raised; pending EO approval</span>'
        transfer_note = "Depot stock: available after T1.7 merge"

    theme.html(
        f"""
        <div class="sk-advisory">
          <div class="sk-ah">Pre-Positioning Transfer Order</div>
          <div class="sk-kv"><span>Source</span><b>Central Depot</b></div>
          <div class="sk-kv"><span>Destination</span><b>Squadron 2 Base (S2)</b></div>
          <div class="sk-kv"><span>Item</span><b>Serviceable Spare Engine</b></div>
          <div class="sk-kv"><span>Transit Time</span><b>{transit_days} days ({depot_to_base_days}d × {pm}× pipeline)</b></div>
          <div class="sk-kv"><span>Estimated Arrival</span><b>Day {arrival_day}</b></div>
          <div class="sk-kv"><span>Status</span><b>{transfer_status_html}</b></div>
          <div class="sk-kv"><span>Depot Status</span><b>{transfer_note}</b></div>
        </div>
        """
    )

    st.markdown("#### Squadron 2 Hangar Plan (3 Slots Available)")

    if snap is not None:
        # Real hangar state from snapshot on Day 485
        st.caption("Live hangar state from Day 485 simulation snapshot. Synthetic wing.")

        # S2 is squadron index 1 (tails 18..35)
        s2_tails_snap = [t for t in snap["tails"] if t["sqn"] == 1]
        s2_maint = [t for t in s2_tails_snap if t["status"] in ("NMCM", "NMCS")]

        hangar_rows = []
        # Slot 1
        if len(s2_maint) > 0:
            m1 = s2_maint[0]
            ret_d = m1.get("return_day", "—")
            comp_d = str(ret_d) if (isinstance(ret_d, int) and ret_d > 0) else "—"
            hangar_rows.append({
                "Slot": "Slot 1",
                "Current Tail": f"T{m1['tail']+1:02d}",
                "Status": f"Active Maintenance ({m1['status']})",
                "Completion Day": comp_d,
            })
        else:
            hangar_rows.append({
                "Slot": "Slot 1",
                "Current Tail": "Vacant",
                "Status": "Standby",
                "Completion Day": "—",
            })

        # Slot 2 (Reserved for injected engine change on T19)
        hangar_rows.append({
            "Slot": "Slot 2",
            "Current Tail": f"{st.session_state.selected_tail} (Reserved)",
            "Status": "Scheduled Engine Change",
            "Completion Day": str(arrival_day + planned_days),
        })

        # Slot 3
        if len(s2_maint) > 1:
            m2 = s2_maint[1]
            ret_d = m2.get("return_day", "—")
            comp_d = str(ret_d) if (isinstance(ret_d, int) and ret_d > 0) else "—"
            hangar_rows.append({
                "Slot": "Slot 3",
                "Current Tail": f"T{m2['tail']+1:02d}",
                "Status": f"Active Maintenance ({m2['status']})",
                "Completion Day": comp_d,
            })
        else:
            hangar_rows.append({
                "Slot": "Slot 3",
                "Current Tail": "Vacant",
                "Status": "Standby / Emergency Buffer",
                "Completion Day": "—",
            })
        st.dataframe(
            pd.DataFrame(hangar_rows), use_container_width=True, hide_index=True
        )
    else:
        # Fallback before T1.7 merge
        st.caption("illustrative layout; live hangar state after T1.7 merge")

        s2_tails = list(range(18, 36))
        s2_nmcm_tails = [
            f"T{t+1:02d}"
            for t in s2_tails
            if sim_res.tail_status[DAY][t] == "NMCM"
        ]
        s2_nmcs_tails = [
            f"T{t+1:02d}"
            for t in s2_tails
            if sim_res.tail_status[DAY][t] == "NMCS"
        ]

        slot1_tail = (
            s2_nmcm_tails[0]
            if s2_nmcm_tails
            else (s2_nmcs_tails[0] if s2_nmcs_tails else "Vacant")
        )
        slot1_status = (
            "Active Maintenance (Engine Change)"
            if s2_nmcm_tails
            else ("Awaiting Spare / Inspection" if s2_nmcs_tails else "Standby")
        )
        slot1_comp = "—"

        hangar_data = [
            {
                "Slot": "Slot 1",
                "Current Tail": slot1_tail,
                "Status": slot1_status,
                "Completion Day": slot1_comp,
            },
            {
                "Slot": "Slot 2",
                "Current Tail": f"{st.session_state.selected_tail} (Reserved)",
                "Status": "Scheduled Engine Change",
                "Completion Day": str(arrival_day + planned_days),
            },
            {
                "Slot": "Slot 3",
                "Current Tail": "Vacant",
                "Status": "Standby / Emergency Buffer",
                "Completion Day": "—",
            },
        ]
        st.dataframe(
            pd.DataFrame(hangar_data), use_container_width=True, hide_index=True
        )

    # ── D+3 Squadron Readiness Forecast ─────────────────────────────────────
    st.markdown("#### D+3 Squadron Readiness Forecast")

    if readiness_mod is not None and snap is not None:
        with st.spinner("Computing D+3 readiness distribution (before vs after injection)..."):
            snap_before = copy.deepcopy(snap)
            snap_after = copy.deepcopy(snap)
            # Injected action: dispatch engine and schedule planned change
            snap_after["queues"]["engine_transfers"].append((DAY + transit_days, 1))
            # Set planned return for injected tail T19 (tail index 18)
            snap_after["tails"][18]["return_day"] = arrival_day + planned_days

            fc_before = get_forecast(
                snap_before,
                config,
                params,
                policy="shikra",
                horizon_days=3,
                n=150,
                seed=42,
            )
            fc_after = get_forecast(
                snap_after,
                config,
                params,
                policy="shikra",
                horizon_days=3,
                n=150,
                seed=42,
            )

        if fc_before is not None and fc_after is not None:
            row_b = fc_before[
                (fc_before["lead_days"] == 3) & (fc_before["sqn"] == 2)
            ].iloc[0]
            row_a = fc_after[
                (fc_after["lead_days"] == 3) & (fc_after["sqn"] == 2)
            ].iloc[0]

            exp_b = float(row_b["expected_mc"])
            q10_b = float(row_b["mc_q10"])
            q90_b = float(row_b["mc_q90"])

            exp_a = float(row_a["expected_mc"])
            q10_a = float(row_a["mc_q10"])
            q90_a = float(row_a["mc_q90"])

            delta_exp = exp_a - exp_b
            theme.kpi_row([
                theme.kpi(
                    "BEFORE INJECTION (Baseline S2 at D+3)",
                    f"{exp_b:.2f}",
                    unit="MC",
                    tone="blue",
                    sub=f"Forecast band (q10–q90): [{q10_b:.1f}, {q90_b:.1f}] MC",
                ),
                theme.kpi(
                    "AFTER INJECTED CHANGE (Mitigated S2 at D+3)",
                    f"{exp_a:.2f}",
                    unit=f"MC ({delta_exp:+.2f})",
                    tone="green" if delta_exp >= 0 else "red",
                    sub=f"Forecast band (q10–q90): [{q10_a:.1f}, {q90_a:.1f}] MC",
                ),
            ])

            cov_x = load_readiness_calib_coverage()
            if cov_x:
                st.caption(f"raw forecast band; held-out coverage {cov_x}")
            else:
                st.caption("coverage: pending")
        else:
            st.info("Readiness forecast computation pending.")
    else:
        theme.html(
            '<div class="sk-panel"><p class="sk-note">Readiness forecast: available after merge of T1.7</p></div>'
        )


    col_nav1, col_nav2 = st.columns([1, 1])
    if col_nav1.button("⬅️ Previous"):
        st.session_state.scenario_step = 4
        st.rerun()
    if col_nav2.button("➡️ Next: Engineering Officer Sign-Off", type="primary"):
        st.session_state.scenario_step = 6
        st.rerun()

# ── Step 6: EO Sign-Off ──────────────────────────────────────────────────────
elif st.session_state.scenario_step == 6:
    st.markdown("### Step 6: Engineering Officer Formal Sign-Off")
    st.caption("SHIKRA is advisory. All actions require explicit human sign-off recorded in audit log.")

    rec_id = f"REC-SCENARIO-{st.session_state.selected_tail}-ENG{st.session_state.selected_eng}"

    theme.html(
        f"""
        <div class="sk-advisory">
          <div class="sk-ah">Decision Package: {rec_id}</div>
          <p><strong>Action:</strong> Dispatch spare engine from Depot to S2 + Schedule hangar slot 2 for {st.session_state.selected_tail}.</p>
          <p><strong>Trigger:</strong> Predictive RUL q10 crossed R*={R_star} cycles.</p>
        </div>
        """
    )

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("✅ Sign Off & Approve Action", type="primary", use_container_width=True):
        audit_append(
            role="Engineering Officer",
            action="preposition_and_hangar_approval",
            target=f"{st.session_state.selected_tail}/eng{st.session_state.selected_eng}",
            recommendation_id=rec_id,
            decision="approved",
            note="Approved pre-positioning and planned replacement ahead of R* breach.",
        )
        st.success(f"Action approved and recorded to append-only audit log: {rec_id}")

    if btn_col2.button("⏸️ Defer Action", use_container_width=True):
        audit_append(
            role="Engineering Officer",
            action="preposition_and_hangar_approval",
            target=f"{st.session_state.selected_tail}/eng{st.session_state.selected_eng}",
            recommendation_id=rec_id,
            decision="deferred",
            note="Deferred pending operational review.",
        )
        st.warning(f"Action deferred and recorded to audit log: {rec_id}")

    st.markdown("#### Recent Audit Log Entries")
    recent_entries = audit_recent(5)
    if recent_entries:
        st.dataframe(pd.DataFrame(recent_entries), use_container_width=True, hide_index=True)

    st.divider()
    if st.button("🔄 Restart Scenario"):
        st.session_state.scenario_step = 1
        st.session_state.injected = False
        st.rerun()

# ── Performance Measurement ──────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
st.caption(f"Measured scenario execution time: {elapsed:.2f} s (< 20.0 s target). Synthetic wing.")
theme.footer()
