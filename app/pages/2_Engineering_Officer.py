"""SHIKRA: Engineering Officer's View.

Engine health table, RUL band chart, sign-off queue, audit log and the
"Why is this engine flagged?" panel.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

start_time = time.perf_counter()

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import theme  # noqa: E402
from app.lib.audit import append as audit_append  # noqa: E402
from app.lib.audit import get_recent as audit_recent  # noqa: E402
from app.lib.data import load_config, load_params, load_replay_rul  # noqa: E402
from app.lib.readiness import get_readiness_module, get_snapshot  # noqa: E402
from app.lib.state import get_sim_result  # noqa: E402
from app.lib.visuals import render_aircraft_detail_svg  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA: Engineering Officer",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("engineering_officer")

# ── Load data ────────────────────────────────────────────────────────────────
config = load_config()
params = load_params()
R_star = params["R_star"]
warmup = config.get("warmup_days", 365)
replay_rul = load_replay_rul()

# Simulation result (cached)
result = get_sim_result("shikra_v2", seed=1)
daily = result.daily
events = result.events
tail_mc_arr = result.tail_mc
tail_status_list = result.tail_status

total_days = len(daily)
measure_start = warmup
measure_end = total_days - 1

theme.header(
    "Engineering Officer · engine health",
    "Which engines need a removal decision now, and which can safely keep flying?",
    "Pessimistic (q10), median (q50) and optimistic (q90) remaining-life estimates for every engine "
    "against the removal threshold R*. Synthetic wing · advisory: all recommendations require "
    "officer sign-off.",
    "readiness snapshot (shikra_v2, seed 1) · artifacts/replay_rul.parquet · "
    "artifacts/policy_params.json · results/explain_contrib.csv · app_data audit log",
)

# Day selector — over the measured window
default_day = min(warmup + 120, measure_end)
selected_day = st.slider(
    "Select day",
    min_value=measure_start,
    max_value=measure_end,
    value=default_day,
    help="Day 485 = 365 warm-up + 120 measured",
)

# Check if readiness snapshot_at is available
readiness_mod = get_readiness_module()
snap = get_snapshot("shikra_v2", seed=1, day=selected_day) if readiness_mod else None

day_status = tail_status_list[selected_day]
if snap is not None and "tails" in snap:
    mc_now = sum(1 for t in snap["tails"] if t.get("status") == "MC")
else:
    mc_now = int(tail_mc_arr[selected_day].sum())
theme.status_strip(strip, day=selected_day, mc=mc_now)


def _condition(q10: float) -> str:
    if q10 <= R_star:
        return "Removal Due (q10 ≤ R*)"
    if q10 <= R_star + 15:
        return "Watch (Close to R*)"
    return "Serviceable"


# Recent events that may need sign-off
recent_events = events[events["day"] >= selected_day - 7].sort_values("day", ascending=False)
queue = recent_events.head(10)

# ── KPI row ─────────────────────────────────────────────────────────────────
tiles = []
if snap is not None:
    engs = snap["engines"]
    n_due = sum(1 for e in engs if float(e.get("q10", 0.0)) <= R_star)
    n_watch = sum(1 for e in engs if R_star < float(e.get("q10", 0.0)) <= R_star + 15)
    n_inst = sum(1 for e in engs if e.get("installed"))
    due_tails = sorted({f"T{e['tail'] + 1:02d}" for e in engs if float(e.get("q10", 0.0)) <= R_star and e.get("installed")})
    tiles.append(
        theme.kpi("Engines tracked", f"{len(engs)}", sub=f"{n_inst} installed on wing · day {selected_day}")
    )
    tiles.append(
        theme.kpi(
            "Removal due (q10 ≤ R*)",
            f"{n_due:02d}",
            tone="red" if n_due else "green",
            alert=n_due > 0,
            badge=theme.chip(f"R* = {R_star}", "red" if n_due else "blue"),
            sub=("Installed: " + ", ".join(due_tails[:6]) + (" …" if len(due_tails) > 6 else "")) if due_tails else "none installed",
        )
    )
    tiles.append(
        theme.kpi(
            "Watch (R* < q10 ≤ R*+15)",
            f"{n_watch:02d}",
            tone="amber",
            sub="close to the removal threshold",
        )
    )
tiles.append(
    theme.kpi(
        "Sign-offs in queue",
        f"{len(queue):02d}",
        tone="blue",
        sub=f"removal events from day {selected_day - 7} on (latest 10 shown)",
    )
)
theme.kpi_row(tiles)

col_main, col_rail = st.columns([1.75, 1], gap="medium")

# ═════════════════════════════════ MAIN ══════════════════════════════════════
with col_main:
    # ── Engine Health Table ──────────────────────────────────────────────────
    st.markdown("### Engine Health Table")

    if snap is not None:
        # Live engine RUL table from snapshot (strictly q10/q50/q90 — predictive only)
        engine_rows = []
        for eng in snap["engines"]:
            t_id = eng["tail"]
            q10 = round(float(eng.get("q10", 0.0)), 1)
            q50 = round(float(eng.get("q50", 0.0)), 1)
            q90 = round(float(eng.get("q90", 0.0)), 1)
            if eng.get("in_maintenance"):
                st_text = "In Maintenance"
            elif eng.get("installed"):
                st_text = "Installed on Wing"
            else:
                st_text = "Depot / Shelf Spare"
            engine_rows.append({
                "Tail": f"T{t_id + 1:02d}",
                "Squadron": f"S{eng['sqn'] + 1}",
                "Engine Position": eng["pos"] % 2,
                "Unit": f"{eng.get('subset', 'FD001')} u{eng.get('unit', 0)}",
                "Cycles": eng.get("cycle", 0),
                "Status": st_text,
                "q10 RUL": q10,
                "q50 Median": q50,
                "q90 RUL": q90,
                "Condition": _condition(q10),
            })
        df_eng = pd.DataFrame(engine_rows)
        df_eng["_sort"] = df_eng["Condition"].map({
            "Removal Due (q10 ≤ R*)": 0,
            "Watch (Close to R*)": 1,
            "Serviceable": 2,
        })
        df_eng = df_eng.sort_values(["_sort", "q10 RUL"]).drop(columns=["_sort"])

        # Stitch-style table: RUL prognostic range bar (q10 – q50 – q90) with R* marker
        top = df_eng.head(14)
        scale = max(30.0, float(top["q90 RUL"].max()) * 1.1)
        rows_html = ""
        for _, r in top.iterrows():
            cond = r["Condition"]
            cls = "sk-hot" if cond.startswith("Removal") else ("sk-warm" if cond.startswith("Watch") else "")
            color = theme.SUPPLY_COLOR if cls == "sk-hot" else (theme.MAINT_COLOR if cls == "sk-warm" else theme.MC_COLOR)
            tone = "red" if cls == "sk-hot" else ("amber" if cls == "sk-warm" else "green")
            lo, mid, hi = (max(0.0, float(r[c])) / scale * 100 for c in ("q10 RUL", "q50 Median", "q90 RUL"))
            rs = R_star / scale * 100
            rows_html += (
                f'<tr class="{cls}"><td class="sk-mono"><b>{r["Tail"]}</b> · {r["Squadron"]}</td>'
                f'<td class="sk-mono">ENG-{int(r["Engine Position"])}</td>'
                f'<td class="sk-mono">{theme.esc(r["Unit"])}</td><td class="sk-mono">{int(r["Cycles"])}</td>'
                f'<td>{theme.esc(r["Status"])}</td>'
                f'<td><div class="sk-rul"><div class="sk-ri" style="left:{lo:.1f}%;width:{max(hi - lo, 0.8):.1f}%;background:{color}55;"></div>'
                f'<div class="sk-rm" style="left:calc({mid:.1f}% - 1px);"></div><div class="sk-rs" style="left:{rs:.1f}%;"></div></div>'
                f'<div class="sk-mono" style="font-size:10px;color:var(--outline);margin-top:3px;">{r["q10 RUL"]:.0f} – {r["q50 Median"]:.0f} – {r["q90 RUL"]:.0f} cyc</div></td>'
                f'<td>{theme.chip(cond.split(" (")[0], tone)}</td></tr>'
            )
        theme.html(
            '<div class="sk-panel">'
            + theme.panel_title(
                "Propulsion fleet health",
                f"RUL range q10 – q50 – q90 (cycles) · dashed line = R* = {R_star} · sorted by condition, then q10",
                theme.chip(f"{len(df_eng)} engines", "blue"),
            )
            + '<div class="sk-scroll"><table class="sk-table"><tr><th>Tail · Sqn</th><th>Pos</th><th>Unit</th>'
            "<th>Cycles</th><th>Status</th><th>RUL prognostic range</th><th>Condition</th></tr>"
            + rows_html
            + "</table></div>"
            + f'<div class="sk-pm" style="margin-top:.5rem;">Showing {len(top)} of {len(df_eng)} engines · full table below</div></div>'
        )
        with st.expander(f"All {len(df_eng)} engines (table)"):
            st.dataframe(df_eng, use_container_width=True, hide_index=True, height=400)
    else:
        st.caption("Engine RUL table (q10/q50/q90) available after T1.7 merge")

        # Build engine health table from events up to selected_day
        health_rows = []
        for tail in range(54):
            sqn_label = f"S{tail // 18 + 1}"
            tail_label = f"T{tail + 1:02d}"
            status = day_status[tail]
            for eng_pos in range(2):
                pos = tail * 2 + eng_pos
                pos_events = events[(events["pos"] == pos) & (events["day"] <= selected_day)].sort_values("day")
                if not pos_events.empty:
                    last_event = pos_events.iloc[-1]
                    last_event_day = int(last_event["day"])
                    last_event_type = last_event["type"]
                else:
                    last_event_day = 0
                    last_event_type = "none"
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
                    "Last Event Day": str(last_event_day) if not pos_events.empty else "n/a",
                    "Last Event Type": last_event_type if not pos_events.empty else "n/a",
                    "Note": removal_note,
                })
        health_df = pd.DataFrame(health_rows)
        health_df["_sort"] = health_df["Status"].map({"MC": 1, "NMCM": 0, "NMCS": 0})
        health_df = health_df.sort_values(["_sort", "Squadron", "Tail"]).drop(columns=["_sort"])
        st.dataframe(health_df, use_container_width=True, hide_index=True, height=400)

    st.markdown(
        f"Removal threshold R* = {R_star} cycles · set by the Engineering Officer · changes are audit-logged"
    )
    st.caption("Only predictive quantiles (q10/q50/q90) are shown.")

    # ── Aircraft Visual Detail ───────────────────────────────────────────────
    st.markdown("### Aircraft Detail")
    tail_options = [f"T{i:02d}" for i in range(1, 55)]
    selected_tail = st.selectbox("Select Aircraft Tail", tail_options, index=18)  # default T19
    tail_idx = int(selected_tail[1:]) - 1
    tail_st = day_status[tail_idx]

    c_img, c_meta = st.columns([1, 1])
    engine_states: dict[str, str] | None = None
    comp_states: dict[str, str] | None = None
    tail_engs: list[dict] = []
    if snap is not None:
        tail_engs = [e for e in snap["engines"] if e["tail"] == tail_idx]
        engine_states = {}
        for i, eng in enumerate(tail_engs[:2]):
            cond = _condition(float(eng["q10"]))
            engine_states[f"engine_{i}"] = (
                "REMOVAL" if cond.startswith("Removal") else ("WATCH" if cond.startswith("Watch") else "OK")
            )
        comp_states = {
            lid: ("MISSING" if linfo.get("missing", False) else "OK")
            for lid, linfo in snap.get("lrus", {}).get(tail_idx, {}).items()
        }

    with c_img:
        if snap is not None:
            st.caption(f"Live propulsion and subsystem health on Day {selected_day}. Synthetic wing.")
        else:
            st.caption("engine detail after T1.7 merge")
        svg_detail = render_aircraft_detail_svg(
            selected_tail,
            tail_st,
            engine_states=engine_states,
            component_states=comp_states,
            width=380,
            height=480,
        )
        b64 = base64.b64encode(svg_detail.encode("utf-8")).decode()
        theme.html(
            f'<div class="sk-panel" style="background:var(--surface-container-lowest);text-align:center;">'
            f'<img src="data:image/svg+xml;base64,{b64}" width="300" alt="generic fighter, top view"></div>'
        )
    with c_meta:
        tone = theme.STATUS_TONE.get(tail_st, "red")
        meta = (
            f'<div class="sk-panel sk-raised">'
            f'<div class="sk-ptitle"><div><div class="sk-pt">Tail {selected_tail}</div>'
            f'<div class="sk-pm">Squadron S{tail_idx // 18 + 1} · day {selected_day}</div></div>'
            f'{theme.chip(f"{tail_st} · {theme.STATUS_TEXT.get(tail_st, tail_st)}", tone, led=True)}</div>'
            f'<div class="sk-kv"><span>Removal threshold R*</span><b>{R_star} cycles</b></div>'
        )
        if snap is not None:
            for i, eng in enumerate(tail_engs[:2]):
                st_text = (engine_states or {}).get(f"engine_{i}", "OK")
                c_tone = "red" if st_text == "REMOVAL" else ("amber" if st_text == "WATCH" else "green")
                meta += (
                    f'<div class="sk-kv"><span>Engine {i} ({"Left" if i == 0 else "Right"})</span>'
                    f'<b>{theme.chip(st_text, c_tone)} q10 {float(eng.get("q10", 0)):.1f} · q50 {float(eng.get("q50", 0)):.1f} · q90 {float(eng.get("q90", 0)):.1f}</b></div>'
                )
            missing = [lid for lid, v in (comp_states or {}).items() if v == "MISSING"]
            meta += (
                f'<p class="sk-note sk-t-red" style="margin-top:.6rem;">Grounded LRU shortage: missing parts: {theme.esc(", ".join(missing))}</p>'
                if missing
                else '<p class="sk-note sk-t-green" style="margin-top:.6rem;">All 10 tracked LRU components serviceable.</p>'
            )
        else:
            meta += (
                '<p class="sk-note" style="margin-top:.6rem;">Subsystem health: per-engine RUL quantiles '
                "(q10/q50/q90) and LRU component markers require a live snapshot (snapshot_at not available).</p>"
            )
        theme.html(meta + "</div>")

    # ── Why is this engine flagged? ──────────────────────────────────────────
    from app.lib.explain_panel import render_for_tail  # noqa: E402, I001

    render_for_tail(snap, tail_idx, R_star)

# ═════════════════════════════════ RAIL ══════════════════════════════════════
with col_rail:
    # ── RUL Band Chart ───────────────────────────────────────────────────────
    st.markdown("### RUL Band Chart")
    st.caption("Predictive quantile bands from replay_rul.parquet. Synthetic wing.")
    units = replay_rul[["subset", "unit"]].drop_duplicates().sort_values(["subset", "unit"])
    unit_options = [f"{row['subset']} / unit {row['unit']}" for _, row in units.iterrows()]
    selected_unit_str = st.selectbox("Select replay unit", unit_options, index=0)
    if selected_unit_str:
        sel_subset, sel_unit_s = selected_unit_str.split(" / unit ")
        sel_unit = int(sel_unit_s)
        unit_data = replay_rul[
            (replay_rul["subset"] == sel_subset) & (replay_rul["unit"] == sel_unit)
        ].sort_values("cycle")
        if not unit_data.empty:
            band_data = unit_data[["cycle", "q10", "q50", "q90"]].copy()
            area = alt.Chart(band_data).mark_area(opacity=0.3, color="#38bdf8").encode(
                x=alt.X("cycle:Q", title="Cycle"), y=alt.Y("q10:Q", title="RUL (cycles)"), y2="q90:Q"
            )
            median_line = alt.Chart(band_data).mark_line(color="#8ed5ff", strokeWidth=2).encode(
                x="cycle:Q", y="q50:Q"
            )
            r_star_line = alt.Chart(pd.DataFrame({"y": [R_star]})).mark_rule(
                color="#ef4444", strokeDash=[4, 4], strokeWidth=2
            ).encode(y="y:Q")
            st.altair_chart(
                (area + median_line + r_star_line).properties(
                    height=260, title=f"{sel_subset} unit {sel_unit} (q10/q50/q90, R*={R_star})"
                ),
                use_container_width=True,
            )
        else:
            st.warning("No data for selected unit.")

    # ── Sign-Off Queue ───────────────────────────────────────────────────────
    st.markdown("### Sign-Off Queue")
    st.caption("Each recommendation requires explicit officer sign-off. Append-only audit log.")
    if not queue.empty:
        for _, row in queue.iterrows():
            tail_num = row["pos"] // 2 + 1
            eng_pos = row["pos"] % 2
            rec_id = f"EV-{int(row['day']):04d}-P{int(row['pos']):03d}"
            unsched = row["type"] == "unscheduled"
            theme.html(
                f'<div class="sk-wo{" sk-red" if unsched else ""}">'
                f'<div class="sk-wk">{rec_id} · {theme.esc(row["type"])} removal</div>'
                f'<div class="sk-wh">T{tail_num:02d} · engine {eng_pos}: {theme.esc(row["type"])} engine removal</div>'
                f'<div class="sk-wm">day {int(row["day"])} · cycle {int(row.get("cycle", 0))} · squadron S{(tail_num - 1) // 18 + 1}</div></div>'
            )
            b1, b2 = st.columns(2)
            if b1.button("Approve", key=f"approve_{rec_id}", use_container_width=True):
                audit_append(
                    role="Engineering Officer",
                    action=f"{row['type']}_removal",
                    target=f"T{tail_num:02d}/eng{eng_pos}",
                    recommendation_id=rec_id,
                    decision="approved",
                )
                st.toast(f"Approved {rec_id}")
                st.rerun()
            if b2.button("Defer", key=f"defer_{rec_id}", use_container_width=True):
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

    # ── Audit Log ────────────────────────────────────────────────────────────
    st.markdown("### Recent Audit Log")
    recent = audit_recent(20)
    if recent:
        entries = "".join(
            f'<div class="sk-audit"><div class="sk-at"><span>{theme.esc(e.get("ts_utc", ""))[:19]}</span>'
            f'<span class="sk-t-{"green" if e.get("decision") == "approved" else "amber"}">{theme.esc(str(e.get("decision", "")).upper())}</span></div>'
            f'<div class="sk-ab">{theme.esc(e.get("role", ""))}: {theme.esc(e.get("action", ""))} · {theme.esc(e.get("target", ""))}</div>'
            f'<div class="sk-ah">{theme.esc(e.get("recommendation_id", ""))}</div></div>'
            for e in recent[:4]
        )
        theme.html(f'<div class="sk-panel">{entries}</div>')
        st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
    else:
        st.info("No audit entries yet.")

# ── Performance ──────────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
st.caption(f"Page load: {elapsed:.2f} s. SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")
theme.footer()
