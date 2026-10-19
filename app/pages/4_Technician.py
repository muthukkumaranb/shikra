"""SHIKRA — Technician's View.

Snag list for the selected day: tails that are non-MC with reason,
squadron, and expected return day, as work cards grouped by squadron.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import theme  # noqa: E402
from app.lib.data import load_config  # noqa: E402
from app.lib.state import get_sim_result  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Technician",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("technician")

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

theme.header(
    "Technician · flight-line snags",
    "Which aircraft are down today, why, and when should each be back?",
    "Every non-mission-capable tail on the selected day as a work card, grouped by squadron "
    "(fewest aircraft available first), with the reason and expected return. "
    "Synthetic wing · advisory — all recommendations require officer sign-off.",
    "simulation (config/wing.yaml + artifacts/replay_rul.parquet, shikra_v2, seed 1): tail status and engine events",
)

selected_day = st.slider(
    "Select day",
    min_value=measure_start,
    max_value=measure_end,
    value=default_day,
    help="Day 485 = 365 warm-up + 120 measured",
)
mc_today = int(tail_mc_arr[selected_day].sum())
theme.status_strip(strip, day=selected_day, mc=mc_today)

# Calculate squadron MC counts on selected_day
sqn_mc_counts = {}
for s in range(3):
    sqn_mc_counts[f"S{s + 1}"] = int(sum(tail_mc_arr[selected_day][s * 18 : (s + 1) * 18]))

# Identify non-MC tails
snags = []
for tail in range(54):
    if not tail_mc_arr[selected_day][tail]:
        sqn_label = f"S{tail // 18 + 1}"
        status = tail_status_list[selected_day][tail]
        tail_events = events[(events["pos"] // 2 == tail) & (events["day"] <= selected_day)].sort_values("day")

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
            "Tail": f"T{tail + 1:02d}",
            "Squadron": sqn_label,
            "Status": status,
            "Reason": reason,
            "Expected Return Day": expected_return,
            "_sqn_mc": sqn_mc_counts[sqn_label],
            "_tail_num": tail,
        })

n_nmcm = sum(1 for s in snags if s["Status"] == "NMCM")
n_nmcs = sum(1 for s in snags if s["Status"] == "NMCS")

# ── KPI row ─────────────────────────────────────────────────────────────────
tiles = [
    theme.kpi(
        "Non-mission-capable",
        f"{len(snags)}",
        unit="/ 54",
        tone="red" if snags else "green",
        badge=theme.chip(f"{len(snags) / 54 * 100:.1f}% NMC", "red" if snags else "green"),
        sub=f'<span class="sk-t-amber">{n_nmcm} in maintenance</span> · <span class="sk-t-red">{n_nmcs} awaiting spares</span>',
    ),
]
if snags:
    soonest = min(s["Expected Return Day"] for s in snags)
    n_soon = sum(1 for s in snags if s["Expected Return Day"] == soonest)
    tiles.append(
        theme.kpi(
            "Soonest expected return",
            f"D+{soonest - selected_day}",
            tone="blue",
            sub=f"day {soonest} · {n_soon} tail(s) · estimate",
        )
    )
for sq, mc in sqn_mc_counts.items():
    pct = mc / 18 * 100
    tiles.append(
        theme.kpi(
            f"{sq} serviceable",
            f"{mc}",
            unit="/ 18",
            tone="green" if pct >= 75 else ("amber" if pct >= 60 else "red"),
            sub=f"{18 - mc} snags",
            extra=theme.bar(pct, theme.MC_COLOR if pct >= 75 else (theme.MAINT_COLOR if pct >= 60 else theme.SUPPLY_COLOR)),
        )
    )
theme.kpi_row(tiles)


def _tail_events(tail_num: int) -> pd.DataFrame:
    ev = events[(events["pos"] // 2 == tail_num) & (events["day"] <= selected_day)].sort_values("day", ascending=False)
    out = ev[["day", "pos", "type"] + (["cycle"] if "cycle" in ev.columns else [])].head(6).copy()
    out["pos"] = out["pos"] % 2
    return out.rename(columns={"day": "Day", "pos": "Engine", "type": "Removal type", "cycle": "Cycle"})


def _card_body(row: dict) -> str:
    tone = "red" if row["Status"] == "NMCS" else "amber"
    days_left = row["Expected Return Day"] - selected_day
    return (
        f'<div class="sk-kv"><span>Status</span><b>{theme.chip(theme.STATUS_TEXT[row["Status"]] + " (" + row["Status"] + ")", tone, led=True)}</b></div>'
        f'<div class="sk-kv"><span>Squadron</span><b>{row["Squadron"]}</b></div>'
        f'<div class="sk-kv"><span>Reason</span><b>{theme.esc(row["Reason"])}</b></div>'
        f'<div class="sk-kv"><span>Expected return</span><b>Day {row["Expected Return Day"]} (in {days_left} days)</b></div>'
    )


@st.dialog("Work card")
def _work_card(row: dict) -> None:
    theme.html(
        f'<div class="sk-eyebrow">{row["Tail"]} · day {selected_day}</div>'
        f'<div class="sk-panel sk-raised" style="margin-top:.6rem;">{_card_body(row)}</div>'
    )
    ev = _tail_events(row["_tail_num"])
    st.markdown("**Engine removal history for this tail (up to the selected day)**")
    if ev.empty:
        st.caption("No engine removals recorded for this tail; the snag is LRU-related.")
    else:
        st.dataframe(ev, use_container_width=True, hide_index=True)
    st.caption("Expected return is an estimate from the event type; advisory only. Synthetic wing.")


col_main, col_rail = st.columns([2, 1], gap="medium")

with col_main:
    st.markdown("### Active Groundings & Snags")
    st.caption("Sorted by squadron with fewest aircraft MC, then by soonest expected return. Synthetic wing.")

    if snags:
        snag_df = pd.DataFrame(snags).sort_values(["_sqn_mc", "Expected Return Day", "Squadron", "Tail"])
        for sq in snag_df["Squadron"].unique():
            mc = sqn_mc_counts[sq]
            pct = mc / 18 * 100
            color = theme.MC_COLOR if pct >= 75 else (theme.MAINT_COLOR if pct >= 60 else theme.SUPPLY_COLOR)
            sub = snag_df[snag_df["Squadron"] == sq]
            theme.html(
                f'<div class="sk-sqgroup"><div><div class="sk-gn">Squadron {sq[1:]} ({sq})</div>'
                f'<div class="sk-gs">{mc} / 18 serviceable ({pct:.1f}%) · {len(sub)} grounded tails</div></div>'
                f'<div class="sk-gbar"><div class="sk-label" style="font-size:10px;color:var(--outline);text-transform:uppercase;letter-spacing:.08em;">squadron readiness</div>'
                f"{theme.bar(pct, color)}</div></div>"
            )
            for _, row in sub.iterrows():
                tone = "red" if row["Status"] == "NMCS" else "amber"
                icon = "⏳" if row["Status"] == "NMCS" else "⚙"
                c_card, c_btn = st.columns([5, 1])
                with c_card:
                    theme.html(
                        f'<div class="sk-card" style="box-shadow:inset 3px 0 0 0 {theme.STATUS_COLOR[row["Status"]]};">'
                        f'<div class="sk-tail"><b>{row["Tail"]}</b><span>{row["Squadron"]} · {row["Status"]}</span></div>'
                        f'<div class="sk-cb">{theme.chip(icon + " " + theme.STATUS_TEXT[row["Status"]], tone)}'
                        f'<div class="sk-ch">{theme.esc(row["Reason"])}</div>'
                        f'<div class="sk-cm">Expected return: Day {row["Expected Return Day"]} '
                        f'(in {row["Expected Return Day"] - selected_day} days)</div></div></div>'
                    )
                with c_btn:
                    st.write("")
                    if st.button("Open card", key=f"open_{row['Tail']}", use_container_width=True):
                        _work_card(row.to_dict())

        with st.expander("Tabular Summary"):
            st.dataframe(snag_df.drop(columns=["_sqn_mc", "_tail_num"]), use_container_width=True, hide_index=True)
    else:
        st.success(f"All 54 aircraft are Mission Capable (MC) on day {selected_day}! No snags reported.")

with col_rail:
    st.markdown("### Active Work Card")
    if snags:
        first = pd.DataFrame(snags).sort_values(["_sqn_mc", "Expected Return Day", "Squadron", "Tail"]).iloc[0].to_dict()
        color = theme.STATUS_COLOR[first["Status"]]
        ev = _tail_events(first["_tail_num"])
        hist = "".join(
            f'<div class="sk-check{" sk-done" if int(e["Day"]) < selected_day else ""}"><i></i>'
            f'<div>Day {int(e["Day"])} · engine {int(e["Engine"])} · {theme.esc(e["Removal type"])} removal'
            + (f' <span class="sk-mono sk-t-muted">cycle {int(e["Cycle"])}</span>' if "Cycle" in e else "")
            + "</div></div>"
            for _, e in ev.iterrows()
        ) or '<p class="sk-note">No engine removals recorded for this tail; LRU-related snag.</p>'
        theme.html(
            '<div class="sk-panel">'
            + theme.panel_title(f"Tail {first['Tail']}", "first in the queue (squadron with fewest MC, soonest return)", theme.chip(first["Squadron"], "blue"))
            + f'<div style="background:var(--surface-container-lowest);border-radius:var(--rounded-lg);padding:.75rem;text-align:center;margin-bottom:.75rem;">{theme.jet_svg(color, 96)}</div>'
            + _card_body(first)
            + '<div class="sk-label" style="font-size:10px;color:var(--outline);text-transform:uppercase;letter-spacing:.08em;margin:.9rem 0 .4rem 0;">Engine removal history</div>'
            + hist
            + "</div>"
        )
    theme.html(
        '<div class="sk-panel sk-raised"><div class="sk-ptitle"><div><div class="sk-pt">How to read a card</div></div></div>'
        f'<p class="sk-note">{theme.chip("NMCM", "amber")} in maintenance: an engine change or LRU repair is under way. '
        f'{theme.chip("NMCS", "red")} waiting for spares: the tail is grounded until a part arrives. '
        "Expected return is estimated from the event type and is advisory only.</p></div>"
    )

theme.footer()
