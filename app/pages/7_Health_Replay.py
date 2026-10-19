"""SHIKRA: Health Replay.

Replays a post-flight health download for one held-out C-MAPSS engine,
cycle by cycle: the calibrated q10/q50/q90 remaining-life band, the R*
removal threshold, and the removal alert the first cycle q10 <= R*.
"""

from __future__ import annotations

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
from app.lib.data import load_params, load_replay_rul  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA: Health Replay",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("health_replay")
theme.status_strip(strip)

theme.header(
    "Health replay · post-flight download",
    "What does SHIKRA see after each flight, and when does it raise an alert?",
    "Each step is one engine cycle: the calibrated remaining-life band (q10–q90) and median, "
    "against the removal threshold R*. The alert fires the first cycle the pessimistic estimate q10 reaches R*.",
    "artifacts/replay_rul.parquet · artifacts/policy_params.json",
)
st.caption(
    "Replay of a held-out NASA C-MAPSS engine (post-flight download, not live telemetry). Advisory only."
)

# ── Data (only the predictive quantiles are read) ────────────────────────────
COLS = ["subset", "unit", "cycle", "q10", "q50", "q90"]
replay = load_replay_rul()[COLS].copy()
replay["subset"] = replay["subset"].astype(str)
replay["unit"] = replay["unit"].astype(int)
replay["cycle"] = replay["cycle"].astype(int)
replay = replay.sort_values(["subset", "unit", "cycle"]).reset_index(drop=True)
R_star = int(load_params()["R_star"])

engines = list(replay[["subset", "unit"]].drop_duplicates().itertuples(index=False, name=None))


def _label(eng: tuple[str, int]) -> str:
    return f"{eng[0]} · unit {eng[1]}"


def _traj(eng: tuple[str, int]) -> pd.DataFrame:
    return replay[(replay["subset"] == eng[0]) & (replay["unit"] == eng[1])]


def _alert_cycle(traj: pd.DataFrame) -> int | None:
    hit = traj[traj["q10"] <= R_star]
    return None if hit.empty else int(hit["cycle"].iloc[0])


def _default_engine() -> tuple[str, int]:
    """First engine by (subset, unit) whose q10 starts above R* and later crosses it."""
    for eng in engines:
        t = _traj(eng)
        if float(t["q10"].iloc[0]) > R_star and _alert_cycle(t) is not None:
            return eng
    return engines[0]


labels = [_label(e) for e in engines]
by_label = dict(zip(labels, engines))
default_label = _label(_default_engine())


def _reset_cycle() -> None:
    eng = by_label[st.session_state["hr_engine"]]
    st.session_state["hr_cycle"] = int(_traj(eng)["cycle"].min())
    st.session_state["hr_playing"] = False


if "hr_engine" not in st.session_state:
    st.session_state["hr_engine"] = default_label
    _reset_cycle()
st.session_state.setdefault("hr_playing", False)

c_eng, c_play, c_pause = st.columns([3, 1, 1], vertical_alignment="bottom")
with c_eng:
    st.selectbox("Engine (held-out C-MAPSS unit)", labels, key="hr_engine", on_change=_reset_cycle)

engine = by_label[st.session_state["hr_engine"]]
traj = _traj(engine)
c_min, c_max = int(traj["cycle"].min()), int(traj["cycle"].max())
step = max(1, (c_max - c_min) // 60)


def _play() -> None:
    if st.session_state["hr_cycle"] >= c_max:
        st.session_state["hr_cycle"] = c_min
    st.session_state["hr_playing"] = True


def _pause() -> None:
    st.session_state["hr_playing"] = False


with c_play:
    st.button("Play", key="hr_play_btn", on_click=_play, use_container_width=True, type="primary")
with c_pause:
    st.button("Pause", key="hr_pause_btn", on_click=_pause, use_container_width=True)

# Advance one tick before the slider is drawn (widget state may only change here).
if st.session_state["hr_playing"]:
    nxt = int(st.session_state["hr_cycle"]) + step
    if nxt >= c_max:
        nxt = c_max
        st.session_state["hr_playing"] = False
    st.session_state["hr_cycle"] = nxt
st.session_state["hr_cycle"] = min(max(int(st.session_state["hr_cycle"]), c_min), c_max)

cycle = st.slider("Engine cycle (post-flight download)", c_min, c_max, key="hr_cycle", on_change=_pause)

seen = traj[traj["cycle"] <= cycle]
now = seen.iloc[-1]
q10, q50, q90 = float(now["q10"]), float(now["q50"]), float(now["q90"])
alert_at = _alert_cycle(traj)
alert_on = alert_at is not None and cycle >= alert_at

# ── KPI tiles ────────────────────────────────────────────────────────────────
theme.kpi_row(
    [
        theme.kpi(
            "Current cycle",
            f"{int(now['cycle'])}",
            unit=f"/ {c_max}",
            badge=theme.chip(_label(engine), "blue"),
            sub=f"downloads replayed: {len(seen)}",
        ),
        theme.kpi(
            "Pessimistic remaining life (q10)",
            f"{q10:.1f}",
            unit="cycles",
            tone="red" if alert_on else "",
            sub=f"R* = {R_star} cycles · margin {q10 - R_star:+.1f}",
            alert=alert_on,
        ),
        theme.kpi(
            "Median remaining life (q50)",
            f"{q50:.1f}",
            unit="cycles",
            sub=f"80% band [{q10:.1f}, {q90:.1f}]",
        ),
        theme.kpi(
            "Status",
            "REMOVAL ADVISED" if alert_on else "Monitor",
            tone="red" if alert_on else "green",
            badge=theme.chip("alert" if alert_on else "no alert", "red" if alert_on else "green", led=True),
            sub=f"alert since cycle {alert_at}" if alert_on else "q10 above R*",
            alert=alert_on,
        ),
    ]
)

# ── Alert panel ──────────────────────────────────────────────────────────────
if alert_on:
    theme.html(
        '<div class="sk-advisory" style="border-top:2px solid #ef4444;box-shadow:inset 0 0 0 1px #ef4444;">'
        '<div class="sk-ah" style="color:#ef4444;">REMOVAL ADVISED: pessimistic remaining life &lt;= R* '
        "- officer sign-off required</div>"
        f"<p>Alert raised at cycle <b>{alert_at}</b> on {theme.esc(_label(engine))}: "
        f"q10 fell to {float(traj.loc[traj['cycle'] == alert_at, 'q10'].iloc[0]):.1f} cycles, "
        f"at or below R* = {R_star} cycles.</p>"
        '<p style="color:var(--outline)!important;font-size:11px!important;font-family:var(--font-mono);">'
        "Source: artifacts/replay_rul.parquet · artifacts/policy_params.json · advisory only</p></div>"
    )
else:
    theme.html(
        '<div class="sk-advisory"><div class="sk-ah">No alert</div>'
        f"<p>Pessimistic remaining life q10 = {q10:.1f} cycles is above R* = {R_star} cycles "
        f"at cycle {int(now['cycle'])}. SHIKRA keeps monitoring each post-flight download.</p></div>"
    )

# ── Band chart (up to the current cycle; axes fixed to the whole replay) ─────
x_scale = alt.Scale(domain=[c_min, c_max])
y_scale = alt.Scale(domain=[0, float(traj["q90"].max()) * 1.05])
band = (
    alt.Chart(seen)
    .mark_area(opacity=0.25, color="#38bdf8")
    .encode(
        x=alt.X("cycle:Q", title="Engine cycle", scale=x_scale),
        y=alt.Y("q10:Q", title="Remaining life (cycles)", scale=y_scale),
        y2="q90:Q",
    )
)
median = (
    alt.Chart(seen)
    .mark_line(color="#0284c7", strokeWidth=2)
    .encode(
        x=alt.X("cycle:Q", scale=x_scale),
        y=alt.Y("q50:Q", scale=y_scale),
        tooltip=[
            alt.Tooltip("cycle:Q", title="Cycle"),
            alt.Tooltip("q10:Q", title="q10", format=".1f"),
            alt.Tooltip("q50:Q", title="q50 (median)", format=".1f"),
            alt.Tooltip("q90:Q", title="q90", format=".1f"),
        ],
    )
)
q10_line = (
    alt.Chart(seen)
    .mark_line(color="#38bdf8", strokeWidth=1)
    .encode(x=alt.X("cycle:Q", scale=x_scale), y=alt.Y("q10:Q", scale=y_scale))
)
r_rule = (
    alt.Chart(pd.DataFrame({"y": [R_star]}))
    .mark_rule(color="#ef4444", strokeDash=[4, 4], strokeWidth=2)
    .encode(y=alt.Y("y:Q", scale=y_scale))
)
layers = band + median + q10_line + r_rule
if alert_on:
    layers = layers + (
        alt.Chart(pd.DataFrame({"x": [alert_at]}))
        .mark_rule(color="#ef4444", strokeWidth=1)
        .encode(x=alt.X("x:Q", scale=x_scale))
    )
st.altair_chart(layers.properties(height=340), use_container_width=True)
st.caption(
    f"Shaded: q10–q90 band; line: q50 median. Red dashed line: R* = {R_star} cycles. "
    "Only calibrated predictive quantiles are displayed. Drag the slider to step manually."
)

theme.footer()

# ── Play loop: one tick per rerun ────────────────────────────────────────────
if st.session_state["hr_playing"]:
    time.sleep(0.25)
    st.rerun()
