"""SHIKRA — Home page.

Fleet overview: flight line for the selected day, the three findings
(generated from data), squadron summaries and the reproduce command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path for imports
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import theme  # noqa: E402
from app.lib.data import load_heatmap, load_supply_shock  # noqa: E402
from app.lib.state import get_sim_result, get_wing_state_on_day  # noqa: E402

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"

st.set_page_config(
    page_title="SHIKRA — Advisory Decision Support",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("home")

# ── Pre-warm Simulation Cache ──────────────────────────────────────────────
with st.spinner("Pre-warming wing simulation cache..."):
    _ = get_sim_result("shikra_v2", seed=1)

theme.header(
    "Fleet overview",
    "How many of the wing's 54 aircraft can fly on the selected day, and what holds the rest back?",
    "Mission-capable status per tail from the wing simulation, with the three headline findings "
    "computed from the result files. Simulation cache pre-warmed for instant offline page access.",
    "simulation (config/wing.yaml + artifacts/replay_rul.parquet, shikra_v2, seed 1) · "
    "results/heatmap.csv · results/supply_shock.csv",
)

flight_day = st.slider(
    "Simulation Day (Measured Window: 365–729)",
    min_value=365,
    max_value=729,
    value=485,
    step=1,
    help="Day 485 = Day 120 of the measured window (post-warmup)",
)

wing_state = get_wing_state_on_day(flight_day, policy="shikra_v2", seed=1)
mc_total = wing_state["mc_count"]
pct_total = (mc_total / 54.0) * 100.0
counts = theme.count_statuses([t["status"] for t in wing_state["tail_statuses"]])
theme.status_strip(strip, day=flight_day, mc=mc_total)

# ── Findings (from data, reusing readme_table.py logic) ────────────────────
heatmap = load_heatmap()
shock = load_supply_shock()

# Finding 1: The pipeline lever (from heatmap.csv) at pipeline_mult=1.7, quality=1 (SHIKRA)
row_17 = heatmap[(heatmap["pipeline_mult"] == 1.7) & (heatmap["quality"] == 1.0)]
# Finding 2: Calibrated prediction's share of the perfect-prediction gain at 0.5×
row_05_shikra = heatmap[(heatmap["pipeline_mult"] == 0.5) & (heatmap["quality"] == 1.0)]
row_05_oracle = heatmap[(heatmap["pipeline_mult"] == 0.5) & (heatmap["quality"] == 0.0)]
# Finding 3: The top indigenisation item (from supply_shock.csv)
top_lrus = shock[shock["lru_id"] != "all_foreign"].sort_values("rank")

# ── KPI row ─────────────────────────────────────────────────────────────────
tiles = [
    theme.kpi(
        "Aircraft mission-capable",
        f"{mc_total}",
        unit="/ 54",
        sub=(
            f'<span class="sk-t-green">{counts["MC"]} MC</span> · '
            f'<span class="sk-t-amber">{counts["NMCM"]} maint</span> · '
            f'<span class="sk-t-red">{counts["NMCS"]} spares</span>'
        ),
        badge=theme.chip(f"day {flight_day}", "blue"),
    ),
    theme.kpi(
        "Wing serviceability",
        f"{pct_total:.1f}%",
        sub=f"{mc_total} of 54 tails MC on day {flight_day}",
        extra=theme.bar(pct_total),
    ),
]
if not row_17.empty:
    r17 = row_17.iloc[0]
    tiles.append(
        theme.kpi(
            "Gain vs reactive at 1.7× pipeline",
            f"{r17['gain_vs_reactive_mean'] * 100:+.1f}",
            unit="pp",
            tone="blue",
            sub=f"95% CI [{r17['gain_ci_lo'] * 100:+.1f}, {r17['gain_ci_hi'] * 100:+.1f}] pp · n=30 seeds",
        )
    )
if not top_lrus.empty:
    t1 = top_lrus.iloc[0]
    tiles.append(
        theme.kpi(
            "Top supply chokepoint",
            f"{t1['loss_pp_shock_window']:.1f}",
            unit="pp",
            tone="amber",
            badge=theme.chip(str(t1["lru_id"]), "amber"),
            sub=f"95% CI [{t1['ci_lo']:.1f}, {t1['ci_hi']:.1f}] pp · 180-day halt",
        )
    )
theme.kpi_row(tiles)

# ── Three findings strip ───────────────────────────────────────────────────
if not row_17.empty:
    gain_pp = row_17.iloc[0]["gain_vs_reactive_mean"] * 100
    svc_mean = row_17.iloc[0]["serviceability_mean"] * 100
    reactive_svc = row_17.iloc[0]["reactive_serviceability_mean"] * 100
    finding1 = (
        f"The pipeline is the lever: at 1.7× pipeline speed, SHIKRA serviceability is "
        f"{svc_mean:.1f}% vs reactive {reactive_svc:.1f}%. Prediction warning time is not "
        f"the binding constraint; the engine-spare pipeline is."
    )
    f1_num, f1_lbl = f"{gain_pp:+.1f} pp", "gain vs reactive"
else:
    finding1 = "The pipeline is the lever: (data unavailable for 1.7× pipeline speed)."
    f1_num, f1_lbl = "—", "no data"

f2_num, f2_lbl = "—", "no data"
if not row_05_shikra.empty and not row_05_oracle.empty:
    shikra_gain = row_05_shikra.iloc[0]["gain_vs_reactive_mean"] * 100
    oracle_gain = row_05_oracle.iloc[0]["gain_vs_reactive_mean"] * 100
    if oracle_gain > 0:
        share = shikra_gain / oracle_gain * 100
        finding2 = (
            f"Calibrated prediction captures {share:.0f}% of the perfect-prediction gain "
            f"at 0.5× pipeline speed ({shikra_gain:.1f} pp vs oracle {oracle_gain:.1f} pp)."
        )
        f2_num, f2_lbl = f"{share:.0f}%", "of perfect-prediction gain"
    else:
        finding2 = (
            "Calibrated prediction: perfect prediction provides no significant gain "
            "at the current pipeline speed."
        )
else:
    finding2 = "Calibrated prediction share: (data unavailable)."

if not top_lrus.empty:
    top1 = top_lrus.iloc[0]
    finding3 = (
        f"Top indigenisation priority: {top1['lru_id']}. A 180-day foreign-supply halt causes "
        f"{top1['loss_pp_shock_window']:.1f} pp loss (peak {top1['peak_extra_grounded']:.0f} extra "
        f"grounded, recovery {top1['recovery_days']:.0f} days). Synthetic parameters; the method is the claim."
    )
    f3_num, f3_lbl = f"+{top1['peak_extra_grounded']:.0f} A/C", "peak extra grounded"
else:
    finding3 = "Top indigenisation item: (data unavailable)."
    f3_num, f3_lbl = "—", "no data"


def _finding(kicker: str, tone: str, text: str, src: str, num: str, lbl: str) -> str:
    return (
        f'<div class="sk-find"><div><div class="sk-fk sk-t-{tone}">{theme.esc(kicker)}</div>'
        f'<div class="sk-fh">{theme.esc(text)}</div><div class="sk-fs">{theme.esc(src)}</div></div>'
        f'<div class="sk-fn"><div class="sk-n sk-t-{tone}">{theme.esc(num)}</div>'
        f'<div class="sk-l">{theme.esc(lbl)}</div></div></div>'
    )


st.markdown("### Three Key Findings")
theme.html(
    '<div class="sk-cards3">'
    + _finding("Pipeline lever", "blue", finding1, "results/heatmap.csv · synthetic wing", f1_num, f1_lbl)
    + _finding("Prediction quality", "green", finding2, "results/heatmap.csv · synthetic wing", f2_num, f2_lbl)
    + _finding("Supply chokepoint", "amber", finding3, "results/supply_shock.csv · synthetic wing", f3_num, f3_lbl)
    + "</div>"
)

# ── Flight-Line View ────────────────────────────────────────────────────────
st.markdown(
    theme.oneline(theme.flight_line(wing_state["tail_statuses"], wing_state["sqn_mc"])),
    unsafe_allow_html=True,
)

# ── Squadron summary tiles ─────────────────────────────────────────────────
sq_tiles = []
for s in range(3):
    sq_st = [t["status"] for t in wing_state["tail_statuses"][s * 18 : (s + 1) * 18]]
    c = theme.count_statuses(sq_st)
    pct = c["MC"] / 18 * 100
    tone = "green" if pct >= 75 else ("amber" if pct >= 60 else "red")
    sq_tiles.append(
        '<div class="sk-panel sk-raised" style="margin:0;">'
        f'<div class="sk-ptitle"><div><div class="sk-label" style="font-size:10px;letter-spacing:.1em;'
        f'text-transform:uppercase;color:var(--outline);font-weight:600;">Squadron {s + 1}</div>'
        f'<div style="font-size:16px;font-weight:600;">S{s + 1} flight line</div></div>'
        f"{theme.chip(f'{pct:.1f}%', tone)}</div>"
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;">'
        f'<span class="sk-mono" style="font-size:18px;font-weight:700;">{c["MC"]} / 18 '
        f'<span style="font-size:12px;font-weight:400;color:var(--on-surface-variant);font-family:var(--font-sans);">mission-capable</span></span>'
        f'<span class="sk-mono sk-t-muted" style="font-size:11px;">{c["MC"]} MC · {c["NMCM"]} NMCM · {c["NMCS"]} NMCS</span></div>'
        + theme.segbar([(c["MC"], theme.MC_COLOR), (c["NMCM"], theme.MAINT_COLOR), (c["NMCS"], theme.SUPPLY_COLOR)])
        + "</div>"
    )
theme.html(f'<div class="sk-cards3">{"".join(sq_tiles)}</div>')

# ── What is SHIKRA? + offline box ───────────────────────────────────────────
c_what, c_box = st.columns([2, 1])
with c_what:
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
with c_box:
    theme.html(
        """
        <div class="sk-panel sk-raised" style="margin-top:.75rem;">
          <div class="sk-ptitle"><div><div class="sk-pt">Offline system</div></div></div>
          <p class="sk-note">Runs on this machine from local files. No cloud service, no network calls.</p>
          <div class="sk-kv"><span>Mode</span><b>offline · advisory</b></div>
          <div class="sk-kv"><span>Data</span><b>local files</b></div>
          <div class="sk-kv"><span>Network</span><b>none</b></div>
          <div class="sk-kv"><span>Reproduce</span><b>make reproduce-sim</b></div>
          <div style="margin-top:.6rem;"><span class="sk-chip">Advisory · offline · synthetic wing + NASA C-MAPSS</span></div>
        </div>
        """
    )

theme.footer()
