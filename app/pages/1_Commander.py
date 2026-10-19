"""SHIKRA: Commander's View.

Lever map (gain-vs-reactive matrix), supply-shock ranking,
per-squadron readiness forecast rail, and one-page brief.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import theme  # noqa: E402
from app.lib.brief import brief_document  # noqa: E402
from app.lib.data import (  # noqa: E402
    load_config,
    load_heatmap,
    load_params,
    load_policy_summary,
    load_supply_shock,
)
from app.lib.readiness import (  # noqa: E402
    get_forecast,
    get_readiness_module,
    get_snapshot,
)
from app.lib.utilisation import display_table, utilisation_table  # noqa: E402

start_time = time.perf_counter()

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA: Commander",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("commander")

SNAP_DAY = 485

# ── Load data ────────────────────────────────────────────────────────────────
heatmap_raw = load_heatmap()
config = load_config()
params = load_params()
shock_df = load_supply_shock()
util_df = utilisation_table(load_policy_summary())

pipeline_mult = config.get("pipeline_mult", 1.7)

readiness_mod = get_readiness_module()
snap = get_snapshot("shikra_v2", seed=1, day=SNAP_DAY) if readiness_mod is not None else None
fc_df = None
if snap is not None:
    fc_df = get_forecast(
        snapshot=snap,
        config=config,
        params=params,
        policy="shikra",
        horizon_days=7,
        n=150,
        seed=42,
    )

# Per-squadron status from the snapshot (MC / NMCM / NMCS)
sq_counts: list[dict[str, int]] | None = None
if snap is not None and "tails" in snap:
    sq_counts = [{"MC": 0, "NMCM": 0, "NMCS": 0} for _ in range(3)]
    for t in snap["tails"]:
        sq = int(t.get("sqn", 0))
        st_val = t.get("status", "MC")
        sq_counts[sq][st_val if st_val in ("MC", "NMCM") else "NMCS"] += 1
snap_mc = sum(c["MC"] for c in sq_counts) if sq_counts else None
theme.status_strip(strip, day=SNAP_DAY, mc=snap_mc)

theme.header(
    "Commander · lever map · supply shock",
    "Which lever buys the wing the most readiness: better prediction or a faster pipeline?",
    "Serviceability gain over reactive maintenance for each combination of prediction quality "
    "and repair-and-supply time, then the parts whose foreign-supply halt hurts most. "
    "Synthetic wing · advisory: all recommendations require officer sign-off.",
    "results/heatmap.csv · results/supply_shock.csv · results/readiness_calib.csv · "
    f"readiness snapshot day {SNAP_DAY} (shikra_v2, seed 1)",
)

# Format heatmap units in pp
heatmap_df = heatmap_raw.copy()
heatmap_df["gain_pp"] = heatmap_df["gain_vs_reactive_mean"] * 100
heatmap_df["gain_ci_lo_pp"] = heatmap_df["gain_ci_lo"] * 100
heatmap_df["gain_ci_hi_pp"] = heatmap_df["gain_ci_hi"] * 100
heatmap_df["serviceability_pct"] = heatmap_df["serviceability_mean"] * 100
heatmap_df["serviceability_ci_lo_pct"] = heatmap_df["serviceability_ci_lo"] * 100
heatmap_df["serviceability_ci_hi_pct"] = heatmap_df["serviceability_ci_hi"] * 100

row_cur_df = heatmap_df[
    (heatmap_df["pipeline_mult"] == pipeline_mult) & (heatmap_df["quality"] == 1.0)
]
best = heatmap_df.loc[heatmap_df["gain_pp"].idxmax()]

# ── KPI row ─────────────────────────────────────────────────────────────────
tiles = []
if not row_cur_df.empty:
    rc = row_cur_df.iloc[0]
    tiles.append(
        theme.kpi(
            "SHIKRA serviceability now",
            f"{rc['serviceability_pct']:.1f}%",
            sub=f"95% CI [{rc['serviceability_ci_lo_pct']:.1f}, {rc['serviceability_ci_hi_pct']:.1f}]% · "
            f"reactive {rc['reactive_serviceability_mean'] * 100:.1f}%",
            badge=theme.chip(f"{pipeline_mult}× pipeline", "blue"),
            extra=theme.bar(rc["serviceability_pct"]),
        )
    )
    tiles.append(
        theme.kpi(
            "Gain vs reactive at current point",
            f"{rc['gain_pp']:+.1f}",
            unit="pp",
            tone="blue" if rc["gain_pp"] >= 0 else "red",
            sub=f"95% CI [{rc['gain_ci_lo_pp']:+.1f}, {rc['gain_ci_hi_pp']:+.1f}] pp · n=30 seeds",
        )
    )
tiles.append(
    theme.kpi(
        "Largest gain in the matrix",
        f"{best['gain_pp']:+.1f}",
        unit="pp",
        tone="green",
        sub=f"{best['quality_label']} at {best['pipeline_mult']}× · "
        f"95% CI [{best['gain_ci_lo_pp']:+.1f}, {best['gain_ci_hi_pp']:+.1f}]",
    )
)
if snap_mc is not None:
    tiles.append(
        theme.kpi(
            "Mission-capable now",
            f"{snap_mc}",
            unit="/ 54",
            sub=" · ".join(f"S{i + 1} {c['MC']}/18" for i, c in enumerate(sq_counts or [])),
            badge=theme.chip(f"day {SNAP_DAY}", "blue"),
        )
    )
util_shikra = util_df[util_df["policy"] == "shikra_v2"]
util_reactive = util_df[util_df["policy"] == "reactive"]
if not util_shikra.empty and not util_reactive.empty:
    us, ur = util_shikra.iloc[0], util_reactive.iloc[0]
    tiles.append(
        theme.kpi(
            "Flying task met",
            f"{us['task_mean'] * 100:.1f}%",
            sub=f"95% CI [{us['task_ci_lo'] * 100:.1f}, {us['task_ci_hi'] * 100:.1f}]% · "
            f"reactive {ur['task_mean'] * 100:.1f}%",
            badge=theme.chip("SHIKRA", "blue"),
            extra=theme.bar(us["task_mean"] * 100),
        )
    )
theme.kpi_row(tiles)

UTIL_CAPTION = (
    "Utilisation = sorties flown ÷ sorties demanded (0.8 per mission-capable tail per day, "
    "config/wing.yaml). Synthetic wing, 30 seeds, today's pipeline."
)
if not util_df.empty:
    st.markdown("### Utilisation: flying task met")
    st.dataframe(display_table(util_df), use_container_width=True, hide_index=True)
    st.caption(UTIL_CAPTION)

col_main, col_rail = st.columns([2.15, 1], gap="medium")

# ═════════════════════════════════ MAIN ══════════════════════════════════════
with col_main:
    # ── Lever Map ────────────────────────────────────────────────────────────
    st.markdown("### Lever Map")
    show_absolute = st.toggle("Show absolute serviceability", value=False)

    mults = sorted(heatmap_df["pipeline_mult"].unique())
    qualities = sorted(heatmap_df["quality"].unique())
    vcol = "serviceability_pct" if show_absolute else "gain_pp"
    vmax = float(heatmap_df["gain_pp"].abs().max())
    s_lo, s_hi = float(heatmap_df[vcol].min()), float(heatmap_df[vcol].max())

    head = '<tr><th class="sk-rowh">Prediction quality ↓ · repair &amp; supply time →</th>'
    for m in mults:
        cur = " sk-cur" if m == pipeline_mult else ""
        sub = "today" if m == pipeline_mult else ("faster" if m < 1.0 else ("baseline" if m == 1.0 else "slower"))
        head += f'<th class="{cur.strip()}">{m:g}×<br><span style="font-weight:500;color:var(--outline);font-size:9px;text-transform:uppercase;">{sub}</span></th>'
    head += "</tr>"
    body = ""
    for q in qualities:
        qrows = heatmap_df[heatmap_df["quality"] == q]
        body += f'<tr><th class="sk-rowh">{theme.esc(qrows.iloc[0]["quality_label"])}</th>'
        for m in mults:
            cell = qrows[qrows["pipeline_mult"] == m]
            if cell.empty:
                body += "<td>n/a</td>"
                continue
            r = cell.iloc[0]
            if show_absolute:
                a = (r[vcol] - s_lo) / ((s_hi - s_lo) or 1.0)
                bg, fg = f"rgba(56,189,248,{0.12 + 0.68 * a:.2f})", ("#060e20" if a > 0.6 else "#dae2fd")
                val = f"{r['serviceability_pct']:.1f}%"
                ci = f"[{r['serviceability_ci_lo_pct']:.1f}, {r['serviceability_ci_hi_pct']:.1f}]"
            else:
                bg, fg = theme.diverging(float(r["gain_pp"]), vmax)
                val = f"{r['gain_pp']:+.1f} pp"
                ci = f"[{r['gain_ci_lo_pp']:+.1f}, {r['gain_ci_hi_pp']:+.1f}]"
            cur = ' class="sk-cur"' if (m == pipeline_mult and q == 1.0) else ""
            body += (
                f'<td{cur} style="background:{bg};color:{fg};"><div class="sk-cv">{val}</div>'
                f'<div class="sk-ci">95% CI {ci}</div></td>'
            )
        body += "</tr>"

    legend = (
        '<div class="sk-diverge" style="margin-top:.6rem;justify-content:space-between;flex-wrap:wrap;">'
        '<span>Diverging scale: <i style="background:rgba(239,68,68,.8)"></i> loss vs reactive · '
        '<i style="background:rgba(56,189,248,.12)"></i> ≈ 0 · <i style="background:rgba(56,189,248,.8)"></i> gain</span>'
        f"<span>Units: percentage points of serviceability · outlined cell = current operating point ({pipeline_mult}×, SHIKRA)</span></div>"
        if not show_absolute
        else '<div class="sk-diverge" style="margin-top:.6rem;">Absolute serviceability (%) with 95% CI · darker = higher</div>'
    )
    theme.html(
        '<div class="sk-panel">'
        + theme.panel_title(
            "Readiness matrix: prediction quality × pipeline speed",
            "repair &amp; supply time (× today's baseline; lower = faster) · 30 seeds per cell",
        )
        + f'<table class="sk-matrix">{head}{body}</table>{legend}</div>'
    )

    # Caption with dynamic values
    if not row_cur_df.empty:
        r = row_cur_df.iloc[0]
        st.caption(
            f"At {pipeline_mult}× pipeline speed, SHIKRA serviceability is "
            f"{r['serviceability_mean']*100:.1f}% vs reactive "
            f"{r['reactive_serviceability_mean']*100:.1f}% (gain "
            f"{r['gain_vs_reactive_mean']*100:+.1f} pp, 95% CI "
            f"[{r['gain_ci_lo']*100:+.1f}, {r['gain_ci_hi']*100:+.1f}] pp). "
            f"Synthetic wing."
        )
    else:
        st.caption("Synthetic wing · advisory.")

    # ── Supply-Shock Ranking ─────────────────────────────────────────────────
    st.markdown("### Supply-Shock Ranking: Strategic Chokepoints")
    st.caption(
        "180-day foreign-supply halt on each component. Ordered by readiness loss. "
        "Synthetic parameters; the method is the claim."
    )

    ranked = shock_df.sort_values("rank")
    max_loss = float(ranked["loss_pp_shock_window"].max()) or 1.0
    items = []
    for _, row in ranked.iterrows():
        is_all = row["lru_id"] == "all_foreign"
        loss = float(row["loss_pp_shock_window"])
        pct = loss / max_loss * 100
        color = "#ef4444" if pct >= 60 else ("#f59e0b" if pct >= 20 else "#38bdf8")
        name = "All foreign parts at once" if is_all else row["lru_id"]
        src_chip = theme.chip(str(row["source"]).upper(), "red" if row["source"] == "foreign" else "green")
        items.append(
            '<div class="sk-rank">'
            f'<div class="sk-rt"><div><span class="sk-rn">#{int(row["rank"])}</span>'
            f'<span class="sk-rname">{theme.esc(name)}</span> {src_chip}'
            + (' <span class="sk-chip">combined case</span>' if is_all else "")
            + f'</div><div class="sk-rv"><b style="color:{color};">−{loss:.1f} pp</b>'
            f'<div>95% CI [{row["ci_lo"]:.1f}, {row["ci_hi"]:.1f}]</div></div></div>'
            f'<div class="sk-hbar"><div style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'<div class="sk-rmeta"><span>Peak extra grounded: +{row["peak_extra_grounded"]:.0f}</span>'
            f'<span>Recovery: {row["recovery_days"]:.0f} d</span>'
            f'<span>Stock: base {int(row["stock_base_total"])} · depot {int(row["stock_depot"])}</span></div></div>'
        )
    theme.html(
        '<div class="sk-panel">'
        + theme.panel_title(
            "Foreign-sourced parts: readiness lost",
            "loss in serviceability (pp) over the 180-day halt window · 95% CI · 30 seeds",
        )
        + "".join(items)
        + "</div>"
    )

    # Format shock table: clean columns
    shock_display = shock_df.copy()
    shock_display["Loss (pp)"] = shock_display["loss_pp_shock_window"].apply(lambda v: f"{v:.1f} pp")
    shock_display["Peak Extra Grounded"] = shock_display["peak_extra_grounded"].apply(lambda v: f"+{v:.0f}")
    shock_display["Recovery (days)"] = shock_display["recovery_days"].apply(lambda v: f"{v:.0f} d")
    col_order = ["rank", "lru_id", "source", "Loss (pp)", "Peak Extra Grounded", "Recovery (days)"]
    shock_display = shock_display[[c for c in col_order if c in shock_display.columns]].rename(
        columns={"rank": "Rank", "lru_id": "Component", "source": "Source"}
    )
    with st.expander("Supply-shock table"):
        st.dataframe(shock_display, use_container_width=True, hide_index=True)

# ═════════════════════════════════ RAIL ══════════════════════════════════════
# Honest calibration caption helpers
calib_path = ROOT / "results" / "readiness_calib.csv"
conf_cov_val = 0.8626
mae_fc_val = 1.3263
mae_pe_val = 0.9818
raw_cov_val = None
if calib_path.exists():
    try:
        _cal = pd.read_csv(calib_path)
        _row_cf = _cal[(_cal["lead_days"] == 3) & (_cal["method"] == "shikra_conformal")]
        _row_fc = _cal[(_cal["lead_days"] == 3) & (_cal["method"] == "shikra_forecast")]
        _row_pe = _cal[(_cal["lead_days"] == 3) & (_cal["method"] == "persistence")]
        if not _row_cf.empty:
            conf_cov_val = float(_row_cf.iloc[0]["coverage_80"])
        if not _row_fc.empty:
            mae_fc_val = float(_row_fc.iloc[0]["mae"])
            raw_cov_val = float(_row_fc.iloc[0]["coverage_80"])
        if not _row_pe.empty:
            mae_pe_val = float(_row_pe.iloc[0]["mae"])
    except Exception:
        pass

calib_caption_text = (
    f"Calibrated band covers {conf_cov_val:.4f} ({conf_cov_val * 100:.1f}%) of held-out outcomes; "
    f"the point forecast does not yet beat persistence (D+3 MAE {mae_fc_val:.4f} vs {mae_pe_val:.4f}). "
    f"Synthetic wing."
)


def _fc_row(lead: int, sqn: int):
    if fc_df is None or fc_df.empty:
        return None
    sub = fc_df[(fc_df["lead_days"] == lead) & (fc_df["sqn"] == sqn)]
    return None if sub.empty else sub.iloc[0]


# ── Commander's One-Page Brief ───────────────────────────────────────────────
def _build_brief() -> tuple[str, str]:
    """Return (print stylesheet, brief body HTML); built once, shown and downloaded."""
    from app.lib.audit import get_recent as _get_rec
    from app.lib.audit import verify as audit_verify

    # a. Wing status now: MC / NMCM / NMCS counts per squadron (from snapshot)
    counts_b = [dict(c) for c in sq_counts] if sq_counts else None
    if counts_b is None:
        from app.lib.state import get_wing_state_on_day

        ws = get_wing_state_on_day(SNAP_DAY, policy="shikra_v2", seed=1)
        counts_b = [{"MC": 0, "NMCM": 0, "NMCS": 0} for _ in range(3)]
        for tinfo in ws["tail_statuses"]:
            sq_i = int(tinfo["squadron"][1:]) - 1
            st_val = tinfo["status"]
            counts_b[sq_i][st_val if st_val in ("MC", "NMCM") else "NMCS"] += 1
    total_mc = sum(c["MC"] for c in counts_b)

    # b. Readiness forecast: expected MC with 80% band per squadron at D+3 and D+7
    def _fc_cell(lead: int, sq: int) -> str:
        r = _fc_row(lead, sq)
        if r is None:
            return "n/a"
        return f"{float(r['expected_mc']):.1f} MC [{float(r['mc_q10']):.1f}–{float(r['mc_q90']):.1f}]"

    # c. Top levers
    row_05 = heatmap_raw[(heatmap_raw["pipeline_mult"] == 0.5) & (heatmap_raw["quality"] == 1.0)]
    row_cb = heatmap_raw[(heatmap_raw["pipeline_mult"] == pipeline_mult) & (heatmap_raw["quality"] == 1.0)]
    if not row_cb.empty and not row_05.empty:
        reac_cur = float(row_cb.iloc[0]["reactive_serviceability_mean"]) * 100
        reac_05 = float(row_05.iloc[0]["reactive_serviceability_mean"]) * 100
        diff_reac = reac_05 - reac_cur
        shikra_gain_05 = float(row_05.iloc[0]["gain_vs_reactive_mean"]) * 100
        lever_str = (
            f"Reactive serviceability at today's pipeline ({pipeline_mult}×): <b>{reac_cur:.1f}%</b> vs "
            f"0.5× pipeline: <b>{reac_05:.1f}%</b> ({diff_reac:+.1f} pp diff computed in code). "
            f"SHIKRA's gain at 0.5× pipeline: <b>{shikra_gain_05:+.1f} pp</b>."
        )
    else:
        lever_str = "Pipeline lever data available in results/heatmap.csv."

    # d. Top 3 parts to indigenise (sorted by loss over the halt)
    top3_indig = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values(
        "loss_pp_shock_window", ascending=False
    ).head(3)
    indig_items = "".join(
        f"<li><b>{theme.esc(row['lru_id'])}</b>: loss {row['loss_pp_shock_window']:.1f} pp "
        f"(peak +{row['peak_extra_grounded']:.0f} grounded, recovery {row['recovery_days']:.0f} days)</li>"
        for _, row in top3_indig.iterrows()
    )

    # e. Pending sign-offs + audit-chain status line
    recent_logs = _get_rec(100)
    pending_recs = sum(1 for r in recent_logs if r.get("decision") == "approved")
    chain = audit_verify()
    if chain["ok"]:
        root_abbr = (chain.get("latest_root") or "none")[:12]
        audit_line = f"Audit chain: intact · {chain['n_rows']} records · root <code>{root_abbr}</code>"
    else:
        audit_line = f"Audit chain: BROKEN at row id={chain['first_bad_id']}"

    status_rows = "".join(
        f'<tr><td>Squadron {i + 1} (S{i + 1})</td><td class="sk-mono sk-t-green">{c["MC"]}</td>'
        f'<td class="sk-mono sk-t-amber">{c["NMCM"]}</td><td class="sk-mono sk-t-red">{c["NMCS"]}</td>'
        '<td class="sk-mono">18</td></tr>'
        for i, c in enumerate(counts_b)
    )
    status_rows += (
        f'<tr><td><b>Total wing</b></td><td class="sk-mono sk-t-green"><b>{total_mc} / 54 MC ({total_mc / 54.0 * 100:.1f}%)</b></td>'
        f'<td class="sk-mono sk-t-amber">{sum(c["NMCM"] for c in counts_b)}</td>'
        f'<td class="sk-mono sk-t-red">{sum(c["NMCS"] for c in counts_b)}</td><td class="sk-mono">54</td></tr>'
    )
    fc_rows = "".join(
        f'<tr><td>D+{lead}</td>' + "".join(f'<td class="sk-mono">{_fc_cell(lead, s)}</td>' for s in (1, 2, 3)) + "</tr>"
        for lead in (3, 7)
    )
    h3 = 'style="font-size:12px!important;border:none!important;padding:0!important;margin:1.1rem 0 .4rem 0!important;color:var(--outline)!important;letter-spacing:.08em;"'
    brief_style = """
<style>
@media print {
  [data-testid="stSidebar"], button, .stButton, header { display: none !important; }
  body, .stApp { background: #ffffff !important; color: #000000 !important; }
  #shikra_brief { background: #ffffff !important; color: #000000 !important; padding: 0 !important; }
  #shikra_brief * { color: #000000 !important; }
}
</style>
"""
    brief_html = f"""
<div id="shikra_brief" class="sk-panel" style="max-width:960px;border-top:2px solid var(--primary-container);padding:1.5rem 2rem;">
<div class="sk-eyebrow">Advisory recommendation · one-page brief</div>
<h2 style="color:var(--primary);margin:.5rem 0 0 0;font-size:22px;">SHIKRA · Commander's brief · Day 485 · synthetic wing</h2>
<h3 {h3}>1. Wing status now (day {SNAP_DAY})</h3>
<table class="sk-table"><tr><th>Squadron</th><th>Mission capable (MC)</th><th>Maintenance (NMCM)</th><th>Supply waiting (NMCS)</th><th>Total aircraft</th></tr>{status_rows}</table>
<h3 {h3}>2. Readiness forecast (expected MC, 80% band)</h3>
<table class="sk-table"><tr><th>Horizon</th><th>S1</th><th>S2</th><th>S3</th></tr>{fc_rows}</table>
<p class="sk-note" style="font-style:italic;margin-top:.4rem;">{calib_caption_text}</p>
<h3 {h3}>3. Top levers (heatmap.csv)</h3>
<p class="sk-note" style="font-size:13px;">{lever_str}</p>
<h3 {h3}>4. Top 3 parts to indigenise (supply_shock.csv)</h3>
<ol class="sk-note" style="font-size:13px;">{indig_items}</ol>
<h3 {h3}>5. Pending sign-offs &amp; audit integrity</h3>
<p class="sk-note" style="font-size:13px;">EO approved recommendations: <b>{pending_recs}</b> · <span class="sk-t-green">{audit_line}</span></p>
<p class="sk-mono sk-t-muted" style="font-size:11px;text-align:center;margin-top:1.2rem;">Advisory decision support · offline · synthetic wing + NASA C-MAPSS</p>
</div>
"""
    return brief_style, brief_html


brief_style, brief_html = _build_brief()
util_line = ""
if not util_shikra.empty and not util_reactive.empty:
    util_line = (
        f"flying task met: SHIKRA {us['task_mean'] * 100:.1f}% "
        f"(95% CI [{us['task_ci_lo'] * 100:.1f}, {us['task_ci_hi'] * 100:.1f}]%) vs reactive "
        f"{ur['task_mean'] * 100:.1f}%; sorties flown / yr SHIKRA {us['sorties_mean']:,.0f} vs reactive "
        f"{ur['sorties_mean']:,.0f}. {UTIL_CAPTION}"
    )
brief_doc = brief_document(brief_html, SNAP_DAY, util_line)
st.session_state["commander_brief_doc"] = brief_doc

with col_rail:
    st.markdown("### Readiness Forecast")
    if readiness_mod is None:
        theme.html(
            '<div class="sk-rcard"><p class="sk-note">Readiness forecast: awaiting readiness module.</p></div>'
        )
    elif fc_df is None or fc_df.empty:
        st.info("Readiness forecast data not available.")
    else:
        theme.html(
            '<div class="sk-note" style="margin-bottom:.6rem;">Expected mission-capable aircraft per squadron '
            f"from day {SNAP_DAY}, with the forecast's 80% band (q10–q90 of 150 Monte Carlo runs).</div>"
        )
        for sq in (1, 2, 3):
            c = sq_counts[sq - 1] if sq_counts else None
            r3, r7 = _fc_row(3, sq), _fc_row(7, sq)
            now_mc = c["MC"] if c else None
            tone = "green" if (now_mc or 0) >= 14 else ("amber" if (now_mc or 0) >= 11 else "red")
            band_html = ""
            if r3 is not None:
                lo, hi, ex = float(r3["mc_q10"]), float(r3["mc_q90"]), float(r3["expected_mc"])
                band_html = (
                    f'<div class="sk-band"><div class="sk-bi" style="left:{lo / 18 * 100:.1f}%;width:{max(hi - lo, 0.2) / 18 * 100:.1f}%;"></div>'
                    f'<div class="sk-bm" style="left:calc({ex / 18 * 100:.1f}% - 2px);"></div>'
                    + (f'<div class="sk-bn" style="left:{now_mc / 18 * 100:.1f}%;" title="now"></div>' if now_mc is not None else "")
                    + "</div>"
                    f'<div class="sk-bandax"><span>0</span><span>D+3 band [{lo:.1f}, {hi:.1f}] · white tick = now</span><span>18</span></div>'
                )

            def _cell(label: str, r) -> str:
                if r is None:
                    return f'<div><div class="sk-fl">{label}</div><div class="sk-fv">n/a</div></div>'
                return (
                    f'<div><div class="sk-fl">{label}</div>'
                    f'<div class="sk-fv">{float(r["expected_mc"]):.1f} <span style="font-size:12px;color:var(--outline);">MC</span></div>'
                    f'<div class="sk-fb">80% band [{float(r["mc_q10"]):.1f} – {float(r["mc_q90"]):.1f}]</div></div>'
                )

            now_chip = (
                theme.chip(f"{now_mc}/18 MC now", tone, led=True) if now_mc is not None else ""
            )
            split = (
                f'{c["MC"]} MC · {c["NMCM"]} NMCM · {c["NMCS"]} NMCS on day {SNAP_DAY}' if c else ""
            )
            theme.html(
                f'<div class="sk-rcard"><div class="sk-rh"><div><div class="sk-rname">Squadron {sq}</div>'
                f'<div class="sk-rsub">{split}</div></div>{now_chip}</div>'
                f'<div class="sk-fc2">{_cell("Expected D+3", r3)}{_cell("Expected D+7", r7)}</div>'
                f"{band_html}</div>"
            )

        raw_note = (
            f" The raw band shown covers {raw_cov_val:.4f} of held-out outcomes at D+3; the conformal "
            "widening is applied in scripts/readiness_calib.py and is not stored, so the wider band is not drawn."
            if raw_cov_val is not None
            else ""
        )
        st.caption(calib_caption_text + raw_note)

        with st.expander("D+1 … D+7 expected MC table"):
            rows = []
            for d in range(1, 8):
                row = {"Day": f"D+{d}"}
                for sq in (1, 2, 3):
                    r = _fc_row(d, sq)
                    row[f"S{sq} expected MC"] = None if r is None else round(float(r["expected_mc"]), 2)
                    row[f"S{sq} 80% band"] = (
                        None if r is None else f"[{float(r['mc_q10']):.1f}, {float(r['mc_q90']):.1f}]"
                    )
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Advisory recommendation card + brief trigger
    top_foreign = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")
    rec_line = ""
    if not top_foreign.empty and not row_cur_df.empty:
        tf = top_foreign.iloc[0]
        row_05 = heatmap_df[(heatmap_df["pipeline_mult"] == 0.5) & (heatmap_df["quality"] == 1.0)]
        reac_cur = row_cur_df.iloc[0]["reactive_serviceability_mean"] * 100
        reac_05 = row_05.iloc[0]["reactive_serviceability_mean"] * 100 if not row_05.empty else None
        lever = (
            f"Halving the pipeline time (0.5×) lifts even reactive serviceability from {reac_cur:.1f}% to "
            f"{reac_05:.1f}%. " if reac_05 is not None else ""
        )
        rec_line = (
            f"{lever}The largest single-part exposure is <b>{theme.esc(tf['lru_id'])}</b> "
            f"(−{tf['loss_pp_shock_window']:.1f} pp over a 180-day halt): first candidate for indigenisation."
        )
    theme.html(
        '<div class="sk-advisory"><div class="sk-ah">Advisory recommendation</div>'
        f"<p>{rec_line}</p>"
        '<p style="color:var(--outline)!important;font-size:11px!important;font-family:var(--font-mono);">'
        "Source: results/heatmap.csv · results/supply_shock.csv · requires officer sign-off</p></div>"
    )
    b_open, b_dl = st.columns(2)
    with b_open:
        open_brief = st.button("Open one-page brief", key="open_brief_btn", use_container_width=True)
    with b_dl:
        st.download_button(
            "Download daily brief (HTML)",
            data=brief_doc,
            file_name=f"shikra_commander_brief_day{SNAP_DAY}.html",
            mime="text/html",
            key="download_brief_btn",
            use_container_width=True,
        )

if open_brief:
    st.markdown(theme.oneline(brief_style + brief_html), unsafe_allow_html=True)


# ── Performance ──────────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
st.caption(f"Page load: {elapsed:.2f} s. Synthetic wing · SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")
theme.footer()
