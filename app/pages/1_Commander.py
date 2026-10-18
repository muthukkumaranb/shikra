"""SHIKRA — Commander's View.

Lever map (interactive Altair heatmap), supply-shock ranking,
readiness forecast panel, and one-page brief.
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

from app.lib.data import load_config, load_heatmap, load_supply_shock  # noqa: E402
from app.lib.readiness import (  # noqa: E402
    get_forecast,
    get_readiness_module,
    get_snapshot,
    load_readiness_calib_coverage,
)

start_time = time.perf_counter()

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Commander",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
st.title("Commander's View")
st.caption("Synthetic wing · advisory — all recommendations require officer sign-off")

# ── Load data ────────────────────────────────────────────────────────────────
heatmap_raw = load_heatmap()
config = load_config()
shock_df = load_supply_shock()

pipeline_mult = config.get("pipeline_mult", 1.7)

# Format heatmap units in pp
heatmap_df = heatmap_raw.copy()
heatmap_df["gain_pp"] = heatmap_df["gain_vs_reactive_mean"] * 100
heatmap_df["gain_ci_lo_pp"] = heatmap_df["gain_ci_lo"] * 100
heatmap_df["gain_ci_hi_pp"] = heatmap_df["gain_ci_hi"] * 100
heatmap_df["gain_label"] = heatmap_df["gain_pp"].apply(lambda v: f"{v:+.1f} pp")
heatmap_df["serviceability_pct"] = heatmap_df["serviceability_mean"] * 100
heatmap_df["serviceability_ci_lo_pct"] = heatmap_df["serviceability_ci_lo"] * 100
heatmap_df["serviceability_ci_hi_pct"] = heatmap_df["serviceability_ci_hi"] * 100
heatmap_df["serviceability_label"] = heatmap_df["serviceability_pct"].apply(lambda v: f"{v:.1f}%")

# ── Lever Map ────────────────────────────────────────────────────────────────
st.markdown("### Lever Map")

# Toggle between gain vs reactive and absolute serviceability
show_absolute = st.toggle("Show absolute serviceability", value=False)

if show_absolute:
    color_col = "serviceability_pct"
    color_title = "Serviceability (%)"
    text_col = "serviceability_label"
    tooltip_cols = [
        alt.Tooltip("pipeline_mult:Q", title="Pipeline speed"),
        alt.Tooltip("quality_label:N", title="Prediction quality"),
        alt.Tooltip("serviceability_label:N", title="Serviceability"),
        alt.Tooltip("serviceability_ci_lo_pct:Q", title="CI low (%)", format=".1f"),
        alt.Tooltip("serviceability_ci_hi_pct:Q", title="CI high (%)", format=".1f"),
    ]
else:
    color_col = "gain_pp"
    color_title = "Gain vs reactive (pp)"
    text_col = "gain_label"
    tooltip_cols = [
        alt.Tooltip("pipeline_mult:Q", title="Pipeline speed"),
        alt.Tooltip("quality_label:N", title="Prediction quality"),
        alt.Tooltip("gain_label:N", title="Gain vs reactive"),
        alt.Tooltip("gain_ci_lo_pp:Q", title="CI low (pp)", format="+.1f"),
        alt.Tooltip("gain_ci_hi_pp:Q", title="CI high (pp)", format="+.1f"),
    ]

# Base heatmap with explicit x-axis title
base = alt.Chart(heatmap_df).encode(
    x=alt.X(
        "pipeline_mult:O",
        title="repair & supply time (× today's baseline; lower = faster)",
        sort="ascending",
    ),
    y=alt.Y("quality_label:N", title="Prediction Quality", sort="ascending"),
)

heatmap_chart = base.mark_rect(stroke="white", strokeWidth=1).encode(
    color=alt.Color(
        f"{color_col}:Q",
        title=color_title,
        scale=alt.Scale(scheme="viridis"),
    ),
    tooltip=tooltip_cols,
)

# Text labels on cells in pp
text_chart = base.mark_text(fontSize=11, fontWeight="bold").encode(
    text=alt.Text(f"{text_col}:N"),
    color=alt.condition(
        alt.datum[color_col] > heatmap_df[color_col].median(),
        alt.value("black"),
        alt.value("white"),
    ),
)

# Mark today's calibrated point
today_df = heatmap_df[
    (heatmap_df["pipeline_mult"] == pipeline_mult) & (heatmap_df["quality"] == 1.0)
]
if not today_df.empty:
    today_mark = (
        alt.Chart(today_df)
        .mark_rect(stroke="red", strokeWidth=3, filled=False)
        .encode(
            x=alt.X("pipeline_mult:O", sort="ascending"),
            y=alt.Y("quality_label:N", sort="ascending"),
        )
    )
    chart = (heatmap_chart + text_chart + today_mark).properties(
        width=500, height=350, title="Lever Map — prediction quality × pipeline speed"
    )
else:
    chart = (heatmap_chart + text_chart).properties(
        width=500, height=350, title="Lever Map — prediction quality × pipeline speed"
    )

st.altair_chart(chart, use_container_width=True)
st.caption(
    f"Red outline marks today's calibrated point (pipeline_mult={pipeline_mult}×, "
    f"SHIKRA quality=1). Synthetic wing."
)

st.divider()

# ── Supply-Shock Ranking ─────────────────────────────────────────────────────
st.markdown("### Supply-Shock Ranking")
st.caption(
    "180-day foreign-supply halt, at 1.0× pipeline speed, synthetic parameters; "
    "the method is the claim."
)

# Bar chart: loss over the halt with CI bars
shock_items = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")
bar_chart = (
    alt.Chart(shock_items)
    .mark_bar(color="#3b82f6")
    .encode(
        x=alt.X("loss_pp_shock_window:Q", title="Loss over halt (pp)"),
        y=alt.Y("lru_id:N", title="LRU", sort="-x"),
        tooltip=[
            alt.Tooltip("lru_id:N", title="LRU"),
            alt.Tooltip("loss_pp_shock_window:Q", title="Loss (pp)", format=".1f"),
            alt.Tooltip("ci_lo:Q", title="CI low", format=".1f"),
            alt.Tooltip("ci_hi:Q", title="CI high", format=".1f"),
            alt.Tooltip("source:N", title="Source"),
        ],
    )
)

error_bars = (
    alt.Chart(shock_items)
    .mark_errorbar(color="white")
    .encode(
        x=alt.X("ci_lo:Q"),
        x2=alt.X2("ci_hi:Q"),
        y=alt.Y("lru_id:N", sort="-x"),
    )
)

st.altair_chart(
    (bar_chart + error_bars).properties(width=500, height=300),
    use_container_width=True,
)

# Table: peak extra grounded and recovery days
st.markdown("**Detailed supply-shock metrics:**")
table_df = shock_df[[
    "lru_id", "source", "loss_pp_shock_window", "peak_extra_grounded",
    "recovery_days", "rank"
]].copy()
table_df.columns = ["LRU", "Source", "Loss (pp)", "Peak extra grounded", "Recovery days", "Rank"]
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.divider()

# ── Readiness Forecast Panel ─────────────────────────────────────────────────
st.markdown("### Readiness Forecast")

readiness_mod = get_readiness_module()

if readiness_mod is not None:
    snap = get_snapshot("shikra_v2", seed=1, day=485)

    if snap is not None:
        st.markdown("#### P(≥N MC) per Squadron — D+1 … D+7")

        n_slider = st.slider(
            "Mission Capable threshold N",
            min_value=6,
            max_value=18,
            value=12,
            step=1,
            help="P(≥N MC) = probability that the squadron has at least N aircraft mission capable",
        )

        with st.spinner("Computing readiness forecast (D+1 … D+7)…"):
            fc_df = get_forecast(
                snap, config, load_config(),
                policy="shikra",
                horizon_days=7,
                n=150,
                seed=42,
            )

        if fc_df is not None and not fc_df.empty:
            p_col = f"p_ge_{n_slider}"
            if p_col not in fc_df.columns:
                st.warning(f"Column {p_col} not in forecast output.")
            else:
                # Build a tidy table: lead_days × sqn → P(≥N MC) %
                pivot_rows = []
                sqn_labels = {1: "S1", 2: "S2", 3: "S3"}
                for lead in range(1, 8):
                    row = {"Day": f"D+{lead}"}
                    for sqn in [1, 2, 3]:
                        sub = fc_df[(fc_df["lead_days"] == lead) & (fc_df["sqn"] == sqn)]
                        if not sub.empty:
                            row[sqn_labels[sqn]] = round(float(sub.iloc[0][p_col]) * 100, 1)
                            row[f"{sqn_labels[sqn]}_exp"] = round(float(sub.iloc[0]["expected_mc"]), 1)
                        else:
                            row[sqn_labels[sqn]] = None
                            row[f"{sqn_labels[sqn]}_exp"] = None
                    pivot_rows.append(row)
                pivot_df = pd.DataFrame(pivot_rows)

                # Display as Altair heatmap
                tidy = pivot_df[["Day", "S1", "S2", "S3"]].melt(
                    id_vars="Day", var_name="Squadron", value_name="P(≥N MC) %"
                )
                heat = (
                    alt.Chart(tidy)
                    .mark_rect()
                    .encode(
                        x=alt.X("Squadron:N", title="Squadron"),
                        y=alt.Y("Day:O", title="Horizon", sort=[f"D+{i}" for i in range(1, 8)]),
                        color=alt.Color(
                            "P(≥N MC) %:Q",
                            scale=alt.Scale(scheme="rdylgn", domain=[0, 100]),
                            title=f"P(≥{n_slider} MC) %",
                        ),
                        tooltip=[
                            alt.Tooltip("Day:O", title="Horizon"),
                            alt.Tooltip("Squadron:N", title="Squadron"),
                            alt.Tooltip("P(≥N MC) %:Q", title=f"P(≥{n_slider} MC) %", format=".1f"),
                        ],
                    )
                )
                text_heat = (
                    alt.Chart(tidy)
                    .mark_text(fontWeight="bold", fontSize=13)
                    .encode(
                        x=alt.X("Squadron:N"),
                        y=alt.Y("Day:O", sort=[f"D+{i}" for i in range(1, 8)]),
                        text=alt.Text("P(≥N MC) %:Q", format=".0f"),
                        color=alt.condition(
                            alt.datum["P(≥N MC) %"] > 50,
                            alt.value("#0f172a"),
                            alt.value("#f8fafc"),
                        ),
                    )
                )
                st.altair_chart(
                    (heat + text_heat).properties(
                        width=340, height=280,
                        title=f"P(≥{n_slider} MC) per Squadron — synthetic wing"
                    ),
                    use_container_width=False,
                )

                # Show expected MC table
                exp_tidy = pivot_df[["Day", "S1_exp", "S2_exp", "S3_exp"]].copy()
                exp_tidy.columns = ["Day", "S1 Expected MC", "S2 Expected MC", "S3 Expected MC"]
                st.dataframe(exp_tidy, use_container_width=True, hide_index=True)

                # Honest calibration caption
                cov_val = load_readiness_calib_coverage()
                calib_path = ROOT / "results" / "readiness_calib.csv"
                if calib_path.exists():
                    import pandas as _pd_calib
                    _cal = _pd_calib.read_csv(calib_path)
                    _row_fc = _cal[(_cal["lead_days"] == 3) & (_cal["method"] == "shikra_forecast")]
                    _row_pe = _cal[(_cal["lead_days"] == 3) & (_cal["method"] == "persistence")]
                    if not _row_fc.empty and not _row_pe.empty:
                        mae_fc = float(_row_fc.iloc[0]["mae"])
                        mae_pe = float(_row_pe.iloc[0]["mae"])
                        n_snap = int(_row_fc.iloc[0]["n_snapshots"])
                        cov_str = cov_val or "pending"
                        st.caption(
                            f"raw band coverage at D+3: {cov_str} (held-out, n={n_snap} snapshots) · "
                            f"point-forecast MAE {mae_fc:.4f} vs persistence MAE {mae_pe:.4f}. "
                            f"Synthetic wing."
                        )
                    else:
                        if cov_val:
                            st.caption(f"raw band coverage at D+3: {cov_val}. Synthetic wing.")
                else:
                    st.caption("Readiness forecast. Synthetic wing.")
        else:
            st.info("Readiness forecast data not available.")
    else:
        st.info("Snapshot not available for day 485.")
else:
    st.markdown(
        '<div style="background: #374151; border-radius: 8px; padding: 1rem; '
        'color: #9ca3af; text-align: center;">'
        "Readiness forecast: awaiting readiness module."
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Commander's One-Page Brief ───────────────────────────────────────────────
st.markdown("### Commander's Brief")

if st.button("Open one-page brief", key="open_brief_btn"):
    import json as _json

    from app.lib.audit import verify as audit_verify

    # Calibration numbers from readiness_calib.csv
    calib_d3_cov = load_readiness_calib_coverage() or "pending"
    calib_path = ROOT / "results" / "readiness_calib.csv"
    d3_exp_mc: dict[str, str] = {}
    d7_exp_mc: dict[str, str] = {}
    d3_band: dict[str, str] = {}
    d7_band: dict[str, str] = {}

    if readiness_mod is not None:
        snap_b = get_snapshot("shikra_v2", seed=1, day=485)
        if snap_b is not None:
            _fc = get_forecast(snap_b, config, load_config(), policy="shikra", horizon_days=7, n=150, seed=42)
            if _fc is not None:
                for _d, _store_exp, _store_band in [(3, d3_exp_mc, d3_band), (7, d7_exp_mc, d7_band)]:
                    for _sqn in [1, 2, 3]:
                        _sub = _fc[(_fc["lead_days"] == _d) & (_fc["sqn"] == _sqn)]
                        if not _sub.empty:
                            _r = _sub.iloc[0]
                            _store_exp[f"S{_sqn}"] = f"{float(_r['expected_mc']):.1f}"
                            _store_band[f"S{_sqn}"] = f"[{float(_r['mc_q10']):.1f}, {float(_r['mc_q90']):.1f}]"

    # Top levers from heatmap.csv
    top_levers_rows = heatmap_df.sort_values("gain_pp", ascending=False).head(3)
    lever_lines = []
    for _, _lr in top_levers_rows.iterrows():
        lever_lines.append(
            f"Pipeline {_lr['pipeline_mult']}× · quality {_lr['quality_label']}: "
            f"gain {_lr['gain_pp']:+.1f} pp"
        )

    # Top 3 indigenisation from supply_shock.csv
    top3 = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank").head(3)
    indig_lines = []
    for _, _row in top3.iterrows():
        indig_lines.append(
            f"{_row['lru_id']}: loss {_row['loss_pp_shock_window']:.1f} pp "
            f"(peak +{_row['peak_extra_grounded']:.0f} grounded, "
            f"recovery {_row['recovery_days']:.0f} d)"
        )

    # Audit chain
    chain = audit_verify()
    if chain["ok"]:
        audit_status = f"intact · {chain['n_rows']} records"
    else:
        audit_status = f"BROKEN at row {chain['first_bad_id']}"

    # EO open recommendations count
    from app.lib.audit import get_recent as _get_rec

    _recent = _get_rec(100)
    open_recs = sum(1 for r in _recent if r.get("decision") == "approved")

    # Build HTML brief
    d3_exp_str = " | ".join(f"{k} {v} MC" for k, v in sorted(d3_exp_mc.items()))
    d7_exp_str = " | ".join(f"{k} {v} MC" for k, v in sorted(d7_exp_mc.items()))
    d3_band_str = " | ".join(f"{k} {v}" for k, v in sorted(d3_band.items()))
    d7_band_str = " | ".join(f"{k} {v}" for k, v in sorted(d7_band.items()))

    brief_html = f"""
<style>
@media print {{
  [data-testid="stSidebar"], button[data-testid], .stButton {{ display: none !important; }}
}}
</style>
<div id="shikra_brief" style="background:#0f172a;color:#e2e8f0;border:1px solid #334155;
     border-radius:10px;padding:2rem;font-family:monospace;max-width:900px;">
  <h2 style="color:#38bdf8;margin-top:0;">SHIKRA · Commander's Brief · Day 485 · synthetic wing</h2>

  <h3 style="color:#94a3b8;font-size:0.95rem;">WING STATUS NOW</h3>
  <p style="margin:0 0 1rem;">Readiness forecast available for S1/S2/S3 (see table above).</p>

  <h3 style="color:#94a3b8;font-size:0.95rem;">READINESS FORECAST</h3>
  <table style="border-collapse:collapse;width:100%;margin-bottom:1rem;font-size:0.85rem;">
    <tr style="border-bottom:1px solid #334155;">
      <th style="text-align:left;padding:4px 8px;color:#94a3b8;">Horizon</th>
      <th style="text-align:left;padding:4px 8px;color:#94a3b8;">Expected MC</th>
      <th style="text-align:left;padding:4px 8px;color:#94a3b8;">Forecast band (q10–q90)</th>
    </tr>
    <tr>
      <td style="padding:4px 8px;">D+3</td>
      <td style="padding:4px 8px;">{d3_exp_str or "—"}</td>
      <td style="padding:4px 8px;">{d3_band_str or "—"}</td>
    </tr>
    <tr>
      <td style="padding:4px 8px;">D+7</td>
      <td style="padding:4px 8px;">{d7_exp_str or "—"}</td>
      <td style="padding:4px 8px;">{d7_band_str or "—"}</td>
    </tr>
  </table>
  <p style="font-size:0.78rem;color:#64748b;margin-bottom:1rem;">
    raw forecast band; held-out coverage at D+3: {calib_d3_cov} · Synthetic wing.
  </p>

  <h3 style="color:#94a3b8;font-size:0.95rem;">TOP LEVERS (heatmap.csv)</h3>
  <ol style="margin:0 0 1rem;padding-left:1.5rem;font-size:0.85rem;">
    {"".join(f"<li>{l}</li>" for l in lever_lines)}
  </ol>

  <h3 style="color:#94a3b8;font-size:0.95rem;">TOP 3 INDIGENISATION PRIORITIES (supply_shock.csv)</h3>
  <ol style="margin:0 0 1rem;padding-left:1.5rem;font-size:0.85rem;">
    {"".join(f"<li>{l}</li>" for l in indig_lines)}
  </ol>

  <h3 style="color:#94a3b8;font-size:0.95rem;">PENDING SIGN-OFFS</h3>
  <p style="margin:0 0 0.5rem;font-size:0.85rem;">
    EO approved recommendations: {open_recs} · Audit chain: {audit_status}
  </p>

  <hr style="border-color:#334155;margin:1.5rem 0;"/>
  <p style="font-size:0.75rem;color:#64748b;margin:0;">
    Advisory decision support · offline · synthetic wing + NASA C-MAPSS
  </p>
</div>
"""
    st.markdown(brief_html, unsafe_allow_html=True)

# ── Performance ──────────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
st.caption(f"Page load: {elapsed:.2f} s. Synthetic wing · SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")
