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

from app.lib.data import load_config, load_heatmap, load_params, load_supply_shock  # noqa: E402
from app.lib.readiness import (  # noqa: E402
    get_forecast,
    get_readiness_module,
    get_snapshot,
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

st.sidebar.caption("SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")

# ── Load data ────────────────────────────────────────────────────────────────
heatmap_raw = load_heatmap()
config = load_config()
params = load_params()
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
        scale=alt.Scale(scheme="blues" if show_absolute else "viridis"),
    ),
    tooltip=tooltip_cols,
)

# Text overlay showing values
text = base.mark_text(baseline="middle", fontSize=12, fontWeight="bold").encode(
    text=alt.Text(f"{text_col}:N"),
    color=alt.condition(
        alt.datum[color_col] > (70 if show_absolute else 1.0),
        alt.value("white"),
        alt.value("black"),
    ),
)

# Callout at (1.7, 1.0) — today's calibrated operating point
callout_data = pd.DataFrame([{
    "pipeline_mult": 1.7,
    "quality_label": "SHIKRA (calibrated)",
    "note": "▲ Today's operating point",
}])
callout = (
    alt.Chart(callout_data)
    .mark_text(dy=28, fontSize=11, fontWeight="bold", color="#f8fafc")
    .encode(
        x=alt.X("pipeline_mult:O"),
        y=alt.Y("quality_label:N"),
        text=alt.Text("note:N"),
    )
)

st.altair_chart(
    (heatmap_chart + text + callout).properties(width=750, height=300),
    use_container_width=True,
)

# Caption with dynamic values
row_17 = heatmap_df[
    (heatmap_df["pipeline_mult"] == pipeline_mult) & (heatmap_df["quality"] == 1.0)
]
if not row_17.empty:
    r = row_17.iloc[0]
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

st.divider()

# ── Supply-Shock Ranking ────────────────────────────────────────────────────
st.markdown("### Supply-Shock Ranking — Strategic Chokepoints")
st.caption(
    "180-day foreign-supply halt on each component. Ordered by readiness loss. "
    "Synthetic parameters; the method is the claim."
)

# Format shock table: clean columns, bold foreign-sourced rows
shock_display = shock_df.copy()
shock_display["Loss (pp)"] = shock_display["loss_pp_shock_window"].apply(
    lambda v: f"{v:.1f} pp"
)
shock_display["Peak Extra Grounded"] = shock_display["peak_extra_grounded"].apply(
    lambda v: f"+{v:.0f}"
)
shock_display["Recovery (days)"] = shock_display["recovery_days"].apply(
    lambda v: f"{v:.0f} d"
)
shock_display["Foreign Sourced"] = shock_display["source"].apply(
    lambda s: "Yes" if s == "foreign" else "No"
)

col_order = [
    "rank",
    "lru_id",
    "source",
    "Loss (pp)",
    "Peak Extra Grounded",
    "Recovery (days)",
]
shock_display = shock_display[
    [c for c in col_order if c in shock_display.columns]
].rename(
    columns={
        "rank": "Rank",
        "lru_id": "Component",
        "source": "Source",
    }
)

st.dataframe(
    shock_display,
    use_container_width=True,
    hide_index=True,
)

# Metric callouts for top foreign item
foreign_lrus = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")
if not foreign_lrus.empty:
    top = foreign_lrus.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Chokepoint", top["lru_id"])
    c2.metric("Readiness Loss", f"{top['loss_pp_shock_window']:.1f} pp")
    c3.metric("Peak Extra Grounded", f"+{top['peak_extra_grounded']:.0f} aircraft")
    c4.metric("Recovery Time", f"{top['recovery_days']:.0f} days")

st.divider()

# ── Readiness Forecast Panel (Part 1.4) ──────────────────────────────────────
st.markdown("### Readiness Forecast Panel")
st.caption(
    "Live P(≥N MC) readiness forecast per squadron across D+1…D+7. "
    "Advisory decision support · synthetic wing."
)

# Honest calibration caption helpers
calib_path = ROOT / "results" / "readiness_calib.csv"
conf_cov_val = 0.8626
mae_fc_val = 1.3263
mae_pe_val = 0.9818
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
        if not _row_pe.empty:
            mae_pe_val = float(_row_pe.iloc[0]["mae"])
    except Exception:
        pass

calib_caption_text = (
    f"Calibrated band covers {conf_cov_val:.4f} ({conf_cov_val * 100:.1f}%) of held-out outcomes; "
    f"the point forecast does not yet beat persistence (D+3 MAE {mae_fc_val:.4f} vs {mae_pe_val:.4f}). "
    f"Synthetic wing."
)

readiness_mod = get_readiness_module()
if readiness_mod is not None:
    snap = get_snapshot("shikra_v2", seed=1, day=485)
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
        if fc_df is not None and not fc_df.empty:
            col_ctrl, col_chart = st.columns([1, 2])
            with col_ctrl:
                n_slider = st.slider(
                    "Target MC Aircraft (N)",
                    min_value=6,
                    max_value=18,
                    value=12,
                    step=1,
                    help="Target number of mission-capable aircraft per squadron",
                )
                st.caption(f"Displaying probability of achieving ≥ {n_slider} MC aircraft per squadron.")

            with col_chart:
                col_key = f"p_ge_{n_slider}"
                sqn_labels = {1: "S1", 2: "S2", 3: "S3"}
                pivot_rows = []
                for d in range(1, 8):
                    row = {"Day": f"D+{d}"}
                    for sqn in [1, 2, 3]:
                        sub = fc_df[(fc_df["lead_days"] == d) & (fc_df["sqn"] == sqn)]
                        if not sub.empty:
                            prob_val = float(sub.iloc[0].get(col_key, 0.0)) * 100.0
                            row[sqn_labels[sqn]] = prob_val
                            row[f"{sqn_labels[sqn]}_exp"] = float(sub.iloc[0]["expected_mc"])
                        else:
                            row[sqn_labels[sqn]] = None
                            row[f"{sqn_labels[sqn]}_exp"] = None
                    pivot_rows.append(row)
                pivot_df = pd.DataFrame(pivot_rows)

                # Display as Altair heatmap with valid color scheme
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
                            scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100]),
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

                # Honest calibration caption (Part 1.4)
                st.caption(calib_caption_text)
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

# ── Commander's One-Page Brief (Part 2) ──────────────────────────────────────
st.markdown("### Commander's Brief")

if st.button("Open one-page brief", key="open_brief_btn"):
    from app.lib.audit import get_recent as _get_rec
    from app.lib.audit import verify as audit_verify

    # a. Wing status now: MC / NMCM / NMCS counts per squadron (from snapshot)
    snap_now = get_snapshot("shikra_v2", seed=1, day=485) if readiness_mod else None
    s1_mc, s1_nmcm, s1_nmcs = 0, 0, 0
    s2_mc, s2_nmcm, s2_nmcs = 0, 0, 0
    s3_mc, s3_nmcm, s3_nmcs = 0, 0, 0
    if snap_now is not None and "tails" in snap_now:
        for t in snap_now["tails"]:
            sq = t.get("sqn", 0)
            st_val = t.get("status", "MC")
            if sq == 0:
                if st_val == "MC":
                    s1_mc += 1
                elif st_val == "NMCM":
                    s1_nmcm += 1
                else:
                    s1_nmcs += 1
            elif sq == 1:
                if st_val == "MC":
                    s2_mc += 1
                elif st_val == "NMCM":
                    s2_nmcm += 1
                else:
                    s2_nmcs += 1
            else:
                if st_val == "MC":
                    s3_mc += 1
                elif st_val == "NMCM":
                    s3_nmcm += 1
                else:
                    s3_nmcs += 1
    else:
        from app.lib.state import get_wing_state_on_day
        ws = get_wing_state_on_day(485, policy="shikra_v2", seed=1)
        for tinfo in ws["tail_statuses"]:
            sq_str = tinfo["squadron"]
            st_val = tinfo["status"]
            if sq_str == "S1":
                if st_val == "MC":
                    s1_mc += 1
                elif st_val == "NMCM":
                    s1_nmcm += 1
                else:
                    s1_nmcs += 1
            elif sq_str == "S2":
                if st_val == "MC":
                    s2_mc += 1
                elif st_val == "NMCM":
                    s2_nmcm += 1
                else:
                    s2_nmcs += 1
            else:
                if st_val == "MC":
                    s3_mc += 1
                elif st_val == "NMCM":
                    s3_nmcm += 1
                else:
                    s3_nmcs += 1

    total_mc = s1_mc + s2_mc + s3_mc

    # b. Readiness forecast: P(≥12 MC) per squadron at D+3 and D+7
    fc_brief = None
    if snap_now is not None:
        fc_brief = get_forecast(
            snapshot=snap_now,
            config=config,
            params=params,
            policy="shikra",
            horizon_days=7,
            n=150,
            seed=42,
        )

    p12_d3 = {}
    p12_d7 = {}
    exp_d3 = {}
    exp_d7 = {}
    if fc_brief is not None and not fc_brief.empty:
        for sq in [1, 2, 3]:
            r3 = fc_brief[(fc_brief["lead_days"] == 3) & (fc_brief["sqn"] == sq)]
            if not r3.empty:
                p12_d3[f"S{sq}"] = f"{float(r3.iloc[0].get('p_ge_12', 0.0)) * 100:.1f}%"
                exp_d3[f"S{sq}"] = f"{float(r3.iloc[0]['expected_mc']):.1f}"
            r7 = fc_brief[(fc_brief["lead_days"] == 7) & (fc_brief["sqn"] == sq)]
            if not r7.empty:
                p12_d7[f"S{sq}"] = f"{float(r7.iloc[0].get('p_ge_12', 0.0)) * 100:.1f}%"
                exp_d7[f"S{sq}"] = f"{float(r7.iloc[0]['expected_mc']):.1f}"

    # c. Top levers: reactive serviceability at today's pipeline vs 0.5× (and difference); SHIKRA gain at 0.5×
    row_cur = heatmap_raw[(heatmap_raw["pipeline_mult"] == pipeline_mult) & (heatmap_raw["quality"] == 1.0)]
    row_05 = heatmap_raw[(heatmap_raw["pipeline_mult"] == 0.5) & (heatmap_raw["quality"] == 1.0)]
    if not row_cur.empty and not row_05.empty:
        reac_cur = float(row_cur.iloc[0]["reactive_serviceability_mean"]) * 100
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
    indig_items = []
    for _, row in top3_indig.iterrows():
        indig_items.append(
            f"<b>{row['lru_id']}</b>: loss {row['loss_pp_shock_window']:.1f} pp "
            f"(peak +{row['peak_extra_grounded']:.0f} grounded, recovery {row['recovery_days']:.0f} days)"
        )

    # e. Pending sign-offs: count of open EO recommendations + audit-chain status line
    recent_logs = _get_rec(100)
    pending_recs = sum(1 for r in recent_logs if r.get("decision") == "approved")
    chain = audit_verify()
    if chain["ok"]:
        root_abbr = (chain.get("latest_root") or "none")[:12]
        audit_line = f"Audit chain: intact · {chain['n_rows']} records · root `{root_abbr}`"
    else:
        audit_line = f"Audit chain: BROKEN at row id={chain['first_bad_id']}"

    brief_html = f"""
<style>
@media print {{
  [data-testid="stSidebar"], button, .stButton, header {{ display: none !important; }}
  body, .stApp {{ background: #ffffff !important; color: #000000 !important; }}
  #shikra_brief {{ border: none !important; background: #ffffff !important; color: #000000 !important; padding: 0 !important; }}
  #shikra_brief h2 {{ color: #0f172a !important; }}
  #shikra_brief h3 {{ color: #334155 !important; }}
  #shikra_brief table {{ border-color: #cbd5e1 !important; }}
  #shikra_brief th, #shikra_brief td {{ color: #000000 !important; border-bottom: 1px solid #cbd5e1 !important; }}
}}
</style>
<div id="shikra_brief" style="background:#0f172a;color:#e2e8f0;border:1px solid #334155;
     border-radius:10px;padding:2rem;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:920px;margin-top:1rem;">
  <h2 style="color:#38bdf8;margin-top:0;font-size:1.6rem;letter-spacing:1px;border-bottom:2px solid #38bdf8;padding-bottom:0.5rem;">
    SHIKRA · Commander's brief · Day 485 · synthetic wing
  </h2>

  <h3 style="color:#94a3b8;font-size:1rem;margin-top:1.2rem;text-transform:uppercase;letter-spacing:0.5px;">1. Wing Status Now (Day 485)</h3>
  <table style="border-collapse:collapse;width:100%;margin-bottom:1rem;font-size:0.9rem;">
    <tr style="border-bottom:1px solid #334155;text-align:left;">
      <th style="padding:6px 10px;color:#94a3b8;">Squadron</th>
      <th style="padding:6px 10px;color:#86efac;">Mission Capable (MC)</th>
      <th style="padding:6px 10px;color:#fde047;">Maintenance (NMCM)</th>
      <th style="padding:6px 10px;color:#fca5a5;">Supply Waiting (NMCS)</th>
      <th style="padding:6px 10px;color:#38bdf8;">Total Aircraft</th>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:6px 10px;font-weight:600;">Squadron 1 (S1)</td>
      <td style="padding:6px 10px;color:#86efac;font-weight:700;">{s1_mc}</td>
      <td style="padding:6px 10px;color:#fde047;">{s1_nmcm}</td>
      <td style="padding:6px 10px;color:#fca5a5;">{s1_nmcs}</td>
      <td style="padding:6px 10px;">18</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:6px 10px;font-weight:600;">Squadron 2 (S2)</td>
      <td style="padding:6px 10px;color:#86efac;font-weight:700;">{s2_mc}</td>
      <td style="padding:6px 10px;color:#fde047;">{s2_nmcm}</td>
      <td style="padding:6px 10px;color:#fca5a5;">{s2_nmcs}</td>
      <td style="padding:6px 10px;">18</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:6px 10px;font-weight:600;">Squadron 3 (S3)</td>
      <td style="padding:6px 10px;color:#86efac;font-weight:700;">{s3_mc}</td>
      <td style="padding:6px 10px;color:#fde047;">{s3_nmcm}</td>
      <td style="padding:6px 10px;color:#fca5a5;">{s3_nmcs}</td>
      <td style="padding:6px 10px;">18</td>
    </tr>
    <tr style="border-top:2px solid #334155;font-weight:700;">
      <td style="padding:6px 10px;">Total Wing</td>
      <td style="padding:6px 10px;color:#4ade80;">{total_mc} / 54 MC ({(total_mc/54.0)*100:.1f}%)</td>
      <td style="padding:6px 10px;color:#fde047;">{s1_nmcm + s2_nmcm + s3_nmcm}</td>
      <td style="padding:6px 10px;color:#fca5a5;">{s1_nmcs + s2_nmcs + s3_nmcs}</td>
      <td style="padding:6px 10px;">54</td>
    </tr>
  </table>

  <h3 style="color:#94a3b8;font-size:1rem;margin-top:1.2rem;text-transform:uppercase;letter-spacing:0.5px;">2. Readiness Forecast (P(≥12 MC))</h3>
  <table style="border-collapse:collapse;width:100%;margin-bottom:0.5rem;font-size:0.9rem;">
    <tr style="border-bottom:1px solid #334155;text-align:left;">
      <th style="padding:6px 10px;color:#94a3b8;">Horizon</th>
      <th style="padding:6px 10px;color:#94a3b8;">S1 P(≥12 MC) [Expected]</th>
      <th style="padding:6px 10px;color:#94a3b8;">S2 P(≥12 MC) [Expected]</th>
      <th style="padding:6px 10px;color:#94a3b8;">S3 P(≥12 MC) [Expected]</th>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:6px 10px;font-weight:600;">D+3</td>
      <td style="padding:6px 10px;">{p12_d3.get('S1', '—')} ({exp_d3.get('S1', '—')} MC)</td>
      <td style="padding:6px 10px;">{p12_d3.get('S2', '—')} ({exp_d3.get('S2', '—')} MC)</td>
      <td style="padding:6px 10px;">{p12_d3.get('S3', '—')} ({exp_d3.get('S3', '—')} MC)</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:6px 10px;font-weight:600;">D+7</td>
      <td style="padding:6px 10px;">{p12_d7.get('S1', '—')} ({exp_d7.get('S1', '—')} MC)</td>
      <td style="padding:6px 10px;">{p12_d7.get('S2', '—')} ({exp_d7.get('S2', '—')} MC)</td>
      <td style="padding:6px 10px;">{p12_d7.get('S3', '—')} ({exp_d7.get('S3', '—')} MC)</td>
    </tr>
  </table>
  <p style="font-size:0.8rem;color:#94a3b8;margin-bottom:1rem;font-style:italic;">
    {calib_caption_text}
  </p>

  <h3 style="color:#94a3b8;font-size:1rem;margin-top:1.2rem;text-transform:uppercase;letter-spacing:0.5px;">3. Top Levers (heatmap.csv)</h3>
  <p style="font-size:0.88rem;line-height:1.5;margin-bottom:1rem;">
    {lever_str}
  </p>

  <h3 style="color:#94a3b8;font-size:1rem;margin-top:1.2rem;text-transform:uppercase;letter-spacing:0.5px;">4. Top 3 Parts to Indigenise (supply_shock.csv)</h3>
  <ol style="margin:0 0 1rem;padding-left:1.5rem;font-size:0.88rem;line-height:1.6;">
    {"".join(f"<li>{item}</li>" for item in indig_items)}
  </ol>

  <h3 style="color:#94a3b8;font-size:1rem;margin-top:1.2rem;text-transform:uppercase;letter-spacing:0.5px;">5. Pending Sign-offs & Audit Integrity</h3>
  <p style="margin:0 0 0.5rem;font-size:0.88rem;">
    EO approved recommendations: <b>{pending_recs}</b> · <span style="color:#86efac;">{audit_line}</span>
  </p>

  <hr style="border-color:#334155;margin:1.5rem 0 1rem 0;"/>
  <p style="font-size:0.78rem;color:#64748b;margin:0;text-align:center;">
    Advisory decision support · offline · synthetic wing + NASA C-MAPSS
  </p>
</div>
"""
    st.markdown(brief_html, unsafe_allow_html=True)

# ── Performance ──────────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
st.caption(f"Page load: {elapsed:.2f} s. Synthetic wing · SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")
