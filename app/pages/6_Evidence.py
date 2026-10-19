"""SHIKRA — Evidence & Verification.

Renders GATES.md, policy summary table with CIs, RUL C-MAPSS evaluations,
key figures, and a reproducibility box.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

start_time = time.perf_counter()

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lib import theme  # noqa: E402
from app.lib.audit import verify as audit_verify  # noqa: E402
from app.lib.data import (  # noqa: E402
    get_result_figures,
    load_gates_md,
    load_policy_summary,
    load_replay_rul,
    load_rul_cmapss,
    load_rul_cmapss_multi,
)

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Evidence",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("evidence")
# This page reads result files only: no simulation day, no snapshot.
theme.status_strip(strip)
theme.header(
    "Evidence · reproducibility",
    "Can every number in this app be traced to a file and reproduced offline?",
    "Verification gates, simulation results, model benchmarks, and audit trail.",
    "results/GATES.md · results/policy_summary.csv · results/rul_cmapss*.csv · results/FINDINGS_*.md · "
    "results/*.png · artifacts/replay_rul.parquet · app_data audit log",
)

# Audit chain integrity line at top
_chain_top = audit_verify()
_fig_paths = get_result_figures()
_policy_df = load_policy_summary()
_n_findings = sum(1 for f in ("FINDINGS_A10.md", "FINDINGS_T14.md", "FINDINGS_CPSAT.md") if (ROOT / "results" / f).exists())
theme.kpi_row([
    theme.kpi(
        "Audit chain",
        "intact" if _chain_top["ok"] else "broken",
        tone="green" if _chain_top["ok"] else "red",
        sub=f"{_chain_top.get('n_rows', 0)} records · SHA-256 chain + daily HMAC roots",
        alert=not _chain_top["ok"],
    ),
    theme.kpi("Policy summary rows", f"{len(_policy_df)}", sub="results/policy_summary.csv · mean with 95% CI"),
    theme.kpi("Result figures", f"{len(_fig_paths)}", sub="PNG files in results/"),
    theme.kpi("Findings documents", f"{_n_findings}", sub="results/FINDINGS_*.md"),
])
if _chain_top["ok"]:
    _top_root = (_chain_top["latest_root"] or "none")[:12]
    st.markdown(f":green[Audit chain: intact · {_chain_top['n_rows']} records · root `{_top_root}`]")
else:
    st.markdown(f":red[Audit chain: BROKEN at row id={_chain_top['first_bad_id']}]")

# ── Findings Sections ───────────────────────────────────────────────────────
FINDINGS_FILES = ["FINDINGS_A10.md", "FINDINGS_T14.md", "FINDINGS_CPSAT.md"]
_any_findings = False
for _fname in FINDINGS_FILES:
    _fpath = ROOT / "results" / _fname
    if _fpath.exists():
        if not _any_findings:
            st.markdown("### Findings")
            _any_findings = True
        with st.expander(f"📋 {_fname}", expanded=False):
            st.markdown(_fpath.read_text(encoding="utf-8"))

st.divider()

# ── Reproducibility Box ──────────────────────────────────────────────────────
st.markdown("### Reproducibility Verification")

# Get git rev-parse --short HEAD
try:
    res = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    git_hash = res.stdout.strip() if res.returncode == 0 else "unknown"
except Exception:
    git_hash = "unknown"

replay_df = load_replay_rul()
model_id = str(replay_df["model_id"].iloc[0]) if "model_id" in replay_df.columns and not replay_df.empty else "unknown"
calib_id = str(replay_df["calib_id"].iloc[0]) if "calib_id" in replay_df.columns and not replay_df.empty else "unknown"

theme.html(
    f"""
    <div class="sk-panel">
      {theme.panel_title("Provenance & Environment", "identifiers of the code, model and calibration behind every page")}
      <div class="sk-kv"><span>Git Commit</span><b>{theme.esc(git_hash)}</b></div>
      <div class="sk-kv"><span>Model ID</span><b>{theme.esc(model_id)}</b></div>
      <div class="sk-kv"><span>Calibration ID</span><b>{theme.esc(calib_id)}</b></div>
      <div class="sk-label" style="font-size:10px;color:var(--outline);text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem 0;">Reproduction Commands</div>
      <div class="sk-mono sk-t-blue" style="background:var(--surface-container-lowest);border-radius:var(--rounded-lg);padding:.75rem;font-size:12px;line-height:20px;">
        <span class="sk-t-muted"># Run simulation reproduction (no raw NASA download needed):</span><br>make reproduce-sim<br><br>
        <span class="sk-t-muted"># Or run offline container:</span><br>docker compose up app
      </div>
    </div>
    """
)

st.divider()

# ── Verification Gates (GATES.md) ───────────────────────────────────────────
st.markdown("### Verification Gates")
st.caption("Verification gates status from results/GATES.md")
gates_text = load_gates_md()
with st.expander("View Full GATES.md", expanded=True):
    st.markdown(gates_text)

st.divider()

# ── Policy Simulation Table ──────────────────────────────────────────────────
st.markdown("### Fleet Simulation Policy Summary")
st.caption("30 replications on synthetic wing (54 aircraft, 730 days). Mean with 95% CI.")

policy_df = load_policy_summary()
if not policy_df.empty:
    # Filter and format for clear view
    p_display = policy_df.copy()
    if "metric" in p_display.columns:
        metrics = p_display["metric"].unique()
        selected_metric = st.selectbox("Filter by metric", ["all"] + list(metrics))
        if selected_metric != "all":
            p_display = p_display[p_display["metric"] == selected_metric]

    st.dataframe(p_display, use_container_width=True, hide_index=True)
else:
    st.warning("policy_summary.csv not found in results/")

st.divider()

# ── RUL Model Evaluations ────────────────────────────────────────────────────
st.markdown("### NASA C-MAPSS RUL Evaluations")
st.caption("LightGBM quantile ensemble (5 seeds) evaluated on official NASA C-MAPSS test sets.")

tab_single, tab_multi = st.tabs(["Single Condition (FD001, FD003)", "Multi Condition (FD002, FD004)"])

with tab_single:
    rul_single = load_rul_cmapss()
    if not rul_single.empty:
        st.dataframe(rul_single, use_container_width=True, hide_index=True)
    else:
        st.info("results/rul_cmapss.csv not available.")

with tab_multi:
    rul_multi = load_rul_cmapss_multi()
    if not rul_multi.empty:
        st.dataframe(rul_multi, use_container_width=True, hide_index=True)
    else:
        st.info("results/rul_cmapss_multi.csv not available.")

st.divider()

# ── Audit Trail Integrity ────────────────────────────────────────────────────
st.markdown("### Audit Trail")
st.caption("Sign-off log: SHA-256 hash chain plus HMAC-authenticated daily Merkle roots.")

if st.button("Verify chain"):
    st.caption("Chain re-verified just now.")
chain = audit_verify()
if chain["ok"]:
    today = datetime.now(UTC).date().isoformat()
    root_label = "today's root" if chain["latest_date"] == today else "latest root"
    root = (chain["latest_root"] or "none")[:12]
    st.markdown(
        f":green[Audit chain: intact · {chain['n_rows']} records · {root_label} `{root}`]"
    )
else:
    st.markdown(f":red[Audit chain: BROKEN at row id={chain['first_bad_id']}]")
    st.caption(chain["reason"])

st.divider()

# ── Key Figures ──────────────────────────────────────────────────────────────
st.markdown("### Key Figures")
st.caption("Generated figures from results/ directory.")

fig_paths = get_result_figures()
if fig_paths:
    cols = st.columns(2)
    for idx, fpath in enumerate(fig_paths):
        col = cols[idx % 2]
        with col:
            st.image(str(fpath), caption=fpath.name, use_column_width=True)
else:
    st.info("No PNG figures found in results/.")

# ── Performance ──────────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
st.caption(f"Page load: {elapsed:.2f} s. SHIKRA · advisory · offline · synthetic wing + NASA C-MAPSS")
theme.footer()
