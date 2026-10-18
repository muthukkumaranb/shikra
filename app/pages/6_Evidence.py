"""SHIKRA — Evidence & Verification.

Renders GATES.md, policy summary table with CIs, RUL C-MAPSS evaluations,
key figures, and a reproducibility box.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
st.title("Evidence & Reproducibility")
st.caption("Verification gates, simulation results, model benchmarks, and audit trail.")

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

st.markdown(
    f"""
    <div style="background: #0f172a; border: 1px solid #3b82f6; border-radius: 10px;
                padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; color: #e2e8f0;">
        <h4 style="margin-top:0; color: #60a5fa;">🔒 Provenance & Environment</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
            <div><strong>Git Commit:</strong> <code>{git_hash}</code></div>
            <div><strong>Model ID:</strong> <code>{model_id}</code></div>
            <div><strong>Calibration ID:</strong> <code>{calib_id}</code></div>
        </div>
        <div>
            <strong>Reproduction Commands:</strong>
            <pre style="background: #1e293b; padding: 0.8rem; border-radius: 6px; margin-top: 0.5rem; color: #38bdf8;">
# Run simulation reproduction (no raw NASA download needed):
make reproduce-sim

# Or run offline container:
docker compose up app
            </pre>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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
