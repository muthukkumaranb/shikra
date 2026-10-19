"""SHIKRA — Logistics View.

Stock view (engines and 10 LRUs), supply-shock exposure per LRU,
and VARI-METRIC allocation panel.
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
from app.lib.data import (  # noqa: E402
    load_config,
    load_spares_alloc,
    load_spares_compare,
    load_supply_shock,
)

ICON_PATH = ROOT / "app" / "assets" / "shikra.png"
st.set_page_config(
    page_title="SHIKRA — Logistics",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)
strip = theme.apply("logistics")
# This page reads configuration and result files only: no simulation day, no snapshot.
theme.status_strip(strip)

config = load_config()
shock_df = load_supply_shock()
spares_alloc = load_spares_alloc()
spares_compare = load_spares_compare()
pipeline_mult = config.get("pipeline_mult", 1.7)

theme.header(
    "Logistics · stock & allocation",
    "Where are the spares, and would placing them differently buy readiness for the same money?",
    "Base and depot stock for engines and the 10 tracked parts, each part's exposure to a 180-day "
    "foreign-supply halt, and the VARI-METRIC re-allocation at the same budget. "
    "Synthetic wing · advisory — all recommendations require officer sign-off.",
    "config/wing.yaml · results/supply_shock.csv · results/spares_alloc.csv · results/spares_compare.csv",
)

# Engine stock summary
eng_cfg = config.get("spares", {}).get("engines", {})
eng_base = eng_cfg.get("base_per_sqn", 1)
eng_depot = eng_cfg.get("depot", 3)
eng_d2b = eng_cfg.get("depot_to_base_days", 3)

lrus_cfg = config.get("lrus", [])
stock_rows = []
for lru in lrus_cfg:
    s_base = lru.get("stock_base", 0)
    s_depot = lru.get("stock_depot", 0)
    stock_rows.append({
        "LRU ID": lru.get("id", ""),
        "Description": lru.get("description", lru.get("id", "")),
        "Source": lru.get("source", "domestic").upper(),
        "Cost (₹ Lakh)": lru.get("cost_lakh", 0),
        "Repair TAT (days)": lru.get("repair_tat_days", 0),
        "Lead Time (days)": lru.get("lead_time_days", 0),
        "Base Stock (per Sqn)": s_base,
        "Depot Stock": s_depot,
        "Total Wing Stock": s_base * 3 + s_depot,
    })
stock_table = pd.DataFrame(stock_rows)
lru_shock = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")
loss_by_lru = dict(zip(lru_shock["lru_id"], lru_shock["loss_pp_shock_window"]))

cmp_shikra = (
    spares_compare[spares_compare["policy"] == "shikra_v2"].copy()
    if spares_compare is not None and not spares_compare.empty
    else None
)

# ── KPI row ─────────────────────────────────────────────────────────────────
n_foreign = int((stock_table["Source"] == "FOREIGN").sum()) if not stock_table.empty else 0
tiles = [
    theme.kpi(
        "Tracked part stock (wing)",
        f"{int(stock_table['Total Wing Stock'].sum()) if not stock_table.empty else 0}",
        unit="units",
        sub=f"{len(stock_table)} part types · base ×3 squadrons + depot",
    ),
    theme.kpi(
        "Foreign-sourced parts",
        f"{n_foreign:02d}",
        unit=f"/ {len(stock_table)}",
        tone="amber",
        badge=theme.chip("supply risk", "amber"),
        sub="exposed to a foreign-supply halt",
    ),
]
if cmp_shikra is not None:
    vm_cur = cmp_shikra[(cmp_shikra["pipeline_mult"] == pipeline_mult) & (cmp_shikra["stock"] == "vari_metric")]
    if not vm_cur.empty:
        v = vm_cur.iloc[0]
        tiles.append(
            theme.kpi(
                "Same-budget re-allocation",
                f"{float(v['diff_vs_current_pp']):+.2f}",
                unit="pp",
                tone="green",
                badge=theme.chip(f"{pipeline_mult}× pipeline", "blue"),
                sub=f"95% CI [{float(v['diff_ci_lo']):+.2f}, {float(v['diff_ci_hi']):+.2f}] · n={int(v['n_reps'])}",
            )
        )
tiles.append(
    theme.kpi(
        "Spare engines",
        f"{eng_base * 3 + eng_depot}",
        unit="units",
        sub=f"{eng_base}/squadron base · {eng_depot} depot · {eng_d2b} d depot→base",
    )
)
theme.kpi_row(tiles)

col_main, col_rail = st.columns([2, 1], gap="medium")

# ═════════════════════════════════ MAIN ══════════════════════════════════════
with col_main:
    st.markdown("### Stock View")
    st.caption(
        "Base stock per squadron and depot stock for engines and LRUs. "
        "Values from wing.yaml configuration. Synthetic wing."
    )

    rows_html = ""
    for _, r in stock_table.iterrows():
        foreign = r["Source"] == "FOREIGN"
        loss = loss_by_lru.get(r["LRU ID"])
        if not foreign:
            posture = theme.chip("domestic", "green", led=True)
        elif loss is not None and loss >= 5:
            posture = theme.chip(f"high · −{loss:.1f} pp", "red", led=True)
        elif loss is not None and loss >= 2:
            posture = theme.chip(f"watch · −{loss:.1f} pp", "amber", led=True)
        else:
            posture = theme.chip(f"low · −{loss:.1f} pp" if loss is not None else "foreign", "blue", led=True)
        b = int(r["Base Stock (per Sqn)"])
        rows_html += (
            f'<tr class="{"sk-hot" if foreign and (loss or 0) >= 5 else ""}"><td><b>{theme.esc(r["Description"])}</b>'
            f'<div class="sk-mono" style="font-size:10px;color:var(--outline);">{theme.esc(r["LRU ID"])}</div></td>'
            f'<td>{theme.chip(r["Source"], "red" if foreign else "green")}</td>'
            f'<td class="sk-mono">{b}</td><td class="sk-mono">{b}</td><td class="sk-mono">{b}</td>'
            f'<td class="sk-mono">{int(r["Depot Stock"])}</td><td class="sk-mono"><b>{int(r["Total Wing Stock"])}</b></td>'
            f'<td class="sk-mono">{int(r["Repair TAT (days)"])} d / {int(r["Lead Time (days)"])} d</td>'
            f'<td class="sk-mono">₹{r["Cost (₹ Lakh)"]:g} L</td><td>{posture}</td></tr>'
        )
    theme.html(
        '<div class="sk-panel">'
        + theme.panel_title(
            "Part stock & base allocation matrix",
            "S1 / S2 / S3 base stock vs depot · posture = loss under a 180-day foreign-supply halt",
        )
        + '<table class="sk-table"><tr><th>Part</th><th>Origin</th><th>S1</th><th>S2</th><th>S3</th>'
        "<th>Depot</th><th>Total</th><th>Repair / lead</th><th>Unit cost</th><th>Supply posture</th></tr>"
        + rows_html
        + "</table></div>"
    )
    with st.expander("LRU Inventory & Characteristics (table)"):
        st.dataframe(stock_table, use_container_width=True, hide_index=True)

# ═════════════════════════════════ RAIL ══════════════════════════════════════
with col_rail:
    st.markdown("### Supply-Shock Exposure per LRU")
    st.caption(
        "180-day foreign-supply halt, at 1.0× pipeline speed, synthetic parameters; "
        "the method is the claim."
    )
    max_loss = float(lru_shock["loss_pp_shock_window"].max()) if not lru_shock.empty else 1.0
    items = ""
    for pos, (_, r) in enumerate(lru_shock.iterrows(), start=1):
        loss = float(r["loss_pp_shock_window"])
        pct = loss / (max_loss or 1.0) * 100
        color = theme.SUPPLY_COLOR if pct >= 60 else (theme.MAINT_COLOR if pct >= 20 else "#38bdf8")
        items += (
            f'<div class="sk-rank"><div class="sk-rt"><div><span class="sk-rn">#{pos}</span>'
            f'<span class="sk-rname">{theme.esc(r["lru_id"])}</span></div>'
            f'<div class="sk-rv"><b style="color:{color};font-size:15px;">−{loss:.1f} pp</b>'
            f'<div>[{r["ci_lo"]:.1f}, {r["ci_hi"]:.1f}]</div></div></div>'
            f'<div class="sk-hbar" style="height:6px;"><div style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'<div class="sk-rmeta"><span>failures in halt {r["failures_in_shock_window_mean"]:.1f}</span>'
            f'<span>stock {int(r["stock_base_total"])}+{int(r["stock_depot"])}</span>'
            f'<span>recovery {r["recovery_days"]:.0f} d</span></div></div>'
        )
    theme.html('<div class="sk-panel">' + theme.panel_title("Foreign-part exposure", "loss (pp) · 95% CI") + items + "</div>")

    display_shock = lru_shock[[
        "rank", "lru_id", "source", "loss_pp_shock_window", "peak_extra_grounded",
        "recovery_days", "failures_in_shock_window_mean", "stock_base_total", "stock_depot",
    ]].copy()
    display_shock.columns = [
        "Rank", "LRU", "Source", "Loss (pp)", "Peak Extra Grounded",
        "Recovery Days", "Failures in Shock", "Total Base Stock", "Depot Stock",
    ]
    with st.expander("Supply-shock metrics (table)"):
        st.dataframe(display_shock, use_container_width=True, hide_index=True)

# ═════════════════════════ VARI-METRIC ALLOCATION ════════════════════════════
st.markdown("### VARI-METRIC Spares Allocation")

if spares_alloc is not None and not spares_alloc.empty and cmp_shikra is not None:
    available_mults = sorted(float(m) for m in spares_alloc["pipeline_mult"].unique())
    sel_mult = st.radio(
        "Pipeline Resupply Speed Multiplier",
        options=available_mults,
        index=available_mults.index(1.0) if 1.0 in available_mults else 0,
        format_func=lambda x: f"{x}× speed ({'Baseline' if x == 1.0 else ('Fast' if x < 1.0 else 'Strained')})",
        horizontal=True,
    )
    alloc_sub = spares_alloc[spares_alloc["pipeline_mult"] == sel_mult].copy()
    sub_curr = cmp_shikra[(cmp_shikra["pipeline_mult"] == sel_mult) & (cmp_shikra["stock"] == "current")].iloc[0]
    sub_vm = cmp_shikra[(cmp_shikra["pipeline_mult"] == sel_mult) & (cmp_shikra["stock"] == "vari_metric")].iloc[0]
    curr_serv = float(sub_curr["serviceability_mean"]) * 100
    vm_serv = float(sub_vm["serviceability_mean"]) * 100

    # Per-part current vs VARI-METRIC rows with paired bars
    max_units = float(
        max(alloc_sub[["current_base_per_sqn", "vm_base_per_sqn"]].max().max() * 3
            + alloc_sub[["current_depot", "vm_depot"]].max().max(), 1)
    )
    part_rows = ""
    for _, r in alloc_sub.sort_values("ebo_current", ascending=False).iterrows():
        cur_tot = int(r["current_base_per_sqn"]) * 3 + int(r["current_depot"])
        vm_tot = int(r["vm_base_per_sqn"]) * 3 + int(r["vm_depot"])
        moved = []
        if int(r["vm_base_per_sqn"]) != int(r["current_base_per_sqn"]):
            moved.append(f"base {int(r['current_base_per_sqn'])}→{int(r['vm_base_per_sqn'])} per sqn")
        if int(r["vm_depot"]) != int(r["current_depot"]):
            moved.append(f"depot {int(r['current_depot'])}→{int(r['vm_depot'])}")
        part_rows += (
            '<div style="margin-bottom:.7rem;">'
            f'<div class="sk-kv" style="font-size:12px;"><span style="color:var(--on-surface);font-weight:700;">'
            f'{theme.esc(r["lru_id"])} {theme.chip(str(r["source"]).upper(), "red" if r["source"] == "foreign" else "green")}</span>'
            f'<b class="sk-t-blue">{theme.esc("; ".join(moved) if moved else "unchanged")}</b></div>'
            f'<div class="sk-kv"><span>current {cur_tot} units · expected backorders {float(r["ebo_current"]):.3f}</span></div>'
            f'<div class="sk-hbar" style="height:6px;margin-top:2px;"><div style="width:{cur_tot / max_units * 100:.1f}%;background:#87929a;"></div></div>'
            f'<div class="sk-kv"><span>VARI-METRIC {vm_tot} units · expected backorders {float(r["ebo_vm"]):.3f}</span></div>'
            f'<div class="sk-hbar" style="height:6px;margin-top:2px;"><div style="width:{vm_tot / max_units * 100:.1f}%;background:var(--secondary);"></div></div>'
            "</div>"
        )
    budget_lakh = float(sub_vm["budget_lakh"])
    gain_txt = (
        f"{float(sub_vm['diff_vs_current_pp']):+.2f} pp · 95% CI "
        f"[{float(sub_vm['diff_ci_lo']):+.2f}, {float(sub_vm['diff_ci_hi']):+.2f}]"
    )
    summary = (
        '<div style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap;margin-bottom:1rem;">'
        f'<div><div class="sk-label" style="font-size:10px;color:var(--outline);letter-spacing:.1em;text-transform:uppercase;">Current stock</div>'
        f'<div class="sk-mono" style="font-size:28px;font-weight:700;">{curr_serv:.1f}%</div></div>'
        '<div class="sk-mono sk-t-blue" style="font-size:22px;">→</div>'
        f'<div><div class="sk-label" style="font-size:10px;color:var(--secondary);letter-spacing:.1em;text-transform:uppercase;">VARI-METRIC</div>'
        f'<div class="sk-mono sk-t-green" style="font-size:28px;font-weight:700;">{vm_serv:.1f}%</div></div>'
        f'<div style="margin-left:auto;">{theme.chip(gain_txt, "green" if float(sub_vm["diff_ci_lo"]) > 0 else "blue")}'
        f'<div class="sk-mono" style="font-size:11px;color:var(--outline);margin-top:.3rem;text-align:right;">same budget ₹{budget_lakh:,.0f} Lakh · shikra_v2 · n={int(sub_vm["n_reps"])}</div></div></div>'
    )
    theme.html(
        '<div class="sk-panel">'
        + theme.panel_title(
            "Same budget, re-allocated stock",
            f"VARI-METRIC multi-echelon allocation at {sel_mult}× pipeline · serviceability of the synthetic wing",
        )
        + summary
        + part_rows
        + "</div>"
    )

    # Result cards across each pipeline speed
    st.markdown("#### Serviceability & NMCS Bottleneck Comparison")
    cards = ""
    for mult in available_mults:
        curr_row = cmp_shikra[(cmp_shikra["pipeline_mult"] == mult) & (cmp_shikra["stock"] == "current")].iloc[0]
        vm_row = cmp_shikra[(cmp_shikra["pipeline_mult"] == mult) & (cmp_shikra["stock"] == "vari_metric")].iloc[0]
        active = mult == sel_mult
        cards += (
            f'<div class="sk-kpi" style="{"box-shadow:inset 0 0 0 1px var(--primary-container);" if active else ""}">'
            f'<div class="sk-kl"><span class="sk-label">{mult}× pipeline speed</span>{theme.chip("selected", "blue") if active else ""}</div>'
            f'<div><div class="sk-num sk-t-{"green" if float(vm_row["diff_ci_lo"]) > 0 else "blue"}">{float(vm_row["diff_vs_current_pp"]):+.2f}<span class="sk-unit">pp</span></div></div>'
            f'<div class="sk-sub">95% CI [{float(vm_row["diff_ci_lo"]):+.2f}, {float(vm_row["diff_ci_hi"]):+.2f}] pp<br>'
            f'Rate: {float(curr_row["serviceability_mean"]) * 100:.1f}% → {float(vm_row["serviceability_mean"]) * 100:.1f}%<br>'
            f'NMCS tails · LRU {float(vm_row["nmcs_lru_sim"]):.2f} (was {float(curr_row["nmcs_lru_sim"]):.2f}) · '
            f'engine {float(vm_row["nmcs_engine_sim"]):.2f} (was {float(curr_row["nmcs_engine_sim"]):.2f}) · '
            f'total {float(vm_row["nmcs_mean"]):.2f}</div></div>'
        )
    theme.html(f'<div class="sk-kpis">{cards}</div>')

    eng_share = float(sub_vm["nmcs_engine_sim"] / sub_vm["nmcs_mean"] * 100.0)
    lru_diff = float(sub_curr["nmcs_lru_sim"] - sub_vm["nmcs_lru_sim"])
    eng_diff = float(sub_vm["nmcs_engine_sim"] - sub_curr["nmcs_engine_sim"])
    st.caption(
        f"same budget (₹{budget_lakh:,.0f} Lakh); synthetic wing; the engine queue is the binding constraint "
        f"({eng_share:.1f}% of NMCS is engine-caused at {sel_mult}× pipeline: {float(sub_vm['nmcs_engine_sim']):.2f} engine vs "
        f"{float(sub_vm['nmcs_lru_sim']):.2f} LRU tails; LRU queue cut by {lru_diff:.2f} shifts bottleneck to engines by +{eng_diff:.2f})."
    )
else:
    theme.html('<div class="sk-panel"><p class="sk-note">Spares optimiser: results/spares_alloc.csv not found.</p></div>')

theme.footer()
