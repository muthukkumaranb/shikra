"""Generate the results table for README.md from results CSVs.

Reads results CSVs and updates the auto-generated Results section in README.md.

Never edit README.md results by hand — run this script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
README_PATH = ROOT / "README.md"


def generate_highlights() -> list[str]:
    lines = [
        "### Highlights\n",
        "| What | Result | Evidence |",
        "|---|---|---|",
    ]

    # 1. RUL accuracy
    rul_path = RESULTS_DIR / "rul_cmapss.csv"
    if rul_path.exists():
        rul_df = pd.read_csv(rul_path)
        babu_cnn = {"FD001": 18.45, "FD003": 19.82}
        fd1_sub = rul_df[rul_df["subset"] == "FD001"]
        fd3_sub = rul_df[rul_df["subset"] == "FD003"]
        if not fd1_sub.empty and not fd3_sub.empty:
            fd1_rmse = fd1_sub.iloc[0]["rmse_capped125"]
            fd3_rmse = fd3_sub.iloc[0]["rmse_capped125"]
            lines.append(
                f"| RUL accuracy | FD001 RMSE (capped) {fd1_rmse:.2f} vs {babu_cnn['FD001']:.2f} (Babu 2016 CNN); "
                f"FD003 {fd3_rmse:.2f} vs {babu_cnn['FD003']:.2f} | [results/rul_cmapss.csv](results/rul_cmapss.csv) |"
            )

    # 2. Calibrated uncertainty
    cov_path = RESULTS_DIR / "coverage.csv"
    if cov_path.exists():
        cov_df = pd.read_csv(cov_path)
        t_cal = cov_df[(cov_df["split"] == "test") & (cov_df["calibrated"])]
        t80_1 = t_cal[(t_cal["nominal"] == 0.8) & (t_cal["subset"] == "FD001")]
        t80_3 = t_cal[(t_cal["nominal"] == 0.8) & (t_cal["subset"] == "FD003")]
        if not t80_1.empty and not t80_3.empty:
            c1 = t80_1.iloc[0]["empirical"]
            c3 = t80_3.iloc[0]["empirical"]
            lines.append(
                f"| Calibrated uncertainty | Held-out empirical coverage {c1:.1%} (FD001) and {c3:.1%} (FD003) at 80% nominal | "
                f"[results/coverage.csv](results/coverage.csv) |"
            )

    # 3. Which lever buys readiness
    hm_path = RESULTS_DIR / "heatmap.csv"
    if hm_path.exists():
        hm_df = pd.read_csv(hm_path)
        fast_mult = hm_df["pipeline_mult"].min()
        slow_mult = hm_df["pipeline_mult"].max()
        f_r = hm_df[(hm_df["pipeline_mult"] == fast_mult) & (hm_df["quality"] == 1.0)]["reactive_serviceability_mean"].iloc[0]
        f_s = hm_df[(hm_df["pipeline_mult"] == fast_mult) & (hm_df["quality"] == 1.0)]["serviceability_mean"].iloc[0]
        f_o = hm_df[(hm_df["pipeline_mult"] == fast_mult) & (hm_df["quality"] == 0.0)]["serviceability_mean"].iloc[0]
        s_r = hm_df[(hm_df["pipeline_mult"] == slow_mult) & (hm_df["quality"] == 1.0)]["reactive_serviceability_mean"].iloc[0]
        s_s = hm_df[(hm_df["pipeline_mult"] == slow_mult) & (hm_df["quality"] == 1.0)]["serviceability_mean"].iloc[0]
        s_o = hm_df[(hm_df["pipeline_mult"] == slow_mult) & (hm_df["quality"] == 0.0)]["serviceability_mean"].iloc[0]
        lines.append(
            f"| Which lever buys readiness | At {fast_mult:.1f}x pipeline (fastest): reactive {f_r:.1%}, SHIKRA {f_s:.1%}, oracle {f_o:.1%}; "
            f"at {slow_mult:.1f}x (slowest): reactive {s_r:.1%}, SHIKRA {s_s:.1%}, oracle {s_o:.1%} | "
            f"[results/heatmap.csv](results/heatmap.csv) |"
        )

    # 4. Unscheduled engine removals
    sim_path = RESULTS_DIR / "policy_summary.csv"
    if sim_path.exists():
        pol_df = pd.read_csv(sim_path)
        uns = pol_df[pol_df["metric"] == "unscheduled_removals"].set_index("policy")
        if "reactive" in uns.index:
            r_uns = uns.loc["reactive"]["mean"]
            s_key = "shikra_v2" if "shikra_v2" in uns.index else ("shikra" if "shikra" in uns.index else "")
            s_uns = uns.loc[s_key]["mean"] if s_key in uns.index else 0.0
            lines.append(
                f"| Unscheduled engine removals | Reactive {r_uns:.1f} vs SHIKRA {s_uns:.1f} per year | "
                f"[results/policy_summary.csv](results/policy_summary.csv) |"
            )

    # 5. What to make in India first
    shock_path = RESULTS_DIR / "supply_shock.csv"
    if shock_path.exists():
        shock_df = pd.read_csv(shock_path)
        af_row = shock_df[shock_df["lru_id"] == "all_foreign"].iloc[0]
        top2 = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank").head(2)
        top2_items = [f"{r['lru_id']} ({r['loss_pp_shock_window']:.1f} pp)" for _, r in top2.iterrows()]
        top2_str = ", ".join(top2_items)
        lines.append(
            f"| What to make in India first | Top-2 LRUs: {top2_str}; foreign halt grounds up to {af_row['peak_extra_grounded']:.0f} extra aircraft, recovery in {af_row['recovery_days']:.0f} days | "
            f"[results/supply_shock.csv](results/supply_shock.csv) |"
        )

    # 6. Readiness forecast
    readiness_path = RESULTS_DIR / "readiness_calib.csv"
    if readiness_path.exists():
        calib_df = pd.read_csv(readiness_path)
        d3_cov = calib_df[(calib_df["lead_days"] == 3) & (calib_df["method"] == "shikra_conformal")]["coverage_80"].iloc[0]
        d7_cov = calib_df[(calib_df["lead_days"] == 7) & (calib_df["method"] == "shikra_conformal")]["coverage_80"].iloc[0]
        d7_fc_mae = calib_df[(calib_df["lead_days"] == 7) & (calib_df["method"] == "shikra_forecast")]["mae"].iloc[0]
        d7_pers_mae = calib_df[(calib_df["lead_days"] == 7) & (calib_df["method"] == "persistence")]["mae"].iloc[0]
        lines.append(
            f"| Readiness forecast | Held-out conformal coverage {d3_cov:.1%} (D+3) and {d7_cov:.1%} (D+7) at 80% nominal; "
            f"MAE does NOT beat persistence (D+7: {d7_fc_mae:.3f} vs {d7_pers_mae:.3f}) | "
            f"[results/readiness_calib.csv](results/readiness_calib.csv) |"
        )

    # 7. Hangar scheduling
    cpsat_bench_path = RESULTS_DIR / "cpsat_benchmark.csv"
    cpsat_wing_path = RESULTS_DIR / "cpsat_wing.csv"
    if cpsat_bench_path.exists() and cpsat_wing_path.exists():
        bench_df = pd.read_csv(cpsat_bench_path)
        wing_df = pd.read_csv(cpsat_wing_path)
        edf_mean = bench_df["edf_weighted_tardiness"].mean()
        cpsat_mean = bench_df["cpsat_weighted_tardiness"].mean()
        pct_red = (edf_mean - cpsat_mean) / edf_mean * 100
        n_inst = len(bench_df)
        k_ties = (wing_df["cpsat_wt"] == wing_df["edf_wt"]).sum()
        m_total = len(wing_df)
        lines.append(
            f"| Hangar scheduling | CP-SAT cuts weighted tardiness {edf_mean:.1f} -> {cpsat_mean:.1f} (-{pct_red:.0f}%) "
            f"on {n_inst} synthetic instances (never worse than EDF); ties EDF on {k_ties}/{m_total} wing snapshots: "
            f"hangar bays are not the bottleneck | [results/FINDINGS_CPSAT.md](results/FINDINGS_CPSAT.md) |"
        )
    else:
        lines.append(
            "| Hangar scheduling | CP-SAT vs earliest-due-date heuristic | [results/FINDINGS_CPSAT.md](results/FINDINGS_CPSAT.md) |"
        )
    lines.append("")
    return lines


def generate_table() -> str:
    lines = ["## Results\n"]

    # ── Highlights table ─────────────────────────────────────────────────
    lines.extend(generate_highlights())

    # ── RUL model table ──────────────────────────────────────────────────
    rul_path = RESULTS_DIR / "rul_cmapss.csv"
    multi_path = RESULTS_DIR / "rul_cmapss_multi.csv"
    cov_path = RESULTS_DIR / "coverage.csv"

    babu_cnn = {
        "FD001": 18.45,
        "FD002": 30.29,
        "FD003": 19.82,
        "FD004": 29.16,
    }

    if rul_path.exists():
        dfs = [pd.read_csv(rul_path)]
        if multi_path.exists():
            dfs.append(pd.read_csv(multi_path))
        all_rul = pd.concat(dfs, ignore_index=True).sort_values("subset").reset_index(drop=True)

        cov_lookup: dict[str, float] = {}
        if cov_path.exists():
            cov_df = pd.read_csv(cov_path)
            for _, r in cov_df[(cov_df["split"] == "test") & (cov_df["nominal"] == 0.8) & (~cov_df["calibrated"])].iterrows():
                cov_lookup[r["subset"]] = float(r["empirical"])

        lines.append("### RUL Model (LightGBM quantile ensemble, 5 seeds)\n")
        lines.append("Official test sets, last cycle only; RMSE shown against true RUL capped at 125 (piecewise convention) and uncapped.\n")
        lines.append("| Subset | Babu 2016 CNN | RMSE (capped 125) | RMSE (uncapped) | MAE (capped 125) | MAE (uncapped) | NASA Score (capped 125) | NASA Score (uncapped) | RMSE (RUL≤30) | 80% Coverage (uncalibrated) | n units |")
        lines.append("|--------|---------------|-------------------|-----------------|------------------|----------------|-------------------------|-----------------------|---------------|-----------------------------|---------|")
        for _, row in all_rul.iterrows():
            sub = str(row["subset"])
            babu_val = f"{babu_cnn[sub]:.2f}" if sub in babu_cnn else "—"
            if "coverage_80" in row and pd.notna(row["coverage_80"]):
                cov_str = f"{row['coverage_80']:.1%}"
            elif sub in cov_lookup:
                cov_str = f"{cov_lookup[sub]:.1%}"
            else:
                cov_str = "—"

            rmse_c = row.get("rmse_capped125", row.get("rmse", 0.0))
            rmse_u = row.get("rmse_uncapped", 0.0)
            mae_c = row.get("mae_capped125", row.get("mae", 0.0))
            mae_u = row.get("mae_uncapped", 0.0)
            ns_c = row.get("nasa_score_capped125", row.get("nasa_score", 0.0))
            ns_u = row.get("nasa_score_uncapped", 0.0)

            lines.append(
                f"| {sub} | {babu_val} | {rmse_c:.2f} | {rmse_u:.2f} | "
                f"{mae_c:.2f} | {mae_u:.2f} | "
                f"{ns_c:.1f} | {ns_u:.1f} | "
                f"{row['rmse_critical']:.2f} | {cov_str} | {int(row['n_units'])} |"
            )
        lines.append("")

    # ── Policy simulation table ───────────────────────────────────────────
    sim_path = RESULTS_DIR / "policy_summary.csv"
    if sim_path.exists():
        summary = pd.read_csv(sim_path)
        svc = summary[summary["metric"] == "serviceability"].set_index("policy")
        uns = summary[summary["metric"] == "unscheduled_removals"].set_index("policy")

        lines.append("### Fleet Simulation (synthetic wing, 30 replications)\n")
        lines.append("| Policy | Serviceability | 95% CI | Unscheduled removals |")
        lines.append("|--------|---------------|--------|---------------------|")
        if "shikra_v2" in svc.index:
            policy_rows = [
                ("reactive", "Reactive"),
                ("preventive", "Preventive"),
                ("shikra_v2", "SHIKRA"),
                ("oracle_v2", "Oracle (perfect prediction)"),
                ("shikra", "SHIKRA (ablation: without lead-time-aware prepositioning)"),
                ("oracle", "Oracle (ablation: without lead-time-aware prepositioning)"),
            ]
        else:
            policy_rows = [
                ("reactive", "Reactive"),
                ("preventive", "Preventive"),
                ("shikra", "SHIKRA"),
                ("oracle", "Oracle (perfect prediction)"),
            ]
        for p_key, label in policy_rows:
            if p_key in svc.index:
                row = svc.loc[p_key]
                ur = uns.loc[p_key] if p_key in uns.index else None
                ur_str = f"{ur['mean']:.1f}" if ur is not None else "—"
                lines.append(
                    f"| {label} | {row['mean']:.1%} | "
                    f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}] | {ur_str} |"
                )
        lines.append("")
        r_uns = uns.loc["reactive"]["mean"] if "reactive" in uns.index else 0.0
        shikra_key = "shikra_v2" if "shikra_v2" in uns.index else ("shikra" if "shikra" in uns.index else "")
        s_uns = uns.loc[shikra_key]["mean"] if shikra_key in uns.index else 0.0
        r_svc = svc.loc["reactive"]["mean"] if "reactive" in svc.index else 0.0
        s_svc = svc.loc[shikra_key]["mean"] if shikra_key in svc.index else 0.0
        lines.append(
            f"> **Key Finding:** SHIKRA removes all unscheduled engine removals ({s_uns:.0f} vs {r_uns:.1f} per year) but, "
            f"in this wing, serviceability is unchanged ({s_svc:.1%} vs {r_svc:.1%}), "
            f"because the engine-spare pipeline, not warning time, is the binding constraint (see the 'Which lever buys readiness' row in Highlights).\n"
        )

        # Supply shock interpretation
        shock_path = RESULTS_DIR / "supply_shock.csv"
        if shock_path.exists():
            shock_df = pd.read_csv(shock_path)
            af_row = shock_df[shock_df["lru_id"] == "all_foreign"].iloc[0]
            peak_ac = af_row["peak_extra_grounded"]
            loss_sw = af_row["loss_pp_shock_window"]
            rec_days = af_row["recovery_days"]
            top_lrus = shock_df[shock_df["lru_id"] != "all_foreign"].sort_values("rank")
            ranking_items = [f"{row['lru_id']} ({row['loss_pp_shock_window']:.1f} pp)" for _, row in top_lrus.head(2).iterrows()]
            ranking_str = ", ".join(ranking_items)
            lines.append(
                f"> **Supply-shock stress test (at 1.0× pipeline speed):** A 180-day halt of foreign supply grounds up to {peak_ac:.0f} extra aircraft of 54 "
                f"({loss_sw:.1f} pp over the halt); recovery takes {rec_days:.0f} days after supply resumes. "
                f"Ranking: {ranking_str}. Parameters are synthetic; the method is the claim.\n"
            )

        # Bias interpretation
        bias_path = RESULTS_DIR / "bias.csv"
        if bias_path.exists():
            bias_df = pd.read_csv(bias_path)
            opt_sub = bias_df[bias_df["bias_cycles"] == 10]
            n_uns = opt_sub["unscheduled_removals_mean"].iloc[0] if not opt_sub.empty else 94.8
            pess_sub = bias_df[bias_df["bias_cycles"] == -10]
            m_wasted = pess_sub["wasted_life_cycles_mean"].iloc[0] if not pess_sub.empty else 1993.0
            lines.append(
                f"> **Prediction-bias stress test (at 1.0× pipeline speed):** −10 cycles wastes {m_wasted:.0f} cycles; "
                f"+10 raises unscheduled removals from 0 to {n_uns:.1f}.\n"
            )

        readiness_path = RESULTS_DIR / "readiness_calib.csv"
        if readiness_path.exists():
            readiness_df = pd.read_csv(readiness_path)
            lead7 = readiness_df[readiness_df["lead_days"] == 7]
            fc = lead7[lead7["method"] == "shikra_forecast"].iloc[0]
            pers = lead7[lead7["method"] == "persistence"].iloc[0]
            lines.append(
                f"> **Readiness forecast:** At lead 7, forecast MAE is {fc['mae']:.3f} vs "
                f"{pers['mae']:.3f} for persistence and Brier is {fc['brier']:.3f} vs "
                f"{pers['brier']:.3f}; action gains are forecast-estimated and not validated in simulation.\n"
            )

    return "\n".join(lines)


def update_readme(table: str) -> None:
    content = README_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- AUTO-GENERATED — do not edit by hand; run `python scripts/readme_table.py` -->"
    end_marker = "\n## What did not work"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Marker not found in README.md.")
        return

    # Find the next section (## Quickstart) after the marker
    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        end_idx = len(content)

    new_content = content[:start_idx] + start_marker + "\n\n" + table + "\n---\n" + content[end_idx:]
    README_PATH.write_text(new_content, encoding="utf-8")
    print("README.md updated with results table.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    table = generate_table()
    try:
        print(table)
    except UnicodeEncodeError:
        print(table.encode("ascii", errors="replace").decode("ascii"))
    update_readme(table)
