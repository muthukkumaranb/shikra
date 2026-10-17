"""Generate the results table for README.md from results CSVs.

Reads results/policy_summary.csv and results/rul_cmapss.csv and prints
a markdown table that can be pasted into README.md.

Never edit README.md by hand — run this script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
README_PATH = ROOT / "README.md"


def generate_table() -> str:
    lines = ["## Results\n"]

    # ── RUL model table ──────────────────────────────────────────────────
    rul_path = RESULTS_DIR / "rul_cmapss.csv"
    if rul_path.exists():
        rul = pd.read_csv(rul_path)
        lines.append("### RUL Model (LightGBM quantile ensemble, 5 seeds)\n")
        lines.append("| Subset | RMSE | MAE | NASA Score | RMSE (RUL≤30) | n units |")
        lines.append("|--------|------|-----|-----------|---------------|---------|")
        for _, row in rul.iterrows():
            lines.append(
                f"| {row['subset']} | {row['rmse']:.2f} | {row['mae']:.2f} | "
                f"{row['nasa_score']:.1f} | {row['rmse_critical']:.2f} | {int(row['n_units'])} |"
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
        for p in ["reactive", "preventive", "shikra", "oracle", "shikra_v2", "oracle_v2"]:
            if p in svc.index:
                row = svc.loc[p]
                ur = uns.loc[p] if p in uns.index else None
                ur_str = f"{ur['mean']:.1f}" if ur is not None else "—"
                lines.append(
                    f"| {p.capitalize()} | {row['mean']:.1%} | "
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
            f"because the engine-spare pipeline, not warning time, is the binding constraint (see A11).\n"
        )

    return "\n".join(lines)


def update_readme(table: str) -> None:
    content = README_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- AUTO-GENERATED — do not edit by hand; run `python scripts/readme_table.py` -->"
    end_marker = "\n## Quickstart"

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
