"""Convert headline SHIKRA results into "aircraft per day" equivalents.

aircraft_equivalent = (percentage points / 100) x N, where N = squadrons x
aircraft_per_sqn from config/wing.yaml: the average number of aircraft per day
the effect adds (or removes) over the measured window. Every input is read from
results/*.csv and config/wing.yaml at run time.

Writes results/aircraft_equivalents.csv. Run via ``make aircraft-eq``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "aircraft_equivalents.csv"
DAYS_PER_YEAR = 365
TOP_SHOCKS = 3
FIELDS = ["item", "pp_or_value", "aircraft_equivalent", "source_csv", "note"]


def _wing() -> dict:
    with open(ROOT / "config" / "wing.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def n_aircraft(wing: dict) -> int:
    return int(wing["fleet"]["squadrons"]) * int(wing["fleet"]["aircraft_per_sqn"])


def rows() -> list[dict]:
    wing = _wing()
    n = n_aircraft(wing)
    today = float(wing["pipeline_mult"])
    measured_days = int(wing["ops"]["days"]) - int(wing["warmup_days"])
    year_note = (
        "synthetic wing; average over the measured year"
        if measured_days == DAYS_PER_YEAR
        else f"synthetic wing; average over the {measured_days}-day measured window"
    )

    hm = pd.read_csv(RESULTS_DIR / "heatmap.csv")
    fast = float(hm["pipeline_mult"].min())
    reactive = hm.drop_duplicates("pipeline_mult").set_index("pipeline_mult")[
        "reactive_serviceability_mean"
    ]
    labels = hm.drop_duplicates("quality").set_index("quality")["quality_label"]
    q_shikra = next(q for q, lab in labels.items() if "SHIKRA" in lab)
    shikra_fast = hm[(hm["pipeline_mult"] == fast) & (hm["quality"] == q_shikra)].iloc[0]

    out: list[dict] = []

    def add(item: str, pp: float, source: str, note: str, convert: bool = True) -> None:
        out.append({
            "item": item,
            "pp_or_value": round(pp, 3),
            "aircraft_equivalent": round(pp / 100 * n, 2) if convert else "",
            "source_csv": source,
            "note": note,
        })

    # (a) Pipeline lever: reactive serviceability at the fast pipeline vs today's.
    add(
        f"pipeline lever: reactive serviceability {fast:g}x vs {today:g}x pipeline (pp)",
        (reactive[fast] - reactive[today]) * 100,
        "results/heatmap.csv;config/wing.yaml",
        year_note,
    )
    # (b) SHIKRA gain over reactive at the fast pipeline.
    add(
        f"SHIKRA gain vs reactive at {fast:g}x pipeline (pp)",
        float(shikra_fast["gain_vs_reactive_mean"]) * 100,
        "results/heatmap.csv;config/wing.yaml",
        year_note,
    )
    # (c) Largest single-LRU supply-shock losses (the all-foreign combined row is not an LRU).
    lru_ids = {lru["id"] for lru in wing["lrus"]}
    shock = pd.read_csv(RESULTS_DIR / "supply_shock.csv")
    shock = shock[shock["lru_id"].isin(lru_ids)].sort_values(
        "loss_pp_shock_window", ascending=False, kind="stable"
    )
    for _, s in shock.head(TOP_SHOCKS).iterrows():
        add(
            f"supply-shock loss: {s['lru_id']} ({s['source']}) (pp)",
            float(s["loss_pp_shock_window"]),
            "results/supply_shock.csv;config/wing.yaml",
            "synthetic wing; average over the supply-halt window",
        )
    # (d) Unscheduled engine removals per year (counts, not an aircraft equivalent).
    ps = pd.read_csv(RESULTS_DIR / "policy_summary.csv")
    unsched = ps[ps["metric"] == "unscheduled_removals"].set_index("policy")["mean"]
    for policy, name in (("reactive", "reactive"), ("shikra_v2", "SHIKRA")):
        add(
            f"unscheduled engine removals per year: {name}",
            float(unsched[policy]) * DAYS_PER_YEAR / measured_days,
            "results/policy_summary.csv;config/wing.yaml",
            f"{year_note}; count of removals, not aircraft",
            convert=False,
        )
    return out


def main(out_path: Path = OUT_PATH) -> list[dict]:
    data = rows()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    return data


if __name__ == "__main__":
    for r in main():
        print(f"{r['item']}: {r['pp_or_value']} -> {r['aircraft_equivalent']}")
