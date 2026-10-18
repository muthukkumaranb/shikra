"""Deck-quality figures for SHIKRA.

Produces (300 dpi, 16:9, white background) in results/deck/:
  lever_map.png, policies.png, indigenise.png, rul_band.png, figures_manifest.csv

Every number drawn is read at run time from results/*.csv,
artifacts/policy_params.json, artifacts/replay_rul.parquet or config/wing.yaml.

Run via ``make deck-figures``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_PATH = ROOT / "config" / "wing.yaml"
DEFAULT_OUT = RESULTS_DIR / "deck"

FIGSIZE = (13.33, 7.5)
DPI = 300
# Okabe-Ito colour-blind-safe palette.
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _wing() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sim_caption(wing: dict, n_reps: int) -> str:
    n_ac = wing["fleet"]["squadrons"] * wing["fleet"]["aircraft_per_sqn"]
    return f"Synthetic {n_ac}-aircraft wing · {n_reps} seeds · 95% CI"


def _n_reps() -> int:
    """Replications per policy, from policy_summary.csv (heatmap.csv has no n_reps)."""
    return int(pd.read_csv(RESULTS_DIR / "policy_summary.csv")["n_reps"].iloc[0])


def _new_fig(n_cols: int = 1):
    fig, axes = plt.subplots(1, n_cols, figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.17, wspace=0.12)
    return fig, axes


def _finish(fig, title: str, caption: str, path: Path) -> None:
    fig.suptitle(title, fontsize=24, fontweight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.025, caption, fontsize=14, color="#444444", ha="left")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def lever_map(out: Path, wing: dict) -> dict:
    hm = pd.read_csv(RESULTS_DIR / "heatmap.csv")
    today = float(wing["pipeline_mult"])
    pipes = sorted(hm["pipeline_mult"].unique())
    quals = sorted(hm["quality"].unique())
    labels = hm.drop_duplicates("quality").set_index("quality")["quality_label"]
    grid = (
        hm.pivot(index="quality", columns="pipeline_mult", values="gain_vs_reactive_mean")
        .reindex(index=quals, columns=pipes)
        * 100
    )
    vals = grid.to_numpy()

    fig, ax = _new_fig()
    fig.subplots_adjust(left=0.15, right=0.93)
    norm = TwoSlopeNorm(vmin=min(vals.min(), -1e-9), vcenter=0.0, vmax=max(vals.max(), 1e-9))
    im = ax.imshow(vals, cmap="PuOr", norm=norm, aspect="auto")
    for i in range(len(quals)):
        for j in range(len(pipes)):
            v = vals[i, j]
            rgba = im.cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=15,
                    color="white" if lum < 0.5 else "black")
    ax.set_xticks(range(len(pipes)), [f"{p:g}×" for p in pipes])
    ax.set_yticks(range(len(quals)), [labels[q] for q in quals])
    ax.set_xlabel("Pipeline multiplier on repair and lead times (left = faster)")
    ax.set_ylabel("Prediction quality")
    for s in ax.spines.values():
        s.set_visible(False)

    # Outline today's operating point: today's pipeline with SHIKRA-quality prediction.
    shikra_q = next(q for q in quals if "SHIKRA" in labels[q])
    j0, i0 = pipes.index(today), quals.index(shikra_q)
    ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, lw=4, ec=OI["black"]))
    ax.annotate("Today", xy=(j0, i0 - 0.5), xytext=(j0, -0.75), ha="center",
                fontsize=15, fontweight="bold", annotation_clip=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Serviceability gain vs reactive (pp)")
    cb.outline.set_visible(False)

    _finish(fig, "Where the next rupee buys aircraft", _sim_caption(wing, _n_reps()),
            out / "lever_map.png")
    return {
        "figure": "lever_map.png",
        "source_csv": "results/heatmap.csv;config/wing.yaml",
        "rows_used": len(hm),
        "key_value": f"gain_pp_today={vals[i0, j0]:+.2f}",
    }


def policies(out: Path, wing: dict) -> dict:
    hm = pd.read_csv(RESULTS_DIR / "heatmap.csv")
    ps = pd.read_csv(RESULTS_DIR / "policy_summary.csv")
    svc = ps[ps["metric"] == "serviceability"].set_index("policy")
    today = float(wing["pipeline_mult"])
    fast = float(min(hm["pipeline_mult"]))
    q_label = hm.drop_duplicates("quality").set_index("quality")["quality_label"]
    q_shikra = next(q for q, lab in q_label.items() if "SHIKRA" in lab)
    q_oracle = next(q for q, lab in q_label.items() if "Oracle" in lab)

    # policy_summary.csv is the today-pipeline run; confirm it matches heatmap.csv.
    hm_today = hm[hm["pipeline_mult"] == today]
    assert np.isclose(
        svc.loc["reactive", "mean"], hm_today["reactive_serviceability_mean"].iloc[0]
    ), "policy_summary.csv reactive does not match heatmap.csv at today's pipeline"

    names = ["Reactive", "Fixed-interval", "SHIKRA", "Perfect prediction"]
    colors = [OI["vermillion"], OI["orange"], OI["blue"], OI["green"]]

    def panel(pipe: float) -> list[tuple[float, float, float] | None]:
        rows = hm[hm["pipeline_mult"] == pipe]
        r = rows.iloc[0]
        s = rows[rows["quality"] == q_shikra].iloc[0]
        o = rows[rows["quality"] == q_oracle].iloc[0]
        fixed = None
        if pipe == today:
            p = svc.loc["preventive"]
            fixed = (p["mean"], p["ci_lo"], p["ci_hi"])
        return [
            (r["reactive_serviceability_mean"], r["reactive_serviceability_ci_lo"],
             r["reactive_serviceability_ci_hi"]),
            fixed,
            (s["serviceability_mean"], s["serviceability_ci_lo"], s["serviceability_ci_hi"]),
            (o["serviceability_mean"], o["serviceability_ci_lo"], o["serviceability_ci_hi"]),
        ]

    fig, axes = _new_fig(2)
    panels = [(today, f"Today's pipeline ({today:g}×)"), (fast, f"Fast pipeline ({fast:g}×)")]
    all_vals = []
    for ax, (pipe, label) in zip(axes, panels):
        data = panel(pipe)
        for k, d in enumerate(data):
            if d is None:
                ax.text(k, 2, "not\nsimulated", ha="center", va="bottom", fontsize=14,
                        color=OI["grey"])
                continue
            m, lo, hi = (v * 100 for v in d)
            all_vals.append(m)
            ax.bar(k, m, color=colors[k], width=0.65)
            ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="none", color="black",
                        capsize=8, lw=2)
            ax.text(k, hi + 1.5, f"{m:.1f}%", ha="center", va="bottom", fontsize=15)
        ax.set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names])
        ax.set_title(label, fontsize=18, loc="left")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Serviceability (%)")
    axes[1].tick_params(axis="y", labelleft=False)

    t, f_ = panel(today), panel(fast)
    d_today = (t[2][0] - t[0][0]) * 100
    d_fast = (f_[2][0] - f_[0][0]) * 100
    _finish(fig, "Prediction pays only once the pipeline is fast",
            _sim_caption(wing, int(svc["n_reps"].iloc[0])), out / "policies.png")
    return {
        "figure": "policies.png",
        "source_csv": "results/heatmap.csv;results/policy_summary.csv;config/wing.yaml",
        # heatmap SHIKRA + perfect rows (which also carry reactive) + policy_summary preventive.
        "rows_used": int(
            (hm["pipeline_mult"].isin([today, fast]) & hm["quality"].isin([q_shikra, q_oracle]))
            .sum()
        )
        + 1,
        "key_value": f"shikra_minus_reactive_pp@{today:g}={d_today:+.2f};"
        f"@{fast:g}={d_fast:+.2f}",
    }


def indigenise(out: Path, wing: dict) -> dict:
    sh = pd.read_csv(RESULTS_DIR / "supply_shock.csv")
    lrus = {lru["id"]: lru for lru in wing["lrus"]}
    per_lru = sh[sh["lru_id"].isin(lrus)].sort_values("loss_pp_shock_window")
    domestic = [lru["description"] for lru in wing["lrus"] if lru["source"] == "domestic"]

    fig, ax = _new_fig()
    fig.subplots_adjust(left=0.27)
    y = np.arange(len(per_lru))
    m = per_lru["loss_pp_shock_window"].to_numpy()
    lo, hi = per_lru["ci_lo"].to_numpy(), per_lru["ci_hi"].to_numpy()
    src_color = {"foreign": OI["vermillion"], "domestic": OI["blue"]}
    ax.barh(y, m, color=[src_color[s] for s in per_lru["source"]], height=0.6)
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", color="black", capsize=7, lw=2)
    for yi, mi, hii in zip(y, m, hi):
        ax.text(hii + 0.3, yi, f"{mi:.1f} pp", va="center", fontsize=15)
    ax.set_yticks(y, [f"{lrus[i]['description']}" for i in per_lru["lru_id"]])
    ax.set_xlim(0, hi.max() * 1.15)
    ax.set_xlabel("Availability lost over the supply halt (pp)")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[Patch(color=src_color["foreign"], label="Foreign-sourced (halted)"),
                 Patch(color=src_color["domestic"], label="Domestic")],
        loc="lower right", frameon=False,
    )
    ax.text(0.99, 0.2, "Domestic LRUs, not halted:\n" + ", ".join(domestic),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=14,
            color="#444444", wrap=True)
    _finish(fig, "What to make in India", _sim_caption(wing, _n_reps()), out / "indigenise.png")
    top = per_lru.iloc[-1]
    return {
        "figure": "indigenise.png",
        "source_csv": "results/supply_shock.csv;config/wing.yaml",
        "rows_used": len(per_lru),
        "key_value": f"top={top['lru_id']}:{top['loss_pp_shock_window']:.2f}pp",
    }


def rul_band(out: Path, wing: dict) -> dict:
    # Ground truth is deliberately never loaded: only prediction columns are read.
    cols = ["subset", "unit", "cycle", "q10", "q50", "q90"]
    rr = pd.read_parquet(ARTIFACTS_DIR / "replay_rul.parquet", columns=cols)
    with open(ARTIFACTS_DIR / "policy_params.json", encoding="utf-8") as f:
        r_star = float(json.load(f)["R_star"])

    # Engine selection rule: sort replay units by (subset, unit) and take the
    # lowest one. Fixed, so the figure never depends on cherry-picking.
    subset, unit = sorted(set(zip(rr["subset"], rr["unit"].astype(int))))[0]
    eng = rr[(rr["subset"] == subset) & (rr["unit"] == unit)].sort_values("cycle")

    fig, ax = _new_fig()
    c = eng["cycle"].to_numpy()
    ax.fill_between(c, eng["q10"], eng["q90"], color=OI["sky"], alpha=0.45,
                    label="q10–q90 band", lw=0)
    ax.plot(c, eng["q50"], color=OI["blue"], lw=2.5, label="q50 (median)")
    ax.plot(c, eng["q10"], color=OI["vermillion"], lw=2.5, label="q10 (pessimistic edge)")
    ax.axhline(r_star, color=OI["black"], ls="--", lw=2)
    ax.text(c[0], r_star + 2, f"R* = {r_star:g} cycles (act when q10 reaches it)",
            fontsize=15, va="bottom")
    hit = eng[eng["q10"] <= r_star]
    key = f"{subset}/unit{unit};R*={r_star:g}"
    if not hit.empty:
        hc = int(hit["cycle"].iloc[0])
        ax.axvline(hc, color=OI["grey"], ls=":", lw=2)
        ax.text(hc - 1, ax.get_ylim()[1], f"q10 hits R* at cycle {hc} ", fontsize=15,
                ha="right", va="top")
        key += f";q10_hits_R*_at_cycle={hc}"
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Predicted remaining life (cycles)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center left", frameon=False)
    _finish(fig, "Plan on the pessimistic edge",
            f"NASA C-MAPSS {subset} unit {unit} · quantile replay · "
            "R* from artifacts/policy_params.json", out / "rul_band.png")
    return {
        "figure": "rul_band.png",
        "source_csv": "artifacts/replay_rul.parquet;artifacts/policy_params.json",
        "rows_used": len(eng),
        "key_value": key,
    }


def main(out_dir: Path = DEFAULT_OUT) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wing = _wing()
    rows = [f(out_dir, wing) for f in (lever_map, policies, indigenise, rul_band)]
    with open(out_dir / "figures_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "source_csv", "rows_used", "key_value"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    for r in main(parser.parse_args().out):
        print(r["figure"], r["key_value"])
