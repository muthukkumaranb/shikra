"""RUL model evaluation against official C-MAPSS test sets.

Computes RMSE, MAE, NASA score, coverage (before/after conformal), and
writes results to ``results/rul_cmapss.csv`` and ``results/coverage.csv``.
Also trains the full model pipeline (features → scaler → LightGBM → conformal).

Orchestration entry point: ``python -m shikra.rul.evaluate``

Gates written to ``results/GATES.md``:
  G0.1 — FD001 RMSE ≤ 18.5 (target ≤ 13)
  G0.2 — after conformal, 90% coverage on replay units ∈ [0.85, 0.95]
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

MODEL_PREFIX = ARTIFACTS_DIR / "rul"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
GATES_PATH = RESULTS_DIR / "GATES.md"


# ---------------------------------------------------------------------------
# NASA score
# ---------------------------------------------------------------------------


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the asymmetric NASA scoring function.

    Late predictions (d > 0) are penalised more heavily than early ones.
    d = pred − true; s = Σ exp(-d/13)-1 if d<0 else exp(d/10)-1
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(scores.sum())


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def empirical_coverage(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    return float(np.mean((y >= q_lo) & (y <= q_hi)))


def mean_width(q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    return float(np.mean(q_hi - q_lo))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Train, calibrate, evaluate, and write all results."""
    from shikra.data.cmapss import load
    from shikra.data.split import load_split, split_to_sets
    from shikra.rul.conformal import (
        adjust,
        build_calib_rows,
        calibrate,
        save_conformal,
    )
    from shikra.rul.features import RUL_CAP, build_features, fit_scaler
    from shikra.rul.model import (
        load_models,
        predict,
        save_models,
        train,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHIKRA RUL pipeline")
    print("=" * 60)

    # ── Load split ─────────────────────────────────────────────────────────
    split_dict = load_split()
    train_pairs, calib_pairs, replay_pairs = split_to_sets(split_dict)

    # ── Load raw data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading C-MAPSS FD001 + FD003 …")
    df_fd001 = load("FD001", "train")
    df_fd003 = load("FD003", "train")
    df_all = pd.concat([df_fd001, df_fd003], ignore_index=True)

    def _filter(df: pd.DataFrame, pairs: list[tuple[str, int]]) -> pd.DataFrame:
        mask = pd.Series(False, index=df.index)
        for subset, unit in pairs:
            mask |= (df["subset"] == subset) & (df["unit"] == unit)
        return df[mask].copy()

    df_train = _filter(df_all, train_pairs)
    df_calib = _filter(df_all, calib_pairs)
    df_replay = _filter(df_all, replay_pairs)

    print(f"  train rows: {len(df_train):,}  calib: {len(df_calib):,}  replay: {len(df_replay):,}")

    # ── Fit scaler on train only ───────────────────────────────────────────
    print("\n[2/6] Fitting scaler and building features …")
    scaler = fit_scaler(df_train)
    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f)

    X_train, y_train, meta_train = build_features(df_train, scaler)

    # Internal val split for early stopping: last 20% of train units (by index)
    n_val = max(1, int(0.2 * len(train_pairs)))
    val_pairs = train_pairs[-n_val:]
    df_val_raw = _filter(df_all, val_pairs)
    X_val, y_val, _ = build_features(df_val_raw, scaler)
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

    # ── Train model ────────────────────────────────────────────────────────
    if MODEL_PREFIX.with_suffix("").with_name("rul_models.pkl").exists():
        print("\n[3/6] Loading existing models …")
        models = load_models(MODEL_PREFIX)
    else:
        print("\n[3/6] Training LightGBM quantile ensemble …")
        models = train(X_train, y_train, X_val, y_val)
        save_models(models, MODEL_PREFIX)

    # ── Conformal calibration on calib units ──────────────────────────────
    print("\n[4/6] Conformal calibration on calib units …")
    calib_rows = build_calib_rows(df_calib, n_trunc=10, seed=2026)
    X_calib, y_calib, meta_calib = build_features(calib_rows, scaler)
    preds_calib_raw = predict(models, X_calib)
    Q_dict = calibrate(y_calib, preds_calib_raw)
    save_conformal(Q_dict)
    adjust(preds_calib_raw, Q_dict)  # runs conformal adjustment (Q_dict is used later)

    # ── Evaluate on official test sets ────────────────────────────────────
    print("\n[5/6] Evaluating on official test sets …")
    rul_rows = []
    cov_rows = []

    for subset in ("FD001", "FD003"):
        df_test = load(subset, "test")
        # Use last cycle per unit only
        last_rows = df_test.sort_values("cycle").groupby("unit").tail(1).copy()
        X_test, y_test_arr, meta_test = build_features(last_rows, scaler)
        y_test_true = last_rows["rul_true"].values  # uncapped true RUL
        y_test_capped = np.minimum(y_test_true, RUL_CAP)
        n_units = len(last_rows)

        preds_raw = predict(models, X_test)
        preds_adj = adjust(preds_raw, Q_dict)

        y_pred_median = preds_adj[0.50]

        rmse = float(np.sqrt(np.mean((y_pred_median - y_test_capped) ** 2)))
        mae = float(np.mean(np.abs(y_pred_median - y_test_capped)))
        ns = nasa_score(y_test_true, y_pred_median)

        # RMSE where true RUL ≤ 30
        mask_crit = y_test_true <= 30
        if mask_crit.sum() > 0:
            rmse_crit = float(
                np.sqrt(np.mean((y_pred_median[mask_crit] - y_test_true[mask_crit]) ** 2))
            )
        else:
            rmse_crit = float("nan")

        rul_rows.append(
            {
                "subset": subset,
                "model": "lgbm_quantile",
                "seed": "avg5",
                "rmse": rmse,
                "mae": mae,
                "nasa_score": ns,
                "rmse_critical": rmse_crit,
                "n_units": n_units,
            }
        )
        print(f"  {subset}: RMSE={rmse:.2f}  MAE={mae:.2f}  NASA={ns:.1f}  n={n_units}")

        # Coverage — pre and post conformal
        for nominal, (alo, ahi) in [(0.80, (0.10, 0.90)), (0.90, (0.05, 0.95))]:
            for calibrated, preds_use in [(False, preds_raw), (True, preds_adj)]:
                emp = empirical_coverage(y_test_capped, preds_use[alo], preds_use[ahi])
                mw = mean_width(preds_use[alo], preds_use[ahi])
                cov_rows.append(
                    {
                        "subset": subset,
                        "split": "test",
                        "nominal": nominal,
                        "empirical": emp,
                        "mean_width": mw,
                        "n": n_units,
                        "calibrated": calibrated,
                    }
                )

    # ── Coverage on replay units ───────────────────────────────────────────
    replay_rows_df = build_calib_rows(df_replay, n_trunc=10, seed=2026)
    X_rep, y_rep, _ = build_features(replay_rows_df, scaler)
    preds_rep_raw = predict(models, X_rep)
    preds_rep_adj = adjust(preds_rep_raw, Q_dict)

    for nominal, (alo, ahi) in [(0.80, (0.10, 0.90)), (0.90, (0.05, 0.95))]:
        for calibrated, preds_use in [(False, preds_rep_raw), (True, preds_rep_adj)]:
            emp = empirical_coverage(y_rep, preds_use[alo], preds_use[ahi])
            mw = mean_width(preds_use[alo], preds_use[ahi])
            cov_rows.append(
                {
                    "subset": "FD001+FD003",
                    "split": "replay",
                    "nominal": nominal,
                    "empirical": emp,
                    "mean_width": mw,
                    "n": len(y_rep),
                    "calibrated": calibrated,
                }
            )

    # ── Write results ──────────────────────────────────────────────────────
    pd.DataFrame(rul_rows).to_csv(RESULTS_DIR / "rul_cmapss.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(RESULTS_DIR / "coverage.csv", index=False)
    print(f"\n  Results written -> {RESULTS_DIR/'rul_cmapss.csv'}")
    print(f"  Coverage written -> {RESULTS_DIR/'coverage.csv'}")

    # ── Reliability figure ─────────────────────────────────────────────────
    _plot_reliability(preds_rep_adj, y_rep, RESULTS_DIR / "fig_reliability.png")

    # ── Gates ─────────────────────────────────────────────────────────────
    print("\n[6/6] Gate verdicts …")
    _write_gates(rul_rows, cov_rows)


def _plot_reliability(
    preds: dict[float, np.ndarray],
    y: np.ndarray,
    out_path: Path,
) -> None:
    """Plot a reliability / calibration diagram."""
    nominals = [0.80, 0.90]
    empiricals = []
    for nominal, (alo, ahi) in [(0.80, (0.10, 0.90)), (0.90, (0.05, 0.95))]:
        emp = empirical_coverage(y, preds[alo], preds[ahi])
        empiricals.append(emp)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(nominals, empiricals, color="steelblue", s=80, zorder=5, label="Observed")
    for n, e in zip(nominals, empiricals):
        ax.annotate(f"{e:.2f}", (n, e), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (replay units)")
    ax.set_title("SHIKRA RUL — Conformal calibration reliability (replay)")
    ax.legend()
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Reliability figure -> {out_path}")


def _write_gates(
    rul_rows: list[dict],
    cov_rows: list[dict],
) -> None:
    """Evaluate gates G0.1 and G0.2 and append verdicts to GATES.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fd001_row = next(r for r in rul_rows if r["subset"] == "FD001")
    fd001_rmse = fd001_row["rmse"]
    g01 = "PASS" if fd001_rmse <= 18.5 else "FAIL"
    g01_detail = f"FD001 RMSE = {fd001_rmse:.2f} (threshold <= 18.5, target <= 13)"

    # G0.2: 90% coverage on replay units (calibrated)
    cov_90_replay = next(
        r
        for r in cov_rows
        if r["split"] == "replay" and r["nominal"] == 0.90 and r["calibrated"]
    )
    emp_90 = cov_90_replay["empirical"]
    g02 = "PASS" if 0.85 <= emp_90 <= 0.95 else "FAIL"
    g02_detail = f"90% replay coverage = {emp_90:.3f} (target 0.85-0.95)"

    print(f"  G0.1: {g01}  - {g01_detail}")
    print(f"  G0.2: {g02}  - {g02_detail}")

    # Append to GATES.md
    existing = GATES_PATH.read_text(encoding="utf-8") if GATES_PATH.exists() else ""
    section = f"""
## RUL Model Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.1 — FD001 RMSE ≤ 18.5 | **{g01}** | {g01_detail} |
| G0.2 — 90% replay coverage 0.85–0.95 | **{g02}** | {g02_detail} |

"""
    if "## RUL Model Gates" in existing:
        # Replace the existing section
        import re
        existing = re.sub(
            r"## RUL Model Gates.*?(?=\n## |\Z)", section.strip(), existing, flags=re.DOTALL
        )
        GATES_PATH.write_text(existing, encoding="utf-8")
    else:
        with GATES_PATH.open("a", encoding="utf-8") as f:
            f.write(section)

    print(f"  Gates written -> {GATES_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
