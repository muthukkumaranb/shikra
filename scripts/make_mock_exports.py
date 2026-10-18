"""Generate synthetic data exports for ingestion testing.

Produces four CSV files in ``data/mock_exports/`` that mimic real-world
maintenance-information-system extracts, plus a ``faults_injected.csv``
manifest that records every intentionally corrupted row.

All data is derived (read-only) from ``config/wing.yaml`` and a single
simulation run (``shikra_v2``, seed 1).

Injected fault rates:
  ~3 %  missing keys  (blank primary-key fields)
  ~1 %  mismatched serials  (engine serial on wrong tail)
  ~5 %  stale stock rows  (last_updated older than 30 days)

Usage:
  python scripts/make_mock_exports.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "wing.yaml"
ARTIFACTS_DIR = ROOT / "artifacts"
REPLAY_PATH = ARTIFACTS_DIR / "replay_rul.parquet"
PARAMS_PATH = ARTIFACTS_DIR / "policy_params.json"
OUT_DIR = ROOT / "data" / "mock_exports"

SEED = 2026
SIM_SEED = 1
BASE_DATE = date(2026, 1, 1)

WORK_ORDER_TEMPLATES = [
    "No.{pos} eng EGT high on climb, borescope",
    "No.{pos} eng vibration trend rising, investigate",
    "No.{pos} eng oil pressure fluctuation noted",
    "No.{pos} eng compressor stall on startup",
    "No.{pos} eng hot start reported by pilot",
    "No.{pos} eng foreign object damage inspection",
    "No.{pos} eng oil consumption above limit",
    "No.{pos} eng N1 overshoot on takeoff",
    "No.{pos} eng intermittent fire warning",
    "No.{pos} eng fuel leak detected during preflight",
    "No.{pos} eng surge during acceleration",
    "No.{pos} eng chip detector warning",
    "No.{pos} eng exhaust nozzle malfunction",
    "No.{pos} eng bleed air leak indication",
    "No.{pos} eng abnormal noise on spool up",
    "No.{pos} eng ITT exceedance on start",
    "No.{pos} eng gearbox magnetic plug check",
    "No.{pos} eng fan blade tip rub marks",
    "No.{pos} eng low idle unstable",
    "No.{pos} eng thrust reverser advisory",
    "No.{pos} eng igniter plug replacement due",
    "No.{pos} eng turbine blade creep inspection",
    "No.{pos} eng starter duty cycle exceeded",
    "No.{pos} eng hydraulic pump vibration check",
    "No.{pos} eng augmentor liner cracks found",
    "No.{pos} eng bearing temperature anomaly",
    "No.{pos} eng flame-out during deceleration",
    "No.{pos} eng power loss at altitude, check",
    "No.{pos} eng fuel control unit malfunction",
    "No.{pos} eng accessory drive seal weeping",
]

REPAIR_AGENCIES = [
    "BRD-A", "BRD-B", "HAL-Koraput", "HAL-Bangalore",
    "OEM-foreign", "OEM-domestic",
]

HUMS_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]


def _tail_id(idx: int) -> str:
    """T01 … T54."""
    return f"T{idx + 1:02d}"


def _engine_sn(pos: int) -> str:
    """Generate deterministic engine serial from position index."""
    return f"ENG-{pos + 1001:04d}"


def _run_sim():
    """Run shikra_v2 seed 1 and return (config, SimResult, rul)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from shikra.rul.variants import apply
    from shikra.sim.engine import run

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    rul = pd.read_parquet(REPLAY_PATH)
    rul_use = apply(rul, 1)
    result = run("shikra_v2", cfg, rul_use, params, SIM_SEED, cfg["ops"]["days"])
    return cfg, result, rul


def generate(seed: int = SEED) -> Path:
    """Generate all four mock exports and the fault log."""
    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg, sim_result, rul_df = _run_sim()

    n_sqn = cfg["fleet"]["squadrons"]
    ac_per_sqn = cfg["fleet"]["aircraft_per_sqn"]
    n_ac = n_sqn * ac_per_sqn
    n_eng = n_ac * cfg["fleet"]["engines_per_ac"]
    lru_list = cfg["lrus"]

    # Build tail → engine mapping
    tail_ids = [_tail_id(i) for i in range(n_ac)]
    engine_sns = [_engine_sn(i) for i in range(n_eng)]
    tail_to_engines: dict[str, list[str]] = {}
    for t_idx in range(n_ac):
        engines = []
        for e in range(cfg["fleet"]["engines_per_ac"]):
            engines.append(engine_sns[t_idx * cfg["fleet"]["engines_per_ac"] + e])
        tail_to_engines[tail_ids[t_idx]] = engines

    faults: list[dict] = []

    # ── 1. emms_workorders.csv ────────────────────────────────────────────
    wo_rows: list[dict] = []
    events_df = sim_result.events
    for i, ev in events_df.iterrows():
        pos = int(ev["pos"])
        day = int(ev["day"])
        tail_idx = pos // cfg["fleet"]["engines_per_ac"]
        eng_pos_in_ac = (pos % cfg["fleet"]["engines_per_ac"]) + 1
        tid = _tail_id(tail_idx)
        esn = engine_sns[pos]
        tmpl = WORK_ORDER_TEMPLATES[rng.integers(0, len(WORK_ORDER_TEMPLATES))]
        system_text = tmpl.format(pos=eng_pos_in_ac)
        action = str(ev["type"]) + " removal"
        wo_id = f"WO-{len(wo_rows) + 1:05d}"
        wo_date = BASE_DATE + timedelta(days=day)
        wo_rows.append({
            "wo_id": wo_id,
            "tail_id": tid,
            "date": wo_date.isoformat(),
            "system_text": system_text,
            "action": action,
            "engine_sn": esn,
        })

    # Add some LRU-related work orders (synthetic, not from sim events)
    for _ in range(40):
        t_idx = int(rng.integers(0, n_ac))
        tid = _tail_id(t_idx)
        lru = lru_list[int(rng.integers(0, len(lru_list)))]
        day = int(rng.integers(0, 365))
        wo_id = f"WO-{len(wo_rows) + 1:05d}"
        wo_date = BASE_DATE + timedelta(days=day)
        wo_rows.append({
            "wo_id": wo_id,
            "tail_id": tid,
            "date": wo_date.isoformat(),
            "system_text": f"{lru['description']} replacement, scheduled",
            "action": "LRU swap",
            "engine_sn": "",
        })

    # Inject ~3% missing keys into work orders
    n_wo = len(wo_rows)
    n_missing_wo = max(1, round(n_wo * 0.03))
    missing_wo_idxs = rng.choice(n_wo, size=n_missing_wo, replace=False)
    for idx in missing_wo_idxs:
        field = rng.choice(["wo_id", "tail_id"])
        original = wo_rows[idx][field]
        wo_rows[idx][field] = ""
        faults.append({
            "source": "emms_workorders.csv",
            "row_id": idx,
            "fault_type": "missing_key",
            "detail": f"{field} blanked (was {original})",
        })

    # Inject ~1% mismatched serials
    n_mismatch_wo = max(1, round(n_wo * 0.01))
    mismatch_wo_idxs = rng.choice(n_wo, size=n_mismatch_wo, replace=False)
    for idx in mismatch_wo_idxs:
        if idx in missing_wo_idxs:
            continue
        row = wo_rows[idx]
        if not row["engine_sn"] or not row["tail_id"]:
            continue
        # Assign an engine from a different tail
        original_tid = row["tail_id"]
        other_tails = [t for t in tail_ids if t != original_tid]
        wrong_tail = rng.choice(other_tails)
        wrong_esn = rng.choice(tail_to_engines[wrong_tail])
        wo_rows[idx]["engine_sn"] = wrong_esn
        faults.append({
            "source": "emms_workorders.csv",
            "row_id": idx,
            "fault_type": "mismatched_sn",
            "detail": f"engine_sn {wrong_esn} belongs to {wrong_tail}, not {original_tid}",
        })

    _write_csv(
        OUT_DIR / "emms_workorders.csv",
        ["wo_id", "tail_id", "date", "system_text", "action", "engine_sn"],
        wo_rows,
    )

    # ── 2. immols_stock.csv ───────────────────────────────────────────────
    stock_rows: list[dict] = []
    locations = ["base_S1", "base_S2", "base_S3", "depot"]
    for lru in lru_list:
        for loc in locations:
            if loc == "depot":
                on_hand = lru["stock_depot"]
            else:
                on_hand = lru["stock_base"]
            days_ago = int(rng.integers(0, 15))
            last_updated = BASE_DATE + timedelta(days=365) - timedelta(days=days_ago)
            stock_rows.append({
                "lru_id": lru["id"],
                "location": loc,
                "on_hand": on_hand,
                "on_order": int(rng.integers(0, 4)),
                "lead_time_days": lru["lead_time_days"],
                "last_updated": last_updated.isoformat(),
            })

    # Inject ~5% stale stock rows
    n_stock = len(stock_rows)
    n_stale = max(1, round(n_stock * 0.05))
    stale_idxs = rng.choice(n_stock, size=n_stale, replace=False)
    for idx in stale_idxs:
        old_date = BASE_DATE - timedelta(days=int(rng.integers(40, 120)))
        stock_rows[idx]["last_updated"] = old_date.isoformat()
        faults.append({
            "source": "immols_stock.csv",
            "row_id": idx,
            "fault_type": "stale",
            "detail": f"last_updated {stock_rows[idx]['last_updated']} is >30 days old",
        })

    # Inject ~3% missing keys into stock
    n_missing_stock = max(1, round(n_stock * 0.03))
    missing_stock_idxs = rng.choice(n_stock, size=n_missing_stock, replace=False)
    for idx in missing_stock_idxs:
        original = stock_rows[idx]["lru_id"]
        stock_rows[idx]["lru_id"] = ""
        faults.append({
            "source": "immols_stock.csv",
            "row_id": idx,
            "fault_type": "missing_key",
            "detail": f"lru_id blanked (was {original})",
        })

    _write_csv(
        OUT_DIR / "immols_stock.csv",
        ["lru_id", "location", "on_hand", "on_order", "lead_time_days", "last_updated"],
        stock_rows,
    )

    # ── 3. hums_engine.csv ────────────────────────────────────────────────
    hums_rows: list[dict] = []
    # Use replay trajectories to generate sensor data
    units = rul_df.groupby(["subset", "unit"]).first().reset_index()
    unit_list = list(zip(units["subset"], units["unit"]))

    # Assign each engine position a trajectory (deterministic with rng)
    pos_trajectories = [unit_list[int(rng.integers(0, len(unit_list)))] for _ in range(n_eng)]

    # Generate synthetic sensor baselines per sensor
    sensor_baselines = {s: float(rng.uniform(400, 650)) for s in HUMS_SENSORS}
    sensor_slopes = {s: float(rng.uniform(-0.1, 0.1)) for s in HUMS_SENSORS}
    sensor_noise_std = {s: float(rng.uniform(1, 5)) for s in HUMS_SENSORS}

    for pos in range(n_eng):
        tail_idx = pos // cfg["fleet"]["engines_per_ac"]
        tid = _tail_id(tail_idx)
        esn = engine_sns[pos]
        sub, unit = pos_trajectories[pos]

        # Get trajectory length
        traj = rul_df[(rul_df["subset"] == sub) & (rul_df["unit"] == unit)]
        max_cycle = int(traj["cycle"].max())

        # Sample a portion of cycles for HUMS readings (not every cycle)
        n_readings = min(max_cycle, int(rng.integers(10, 40)))
        cycles = sorted(rng.choice(range(1, max_cycle + 1), size=n_readings, replace=False))

        for c in cycles:
            row: dict = {
                "engine_sn": esn,
                "tail_id": tid,
                "date": (BASE_DATE + timedelta(days=int(c * 365 / max_cycle))).isoformat(),
                "cycle": int(c),
            }
            for s in HUMS_SENSORS:
                val = sensor_baselines[s] + sensor_slopes[s] * c + rng.normal(0, sensor_noise_std[s])
                row[s] = round(val, 2)
            hums_rows.append(row)

    # Inject ~3% missing keys into HUMS
    n_hums = len(hums_rows)
    n_missing_hums = max(1, round(n_hums * 0.03))
    missing_hums_idxs = rng.choice(n_hums, size=n_missing_hums, replace=False)
    for idx in missing_hums_idxs:
        original = hums_rows[idx]["engine_sn"]
        hums_rows[idx]["engine_sn"] = ""
        faults.append({
            "source": "hums_engine.csv",
            "row_id": idx,
            "fault_type": "missing_key",
            "detail": f"engine_sn blanked (was {original})",
        })

    # Inject ~1% mismatched serials into HUMS
    n_mismatch_hums = max(1, round(n_hums * 0.01))
    mismatch_hums_idxs = rng.choice(n_hums, size=n_mismatch_hums, replace=False)
    for idx in mismatch_hums_idxs:
        if idx in missing_hums_idxs:
            continue
        row = hums_rows[idx]
        if not row["engine_sn"] or not row["tail_id"]:
            continue
        original_tid = row["tail_id"]
        other_tails = [t for t in tail_ids if t != original_tid]
        wrong_tail = rng.choice(other_tails)
        wrong_esn = rng.choice(tail_to_engines[wrong_tail])
        hums_rows[idx]["engine_sn"] = wrong_esn
        faults.append({
            "source": "hums_engine.csv",
            "row_id": idx,
            "fault_type": "mismatched_sn",
            "detail": f"engine_sn {wrong_esn} belongs to {wrong_tail}, not {original_tid}",
        })

    _write_csv(
        OUT_DIR / "hums_engine.csv",
        ["engine_sn", "tail_id", "date", "cycle"] + HUMS_SENSORS,
        hums_rows,
    )

    # ── 4. brd_repairs.csv ────────────────────────────────────────────────
    repair_rows: list[dict] = []
    # Generate repairs from sim engine removal events
    for i, ev in events_df.iterrows():
        pos = int(ev["pos"])
        day = int(ev["day"])
        esn = engine_sns[pos]
        repair_days = max(10, int(rng.normal(60, 20)))
        sent_date = BASE_DATE + timedelta(days=day)
        returned_date = sent_date + timedelta(days=repair_days)
        ro_id = f"RO-{len(repair_rows) + 1:05d}"
        agency = REPAIR_AGENCIES[int(rng.integers(0, len(REPAIR_AGENCIES)))]
        repair_rows.append({
            "ro_id": ro_id,
            "item_type": "engine",
            "sn": esn,
            "sent": sent_date.isoformat(),
            "returned": returned_date.isoformat(),
            "agency": agency,
        })

    # Add LRU repair orders
    for _ in range(30):
        lru = lru_list[int(rng.integers(0, len(lru_list)))]
        day = int(rng.integers(0, 365))
        repair_days = max(5, int(rng.normal(lru["repair_tat_days"], 5)))
        sent_date = BASE_DATE + timedelta(days=day)
        returned_date = sent_date + timedelta(days=repair_days)
        ro_id = f"RO-{len(repair_rows) + 1:05d}"
        agency = REPAIR_AGENCIES[int(rng.integers(0, len(REPAIR_AGENCIES)))]
        sn = f"LRU-{lru['id']}-{rng.integers(1, 20):03d}"
        repair_rows.append({
            "ro_id": ro_id,
            "item_type": "lru",
            "sn": sn,
            "sent": sent_date.isoformat(),
            "returned": returned_date.isoformat(),
            "agency": agency,
        })

    # Inject ~3% missing keys
    n_repair = len(repair_rows)
    n_missing_repair = max(1, round(n_repair * 0.03))
    missing_repair_idxs = rng.choice(n_repair, size=n_missing_repair, replace=False)
    for idx in missing_repair_idxs:
        field = rng.choice(["ro_id", "sn"])
        original = repair_rows[idx][field]
        repair_rows[idx][field] = ""
        faults.append({
            "source": "brd_repairs.csv",
            "row_id": idx,
            "fault_type": "missing_key",
            "detail": f"{field} blanked (was {original})",
        })

    _write_csv(
        OUT_DIR / "brd_repairs.csv",
        ["ro_id", "item_type", "sn", "sent", "returned", "agency"],
        repair_rows,
    )

    # ── 5. faults_injected.csv ────────────────────────────────────────────
    _write_csv(
        OUT_DIR / "faults_injected.csv",
        ["source", "row_id", "fault_type", "detail"],
        faults,
    )

    print(f"Mock exports written -> {OUT_DIR}")
    print(f"  emms_workorders.csv   : {len(wo_rows)} rows")
    print(f"  immols_stock.csv      : {len(stock_rows)} rows")
    print(f"  hums_engine.csv       : {len(hums_rows)} rows")
    print(f"  brd_repairs.csv       : {len(repair_rows)} rows")
    print(f"  faults_injected.csv   : {len(faults)} faults")
    return OUT_DIR


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a CSV with a synthetic data header comment."""
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC: not IAF data\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate()
