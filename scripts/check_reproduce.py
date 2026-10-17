"""Verify that simulation results match committed git artifacts exactly."""

import argparse
import subprocess
import sys
from pathlib import Path

QUICK_CSVS = [
    "results/policies.csv",
    "results/policy_summary.csv",
    "results/bias.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check simulation results reproducibility.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check only sim + summary + bias CSVs (used in CI)",
    )
    args = parser.parse_args()

    if args.quick:
        targets = QUICK_CSVS
        mode_str = " (quick mode: policies.csv, policy_summary.csv, bias.csv)"
    else:
        targets = [str(p).replace("\\", "/") for p in sorted(Path("results").glob("*.csv"))]
        mode_str = ""

    n_files = len(targets)
    if n_files == 0:
        print("ERROR: No CSV files found to check.", file=sys.stderr)
        return 1

    # Run git diff --quiet -- <targets>
    cmd = ["git", "diff", "--quiet", "--"] + targets
    res = subprocess.run(cmd, capture_output=True)

    if res.returncode == 0:
        print(f"REPRODUCED: {n_files} files identical{mode_str}")
        return 0

    # If diff exists, list which files differ
    diff_names = subprocess.run(
        ["git", "diff", "--name-only", "--"] + targets,
        capture_output=True,
        text=True,
        check=True,
    )
    differing = [line.strip() for line in diff_names.stdout.splitlines() if line.strip()]

    print(f"FAILED: {len(differing)} file(s) differ from committed version:", file=sys.stderr)
    for f in differing:
        print(f"  - {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
