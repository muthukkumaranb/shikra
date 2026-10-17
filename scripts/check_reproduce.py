"""Verify that simulation results match committed git artifacts exactly."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    csv_files = sorted(Path("results").glob("*.csv"))
    n_files = len(csv_files)
    if n_files == 0:
        print("ERROR: No results/*.csv files found to check.", file=sys.stderr)
        return 1

    # Run git diff --quiet -- results/*.csv
    res = subprocess.run(
        ["git", "diff", "--quiet", "--", "results/*.csv"],
        capture_output=True,
    )

    if res.returncode == 0:
        print(f"REPRODUCED: {n_files} files identical")
        return 0

    # If diff exists, list which files differ
    diff_names = subprocess.run(
        ["git", "diff", "--name-only", "--", "results/*.csv"],
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
