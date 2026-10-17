"""Download and verify the NASA C-MAPSS dataset.

Downloads:
  https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

Extracts until 12 individual text files are present in data/raw/cmapss/.
Prints SHA-256 checksums for DATA_CARD.md.

If the download fails, prints clear instructions for manual download.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "cmapss"

PRIMARY_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

EXPECTED_FILES = (
    [f"train_FD00{i}.txt" for i in range(1, 5)]
    + [f"test_FD00{i}.txt" for i in range(1, 5)]
    + [f"RUL_FD00{i}.txt" for i in range(1, 5)]
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_zip(data: bytes, target_dir: Path) -> None:
    """Recursively extract zip/nested zip until txt files are found."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        # Check for nested zip
        inner_zips = [n for n in names if n.lower().endswith(".zip")]
        txt_files = [n for n in names if n.lower().endswith(".txt")]

        if txt_files:
            z.extractall(target_dir)
            # Flatten: move any .txt files from subdirs
            for txt in target_dir.rglob("*.txt"):
                dest = target_dir / txt.name
                if txt != dest:
                    txt.rename(dest)
        elif inner_zips:
            inner_data = z.read(inner_zips[0])
            _extract_zip(inner_data, target_dir)
        else:
            z.extractall(target_dir)


def check_files() -> bool:
    """Return True if all 12 expected files exist."""
    return all((DATA_DIR / f).exists() for f in EXPECTED_FILES)


def download() -> None:
    """Download and extract C-MAPSS data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if check_files():
        print("All 12 C-MAPSS files already present. Skipping download.")
    else:
        print(f"Downloading C-MAPSS from:\n  {PRIMARY_URL}")
        try:
            with urllib.request.urlopen(PRIMARY_URL, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                data = b""
                downloaded = 0
                chunk_size = 65536
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    data += chunk
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)", end="")
                print()
        except Exception as exc:
            print(f"\n\nERROR: Download failed: {exc}")
            print(
                "\nPlease download the C-MAPSS dataset manually:\n"
                "  1. Go to: https://www.nasa.gov/intelligent-systems-division/"
                "discovery-and-systems-health/pcoe/pcoe-data-set-repository/\n"
                "  2. Find 'Turbofan Engine Degradation Simulation Data Set'\n"
                "  3. Download and extract to:  data/raw/cmapss/\n"
                "  Required files: train_FD001.txt … RUL_FD004.txt (12 files)\n"
            )
            sys.exit(1)

        print("Extracting …")
        _extract_zip(data, DATA_DIR)

    if not check_files():
        missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
        print(f"ERROR: Missing files after extraction: {missing}")
        sys.exit(1)

    print("\nFile verification (SHA-256):")
    download_date = datetime.now(UTC).strftime("%Y-%m-%d")
    for fname in EXPECTED_FILES:
        p = DATA_DIR / fname
        ck = sha256(p)
        size = p.stat().st_size
        print(f"  {fname:<22}  {ck}  ({size:,} bytes)")

    print(f"\nDownload date: {download_date}")
    print(f"Data directory: {DATA_DIR}")
    print("\nAll 12 C-MAPSS files verified. Update DATA_CARD.md with the checksums above.")


if __name__ == "__main__":
    download()
