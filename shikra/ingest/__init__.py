"""Mock e-MMS / IMMOLS / HUMS / BRD ingestion into unified SQLite model."""

from __future__ import annotations

from shikra.ingest.brd import BRDAdapter
from shikra.ingest.emms import AdapterResult, EMMSAdapter
from shikra.ingest.hums import HUMSAdapter
from shikra.ingest.immols import IMMOLSAdapter
from shikra.ingest.pipeline import generate_reports, load_all
from shikra.ingest.schema import init_db

__all__ = [
    "AdapterResult",
    "BRDAdapter",
    "EMMSAdapter",
    "HUMSAdapter",
    "IMMOLSAdapter",
    "generate_reports",
    "init_db",
    "load_all",
]
