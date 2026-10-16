"""Bronze layer for radar: unpack the archive from the supervisor and index every volume by scan time."""
from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from stormcast.config import RadarConfig

log = logging.getLogger(__name__)

SKIP_SUFFIXES = {".txt", ".md", ".json", ".csv", ".xml", ".log", ".zip", ".tar", ".gz", ".rar", ".7z"}


def extract_archive(archive: Path, dest: Path) -> None:
    """Unpack zip/tar archives. RAR/7z: extract manually with 7-Zip into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() in {".rar", ".7z"}:
        raise RuntimeError(f"Extract {archive} with 7-Zip into {dest}, then run `stormcast radar-index`.")
    shutil.unpack_archive(str(archive), str(dest))


def time_from_filename(name: str, cfg: RadarConfig) -> datetime | None:
    m = re.search(cfg.filename_time_regex, name)
    if not m:
        return None
    try:
        local = datetime.strptime(m.group(1), cfg.filename_time_format)
    except ValueError:
        return None
    return local - timedelta(hours=cfg.filename_utc_offset_hours)


def time_from_file(path: Path, reader: str) -> datetime:
    """Slow path: open the volume with Py-ART and read the time of the first ray (UTC)."""
    import pyart

    from stormcast.process.radar_cappi import read_radar

    radar = read_radar(path, reader)
    return pd.Timestamp(str(pyart.util.datetime_from_radar(radar))).to_pydatetime()


def index_radar_files(cfg: RadarConfig, raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir or cfg.raw_dir)
    rows = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        scan_time, source, status = time_from_filename(path.name, cfg), "filename", "ok"
        if scan_time is None:
            try:
                scan_time, source = time_from_file(path, cfg.reader), "header"
            except Exception as exc:  # noqa: BLE001
                scan_time, source, status = None, "none", f"error: {exc}"
        rows.append({"path": str(path), "scan_time": scan_time, "time_source": source, "status": status})
    df = pd.DataFrame(rows, columns=["path", "scan_time", "time_source", "status"])
    df["scan_time"] = pd.to_datetime(df["scan_time"])
    log.info("indexed %d radar files (%d unreadable)", len(df), (df["status"] != "ok").sum())
    return df.sort_values("scan_time", na_position="last").reset_index(drop=True)
