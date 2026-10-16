"""Bronze layer: download only the Himawari-9 HSD files we need from the public NOAA S3 bucket.

Key layout:  AHI-L1b-FLDK/YYYY/MM/DD/HHMM/HS_H09_YYYYMMDD_HHMM_Bbb_FLDK_Rrr_SssTT.DAT.bz2
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from stormcast.catalog import resolve, to_rel
from stormcast.config import Config
from stormcast.geos import segments_for_bbox
from stormcast.grid import approx_bbox

log = logging.getLogger(__name__)

KEY_RE = re.compile(
    r"HS_(?P<sat>H\d{2})_(?P<date>\d{8})_(?P<hhmm>\d{4})_(?P<band>B\d{2})_FLDK_"
    r"(?P<res>R\d{2})_S(?P<seg>\d{2})(?P<nseg>\d{2})\.DAT\.bz2$"
)


def _s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    # Public bucket: anonymous access, generous connection pool for threaded downloads
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED, max_pool_connections=64))


def wanted_segments(cfg: Config) -> list[int]:
    if cfg.himawari.segments != "auto":
        return [int(s) for s in cfg.himawari.segments]
    return segments_for_bbox(*approx_bbox(cfg.grid), resolution=cfg.himawari.resolution)


def list_day(cfg: Config, day: date, client=None) -> pd.DataFrame:
    """List the S3 objects for one day, filtered to our bands / resolution / segments."""
    client = client or _s3_client()
    bands = set(cfg.himawari.bands)
    segments = set(wanted_segments(cfg))
    prefix = f"{cfg.himawari.product}/{day:%Y/%m/%d}/"
    rows = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=cfg.himawari.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            m = KEY_RE.search(obj["Key"])
            if not m:
                continue
            if m["band"] not in bands or m["res"] != cfg.himawari.resolution or int(m["seg"]) not in segments:
                continue
            rows.append(
                {
                    "key": obj["Key"],
                    "nominal_time": pd.Timestamp(f"{m['date']} {m['hhmm']}"),
                    "band": m["band"],
                    "segment": int(m["seg"]),
                    "size_bytes": int(obj["Size"]),
                }
            )
    return pd.DataFrame(rows)


def _download_one(client, bucket: str, key: str, dest: Path, size: int) -> str:
    if dest.exists() and dest.stat().st_size == size:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        client.download_file(bucket, key, str(tmp))
        tmp.replace(dest)
        return "downloaded"
    except Exception as exc:  # noqa: BLE001 - recorded in the catalog, retried on next run
        log.warning("download failed %s: %s", key, exc)
        tmp.unlink(missing_ok=True)
        return f"error: {exc}"


def download_day(cfg: Config, day: date) -> pd.DataFrame:
    """Download one day of files (idempotent: files already on disk with the right size are skipped)."""
    client = _s3_client()
    listing = list_day(cfg, day, client)
    if listing.empty:
        log.warning("no Himawari objects found for %s", day)
        return listing
    root = cfg.paths.bronze / "himawari"
    listing["local_path"] = [
        to_rel(root / k.split(f"{cfg.himawari.product}/", 1)[1], cfg.paths.root) for k in listing["key"]
    ]
    with ThreadPoolExecutor(cfg.himawari.download_workers) as pool:
        listing["status"] = list(
            pool.map(
                lambda r: _download_one(
                    client, cfg.himawari.bucket, r.key, resolve(r.local_path, cfg.paths.root), r.size_bytes
                ),
                listing.itertuples(index=False),
            )
        )
    n_ok = listing["status"].isin(["cached", "downloaded"]).sum()
    log.info("%s: %d/%d files ok (%.1f MB)", day, n_ok, len(listing), listing["size_bytes"].sum() / 1e6)
    return listing
