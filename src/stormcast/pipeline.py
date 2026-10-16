"""Pipeline steps. Plain functions over (config, day) so the CLI and Dagster share one implementation."""
from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from stormcast import catalog
from stormcast.config import Config

log = logging.getLogger(__name__)

T_HIMAWARI_RAW = "himawari_raw"
T_RADAR_INDEX = "radar_index"
T_HIMAWARI_FRAMES = "himawari_frames"
T_RADAR_FRAMES = "radar_frames"
T_PAIRS = "pairs"


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"] if not df.empty else df


def _run_pool(fn, jobs: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        return [fn(*job) for job in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn, *zip(*jobs, strict=True)))


# --- bronze -------------------------------------------------------------------------------------

def himawari_download(cfg: Config, day: date) -> pd.DataFrame:
    from stormcast.ingest.himawari import download_day

    df = download_day(cfg, day)
    catalog.write_partition(df, cfg.paths.catalog, T_HIMAWARI_RAW, day)
    return df


def radar_index(cfg: Config, archive: Path | None = None) -> pd.DataFrame:
    """Index all radar volumes once, then split the catalog by day."""
    from stormcast.ingest.radar import extract_archive, index_radar_files

    raw_dir = Path(cfg.radar.raw_dir)
    if archive is not None:
        extract_archive(Path(archive), raw_dir)
    df = index_radar_files(cfg.radar, raw_dir)
    ok = df.dropna(subset=["scan_time"])
    for day, part in ok.groupby(ok["scan_time"].dt.date):
        catalog.write_partition(part, cfg.paths.catalog, T_RADAR_INDEX, day)
    return df


# --- silver -------------------------------------------------------------------------------------

def himawari_process(cfg: Config, day: date) -> pd.DataFrame:
    from stormcast.process.himawari_grid import process_timestep

    raw = catalog.read_table(cfg.paths.catalog, T_HIMAWARI_RAW, [day])
    if raw.empty:
        log.warning("no himawari raw catalog for %s - run himawari-download first", day)
        return raw
    raw = raw[raw["status"].isin(["cached", "downloaded"])]
    root = cfg.paths.root
    jobs = [
        (cfg, t, sorted(str(catalog.resolve(p, root)) for p in g["local_path"])) for t, g in raw.groupby("nominal_time")
    ]
    t0 = time.perf_counter()
    df = pd.DataFrame(_run_pool(process_timestep, jobs, cfg.processing.workers))
    log.info("himawari %s: %d timesteps in %.1fs", day, len(df), time.perf_counter() - t0)
    catalog.write_partition(df, cfg.paths.catalog, T_HIMAWARI_FRAMES, day)
    if not cfg.himawari.keep_raw:
        # also drops files of failed/incomplete timesteps, so a bad day can never fill the disk
        shutil.rmtree(cfg.paths.bronze / "himawari" / f"{day:%Y/%m/%d}", ignore_errors=True)
    return df


def radar_process(cfg: Config, day: date) -> pd.DataFrame:
    from stormcast.process.radar_cappi import process_file

    idx = catalog.read_table(cfg.paths.catalog, T_RADAR_INDEX, [day])
    if idx.empty:
        log.warning("no radar index for %s - run radar-index first", day)
        return idx
    idx = idx[idx["status"] == "ok"]
    jobs = [(cfg, r.path, pd.Timestamp(r.scan_time)) for r in idx.itertuples(index=False)]
    t0 = time.perf_counter()
    df = pd.DataFrame(_run_pool(process_file, jobs, cfg.processing.workers))
    log.info("radar %s: %d volumes in %.1fs", day, len(df), time.perf_counter() - t0)
    catalog.write_partition(df, cfg.paths.catalog, T_RADAR_FRAMES, day)
    return df


# --- gold ---------------------------------------------------------------------------------------

def align(cfg: Config, day: date) -> pd.DataFrame:
    from stormcast.align import pair_frames

    sat = _ok(catalog.read_table(cfg.paths.catalog, T_HIMAWARI_FRAMES, [day]))
    # scans around midnight can pair with the first/last satellite frame of the day
    neighbours = [day - timedelta(days=1), day, day + timedelta(days=1)]
    radar = _ok(catalog.read_table(cfg.paths.catalog, T_RADAR_FRAMES, neighbours))
    pairs = pair_frames(sat, radar, cfg.align.tolerance_minutes)
    catalog.write_partition(pairs, cfg.paths.catalog, T_PAIRS, day)
    log.info("align %s: %d pairs from %d sat / %d radar frames", day, len(pairs), len(sat), len(radar))
    return pairs


def build(cfg: Config) -> Path:
    from stormcast.build_dataset import build_zarr

    pairs = catalog.read_table(cfg.paths.catalog, T_PAIRS, cfg.period.days())
    if pairs.empty:
        raise RuntimeError("no pairs - run align first")
    return build_zarr(cfg, pairs)


def build_sat(cfg: Config) -> Path:
    """Himawari-only gold dataset (satellite nowcasting); needs no radar."""
    from stormcast.build_dataset import build_sat_zarr

    frames = _ok(catalog.read_table(cfg.paths.catalog, T_HIMAWARI_FRAMES, cfg.period.days()))
    if frames.empty:
        raise RuntimeError("no processed Himawari frames - run himawari-process first")
    return build_sat_zarr(cfg, frames)


def run_day(cfg: Config, day: date) -> None:
    himawari_download(cfg, day)
    himawari_process(cfg, day)
    radar_process(cfg, day)
    align(cfg, day)
