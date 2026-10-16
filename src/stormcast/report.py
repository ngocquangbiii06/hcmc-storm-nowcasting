"""Markdown data-quality report built from the catalog (the numbers you quote in the README / CV)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stormcast import catalog
from stormcast.config import Config
from stormcast.pipeline import T_HIMAWARI_FRAMES, T_HIMAWARI_RAW, T_PAIRS, T_RADAR_FRAMES, T_RADAR_INDEX
from stormcast.quality import gaps, missing_timesteps


def _status_table(df: pd.DataFrame) -> str:
    if df.empty or "status" not in df:
        return "_no data_\n"
    counts = df["status"].fillna("unknown").value_counts()
    lines = ["| status | frames | share |", "|---|---:|---:|"]
    lines += [f"| {k} | {v} | {v / len(df):.1%} |" for k, v in counts.items()]
    return "\n".join(lines) + "\n"


def build_report(cfg: Config) -> Path:
    days = cfg.period.days()
    raw = catalog.read_table(cfg.paths.catalog, T_HIMAWARI_RAW, days)
    sat = catalog.read_table(cfg.paths.catalog, T_HIMAWARI_FRAMES, days)
    rindex = catalog.read_table(cfg.paths.catalog, T_RADAR_INDEX, days)
    radar = catalog.read_table(cfg.paths.catalog, T_RADAR_FRAMES, days)
    pairs = catalog.read_table(cfg.paths.catalog, T_PAIRS, days)

    out = [f"# Data quality report ({cfg.period.start} to {cfg.period.end})\n"]

    out.append("## Himawari-9\n")
    if not raw.empty:
        out.append(
            f"- Raw files: {len(raw)} ({raw['size_bytes'].sum() / 1e9:.2f} GB downloaded, "
            f"segments {sorted(raw['segment'].unique().tolist())} only)\n"
        )
    out.append(_status_table(sat))
    if not sat.empty:
        freq = cfg.quality.himawari_freq_minutes
        missing = missing_timesteps(sat["nominal_time"], cfg.period.start, cfg.period.end, freq)
        out.append(f"\n- Missing 10-min timesteps: {len(missing)}\n")
        ok = sat[sat["status"] == "ok"]
        if not ok.empty:
            offset = (pd.to_datetime(ok["obs_time"]) - pd.to_datetime(ok["nominal_time"])).dt.total_seconds() / 60
            out.append(f"- Observation time minus nominal time: median {offset.median():.1f} min\n")

    out.append("\n## Radar Nha Be\n")
    if not rindex.empty:
        sources = rindex["time_source"].value_counts().to_dict()
        out.append(f"- Volumes indexed: {len(rindex)} (time source: {sources})\n")
    out.append(_status_table(radar))
    if not radar.empty:
        g = gaps(radar.loc[radar["status"] == "ok", "scan_time"], cfg.quality.radar_max_gap_minutes)
        out.append(f"\n- Gaps longer than {cfg.quality.radar_max_gap_minutes} min: {len(g)}")
        if not g.empty:
            out.append(f" (longest {g['gap_minutes'].max():.0f} min)")
        out.append("\n")

    out.append("\n## Paired dataset\n")
    if not pairs.empty:
        out.append(f"- Pairs: {len(pairs)}\n")
        out.append(f"- |radar - satellite| time difference: median {pairs['dt_seconds'].abs().median() / 60:.1f} min, "
                   f"max {pairs['dt_seconds'].abs().max() / 60:.1f} min\n")
    else:
        out.append("_no pairs yet_\n")

    path = Path(cfg.paths.reports) / "data_quality.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(out), encoding="utf-8")
    return path
