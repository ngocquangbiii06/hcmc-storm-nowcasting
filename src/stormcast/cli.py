"""Command line entry point:  stormcast <step> [--start YYYY-MM-DD --end YYYY-MM-DD]"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from stormcast import pipeline
from stormcast.config import load_config

DAILY_STEPS = {
    "himawari-download": pipeline.himawari_download,
    "himawari-process": pipeline.himawari_process,
    "radar-process": pipeline.radar_process,
    "align": pipeline.align,
    "run-day": pipeline.run_day,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stormcast", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="step", required=True)

    for name in DAILY_STEPS:
        p = sub.add_parser(name)
        p.add_argument("--start", help="first day (default: config period.start)")
        p.add_argument("--end", help="last day, inclusive (default: config period.end)")

    p = sub.add_parser("radar-index", help="index radar volumes (optionally unpack an archive first)")
    p.add_argument("--archive", help="zip/tar file downloaded from Google Drive")
    sub.add_parser("build", help="write the gold radar+satellite Zarr dataset from all pairs")
    sub.add_parser("build-sat", help="write the gold Himawari-only Zarr dataset (no radar needed)")
    sub.add_parser("report", help="write reports/data_quality.md")
    sub.add_parser("segments", help="print the Himawari segments covering the grid")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)

    if args.step in DAILY_STEPS:
        period = replace(cfg.period, start=args.start or cfg.period.start, end=args.end or cfg.period.end)
        for day in period.days():
            DAILY_STEPS[args.step](cfg, day)
    elif args.step == "radar-index":
        pipeline.radar_index(cfg, args.archive)
    elif args.step == "build":
        print(pipeline.build(cfg))
    elif args.step == "build-sat":
        print(pipeline.build_sat(cfg))
    elif args.step == "report":
        from stormcast.report import build_report

        print(build_report(cfg))
    elif args.step == "segments":
        from stormcast.ingest.himawari import wanted_segments

        print(wanted_segments(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
