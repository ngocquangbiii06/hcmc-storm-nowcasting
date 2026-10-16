"""Side-by-side quick look of one paired frame: B13 brightness temperature vs radar CAPPI.

    python scripts/quicklook.py --time "2025-08-15 09:00" [--zarr data/gold/hcmc_aug2025.zarr] [--out reports/fig.png]

The first thing to check after building silver/gold: storm cells must line up between the
two panels. If the radar looks mirrored north-south, the grid flip is wrong; if it is
shifted by ~7 h of weather, the radar file names are local time (set filename_utc_offset_hours).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--time", required=True)
    p.add_argument("--zarr", default="data/gold/hcmc_aug2025.zarr")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = xr.open_zarr(args.zarr).sel(time=pd.Timestamp(args.time), method="nearest")
    extent = [float(ds.x.min()) / 1e3, float(ds.x.max()) / 1e3, float(ds.y.min()) / 1e3, float(ds.y.max()) / 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    im0 = axes[0].imshow(ds["bt"].sel(band="B13"), cmap="gray_r", vmin=190, vmax=300, extent=extent)
    axes[0].set_title(f"Himawari-9 B13 BT (K)\nobs {pd.Timestamp(ds.sat_obs_time.values):%H:%M} UTC")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    dbz = np.where(ds["dbz"] >= 5, ds["dbz"], np.nan)
    im1 = axes[1].imshow(dbz, cmap="turbo", vmin=5, vmax=65, extent=extent)
    axes[1].set_title(f"Radar CAPPI (dBZ)\nscan {pd.Timestamp(ds.radar_time.values):%H:%M} UTC")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    for ax in axes:
        ax.plot(0, 0, "k+", ms=12)  # Nha Be radar
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    fig.suptitle(f"{pd.Timestamp(ds.time.values):%Y-%m-%d %H:%M} UTC")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
