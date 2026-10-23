# HCMC Storm Nowcasting

**An end-to-end data pipeline and deep-learning nowcasting model for convective clouds over Ho Chi Minh City, built from raw Himawari-9 geostationary satellite data.**

Full-disk Himawari-9 files are pulled from the NOAA open-data bucket on AWS, cut down to the few segments that cover southern Vietnam, calibrated to brightness temperature, regridded onto a 256×256 km grid centred on the Nha Be weather radar, quality-checked and published as a chunked Zarr dataset. A U-Net then forecasts the next 60 minutes of cloud-top temperature, and its skill at predicting deep-convective (cold) cloud is compared against persistence and optical-flow extrapolation.

The pipeline also contains a radar branch (raw radar volumes → CAPPI with Py-ART, paired with the satellite frames), which is tested on synthetic volumes and ready for when radar data is available.

> Status: August 2025 fully processed (4,393 frames, 100% QC pass) and a U-Net trained and evaluated against two baselines; see [Results](#results).

---

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        S3[(NOAA S3<br/>noaa-himawari9)]
        RAW[(Weather radar<br/>raw volumes, optional)]
    end
    subgraph Bronze
        HB[HSD segments 4-5<br/>B08, B13-B16]
        RB[Radar index]
    end
    subgraph Silver
        HS[Brightness temperature<br/>5 bands, 256x256]
        RS[CAPPI 2 km<br/>256x256]
    end
    subgraph Gold
        ZS[(Himawari Zarr)]
        ZP[(Radar + satellite<br/>pairs Zarr)]
    end
    S3 -->|boto3 + segment selection| HB -->|SatPy| HS --> ZS --> NC[U-Net nowcast<br/>+0 to +60 min]
    RAW --> RB -->|Py-ART| RS
    HS & RS --> ZP --> RT[U-Net<br/>satellite to radar]
    HS & RS -.-> C[(Parquet catalog<br/>+ QC report)]
```

Orchestrated as daily-partitioned **Dagster** assets; the same steps run from a plain CLI and from Kaggle/Colab notebooks (the heavy download runs next to the data, only the regridded output comes back).

## Engineering highlights

| Problem | Solution |
|---|---|
| A full-disk scan for 5 bands is ~550 GB/month | Compute the needed segments from the satellite's fixed-grid navigation equations, so only segments 4-5 are downloaded (**~110 GB/month, 80% less**); keep only regridded output (~1% of raw) |
| Slow residential link to us-east-1 (~30 KB/s per stream) | Ingest runs on Kaggle/Colab next to the data; raw files are deleted right after processing and only the regridded output is kept |
| A satellite file timestamp is not the observation time | The disk is scanned north to south; HCMC is observed **~4 min after** the nominal time (measured), so the real observation time is stored and used for alignment |
| Missing / corrupt files (e.g. 141/144 timesteps on 2025-08-01) | Frame-level QC (range, NaN fraction, flat fields, time gaps) in a Parquet catalog; forecast windows that span a gap are dropped |
| Train/test leakage in time series | Day-blocked splits, and each input+target window must lie entirely inside one split |
| Re-runs | Idempotent steps: cached downloads, skip-if-exists outputs, per-day catalog partitions, atomic writes, catalog paths relative to the data root (portable between Colab and local) |

## Results

### Data pipeline (August 2025, one Kaggle CPU run)

| Metric | Value |
|---|---|
| Raw Himawari files ingested | 43,930 files, **113.5 GB** (segments 4-5 only; ~550 GB for the full disk) |
| Frames passing QC | **4,393 / 4,393 (100%)** over 31 days |
| Timesteps missing upstream on S3 | 71 of 4,464 (recorded, never fatal) |
| Observation time minus file timestamp | **3.9 min** (median), used for alignment |
| Output kept | 3.1 GB silver NetCDF + 3.3 GB gold Zarr (~6% of raw) |
| Throughput | ~45 s download + ~10 min processing per day on 4 CPUs; 5.5 h for the month |

### Nowcasting skill (held-out days 27–31 Aug, 591 forecast windows)

CSI for cold cloud tops, higher is better (B13 ≤ 235 K: convective cloud, ≤ 220 K: deep convection):

| Method | +10 min | +30 min | +60 min | +60 min, ≤ 220 K | MAE +60 min (K) |
|---|---|---|---|---|---|
| Persistence | 0.839 | 0.678 | 0.539 | 0.420 | 12.4 |
| Optical flow (Farneback) | 0.873 | 0.713 | 0.557 | 0.442 | 12.0 |
| **U-Net** (4 input frames × 5 bands, residual) | **0.913** | **0.806** | **0.678** | **0.601** | **8.4** |

The U-Net beats optical flow at every lead time, and the margin grows with lead time (+0.12 CSI at +60 min,
+0.16 for deep convection): extrapolation can move clouds but not grow or decay them, which is what the
network learns. Trained 30 epochs on a Kaggle T4 (~82 s/epoch); one month of wet-season data, so treat the
numbers as a proof of concept.

## Quickstart

```bash
mamba env create -f environment.yml && conda activate stormcast
pip install -e ".[dev,orchestration,ml]"
pytest

stormcast segments                                                   # -> [4, 5]
stormcast himawari-download --start 2025-08-01 --end 2025-08-01
stormcast himawari-process  --start 2025-08-01 --end 2025-08-01
stormcast build-sat && stormcast report
python -m stormcast.ml.train_nowcast

dagster dev -f src/stormcast/orchestration/dagster_defs.py           # UI with lineage + backfills
```

On a slow connection, run next to the data instead:
- **Kaggle** (runs in the background): [notebooks/kaggle_himawari.ipynb](notebooks/kaggle_himawari.ipynb) (CPU, internet on) → add its output as input to [notebooks/kaggle_train_nowcast.ipynb](notebooks/kaggle_train_nowcast.ipynb) (GPU T4).
- **Colab**: `python scripts/package_for_colab.py`, upload `dist/stormcast-code.zip` to `MyDrive/stormcast/`, run [notebooks/colab_himawari.ipynb](notebooks/colab_himawari.ipynb) then [notebooks/colab_train_nowcast.ipynb](notebooks/colab_train_nowcast.ipynb).

Radar branch (when data is available): `stormcast radar-index --archive radar.zip`, `radar-process`, `align`, `build`, then `python -m stormcast.ml.train`.

All settings (grid, bands, tolerances, QC thresholds, splits, model) live in [configs/default.yaml](configs/default.yaml).

## Repository layout

```
configs/default.yaml           pipeline + training configuration
src/stormcast/
  geos.py                      Himawari fixed-grid navigation -> which segments to download
  grid.py                      the shared 256x256 target grid
  ingest/                      S3 download (Himawari), archive index (radar)      [bronze]
  process/                     SatPy calibration + resampling, Py-ART CAPPI       [silver]
  align.py, build_dataset.py   time matching, Zarr writers                        [gold]
  quality.py, report.py        QC rules + markdown data-quality report
  catalog.py                   per-day Parquet catalog (query with DuckDB)
  pipeline.py, cli.py          steps + command line
  orchestration/dagster_defs.py
  ml/nowcast.py, train_nowcast.py   satellite nowcasting (U-Net vs persistence / optical flow)
  ml/dataset.py, train.py           satellite -> radar retrieval
  ml/metrics.py, baselines.py, unet.py
notebooks/                     Kaggle / Colab ingest (CPU) and training (GPU)
scripts/                       quick-look figures, Colab packaging
tests/                         unit + integration tests (incl. synthetic radar and a training smoke run)
```

## Data

- **Himawari-9 AHI L1b**: NOAA Open Data Dissemination, `s3://noaa-himawari9` (public, no credentials).
- Grid centre: Nha Be weather radar site (10.6596°N, 106.7283°E).

## Limitations

- Cloud-top temperature is a proxy for convection; there are no lightning or rain-gauge labels in this version.
- One month (August 2025, wet season) to start with; more months are a config change away.
- No parallax correction yet: tall storm tops appear displaced by a few km at this viewing angle.

## Tech stack

Python · SatPy / pyresample · Py-ART · xarray · Zarr · Dask · boto3 · Parquet / DuckDB · Dagster · Docker · GitHub Actions · PyTorch · OpenCV · MLflow · Google Colab
