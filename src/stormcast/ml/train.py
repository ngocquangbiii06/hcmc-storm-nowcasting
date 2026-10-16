"""Train the satellite -> radar U-Net and compare it with the cold-cloud baseline.

    python -m stormcast.ml.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from stormcast.config import Config, load_config
from stormcast.ml.baselines import cold_cloud_events, tune_bt_threshold
from stormcast.ml.dataset import SatToRadarDataset, band_stats, split_indices
from stormcast.ml.metrics import Contingency, fss
from stormcast.ml.unet import UNet

log = logging.getLogger(__name__)

# loss weight per dBZ bin: rare heavy echoes matter most for convection
WEIGHT_BINS = [(20.0, 2.0), (35.0, 5.0), (45.0, 10.0)]


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, max_dbz: float) -> torch.Tensor:
    w = torch.ones_like(y)
    for dbz, weight in WEIGHT_BINS:
        w = torch.where(y * max_dbz >= dbz, torch.full_like(y, weight), w)
    w = w * valid
    return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device, bt_thresholds: dict | None = None) -> dict:
    model.eval()
    thresholds = list(cfg.ml.thresholds_dbz)
    tables = {t: Contingency() for t in thresholds}
    base_tables = {t: Contingency() for t in thresholds}
    fss_35 = []
    for batch in loader:
        pred = model(batch["x"].to(device)).cpu().numpy()[:, 0] * cfg.ml.max_dbz
        obs = batch["dbz"].numpy()
        bt13 = batch["bt13"].numpy()
        valid = np.isfinite(obs)
        for t in thresholds:
            tables[t].update(pred, obs, t, valid)
            if bt_thresholds:
                base_tables[t].update_events(cold_cloud_events(bt13, bt_thresholds[t]), obs >= t, valid)
        fss_35 += [fss(p, o, 35.0, window=9) for p, o in zip(pred, obs, strict=True)]
    out = {f"model_{k}@{t:g}": v for t in thresholds for k, v in tables[t].as_dict().items()}
    out["model_fss@35_9km"] = float(np.nanmean(fss_35)) if fss_35 else float("nan")
    if bt_thresholds:
        out |= {f"baseline_{k}@{t:g}": v for t in thresholds for k, v in base_tables[t].as_dict().items()}
    return out


def tune_baseline(ds: xr.Dataset, train_idx: np.ndarray, cfg: Config, max_frames: int = 200) -> dict:
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(train_idx, size=min(max_frames, len(train_idx)), replace=False))
    bt13 = ds["bt"].sel(band="B13").isel(time=sample).values
    dbz = ds["dbz"].isel(time=sample).values
    return {t: tune_bt_threshold(bt13, dbz, t)[0] for t in cfg.ml.thresholds_dbz}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    m = cfg.ml
    epochs = args.epochs or m.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(m.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(m.zarr_path)
    times = ds["time"].values
    idx = {name: split_indices(times, *m.split[name], m.history_steps) for name in ("train", "val", "test")}
    log.info("frames: %s", {k: len(v) for k, v in idx.items()})
    mean, std = band_stats(ds, idx["train"])
    bt_thresholds = tune_baseline(ds, idx["train"], cfg)
    log.info("cold-cloud baseline thresholds (K): %s", bt_thresholds)

    def loader(name: str, shuffle: bool) -> DataLoader:
        data = SatToRadarDataset(m.zarr_path, idx[name], mean, std, m.history_steps, m.max_dbz)
        return DataLoader(data, batch_size=m.batch_size, shuffle=shuffle, num_workers=m.num_workers)

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)
    in_ch = len(cfg.himawari.bands) * (m.history_steps + 1)
    model = UNet(in_ch, 1, base=m.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=m.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    try:
        import mlflow

        mlflow.start_run(run_name=run_dir.name)
        mlflow.log_params({"history_steps": m.history_steps, "base_channels": m.base_channels, "lr": m.lr,
                           "batch_size": m.batch_size, "epochs": epochs})
    except ImportError:
        mlflow = None

    best_csi, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_dl:
            pred = model(batch["x"].to(device))
            loss = weighted_mse(pred, batch["y"].to(device), batch["valid"].to(device).float(), m.max_dbz)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate(model, val_dl, cfg, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))} | val
        history.append(row)
        log.info("epoch %d loss %.4f val CSI@35 %.3f", epoch, row["train_loss"], val["model_csi@35"])
        if mlflow:
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in row.items() if k != "epoch"}, step=epoch)
        if epoch == 1 or np.nan_to_num(val["model_csi@35"], nan=-1) > best_csi:
            best_csi = float(np.nan_to_num(val["model_csi@35"], nan=-1))
            torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test = evaluate(model, test_dl, cfg, device, bt_thresholds)
    results = {"test": test, "baseline_bt13_thresholds_k": bt_thresholds,
               "band_mean": mean.tolist(), "band_std": std.tolist(), "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    if mlflow:
        mlflow.log_metrics({f"test_{k.replace('@', '_at_')}": v for k, v in test.items()})
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        mlflow.end_run()
    log.info("test: %s", json.dumps(test, indent=2, default=float))


if __name__ == "__main__":
    main()
