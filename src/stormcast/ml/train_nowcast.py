"""Train the satellite nowcasting U-Net and compare it with persistence and optical-flow baselines.

    python -m stormcast.ml.train_nowcast --config configs/default.yaml [--epochs 2] [--max-train-batches 20]

Input : `input_steps` past frames of all 5 IR bands (t-30 ... t)
Output: B13 brightness temperature at t+10 ... t+60 min, predicted as a *change* from the last frame
Score : MAE (K) and CSI of cold-cloud events (BT <= 235 K convective cloud, <= 220 K deep convection)
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
from stormcast.ml.baselines import optical_flow_forecast, persistence
from stormcast.ml.dataset import band_stats
from stormcast.ml.metrics import Contingency
from stormcast.ml.nowcast import SatNowcastDataset, window_indices
from stormcast.ml.unet import UNet

log = logging.getLogger(__name__)
COLD_WEIGHT = 4.0  # extra loss weight on cold-cloud pixels, which are rare but are what matters


def weighted_l1(pred: torch.Tensor, batch: dict, cold_k: float, device) -> torch.Tensor:
    valid = batch["valid"].to(device).float()
    w = valid * (1 + COLD_WEIGHT * (batch["target"].to(device) <= cold_k).float())
    return (w * (pred - batch["y_norm"].to(device)).abs()).sum() / w.sum().clamp_min(1.0)


def forward(model, batch: dict, device) -> torch.Tensor:
    """Residual forecast in normalised units: last frame + learned change."""
    return batch["last_norm"].to(device)[:, None] + model(batch["x"].to(device))


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device, mean_b: float, std_b: float, baselines: bool) -> dict:
    model.eval()
    leads, thresholds = cfg.ml.lead_steps, list(cfg.ml.bt_thresholds_k)
    methods = ["unet"] + (["persistence", "optical_flow"] if baselines else [])
    abs_err = {m: np.zeros(leads) for m in methods}
    count = {m: np.zeros(leads) for m in methods}
    tables = {m: {t: [Contingency() for _ in range(leads)] for t in thresholds} for m in methods}
    for batch in loader:
        target, valid = batch["target"].numpy(), batch["valid"].numpy()
        forecasts = {"unet": forward(model, batch, device).cpu().numpy() * std_b + mean_b}
        if baselines:
            last, prev = batch["last"].numpy(), batch["prev"].numpy()
            forecasts["persistence"] = np.stack([persistence(a, leads) for a in last])
            forecasts["optical_flow"] = np.stack(
                [optical_flow_forecast(p, a, leads) for p, a in zip(prev, last, strict=True)]
            )
        for m, f in forecasts.items():
            ok = valid & np.isfinite(f)
            abs_err[m] += np.where(ok, np.abs(f - target), 0).sum(axis=(0, 2, 3))
            count[m] += ok.sum(axis=(0, 2, 3))
            for t in thresholds:
                for j in range(leads):
                    tables[m][t][j].update_events(f[:, j] <= t, target[:, j] <= t, ok[:, j])
    return {
        m: {
            f"+{10 * (j + 1)}min": {"mae_k": float(abs_err[m][j] / max(count[m][j], 1))}
            | {f"csi@{t:g}K": tables[m][t][j].csi for t in thresholds}
            for j in range(leads)
        }
        for m in methods
    }


def mean_csi(result: dict, threshold: float) -> float:
    return float(np.nanmean([v[f"csi@{threshold:g}K"] for v in result["unet"].values()]))


def plot_skill(results: dict, thresholds, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = ["mae_k"] + [f"csi@{t:g}K" for t in thresholds]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), constrained_layout=True)
    for ax, metric in zip(axes, metrics, strict=True):
        for method, per_lead in results.items():
            leads = [int(k.strip("+min")) for k in per_lead]
            ax.plot(leads, [v[metric] for v in per_lead.values()], marker="o", label=method)
        ax.set_xlabel("lead time (min)")
        ax.set_title(metric.replace("mae_k", "MAE (K), lower is better").replace("csi@", "CSI, BT <= "))
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(path, dpi=120)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int, help="cap batches per epoch (quick smoke runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    m = cfg.ml
    if m.input_steps < 2:
        raise ValueError("input_steps must be >= 2 (optical flow needs two frames)")
    epochs = args.epochs or m.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(m.output_dir) / f"nowcast-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(m.sat_zarr_path)
    times = ds["time"].values
    idx = {s: window_indices(times, *m.split[s], m.input_steps, m.lead_steps) for s in ("train", "val", "test")}
    log.info("windows: %s (device %s)", {k: len(v) for k, v in idx.items()}, device)
    mean, std = band_stats(ds, idx["train"])
    b = [str(x) for x in ds["band"].values].index(m.target_band)
    bt = ds["bt"].astype("float16").values if m.preload else None

    def loader(split: str, shuffle: bool) -> DataLoader:
        data = SatNowcastDataset(m.sat_zarr_path, idx[split], mean, std, m.input_steps, m.lead_steps, m.target_band, bt)
        return DataLoader(data, batch_size=m.batch_size, shuffle=shuffle, num_workers=m.num_workers)

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)
    model = UNet(len(cfg.himawari.bands) * m.input_steps, m.lead_steps, base=m.base_channels, sigmoid_output=False)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=m.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cold_k = m.bt_thresholds_k[0]

    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses, t0 = [], time.perf_counter()
        for n, batch in enumerate(train_dl):
            if args.max_train_batches and n >= args.max_train_batches:
                break
            loss = weighted_l1(forward(model, batch, device), batch, cold_k, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate(model, val_dl, cfg, device, float(mean[b]), float(std[b]), baselines=False)
        score = mean_csi(val, cold_k)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_csi": score})
        log.info("epoch %d  loss %.4f  val mean CSI@%gK %.3f  (%.0fs)", epoch, np.mean(losses), cold_k, score,
                 time.perf_counter() - t0)
        if epoch == 1 or np.nan_to_num(score, nan=-1) > best:
            best = float(np.nan_to_num(score, nan=-1))
            torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test = evaluate(model, test_dl, cfg, device, float(mean[b]), float(std[b]), baselines=True)
    results = {"test": test, "history": history, "band_mean": mean.tolist(), "band_std": std.tolist(),
               "windows": {k: len(v) for k, v in idx.items()}, "config": {"input_steps": m.input_steps,
               "lead_steps": m.lead_steps, "base_channels": m.base_channels, "epochs": epochs}}
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    plot_skill(test, m.bt_thresholds_k, run_dir / "skill_vs_lead.png")
    for method, per_lead in test.items():
        log.info("%-13s %s", method, "  ".join(f"{k}: CSI@{cold_k:g} {v[f'csi@{cold_k:g}K']:.3f}"
                                                for k, v in per_lead.items()))
    log.info("results in %s", run_dir)


if __name__ == "__main__":
    main()
