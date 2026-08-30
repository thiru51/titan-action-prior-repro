from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baselines import constant_velocity
from .config import Config
from .data.common import collate, denormalize_boxes, normalize_boxes
from .device import (
    Precision,
    autocast,
    describe,
    make_scaler,
    peak_memory_gb,
    reset_peak_memory,
    resolve_batch_size,
    resolve_device,
    resolve_num_workers,
    select_precision,
    setup_backends,
    synchronize,
)
from .losses import UncertaintyWeightedActionLoss, ego_motion_loss, masked_trajectory_loss
from .metrics import ForecastMetrics
from .models.titan_net import TitanNet


def build_dataset(cfg: Config, split: str):
    if cfg.synthetic.enabled:
        from .data.synthetic import SyntheticTitanDataset

        return SyntheticTitanDataset(cfg.data, split, cfg.synthetic, load_video=cfg.priors.action)
    from .data.titan import TitanForecastDataset

    # Skip image IO entirely when the action prior is off; decoding PNG crops
    # dominates loading time and nothing downstream consumes the tubes.
    data_cfg = dataclasses.replace(cfg.data, load_video=cfg.data.load_video and cfg.priors.action)
    return TitanForecastDataset(data_cfg, split)


def build_loader(cfg: Config, split: str, shuffle: bool, device: torch.device) -> DataLoader:
    ds = build_dataset(cfg, split)
    workers = resolve_num_workers(cfg.data.num_workers)

    extra = {}
    if workers > 0:
        # Worker startup costs real time because torch re-imports in every
        # process; keeping them alive matters when epochs are short.
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = cfg.data.prefetch_factor

    return DataLoader(
        ds,
        batch_size=resolve_batch_size(cfg.train.batch_size, device),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=device.type == "cuda",
        **extra,
    )


def is_synthetic(loader: DataLoader) -> bool:
    return getattr(loader.dataset, "is_synthetic", False)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _step(
    net,
    batch: dict,
    cfg: Config,
    action_loss_fn,
    action_log_var: torch.Tensor | None,
    device: torch.device,
    precision: Precision,
) -> tuple[torch.Tensor, dict]:
    # Forward and loss both run under autocast; the optimiser step and the
    # metrics deliberately do not.
    with autocast(device, precision):
        out = net(batch)

        target_norm = normalize_boxes(batch["fut_boxes"])
        valid = batch["valid_fut"] & batch["agent_mask"].unsqueeze(-1)

        traj = masked_trajectory_loss(out["pred_boxes"], target_norm, valid)
        total = traj
        parts = {"traj": traj.detach().item()}

        if cfg.priors.ego:
            ego = ego_motion_loss(out["ego_pred"], batch["fut_ego"])
            total = total + cfg.train.ego_loss_weight * ego
            parts["ego"] = ego.detach().item()

        if cfg.priors.action and out["action_logits"]:
            act = action_loss_fn(
                out["action_logits"], batch["actions"], batch["agent_mask"], action_log_var
            )
            total = total + cfg.train.action_loss_weight * act
            parts["action"] = act.detach().item()

    parts["total"] = total.detach().item()
    return total, parts


@torch.no_grad()
def evaluate(
    net,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    precision: Precision | None = None,
) -> dict:
    precision = precision or select_precision(device, False)
    net.eval()
    metrics = ForecastMetrics()
    cv_metrics = ForecastMetrics()

    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device, precision):
            out = net(batch)
        valid = batch["valid_fut"] & batch["agent_mask"].unsqueeze(-1)

        # ADE/FDE are reported numbers, so they leave autocast immediately: back
        # to fp32 first, then out of normalised space into pixels.
        pred_px = denormalize_boxes(out["pred_boxes"].float())
        metrics.update(pred_px, batch["fut_boxes"].float(), valid)

        cv = constant_velocity(batch["obs_boxes"].float(), cfg.data.pred_len)
        cv_metrics.update(denormalize_boxes(cv), batch["fut_boxes"].float(), valid)

    res = metrics.compute()
    cv_res = cv_metrics.compute()
    res["const_vel_FDE"] = cv_res["FDE"]
    res["const_vel_ADE"] = cv_res["ADE"]
    return res


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    setup_backends(device)
    precision = select_precision(device, cfg.train.amp)

    train_loader = build_loader(cfg, "train", shuffle=True, device=device)
    val_loader = build_loader(cfg, "val", shuffle=False, device=device)
    synthetic = is_synthetic(train_loader)

    model = TitanNet(cfg).to(device)
    # Keep a handle on the uncompiled module: torch.compile returns a wrapper
    # that does not forward attribute access to things like action_log_var.
    net = torch.compile(model) if cfg.train.compile else model

    action_loss_fn = UncertaintyWeightedActionLoss()
    action_log_var = getattr(model, "action_log_var", None)
    scaler = make_scaler(device, precision)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    out_dir = Path(cfg.train.out_dir) / cfg.priors.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(cfg, synthetic, device, precision, model, train_loader)
    reset_peak_memory(device)

    history = []
    best = float("inf")
    wall0 = time.time()
    for epoch in range(cfg.train.epochs):
        model.train()
        synchronize(device)
        t0 = time.time()
        running = 0.0
        seen = 0

        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            loss, parts = _step(net, batch, cfg, action_loss_fn, action_log_var, device, precision)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Gradients have to come back out of the loss scale before they can
            # be clipped, otherwise the clip threshold means nothing.
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()

            running += parts["total"]
            seen += batch["obs_boxes"].shape[0]
            if cfg.train.log_every and i % cfg.train.log_every == 0:
                bits = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                print(f"  epoch {epoch} step {i}/{len(train_loader)} {bits}", flush=True)

        synchronize(device)
        train_secs = time.time() - t0

        val = evaluate(net, val_loader, cfg, device, precision)
        rec = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "train_seconds": round(train_secs, 1),
            "samples_per_sec": round(seen / max(train_secs, 1e-6), 1),
            "peak_vram_gb": round(peak_memory_gb(device), 2),
            **{k: round(v, 4) for k, v in val.items()},
        }
        history.append(rec)
        print(
            f"epoch {epoch}: train_loss={rec['train_loss']:.4f} "
            f"val_ADE={val['ADE']:.2f}px val_FDE={val['FDE']:.2f}px "
            f"val_FIOU={val['FIOU']:.4f} (const-vel FDE={val['const_vel_FDE']:.2f}px) "
            f"[{rec['train_seconds']}s, {rec['samples_per_sec']} samples/s, "
            f"peak {rec['peak_vram_gb']} GB]",
            flush=True,
        )

        if val["FDE"] < best:
            best = val["FDE"]
            torch.save(
                {"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch, "val": val},
                out_dir / "best.pt",
            )

    result = {
        "priors": cfg.priors.tag,
        "synthetic": synthetic,
        "device": str(device),
        "precision": precision.name,
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "wall_clock_seconds": round(time.time() - wall0, 1),
        "peak_vram_gb": round(peak_memory_gb(device), 2),
        "best_val_FDE_px": best,
        "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(result, indent=2))

    print(
        f"total wall clock {result['wall_clock_seconds']}s, "
        f"peak VRAM {result['peak_vram_gb']} GB",
        flush=True,
    )
    if synthetic:
        print(
            "\nThese numbers come from SYNTHETIC data. They say the pipeline runs; "
            "they say nothing about the paper's results."
        )
    return result


def banner(
    cfg: Config,
    synthetic: bool,
    device: torch.device,
    precision: Precision,
    model: TitanNet,
    loader: DataLoader,
) -> None:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"priors: {cfg.priors.tag}  trainable params: {n_params:,}")
    print(describe(device, precision))
    print(
        f"batch size: {loader.batch_size}  dataloader workers: {loader.num_workers}  "
        f"compile: {cfg.train.compile}"
    )
    if cfg.priors.action:
        print(f"action backbone: {model.action_branch.backbone_name} (substitute for I3D, see README)")
    if synthetic:
        from .data.synthetic import BANNER

        print(f"!! {BANNER}")
