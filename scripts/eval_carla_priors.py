"""Run the seven EP/IP/AP ablation rows on CARLA trajectories.

The point of this script is one question: does the action prior contribute more
than the ego or interaction prior, as TITAN reports? CARLA lets us ask it with
exact labels instead of hand annotation.

The action prior here is the *contextual* group only -- crossing, waiting,
hesitating -- derived from the per-frame `cross` string. CARLA has no
communicative gestures, so this tests a weaker form of the paper's claim and
the results must be read that way.

Everything except the prior switches is held fixed across rows: same episodes,
same split, same schedule, same seeds. Split is by episode; windows inside one
episode share frames, so a window-level split would leak near-duplicates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from titan.config import PriorConfig
from titan.data.carla import (
    CarlaConfig,
    CarlaDataset,
    assert_disjoint_episodes,
    carla_collate,
    usable_episodes,
)
from titan.device import autocast, resolve_device, select_precision, setup_backends
from titan.metrics import displacement_errors
from titan.models.carla_net import CarlaTitanNet

ROWS = [
    PriorConfig(ego=False, interaction=False, action=False),
    PriorConfig(ego=False, interaction=False, action=True),
    PriorConfig(ego=True, interaction=False, action=False),
    PriorConfig(ego=False, interaction=True, action=False),
    PriorConfig(ego=True, interaction=False, action=True),
    PriorConfig(ego=True, interaction=True, action=False),
    PriorConfig(ego=True, interaction=True, action=True),
]


def run_once(priors, tr_ds, te_ds, dev, prec, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CarlaTitanNet(priors=priors, pred_len=te_ds.cfg.pred_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=carla_collate, drop_last=True)
    el = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=carla_collate)

    model.train()
    for _ in range(epochs):
        for b in tl:
            b = {k: v.to(dev) for k, v in b.items()}
            with autocast(dev, prec):
                pred = model(b)
                tgt = b["fut_boxes"][..., :2]
                m = b["target_mask"].unsqueeze(-1).unsqueeze(-1)
                loss = (((pred[..., :2] - tgt) ** 2) * m).sum() / m.sum().clamp(min=1) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Metrics in fp32 regardless of AMP: they are the reported number.
    model.eval()
    px = te_ds.scale[:2].to(dev)
    ades, fdes = [], []
    with torch.no_grad():
        for b in el:
            b = {k: v.to(dev) for k, v in b.items()}
            pred = model(b).float()
            tgt = b["fut_boxes"][..., :2].float()
            keep = b["target_mask"].bool()
            if keep.sum() == 0:
                continue
            # The network works in [0, 1]; ADE/FDE are reported in pixels, so
            # scale the centre coords back by the recorded image size first.
            a, f = displacement_errors(pred[keep][..., :2] * px, tgt[keep] * px)
            ades.append(a.flatten().cpu())
            fdes.append(f.flatten().cpu())
    return float(torch.cat(ades).mean()), float(torch.cat(fdes).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="../pedestrian-intent-forecast/data/carla_run")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--holdout-frac", dest="holdout", type=float, default=0.25)
    p.add_argument("--out", default="artifacts/carla_priors.json")
    a = p.parse_args()

    dev = resolve_device(None)
    setup_backends(dev)
    prec = select_precision(dev, want_amp=True)
    cfg = CarlaConfig(root=a.root)
    eps = usable_episodes(cfg)
    rng = np.random.default_rng(0)
    order = list(eps)
    rng.shuffle(order)
    n_te = max(1, int(len(order) * a.holdout))
    te_eps, tr_eps = order[:n_te], order[n_te:]
    assert_disjoint_episodes(tr_eps, te_eps)

    tr_ds, te_ds = CarlaDataset(cfg, tr_eps), CarlaDataset(cfg, te_eps)
    print(f"device={dev} precision={prec.name}")
    print(f"episodes: {len(tr_eps)} train / {len(te_eps)} test (split by episode, disjoint asserted)")
    print(f"windows : {len(tr_ds)} train / {len(te_ds)} test")
    print(f"protocol: {cfg.obs_len} observed / {cfg.pred_len} predicted at {20 // cfg.subsample} Hz"
          f" = {cfg.obs_len / 20:.2f} s observed, {cfg.pred_len / 20:.2f} s predicted")
    print(f"seeds   : {a.seeds}, {a.epochs} epochs, identical schedule for every row\n")

    t0 = time.time()
    results = {}
    for priors in ROWS:
        ad, fd = [], []
        for s in a.seeds:
            x, y = run_once(priors, tr_ds, te_ds, dev, prec, a.epochs, a.batch_size, a.lr, s)
            ad.append(x)
            fd.append(y)
        results[priors.tag] = {
            "ADE": float(np.mean(ad)), "ADE_std": float(np.std(ad)), "ADE_all": ad,
            "FDE": float(np.mean(fd)), "FDE_std": float(np.std(fd)), "FDE_all": fd,
        }
        print(f"  {priors.tag:12s} ADE {np.mean(ad):7.2f} +-{np.std(ad):5.2f}   FDE {np.mean(fd):7.2f} +-{np.std(fd):5.2f}")

    base = results["vanilla"]["ADE"]
    print(f"\n  gain over vanilla (ADE px, lower is better):")
    for tag in ("AP", "EP", "IP"):
        if tag in results:
            print(f"    {tag:3s} {base - results[tag]['ADE']:+7.2f}")

    out = {
        "dataset": "CARLA (synthetic, ground-truth labels)",
        "action_prior": "contextual group only; CARLA has no communicative gestures",
        "protocol": {"obs_len": cfg.obs_len, "pred_len": cfg.pred_len, "hz": 20 // cfg.subsample,
                     "units": "pixels", "split": "by episode"},
        "episodes": {"train": len(tr_eps), "test": len(te_eps)},
        "windows": {"train": len(tr_ds), "test": len(te_ds)},
        "seeds": a.seeds, "epochs": a.epochs,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "rows": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}  ({out['wall_clock_seconds']}s)")
    print("\nCARLA data, ground-truth labels, contextual action prior only.")
    print("Not a TITAN result and not accuracy on real pedestrians.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
