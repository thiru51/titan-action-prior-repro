"""Run TITAN's seven-prior ablation on CARLA recordings and print ADE/FDE.

Reproduce the table in RESULTS.md with:

    PYTHONPATH=src .pixi/envs/default/bin/python scripts/eval_carla.py \
        --epochs 60 --folds 5 --seeds 0 1 2 --out artifacts/carla_priors.json

What this measures, and what it does not. Every number this prints is
**simulated data with ground-truth labels and a restricted action prior**. The
pedestrians are CARLA walkers, not people. The action prior covers TITAN's
contextual group only, because a simulator walker has no gaze, no gestures and
nothing in its hands. Nothing here is a TITAN result or an accuracy on real
pedestrians. See RESULTS.md for the full statement of the limits.

The split is by **episode**, never by window. Windows inside one episode
overlap in time, so splitting by window would put near-identical samples on
both sides. `assert_disjoint_episodes` enforces it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan.baselines import constant_velocity  # noqa: E402
from titan.config import PriorConfig  # noqa: E402
from titan.data.carla import (  # noqa: E402
    CarlaConfig,
    CarlaDataset,
    assert_disjoint_episodes,
    carla_collate,
    episode_folds,
    usable_episodes,
)
from titan.device import resolve_device, setup_backends  # noqa: E402
from titan.metrics import ForecastMetrics  # noqa: E402
from titan.models.carla_net import CarlaTitanNet  # noqa: E402

# The paper's seven ablation rows, in the order its table lists them.
PRIOR_TAGS: tuple[str, ...] = ("vanilla", "AP", "EP", "IP", "EP+AP", "EP+IP", "EP+IP+AP")

BASELINE = "const_vel"


def loader(ds: CarlaDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=carla_collate, drop_last=False
    )


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def step_mask(batch: dict[str, torch.Tensor], pred_len: int) -> torch.Tensor:
    """(B, A) agent mask -> (B, A, pred_len). Only pedestrians are scored."""
    return batch["target_mask"].unsqueeze(-1).expand(-1, -1, pred_len)


def box_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean L2 distance over the full (c_u, c_v, l_u, l_v) box.

    Supervising all four channels rather than the centre alone keeps the box
    extents meaningful, which matters because the decoder feeds its own last
    box back in at every step.
    """
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    m = mask.to(dist.dtype)
    return (dist * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def measure(predict, dl: DataLoader, scale: torch.Tensor, device: torch.device) -> dict[str, float]:
    """ADE / FDE in pixels at the recorded 1280x720."""
    metrics = ForecastMetrics()
    scale = scale.to(device)
    for raw in dl:
        batch = to_device(raw, device)
        pred = predict(batch) * scale
        target = batch["fut_boxes"] * scale
        metrics.update(pred.float(), target.float(), step_mask(batch, pred.shape[-2]))
    res = metrics.compute()
    return {"ADE": res["ADE"], "FDE": res["FDE"], "num_tracks": res["num_tracks"]}


def train_one(
    tag: str,
    seed: int,
    train_dl: DataLoader,
    val_dl: DataLoader,
    scale: torch.Tensor,
    args,
    device: torch.device,
) -> CarlaTitanNet:
    torch.manual_seed(seed)
    net = CarlaTitanNet(
        PriorConfig.from_tag(tag),
        pred_len=args.pred_len,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # Cosine decay to zero over the run. Without it the validation ADE bounces
    # by several pixels between neighbouring epochs, and "best epoch on
    # validation" then becomes a lottery that is larger than the effect the
    # ablation is trying to measure.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for epoch in range(args.epochs):
        net.train()
        for raw in train_dl:
            batch = to_device(raw, device)
            pred = net(batch)
            loss = box_loss(pred, batch["fut_boxes"], step_mask(batch, args.pred_len))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
        sched.step()

        net.eval()
        val = measure(net, val_dl, scale, device)
        if val["ADE"] < best:
            best = val["ADE"]
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if args.verbose and epoch % 10 == 0:
            print(f"      epoch {epoch:3d} val_ADE={val['ADE']:.2f} (best {best:.2f})", flush=True)

    net.load_state_dict(best_state)
    net.eval()
    return net


def prior_content(datasets) -> dict[str, float]:
    """How much each prior actually has to say on this data.

    A prior cannot help if its input is the same in every window, so these
    three fractions are needed to read the ablation table at all. They are
    measured over the test split of every fold, i.e. over exactly the windows
    the reported ADE/FDE average over.
    """
    from titan.data.carla import ACTION_CROSSING, AGENT_PERSON

    windows = [w for _, _, test in datasets for w in test.windows]
    n = max(len(windows), 1)
    varies = 0
    ends_crossing = 0
    has_neighbour = 0
    for w in windows:
        ped = w.obs_actions[(w.agent_class == AGENT_PERSON) & w.agent_mask]
        if ped.size:
            varies += int(len(set(ped[0].tolist())) > 1)
            ends_crossing += int(ped[0][-1] == ACTION_CROSSING)
        has_neighbour += int(w.agent_mask.sum() > 1)
    return {
        "test_windows": float(len(windows)),
        "action_label_varies_within_observation": varies / n,
        "action_label_is_crossing_at_last_observed_step": ends_crossing / n,
        "window_has_a_second_agent": has_neighbour / n,
    }


def build_folds(cfg: CarlaConfig, num_folds: int):
    """-> list of (train, val, test) episode-id lists, one per fold."""
    episodes = usable_episodes(cfg)
    folds = episode_folds(cfg, episodes, num_folds)
    assert_disjoint_episodes(*folds)
    out = []
    for k in range(num_folds):
        test = folds[k]
        val = folds[(k + 1) % num_folds]
        train = [e for i, f in enumerate(folds) if i not in (k, (k + 1) % num_folds) for e in f]
        assert_disjoint_episodes(train, val, test)
        out.append((train, val, test))
    return episodes, out


def main() -> None:
    ap = argparse.ArgumentParser(description="TITAN prior ablation on CARLA recordings")
    ap.add_argument("--root", default="data/carla_run")
    ap.add_argument("--priors", nargs="+", default=list(PRIOR_TAGS))
    ap.add_argument("--obs-len", type=int, default=10)
    ap.add_argument("--pred-len", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-agents", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    setup_backends(device)
    cfg = CarlaConfig(
        root=args.root,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        max_agents=args.max_agents,
    )

    episodes, splits = build_folds(cfg, args.folds)
    hz = 20.0 / cfg.subsample
    print(
        f"CARLA  obs={args.obs_len} pred={args.pred_len} at {hz:.0f} Hz  "
        f"({args.obs_len / hz:.2f} s observed, {args.pred_len / hz:.2f} s forecast)"
    )
    print(
        f"{len(episodes)} usable episodes, {args.folds}-fold split by episode, "
        f"seeds {args.seeds}, device={device}"
    )
    print("SIMULATED DATA, ground-truth labels, action prior restricted to the")
    print("contextual group. Not TITAN results, not accuracy on real pedestrians.\n")

    datasets = []
    for k, (train, val, test) in enumerate(splits):
        d = tuple(CarlaDataset(cfg, part) for part in (train, val, test))
        datasets.append(d)
        print(
            f"  fold {k}: episodes {len(train)}/{len(val)}/{len(test)}  "
            f"windows {len(d[0])}/{len(d[1])}/{len(d[2])}  "
            f"test pedestrian-windows {d[2].num_targets}"
        )
    scale = datasets[0][2].scale
    content = prior_content(datasets)
    print("\n  How much each prior has to say, over the pooled test windows:")
    print(
        f"    action label varies inside the observed window: "
        f"{content['action_label_varies_within_observation']:.1%}"
    )
    print(
        f"    action label is 'crossing' at the last observed step: "
        f"{content['action_label_is_crossing_at_last_observed_step']:.1%}"
    )
    print(f"    window holds a second agent: {content['window_has_a_second_agent']:.1%}")
    print()

    # per config -> per seed -> per fold -> {ADE, FDE}
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    t0 = time.time()

    for k, (train_ds, val_ds, test_ds) in enumerate(datasets):
        train_dl = loader(train_ds, args.batch_size, True)
        val_dl = loader(val_ds, args.batch_size, False)
        test_dl = loader(test_ds, args.batch_size, False)

        res = measure(
            lambda b: constant_velocity(b["obs_boxes"], args.pred_len), test_dl, scale, device
        )
        results.setdefault(BASELINE, {}).setdefault("na", {})[str(k)] = res
        print(f"  fold {k}  {BASELINE:9} ADE {res['ADE']:7.2f}  FDE {res['FDE']:7.2f} px", flush=True)

        for tag in args.priors:
            for seed in args.seeds:
                net = train_one(tag, seed, train_dl, val_dl, scale, args, device)
                res = measure(net, test_dl, scale, device)
                results.setdefault(tag, {}).setdefault(str(seed), {})[str(k)] = res
                print(
                    f"  fold {k}  {tag:9} seed {seed}  ADE {res['ADE']:7.2f}  "
                    f"FDE {res['FDE']:7.2f} px",
                    flush=True,
                )

    print("\n" + "=" * 76)
    print(f"ADE / FDE in pixels at 1280x720, {args.obs_len} observed / {args.pred_len} predicted")
    print(f"at {hz:.0f} Hz. Lower is better. Mean over {args.folds} episode folds; +- is the")
    print(f"standard deviation across the {len(args.seeds)} training seeds of that fold mean.")
    print("SIMULATED DATA. Restricted action prior. NOT TITAN results.\n")

    header = f"{'configuration':<14}{'ADE (px)':>20}{'FDE (px)':>20}"
    print(header)
    print("-" * len(header))

    summary: dict[str, dict[str, float]] = {}
    for tag in [BASELINE] + list(args.priors):
        per_seed_ade, per_seed_fde = [], []
        for seed_key, folds in results[tag].items():
            per_seed_ade.append(statistics.fmean(f["ADE"] for f in folds.values()))
            per_seed_fde.append(statistics.fmean(f["FDE"] for f in folds.values()))
        ade, fde = statistics.fmean(per_seed_ade), statistics.fmean(per_seed_fde)
        sd_a = statistics.stdev(per_seed_ade) if len(per_seed_ade) > 1 else 0.0
        sd_f = statistics.stdev(per_seed_fde) if len(per_seed_fde) > 1 else 0.0
        summary[tag] = {"ADE": ade, "FDE": fde, "ADE_sd": sd_a, "FDE_sd": sd_f}
        print(f"{tag:<14}{ade:>13.2f} +-{sd_a:<5.2f}{fde:>13.2f} +-{sd_f:<5.2f}")

    base = summary["vanilla"]["ADE"] if "vanilla" in summary else None
    if base is not None:
        print("\nChange in ADE against the vanilla row (negative = the prior helps):")
        for tag in args.priors:
            if tag == "vanilla":
                continue
            d = summary[tag]["ADE"] - base
            print(f"  {tag:<10}{d:+7.2f} px  ({100 * d / base:+.1f}%)")

    print(f"\nwall clock {time.time() - t0:.0f}s")

    if args.out:
        payload = {
            "dataset": "CARLA simulator recordings (synthetic)",
            "warning": (
                "Synthetic data with ground-truth labels. The action prior is "
                "restricted to the contextual group: CARLA has no communicative "
                "or transportive actions. Not TITAN results. Not an accuracy on "
                "real pedestrians."
            ),
            "protocol": {
                "obs_len": args.obs_len,
                "pred_len": args.pred_len,
                "hz": hz,
                "obs_seconds": args.obs_len / hz,
                "pred_seconds": args.pred_len / hz,
                "units": "pixels at 1280x720",
                "split": f"{args.folds}-fold cross-validation by episode",
                "stride": args.stride,
                "max_agents": args.max_agents,
                "num_episodes": len(episodes),
            },
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dropout": args.dropout,
                "grad_clip": args.grad_clip,
                "seeds": args.seeds,
            },
            "prior_content": content,
            "per_run": results,
            "summary": summary,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
