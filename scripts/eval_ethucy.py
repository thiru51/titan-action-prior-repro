"""Run the ETH/UCY leave-one-scene-out benchmark and print an ADE/FDE table.

Reproduce the table in RESULTS.md with:

    PYTHONPATH=src .pixi/envs/default/bin/python scripts/eval_ethucy.py \
        --models const_vel linear lstm social_lstm --out artifacts/ethucy.json

ADE and FDE come out of `titan/metrics.py`, the same code the TITAN path uses.
That is the point of the exercise: the metric implementation gets checked
against a public benchmark, because the TITAN dataset is not available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan.baselines import constant_velocity, linear_least_squares  # noqa: E402
from titan.data.ethucy import (  # noqa: E402
    SCENES,
    EthUcyConfig,
    EthUcyDataset,
    ethucy_collate,
)
from titan.device import resolve_device  # noqa: E402
from titan.metrics import ForecastMetrics  # noqa: E402
from titan.models.traj_lstm import TrajLSTM  # noqa: E402

TRAINED = {"lstm", "social_lstm"}


def loader(ds: EthUcyDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ethucy_collate, drop_last=False
    )


@torch.no_grad()
def measure(predict, dl: DataLoader, device: torch.device) -> dict[str, float]:
    metrics = ForecastMetrics()
    for batch in dl:
        obs = batch["obs"].to(device)
        fut = batch["fut"].to(device)
        pred = predict(obs, batch["seq_start_end"].to(device))
        metrics.update(pred.float(), fut.float())
    res = metrics.compute()
    return {"ADE": res["ADE"], "FDE": res["FDE"], "num_tracks": res["num_tracks"]}


def make_predictor(model: str, net: TrajLSTM | None, pred_len: int):
    if model == "const_vel":
        return lambda obs, sse: constant_velocity(obs, pred_len)
    if model == "linear":
        return lambda obs, sse: linear_least_squares(obs, pred_len)
    if net is None:
        raise ValueError(f"model {model!r} needs a trained network")
    net.eval()
    return lambda obs, sse: net(obs, sse)


def train_net(
    model: str, train_dl: DataLoader, val_dl: DataLoader, args, device: torch.device
) -> TrajLSTM:
    torch.manual_seed(args.seed)
    net = TrajLSTM(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embed=args.embed,
        hidden=args.hidden,
        dropout=args.dropout,
        social=(model == "social_lstm"),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_ade = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for epoch in range(args.epochs):
        net.train()
        total, seen = 0.0, 0
        for batch in train_dl:
            obs = batch["obs"].to(device)
            fut = batch["fut"].to(device)
            pred = net(obs, batch["seq_start_end"].to(device))
            # Mean L2 distance, i.e. exactly the ADE the benchmark reports, so
            # training and model selection optimise the reported quantity
            # rather than a proxy for it.
            loss = torch.linalg.vector_norm(pred - fut, dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach()) * obs.shape[0]
            seen += obs.shape[0]

        val = measure(make_predictor(model, net, args.pred_len), val_dl, device)
        if val["ADE"] < best_ade:
            best_ade = val["ADE"]
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if args.verbose and epoch % 10 == 0:
            print(
                f"    epoch {epoch:3d} train_loss={total / max(seen, 1):.4f} "
                f"val_ADE={val['ADE']:.4f} (best {best_ade:.4f})",
                flush=True,
            )

    net.load_state_dict(best_state)
    return net


def main() -> None:
    ap = argparse.ArgumentParser(description="ETH/UCY leave-one-scene-out ADE/FDE")
    ap.add_argument("--root", default="data/datasets")
    ap.add_argument("--models", nargs="+", default=["const_vel", "linear", "lstm", "social_lstm"])
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--obs-len", type=int, default=8)
    ap.add_argument("--pred-len", type=int, default=12)
    ap.add_argument(
        "--min-agents",
        type=int,
        default=1,
        help="drop windows with fewer fully-tracked people; 2 matches the Social-GAN loader",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cfg = EthUcyConfig(
        root=args.root,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        min_agents=args.min_agents,
    )

    print(
        f"ETH/UCY  obs={args.obs_len} pred={args.pred_len} (2.5 Hz)  "
        f"leave-one-scene-out  min_agents={args.min_agents}  device={device}"
    )
    print("Metrics in metres, computed by titan.metrics.ForecastMetrics.\n")

    results: dict[str, dict[str, dict[str, float]]] = {m: {} for m in args.models}
    counts: dict[str, int] = {}
    t0 = time.time()

    for scene in args.scenes:
        test_dl = loader(EthUcyDataset(cfg, scene, "test"), args.batch_size, False)
        counts[scene] = int(test_dl.dataset.num_people)
        train_dl = val_dl = None
        for model in args.models:
            net = None
            if model in TRAINED:
                if train_dl is None:
                    train_dl = loader(EthUcyDataset(cfg, scene, "train"), args.batch_size, True)
                    val_dl = loader(EthUcyDataset(cfg, scene, "val"), args.batch_size, False)
                print(f"  training {model} on the four scenes that are not {scene} ...", flush=True)
                net = train_net(model, train_dl, val_dl, args, device)
            res = measure(make_predictor(model, net, args.pred_len), test_dl, device)
            results[model][scene] = res
            print(
                f"  {scene:6} {model:12} ADE {res['ADE']:.3f}  FDE {res['FDE']:.3f}  "
                f"(n={int(res['num_tracks'])})",
                flush=True,
            )

    print("\n" + "=" * 78)
    print("ADE / FDE in metres, 8 observed / 12 predicted, lower is better.")
    print("Measured by this repo on ETH/UCY. NOT TITAN results.\n")
    header = f"{'model':<14}" + "".join(f"{s:>13}" for s in args.scenes) + f"{'AVG':>13}"
    print(header)
    print("-" * len(header))

    summary: dict[str, dict[str, float]] = {}
    for model in args.models:
        per = [results[model][s] for s in args.scenes]
        # Unweighted mean over the five folds, which is how the published
        # tables average. Weighting by test-set size would change the number,
        # because univ holds far more person-windows than eth does.
        avg_ade = sum(p["ADE"] for p in per) / len(per)
        avg_fde = sum(p["FDE"] for p in per) / len(per)
        summary[model] = {"ADE": avg_ade, "FDE": avg_fde}
        cells = "".join(f"{p['ADE']:>6.2f}/{p['FDE']:<6.2f}" for p in per)
        print(f"{model:<14}{cells}{avg_ade:>6.2f}/{avg_fde:<6.2f}")

    print(f"\nwall clock {time.time() - t0:.0f}s")

    if args.out:
        payload = {
            "dataset": "ETH/UCY",
            "protocol": {
                "obs_len": args.obs_len,
                "pred_len": args.pred_len,
                "hz": 2.5,
                "units": "metres",
                "split": "leave-one-scene-out",
                "min_agents": args.min_agents,
            },
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "embed": args.embed,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "seed": args.seed,
            },
            "test_people_per_scene": counts,
            "per_scene": results,
            "average": summary,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
