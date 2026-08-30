from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from .config import Config, PriorConfig
from .device import resolve_device, select_precision, setup_backends
from .engine import build_loader, evaluate, is_synthetic, train
from .models.titan_net import TitanNet
from .paper import PAPER_RESULTS, format_paper_table

# The rows the paper ablates over, in the order Table 2 presents them.
ABLATION_TAGS = ("vanilla", "AP", "EP", "IP", "EP+AP", "EP+IP", "EP+IP+AP")


def _apply_overrides(cfg: Config, args) -> Config:
    if args.priors:
        cfg.priors = PriorConfig.from_tag(args.priors)
    if args.synthetic:
        cfg.synthetic.enabled = True
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.device:
        cfg.train.device = args.device
    if args.amp is not None:
        cfg.train.amp = args.amp
    if args.compile:
        cfg.train.compile = True
    if args.data_root:
        cfg.data.root = args.data_root
    return cfg


def cmd_train(cfg: Config, args) -> None:
    train(cfg)


def cmd_smoke(cfg: Config, args) -> None:
    """End-to-end plumbing check on synthetic data. Proves nothing about accuracy."""
    cfg.synthetic.enabled = True
    cfg.train.epochs = max(1, min(cfg.train.epochs, args.epochs or 2))
    cfg.synthetic.num_clips = 8
    # Fixed and tiny on purpose: the smoke test is checking that the wiring
    # holds, so it should not depend on how much VRAM the machine happens to
    # have, and spawning workers for 8 fake clips costs more than it saves.
    cfg.train.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.max_agents = 6
    cfg.model.pretrained_backbone = False

    print("=" * 72)
    print("SMOKE TEST -- synthetic data, no real TITAN involved.")
    print("Checks that data -> action branch -> interaction -> AIM -> GRU -> FDE")
    print("runs end to end and produces finite numbers. Nothing here reproduces")
    print("the paper. See README for the paper's actual reported results.")
    print("=" * 72)

    results = {}
    for tag in ABLATION_TAGS:
        print(f"\n--- {tag} ---")
        sub = dataclasses.replace(cfg, priors=PriorConfig.from_tag(tag))
        out = train(sub)
        results[tag] = out["best_val_FDE_px"]

    print("\nSynthetic-data FDE by prior configuration (NOT the paper's numbers):")
    for tag, fde in results.items():
        print(f"  {tag:<10} {fde:8.2f} px  [synthetic]")
    print("\nPaper's actual reported results, for reference only:")
    print(format_paper_table())


def cmd_eval(cfg: Config, args) -> None:
    device = resolve_device(cfg.train.device)
    setup_backends(device)
    precision = select_precision(device, cfg.train.amp)
    model = TitanNet(cfg).to(device)

    ckpt_path = args.checkpoint or Path(cfg.train.out_dir) / cfg.priors.tag / "best.pt"
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"loaded {ckpt_path} (epoch {state.get('epoch')})")
    else:
        print(f"no checkpoint at {ckpt_path}; evaluating an untrained model")

    loader = build_loader(cfg, args.split, shuffle=False, device=device)
    res = evaluate(model, loader, cfg, device, precision)

    label = "SYNTHETIC" if is_synthetic(loader) else "real TITAN"
    print(f"\nsplit={args.split} priors={cfg.priors.tag} data={label}")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
    if is_synthetic(loader):
        print("\nSynthetic data -- these are plumbing numbers, not results.")


def cmd_ablation(cfg: Config, args) -> None:
    results = {}
    for tag in ABLATION_TAGS:
        print(f"\n===== {tag} =====")
        sub = dataclasses.replace(cfg, priors=PriorConfig.from_tag(tag))
        results[tag] = train(sub)["best_val_FDE_px"]

    synthetic = cfg.synthetic.enabled
    label = "SYNTHETIC DATA -- not a reproduction" if synthetic else "real TITAN"
    print(f"\nAblation FDE (px), {label}:")
    for tag, fde in results.items():
        paper = PAPER_RESULTS.get(tag)
        ref = f"   paper: {paper['FDE']:.2f}" if paper else ""
        print(f"  {tag:<10} {fde:8.2f}{ref}")

    out = Path(cfg.train.out_dir) / "ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"synthetic": synthetic, "fde_px": results}, indent=2))
    print(f"\nwrote {out}")


def cmd_paper(cfg: Config, args) -> None:
    print(format_paper_table())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="titan", description="TITAN (CVPR 2020) reproduction attempt")
    ap.add_argument("command", choices=["train", "eval", "smoke", "ablation", "paper"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--priors", help="vanilla | EP | IP | AP | EP+IP | EP+AP | EP+IP+AP")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint")
    ap.add_argument("--data-root")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, help="default scales off detected VRAM")
    ap.add_argument("--num-workers", type=int, help="default scales off os.cpu_count()")
    ap.add_argument("--device", help="cuda | cuda:1 | cpu; default auto-detects")
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mixed precision on CUDA: bf16 where supported, otherwise fp16 (default on)",
    )
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile the model (off by default)"
    )
    ap.add_argument("--synthetic", action="store_true", help="use synthetic smoke-test data")
    args = ap.parse_args(argv)

    cfg_path = args.config if Path(args.config).exists() else None
    cfg = _apply_overrides(Config.load(cfg_path), args)

    {
        "train": cmd_train,
        "eval": cmd_eval,
        "smoke": cmd_smoke,
        "ablation": cmd_ablation,
        "paper": cmd_paper,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
