"""Numbers reported by the TITAN paper. These are the target, not our results.

Source: Malla, Dariush & Choi, "TITAN: Future Forecast using Action Priors",
CVPR 2020, Table 2 (future object localization). ADE/FDE are in pixels at the
native 1920x1200 resolution; FIOU is the IoU of the final predicted box.

Nothing in this repo has been trained on the real TITAN dataset yet, so no
value here has been matched or reproduced. Anything printed from this module is
labelled as the paper's, and the eval code never compares against it as if it
were ours.
"""

from __future__ import annotations

CITATION = (
    "Malla, Dariush & Choi. TITAN: Future Forecast using Action Priors. CVPR 2020. "
    "arXiv:2003.13886"
)

# Baselines the paper compares against.
PAPER_BASELINES: dict[str, dict[str, float | None]] = {
    "Const-Vel (w/o scaling)": {"ADE": 44.39, "FDE": 102.47, "FIOU": 0.1567},
    "Const-Vel (w/ scaling)": {"ADE": 44.39, "FDE": 102.47, "FIOU": 0.1692},
    "Social-LSTM": {"ADE": 37.01, "FDE": 66.78, "FIOU": None},
    "Social-GAN": {"ADE": 35.41, "FDE": 69.41, "FIOU": None},
}

# The paper's own ablation over ego (EP), interaction (IP) and action (AP) priors.
PAPER_RESULTS: dict[str, dict[str, float]] = {
    "vanilla": {"ADE": 38.56, "FDE": 72.42, "FIOU": 0.3233},
    "AP": {"ADE": 33.54, "FDE": 55.80, "FIOU": 0.3670},
    "EP": {"ADE": 29.42, "FDE": 41.21, "FIOU": 0.4010},
    "IP": {"ADE": 22.53, "FDE": 32.80, "FIOU": 0.5589},
    "EP+AP": {"ADE": 26.03, "FDE": 38.78, "FIOU": 0.5360},
    "EP+IP": {"ADE": 17.79, "FDE": 27.69, "FIOU": 0.5650},
    "EP+IP+AP": {"ADE": 11.32, "FDE": 19.53, "FIOU": 0.6559},
}

DATASET_FACTS = {
    "clips": 700,
    "annotated_frames": 75262,
    "unique_persons": 8592,
    "unique_vehicles": 5504,
    "person_instances": 395770,
    "action_labels": 50,
    "resolution": "1920x1200",
    "annotation_hz": 10,
    "split_clips": {"train": 400, "val": 200, "test": 100},
    "observation_seconds": 1.0,
    "prediction_seconds": 2.0,
}


def format_paper_table() -> str:
    lines = [
        "TITAN paper, Table 2 -- REPORTED BY THE AUTHORS, not measured here.",
        f"{CITATION}",
        "",
        f"  {'method':<24} {'ADE':>8} {'FDE':>8} {'FIOU':>8}",
        "  " + "-" * 52,
    ]
    for name, r in PAPER_BASELINES.items():
        fiou = f"{r['FIOU']:.4f}" if r["FIOU"] is not None else "-"
        lines.append(f"  {name:<24} {r['ADE']:>8.2f} {r['FDE']:>8.2f} {fiou:>8}")
    lines.append("  " + "-" * 52)
    for tag, r in PAPER_RESULTS.items():
        lines.append(f"  {'TITAN_' + tag:<24} {r['ADE']:>8.2f} {r['FDE']:>8.2f} {r['FIOU']:>8.4f}")
    lines.append("")
    lines.append("  ADE/FDE in pixels at 1920x1200. Lower is better; FIOU higher is better.")
    return "\n".join(lines)
