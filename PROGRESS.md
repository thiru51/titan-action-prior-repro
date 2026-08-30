# Progress

## Status in one paragraph

The implementation is written and runs. I have verified the full pipeline runs
end-to-end on synthetic data; a real training run against the actual TITAN
dataset is the next step once access is granted. **No result in this repo is a
reproduction of the paper's numbers.** The paper's published figures
(constant-velocity 102.5 px FDE, full EP+IP+AP 19.5 px FDE, Social-GAN 69.4 px,
Social-LSTM 66.8 px) are the *targets* this implementation aims to reproduce —
they were measured by the authors on the real dataset. Real reproduction needs
TITAN dataset access, which is gated behind a request at
<https://usa.honda-ri.com/titan> and **has not been obtained**.

## Environment

- [x] `pixi install` works. Env is ~9.4 GB, torch 2.13.0 (cu129), torchvision
      0.28.0, Python 3.11.
- [x] venv + pip path works too, verified end to end — tests and smoke test
      both pass under it. Resolves torch 2.13.0+cu130 from plain PyPI.
- [x] `requirements.txt` generated and actually installed into a clean venv,
      not just written.
- [x] Fixed the Triton `cuda.h` failure that killed an earlier smoke test on
      the first backward pass — `cuda-cudart-dev` in `pixi.toml`.
- [x] Dropped `tqdm` (declared, never used); added `pillow` (used directly by
      `data/video.py`, previously relied on as a torchvision transitive).
- Hardware used: RTX 4080 Laptop, 11.57 GB, compute capability 8.9, bf16
  supported. 32 CPU cores.

## Code

- [x] Data schema: TITAN CSV columns with alias resolution (releases have
      drifted on spellings), the five action groups, split sizes, IMU columns.
- [x] Real TITAN loader — **written, never run against real data**.
- [x] Synthetic dataset for smoke testing, loudly labelled as fake everywhere
      it surfaces.
- [x] Per-agent video tube cropping and Kinetics normalisation.
- [x] Action branch: `r3d_18` backbone plus five parallel action heads.
- [x] Interaction encoder: masked multi-head attention with relative box offset
      as an additive attention bias.
- [x] Agent Importance Mechanism: mask-aware softmax over present agents,
      weighted pool, weights returned; trained jointly with ego-motion.
- [x] GRU decoder predicting box deltas integrated onto the last observed box.
- [x] Losses: masked smooth-L1 trajectory, uncertainty-weighted multi-head
      action loss, ego MSE.
- [x] Metrics: ADE / FDE / FIOU in pixels, always computed in fp32. FDE uses
      each track's last *valid* step, not blindly index -1.
- [x] Constant-velocity baseline.
- [x] EP / IP / AP switches; all seven ablation rows constructible and runnable.
- [x] CLI: `train`, `eval`, `smoke`, `ablation`, `paper`.
- [x] The paper's numbers live in one module and are never printed without a
      header saying the authors measured them.

## GPU and performance

- [x] Central device/precision module. Device auto-detects, never hardcoded.
- [x] TF32 on (matmul + cudnn), `set_float32_matmul_precision("high")`,
      `cudnn.benchmark=True`.
- [x] AMP in the training step: autocast over forward and loss; bf16 where
      supported, fp16 otherwise; GradScaler enabled **only** for fp16.
- [x] Gradients unscaled before clipping, so the clip threshold means something.
- [x] ADE/FDE deliberately computed outside autocast in fp32 — reported numbers.
- [x] `channels_last_3d` on the video backbone.
- [x] DataLoader: auto workers from `os.cpu_count()`, `pin_memory`,
      `persistent_workers`, `prefetch_factor`, `non_blocking=True` transfers.
- [x] Flags: `--device`, `--batch-size`, `--num-workers`, `--amp/--no-amp`,
      `--compile` (off by default).
- [x] Batch size auto-scales off detected VRAM, overridable.
- [x] `scripts/check_gpu.py` doctor: torch version, CUDA build, device name,
      total/free VRAM, compute capability, bf16, TF32 status, matmul benchmark.
- [x] Runs log real peak VRAM, samples/sec and wall clock, into both stdout and
      `history.json`.

## Verification actually performed

- [x] `pytest -q tests` — 44 passed, 1 skipped. The skip only runs on a machine
      without CUDA. Captured in `artifacts/tests.log`.
- [x] `scripts/check_gpu.py` — captured in `artifacts/check_gpu.log`.
- [x] Full synthetic smoke test, all seven ablation configurations, 2 epochs
      each, on GPU with bf16. Captured in `artifacts/smoke_test.log`.
      Peak VRAM 1.69 GB for the AP configurations, ~0.02 GB without the video
      branch. **These are pipeline-verification numbers on random-walk data and
      are not results.**
- [x] Both install paths (pixi, venv+pip) verified independently.
- [ ] **Dockerfile is NOT build-verified.** The Docker daemon was not
      accessible from where this was written. The base image tag is confirmed
      to exist and the build runs the test suite as a build step, so a broken
      image should fail the build, but nobody has run `docker build` on it yet.

## Open

- [ ] **TITAN dataset access.** Not requested-and-granted. This blocks
      everything below. Request at <https://usa.honda-ri.com/titan>.
- [ ] Run the loader against real tarballs. It has never seen a real TITAN CSV.
      Expect the first run to be a debugging session — column names and frame
      filename padding are the likely friction.
- [ ] Compute the constant-velocity baseline on real data and check it against
      the paper's 102.5 px FDE. This is the gate before trusting anything else;
      it needs no training.
- [ ] Train `EP+IP+AP` on real data.
- [ ] Run the full seven-row ablation on real data and check the *ordering*
      first, absolute numbers second.
- [ ] Replace `r3d_18` with a real I3D on Kinetics-600 weights. The current
      backbone is a documented substitution and the most likely source of a gap
      in the AP rows.
- [ ] Tune. Nothing has been tuned; the hyperparameters in
      `configs/default.yaml` are reasonable defaults, not searched values.
- [ ] Try `--compile` on a long run and record whether it actually pays off.
