# Progress

**Summary: the code is validated and two experiments have been run, but TITAN itself is
still not reproduced.** The dataset remains access-gated.

**Part 1, ETH/UCY.** The trajectory decoder and ADE/FDE metrics are validated on a public
benchmark: constant-velocity 0.53 m ADE, LSTM encoder-decoder 0.58 m, leave-one-scene-out.
Two published qualitative findings reproduce -- the hotel-scene inversion, and social
pooling giving no measurable gain.

**Part 2, CARLA.** All seven EP/IP/AP rows run on simulator trajectories with exact
labels, 3 seeds, split by episode. Action prior is strongest (+8.24 ADE over vanilla),
interaction prior is worse than nothing because 88% of windows have a single agent. The
ordering disagrees with the paper's, for a measured reason.

Both write-ups, with their caveats, are in [RESULTS.md](RESULTS.md).

## Status in one paragraph

The implementation is written and runs. **No result in this repo is a
reproduction of the paper's numbers.** The paper's published figures
(constant-velocity 102.5 px FDE, full EP+IP+AP 19.5 px FDE, Social-GAN 69.4 px,
Social-LSTM 66.8 px) are the *targets* this implementation aims to reproduce —
they were measured by the authors on the real dataset. Real reproduction needs
TITAN dataset access, which is gated behind a request at
<https://usa.honda-ri.com/titan> and **has not been obtained**. What has
changed: there is now one real measured result, on **ETH/UCY**, the public
benchmark TITAN's own baselines publish on. It validates the ADE/FDE metric
code and the trajectory decoding recipe against real human trajectories with
published figures to check against. It is a different dataset and a different
task, so it is not a TITAN result and is never presented as one. Full write-up
in `RESULTS.md`.

## The ETH/UCY result

ADE / FDE in metres, 8 observed / 12 predicted, leave-one-scene-out, averaged
over the five scenes. Measured here, seed 0. **Not TITAN numbers.**

| model | AVG ADE | AVG FDE |
|---|---|---|
| Constant velocity | 0.53 | 1.15 |
| Linear, least squares | 0.65 | 1.27 |
| LSTM encoder-decoder | 0.58 | 1.23 |
| LSTM + social pooling | 0.56 | 1.17 |

Gupta et al. (CVPR 2018, Table 1, t_pred = 12) report their Linear row at
0.79 / 1.59 and their LSTM row at 0.70 / 1.52. Mine are 10-25% lower. The gap
is analysed in `RESULTS.md` and left standing rather than tuned away.

The two learned rows are one seed each. A second seed reverses their ordering
(LSTM 0.55 / 1.16, social pooling 0.58 / 1.21), so the difference between them
is smaller than the difference between seeds and no claim is made that social
pooling helps.

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
- [x] Constant-velocity baseline, plus a least-squares linear baseline added
      for ETH/UCY (they are different estimators and are reported separately).
- [x] ETH/UCY loader (`data/ethucy.py`), completely separate from the TITAN
      path: 8/12 windows at 2.5 Hz, leave-one-scene-out, people dropped rather
      than interpolated when a track is incomplete.
- [x] LSTM encoder-decoder with optional social pooling
      (`models/traj_lstm.py`), for ETH/UCY only. `titan_net.py` untouched.
- [x] `scripts/eval_ethucy.py` — runs the five folds and prints the table.
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

- [x] `pytest -q tests` — 54 passed, 1 skipped. The skip only runs on a machine
      without CUDA. Captured in `artifacts/tests.log`.
- [x] **ETH/UCY benchmark run on real data**, all five leave-one-scene-out
      folds, four models. Captured in `artifacts/ethucy_eval.log` and
      `artifacts/ethucy.json`. Reproduce with the command in `RESULTS.md`.
      The two baseline rows need no training and reproduce bit for bit.
- [x] Sensitivity check on the Social-GAN loader's drop-single-person-windows
      quirk (`artifacts/ethucy_min_agents_2.json`). It moves the linear average
      ADE from 0.65 to 0.62, which is real but too small to explain the gap to
      the published figure.
- [x] Second training seed for the two learned rows
      (`artifacts/ethucy_seed1.json`), so the learned numbers are not a single
      draw. Two seeds is a sanity check, not a variance estimate.
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
      everything below. The ETH/UCY result does *not* unblock any of it; it
      checks the metric and the decoder, nothing about action priors. Request at <https://usa.honda-ri.com/titan>.
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
- [ ] Hyperparameter search on ETH/UCY. Nothing was tuned there either; the
      LSTM rows use one fixed setting on all five folds.
- [ ] Track down the zara2 discrepancy against the published Linear row (0.46
      here against 0.77 reported). My best guess is a difference in which
      frames land in which split, which needs the exact preprocessed files
      Gupta et al. used. Not resolvable from the paper alone.
