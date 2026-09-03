# TITAN reproduction attempt: action priors for pedestrian trajectory forecasting

An implementation of TITAN (Malla, Dariush & Choi, CVPR 2020) from Honda
Research Institute: a 3D-conv action-recognition branch feeding a GRU
trajectory decoder, plus the paper's Agent Importance Mechanism, with the ego /
interaction / action priors on independent switches so the paper's ablation
table can be run row by row.

## Status, stated plainly

**No number in this repository is a reproduction of the paper.** The TITAN
dataset is access-gated and I have not been granted access yet (request process
in [Getting the dataset](#getting-the-dataset)). What has actually happened:

- The code is written, the tests pass (54 passed, 1 skipped), and I have
  verified the full pipeline runs end-to-end on synthetic data — data loader to
  action branch to interaction encoder to Agent Importance Mechanism to GRU
  decoder to the FDE metric — for all seven ablation configurations. A real
  training run against the actual TITAN dataset is the next step once access is
  granted.
- The synthetic data is random walks with noise. Every FDE it produces is a
  number about random walks, not about pedestrians. `artifacts/smoke_test.log`
  is the real captured output of that run and is labelled as such throughout.
- **Two experiments have now been run on data that is available.** Part 1 validates the
  trajectory decoder and the ADE/FDE metrics on the public ETH/UCY benchmark. Part 2 runs
  the seven-row EP/IP/AP ablation on CARLA trajectories, where the labels are exact.
  Both are in [RESULTS.md](RESULTS.md). Neither is a TITAN result.
- **The CARLA ablation disagrees with the paper's ordering, and the reason is measurable.**
  Action prior is the strongest single prior here (+8.24 ADE over vanilla) and the best row
  overall; interaction prior is *worse than no prior*. 4,699 of 5,346 windows contain
  exactly one agent, so there is nothing for an interaction prior to model. The paper's
  Tokyo footage has crowded pavements; these scripted scenarios do not.
- The action branch runs `torchvision`'s `r3d_18`, not I3D. See
  [What is substituted](#what-is-substituted-and-why).
- **There is now one real result on real data, and it is not TITAN.** The
  trajectory decoder and the ADE/FDE metric code have been run on ETH/UCY, a
  public benchmark, and compared against published figures. See
  [The one real result](#the-one-real-result-ethucy) and `RESULTS.md`.

The numbers in the next section are **the paper's own published results**. They
are the target this implementation aims to reproduce. They were measured by the
authors on the real dataset, not by this code.

### The paper's published results (theirs, not mine)

TITAN paper, Table 2. ADE and FDE in pixels at the native 1920x1200; 1 second
observed, 2 seconds forecast, 10 Hz. Lower ADE/FDE is better, higher FIOU is
better.

| method (as reported in the paper) | ADE | FDE | FIOU |
|---|---|---|---|
| Constant velocity | 44.39 | 102.47 | 0.1567 |
| Social-LSTM (Alahi et al. 2016) | 37.01 | 66.78 | - |
| Social-GAN (Gupta et al. 2018) | 35.41 | 69.41 | - |
| TITAN vanilla (no priors) | 38.56 | 72.42 | 0.3233 |
| TITAN + AP (action) | 33.54 | 55.80 | 0.3670 |
| TITAN + EP (ego) | 29.42 | 41.21 | 0.4010 |
| TITAN + IP (interaction) | 22.53 | 32.80 | 0.5589 |
| TITAN + EP + AP | 26.03 | 38.78 | 0.5360 |
| TITAN + EP + IP | 17.79 | 27.69 | 0.5650 |
| **TITAN + EP + IP + AP** | **11.32** | **19.53** | 0.6559 |

The headline is the last row against the first: 102.5 px down to 19.5 px, and
comfortably past both the Social-LSTM and Social-GAN reference rows. Those are
the paper's claims. `python -m titan.cli paper` prints this same table from
`src/titan/paper.py`, always with the "reported by the authors, not measured
here" header attached.

## The one real result: ETH/UCY

Since TITAN itself is gated, the obvious question about this repo is: *you
reimplemented a paper you cannot run, so how would anyone know the code is
right?* The answer is a public benchmark.

ETH/UCY is five scenes of real pedestrian trajectories in world metres, and it
is what TITAN's own two baselines — Social-LSTM and Social-GAN — publish on. I
ran the parts of this repo that are not TITAN-specific on it: the ADE/FDE code
in `src/titan/metrics.py`, and the decoding recipe the TITAN decoder uses
(encode the past, decode per-step deltas, integrate onto the last observed
position). The protocol is the standard one: 8 observed steps and 12 predicted
at 2.5 Hz, leave-one-scene-out over the five scenes.

ADE / FDE in **metres**, averaged over the five folds. **Measured by this repo
on ETH/UCY. These are not TITAN numbers and they are not comparable to the
pixel table above.**

| model | AVG ADE | AVG FDE |
|---|---|---|
| Constant velocity (`titan.baselines`) | 0.53 | 1.15 |
| Linear, least squares | 0.65 | 1.27 |
| LSTM encoder-decoder | 0.58 | 1.23 |
| LSTM + social pooling | 0.56 | 1.17 |

The two baseline rows are deterministic. The two learned rows are one training
seed each, and rerunning with a different seed moves them by more than the gap
between them — so this table does **not** show that social pooling helps, and
`RESULTS.md` says so.

For context, Gupta et al. (CVPR 2018) report **their** Linear row at 0.79 / 1.59
and **their** LSTM row at 0.70 / 1.52 on the same protocol. Mine come out
10-25% lower. `RESULTS.md` has the per-scene breakdown, the exact table and
caption those figures come from, and an honest account of why the gap exists —
including the fact that "linear baseline" is under-specified enough that my own
two linear rows disagree with each other by 0.12 ADE.

Data is committed under `data/datasets/` (7 MB), so this runs from a clean
clone with no download:

```bash
PYTHONPATH=src .pixi/envs/default/bin/python scripts/eval_ethucy.py \
    --models const_vel linear lstm social_lstm \
    --epochs 200 --seed 0 \
    --out artifacts/ethucy.json
```

About 17 minutes on an RTX 4080 Laptop. The two baseline rows need no training
and reproduce exactly. Captured output: `artifacts/ethucy_eval.log`.

This validates the metric implementation and the decoder on real human
trajectory data. **It does not reproduce TITAN**, which still needs the gated
dataset. The ETH/UCY path is entirely separate code — `data/ethucy.py`,
`models/traj_lstm.py`, `scripts/eval_ethucy.py` — and nothing in the TITAN
model was changed to accommodate it.

## Why this paper

Honda Research Institute published TITAN, and it is their own work on
pedestrian behaviour prediction — the exact problem an autonomous driving group
has to solve before it can put a car near a crosswalk. Reproducing it is a more
honest exercise than reproducing something generic: the dataset is theirs, the
numbers are checkable, and the ablation structure is precise enough that a
wrong implementation shows up as the wrong ordering of rows rather than as a
vague accuracy gap.

It is also a good paper to build against. The central claim is narrow and
testable: knowing what a pedestrian is *doing* helps you predict where they
will *be*.

## Why action priors help

Most trajectory forecasting models are built on interaction — Social-LSTM and
Social-GAN both model how nearby agents influence each other. That is real
signal, but it is indirect. It tells you how a pedestrian's neighbours are
arranged, and asks the model to infer intent from that arrangement.

An action label is a much shorter path to the same conclusion. A pedestrian
labelled "waiting to cross street" is standing still now and will be moving
across the road shortly. One labelled "getting into a 4-wheel vehicle" is about
to stop existing as a moving agent. One labelled "looking into phone" while
walking will keep going in a straight line and will not react to the car. Over
a 2-second horizon this is the dominant term: the pedestrian's own immediate
intention constrains the next 20 frames far more tightly than the geometry of
who is standing near whom.

That is why the paper takes the trouble to annotate 50 hierarchical action
labels and run a video CNN over per-pedestrian crops, and it is why the
ablation matters — the point is to show each prior contributes something the
others do not.

## Architecture

```
per-agent past boxes (B, A, 10, 4)
        |
        +-- TrajectoryEncoder: GRU over [box, velocity] -> (B, A, 128)
        |
per-agent video tubes (B, A, 3, 16, 112, 112)      [only when AP is on]
        |
        +-- ActionBranch: r3d_18 -> 5 parallel action heads + a 128-d feature
        |
        +-- InteractionEncoder: masked multi-head attention over agents,       [IP]
        |   with relative box offset as an additive attention bias
        |
ego IMU (B, 10, 2) = (longitudinal accel, yaw rate)
        |
        +-- EgoEncoder: GRU -> (B, 64)                                         [EP]
        |
    concat -> fuse (Linear + ReLU + Dropout) -> (B, A, 128)
        |
        +-- AgentImportance (AIM): masked softmax over agents -> scene vector
        |       -> ego_head: predicts the ego vehicle's own future motion
        |
        +-- GRUCell decoder, 20 steps, predicting box *deltas* which are
            integrated onto the last observed box -> (B, A, 20, 4)
```

A few implementation notes that are choices, not transcriptions of the paper:

- **Boxes are `(c_u, c_v, l_u, l_v)`** — centre plus extent, which is the
  paper's parameterisation. The network works in coordinates normalised by
  1920x1200; every reported metric is converted back to pixels first, in
  `data/common.py`, so there is exactly one place where that conversion lives.
- **The decoder predicts deltas, not absolute positions.** Predicting absolute
  pixel coordinates makes the GRU relearn the identity of the last observed box
  at every step instead of learning the dynamics.
- **Interaction runs after the action feature is concatenated**, so a
  neighbour's predicted action is visible to the attention. That is my reading
  of the paper's ordering and is worth revisiting against real results.
- **AIM's scoring function is mine.** The paper describes AIM as a
  self-attention-style weighting learned jointly with ego-motion prediction, but
  does not publish the shape of the scoring network. `models/aim.py` uses a
  small MLP, a mask-aware softmax over agents actually present, and a weighted
  pool. The weights are returned rather than hidden, because attributing ego
  risk to individual agents is the whole point of the module.
- **The action loss is the paper's uncertainty weighting**, `sum_i [ce_i /
  sigma_i^2 + log sigma_i]`, parameterised by log-variance for stability. TITAN
  labels are five parallel multi-class attribute groups, not one flat softmax,
  which is what makes per-group uncertainty weighting meaningful.

### What is substituted, and why

**The action branch runs `torchvision.models.video.r3d_18`, not I3D.** This
matters and I do not want it buried.

The paper finetunes a single-stream I3D and a 3D ResNet, both pretrained on
Kinetics-600. Kinetics-600 I3D weights are not distributed with torchvision,
and there is no first-party PyTorch I3D that is a drop-in. `r3d_18` is
torchvision's 3D ResNet-18, pretrained on Kinetics-400. So this is a
substitution for the 3D-ResNet arm the paper also reports — a smaller one, on a
smaller pretraining set — and *not* a reimplementation of I3D.

Practically: expect the action branch here to be weaker than the paper's, and
expect the AP rows to gain less than the paper's AP rows do. If the ablation
ordering reproduces but the AP margin is smaller, the backbone is the first
place to look. Swapping it is a one-line config change —
`model.action_backbone` also accepts `mc3_18` and `r2plus1d_18` — and wiring in
a real I3D with Kinetics-600 weights is the single highest-value next step for
faithfulness.

Anywhere this repo prints an action-branch banner, it says `r3d_18 (substitute
for I3D, see README)`. It never claims I3D.

## Prerequisites

- An NVIDIA GPU. Everything auto-detects and falls back to CPU, but the action
  branch is a 3D-conv video model and CPU training is not realistic.
- A working NVIDIA driver. Check with:

```bash
nvidia-smi
```

  If that command errors or is not found, fix the driver first — nothing below
  will work.

- Disk: ~10 GB for the environment (PyTorch and CUDA runtime dominate), plus
  room for the dataset itself when you get it. The released TITAN clips are
  large; budget on the order of several hundred GB for the anonymised images if
  you take the full release.

```bash
df -h .
```

## Install

Two paths. Pick one.

### Path A: pixi (what I use, fully locked)

[pixi](https://pixi.sh) resolves conda packages from a lockfile, which is how
this repo gets a matching CUDA runtime without you hand-picking wheel versions.

```bash
# install pixi itself, if you do not have it
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL   # so `pixi` lands on your PATH

git clone <this repo> titan-action-prior-repro
cd titan-action-prior-repro
pixi install
```

`pixi install` reads `pixi.toml` and `pixi.lock` and builds `.pixi/envs/default`.
It takes a while the first time (it is downloading PyTorch and a CUDA runtime)
and prints `✔ The default environment has been installed.` when it is done.

Then run things with `pixi run`, which activates the environment and sets
`PYTHONPATH=src` for you:

```bash
pixi run gpu        # scripts/check_gpu.py
pixi run test       # pytest -q tests
pixi run smoke      # the synthetic end-to-end check
pixi run train
pixi run ablation
```

### Path B: venv + pip

Use this if you do not want pixi. You get less pinning; `requirements.txt`
carries lower bounds rather than exact versions.

```bash
git clone <this repo> titan-action-prior-repro
cd titan-action-prior-repro

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH=src        # pixi does this for you; here you do it yourself
```

I have run this path end to end on this machine and it works: plain PyPI gave
me `torch 2.13.0+cu130`, CUDA detected, and both the tests and the smoke test
pass under it. Note it resolves a slightly different CUDA build than pixi does
(cu130 versus cu129) — same code, different runtime.

One catch: `pip install torch` gives you whatever wheel matches your platform.
On Linux that is normally a CUDA build, but if you end up with a CPU-only wheel
on a machine that has a GPU, reinstall from the CUDA index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

`scripts/check_gpu.py` prints `built for CUDA ...` versus `cpu-only build`, so
you can tell immediately which one you got.

With this path, replace every `pixi run X` below with the plain command (and
keep `PYTHONPATH=src` exported).

### Path C: Docker

There is a `Dockerfile` if you would rather not touch the host environment:

```bash
docker build -t titan-repro .
docker run --gpus all --rm titan-repro                  # runs check_gpu.py

docker run --gpus all --rm \
  -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" \
  titan-repro python -m titan.cli train --data-root data/titan --priors EP+IP+AP
```

`--gpus` needs the NVIDIA Container Toolkit on the host. Data is bind-mounted
rather than baked into the image, because TITAN is access-gated and must not
end up inside something redistributable.

Being straight about this one: **I have not been able to build this image** —
the Docker daemon is not accessible from where I was working. The base image
tag is confirmed to exist and the build runs the test suite as a build step, so
a broken image should fail the build rather than ship, but treat it as unverified
until someone runs `docker build` on it. Paths A and B I did run end to end.

## Check the GPU first

```bash
pixi run python scripts/check_gpu.py
```

This is the doctor script. Run it before anything else — it tells you what
torch actually sees, which precision the training code will pick on your
hardware, and whether the GPU can do arithmetic at all.

Real output from my machine (an RTX 4080 Laptop, 12 GB); yours will differ:

```
== torch ==
  torch            2.13.0
  built for CUDA   12.9
  cuda.is_available True

== device ==
  name             NVIDIA GeForce RTX 4080 Laptop GPU
  compute cap      8.9
  multiprocessors  58
  total VRAM       11.57 GB
  free VRAM        10.10 GB
  bf16 supported   True

== precision settings this repo will use ==
  autocast dtype   bf16
  grad scaler      off
  matmul tf32      True
  cudnn tf32       True
  cudnn benchmark  True
  matmul precision high

== defaults picked for this machine ==
  batch size       8   (override with --batch-size)
  dataloader workers 8   (override with --num-workers)

== benchmark ==
  fp32 4096^2 matmul 20.34 TFLOP/s
  peak VRAM in bench 0.20 GB

Looks usable. Next: pytest -q tests, then the synthetic smoke test.
```

The captured version of that run is in `artifacts/check_gpu.log`.

What to look for:

- `cuda.is_available False` means torch cannot see the GPU. Either the driver
  is broken or you installed a CPU-only wheel — the `built for CUDA` line tells
  you which.
- `bf16 supported False` is fine. It means you are on pre-Ampere hardware and
  the code will use fp16 with a gradient scaler instead. Both paths are wired.
- A matmul figure in the single-digit TFLOP/s on a card that should be much
  faster usually means thermal throttling or another process on the GPU.

## Run the tests

```bash
pixi run pytest -q tests
```

Expected:

```
..s....................................................                  [100%]
54 passed, 1 skipped in 2.89s
```

The skip is a test that only runs on a machine *without* CUDA (it checks that
asking for `--device cuda` on a CPU-only box fails loudly instead of silently
falling back). On a GPU machine it is correctly skipped.

## Run the synthetic smoke test

```bash
pixi run python -m titan.cli smoke --config configs/default.yaml
```

This trains every one of the seven ablation configurations for 2 epochs on
generated random-walk data. It is a plumbing check: it proves the tensors flow
from the loader through the action branch, the interaction encoder, AIM, the
GRU decoder and out into the FDE metric without a shape error or a NaN. **It
proves nothing about accuracy.** The FDE values it prints describe random
walks.

A representative chunk of the real output (full log in
`artifacts/smoke_test.log`):

```
--- EP+IP+AP ---
priors: EP+IP+AP  trainable params: 33,716,315
device: cuda (NVIDIA GeForce RTX 4080 Laptop GPU, sm_89, 11.6 GB)  precision: bf16  tf32: True
batch size: 2  dataloader workers: 0  compile: False
action backbone: r3d_18 (substitute for I3D, see README)
!! SYNTHETIC DATA -- pipeline smoke test only, results are meaningless
  epoch 0 step 0/4 traj=0.7914 ego=0.8145 action=9.4421 total=11.0480
epoch 0: train_loss=10.9347 val_ADE=906.57px val_FDE=1571.22px val_FIOU=0.0000 (const-vel FDE=31.03px) [0.6s, 13.5 samples/s, peak 1.69 GB]
  epoch 1 step 0/4 traj=0.3028 ego=0.7931 action=8.7000 total=9.7959
epoch 1: train_loss=10.0152 val_ADE=506.26px val_FDE=713.71px val_FIOU=0.0000 (const-vel FDE=31.03px) [0.6s, 13.6 samples/s, peak 1.69 GB]
total wall clock 1.9s, peak VRAM 1.69 GB

These numbers come from SYNTHETIC data. They say the pipeline runs; they say nothing about the paper's results.
```

The run then prints a synthetic-FDE-per-configuration summary (labelled
`[synthetic]` on every line) followed by the paper's real table under a header
saying the paper measured it, not this code. Those two blocks are deliberately
never merged into one table.

Sanity checks worth eyeballing in that output:

- Every configuration reaches `epoch 1` without crashing — that is the actual
  pass condition.
- `train_loss` goes down between the two epochs in each configuration. On
  random walks it should still fit something.
- The `AP` and `EP+IP+AP` rows report ~33.5 M trainable parameters against
  ~250 K for `vanilla`. That is the r3d_18 backbone, and it confirms the action
  branch is genuinely in the graph rather than silently skipped.
- `val_FIOU=0.0000` here is real, not a broken metric: predictions are hundreds
  of pixels off on ~60-pixel-wide boxes, so there is no overlap. The metric is
  unit-tested separately.

## Getting the dataset

**This is the blocker.** TITAN is not a public download.

1. Go to <https://usa.honda-ri.com/titan>.
2. Submit the access request form on that page. It asks who you are, your
   affiliation, and what you intend to use the data for; academic and research
   use is the expected case.
3. HRI reviews the request and, if approved, emails you download links.

Check the page for the current process — it is theirs and it can change. I have
submitted nothing that has come back yet, which is why this repo has no real
results.

### Where to put the data

The loader expects the tarballs extracted into this layout under `data.root`
(default `data/titan`, override with `--data-root`):

```
data/titan/
├── titan_0_4/
│   ├── clip_1.csv
│   ├── clip_2.csv
│   └── ...                     one CSV of annotations per clip
├── images_anonymized/
│   ├── clip_1/
│   │   └── images/
│   │       ├── 000000.png
│   │       └── ...             one PNG per frame
│   └── ...
├── imu_data/
│   ├── clip_1/
│   │   └── synced_sensors.csv  ego vehicle IMU, synced to frames
│   └── ...
└── splits/
    ├── train_set.txt           400 clip names, one per line
    ├── val_set.txt             200
    └── test_set.txt            100
```

The loader checks the split files against the published clip counts (400 / 200
/ 100, 700 total) and refuses to start if they do not match, so a half-finished
download fails loudly instead of quietly training on less data. If you have a
deliberate subset, set `data.strict_split_sizes: false` in the config.

Frame filename zero-padding varies between dumps, so `data/video.py` tries
several widths before giving up on a frame. The annotation CSV column names
have also drifted between releases (some dumps misspell "Communicative"), so
columns are resolved by alias rather than by position — see
`data/schema.py`.

**The schema in this repo has never been run against the real tarballs.** It
was written from the HRI release description and cross-checked against the
TITAN parser in `vita-epfl/pedestrian-transition-dataset`. Treat the first real
run as a debugging session, not a training run.

## Train on real data

Once the data is in place:

```bash
# the full model, all three priors
pixi run python -m titan.cli train \
  --config configs/default.yaml \
  --data-root data/titan \
  --priors EP+IP+AP

# evaluate the best checkpoint on the test split
pixi run python -m titan.cli eval \
  --config configs/default.yaml \
  --data-root data/titan \
  --priors EP+IP+AP \
  --split test
```

Checkpoints and a per-epoch `history.json` land in
`checkpoints/<prior tag>/`. The history file records device, precision, batch
size, worker count, peak VRAM, samples/sec and wall clock alongside the
metrics, so a result can be traced back to the run that produced it.

Every epoch also prints the constant-velocity FDE on the same batches. That is
the one row of the paper's table this repo can compute honestly on day one,
with no training involved, so it is the first thing to check: if the
constant-velocity FDE on real TITAN does not land near the paper's 102.5 px,
the data loading or the eval protocol is wrong and no amount of training will
fix it. Debug that before trusting anything else.

## Run the ablation

This is the paper's Table 2 structure. It trains all seven configurations in
sequence and writes `checkpoints/ablation.json`:

```bash
pixi run python -m titan.cli ablation \
  --config configs/default.yaml \
  --data-root data/titan
```

Or run one row at a time:

```bash
for p in vanilla AP EP IP EP+AP EP+IP EP+IP+AP; do
  pixi run python -m titan.cli train --data-root data/titan --priors "$p"
done
```

The prior tags map onto the switches in `configs/default.yaml`:

| tag | ego | interaction | action |
|---|---|---|---|
| `vanilla` | off | off | off |
| `AP` | off | off | on |
| `EP` | on | off | off |
| `IP` | off | on | off |
| `EP+AP` | on | off | on |
| `EP+IP` | on | on | off |
| `EP+IP+AP` | on | on | on |

The ablation summary prints each configuration's FDE next to the paper's
corresponding number, so the comparison is explicit rather than implied. What
you are looking for first is the *ordering* — each added prior should improve
FDE. Matching the ordering with a worse absolute number is a partial
reproduction and is worth saying so; claiming the paper's number is not.

Turning off the action prior also turns off image loading entirely
(`engine.build_dataset`), because decoding PNG crops dominates load time and
nothing downstream consumes the tubes. So the non-AP rows run much faster than
the AP rows.

## Performance and tuning

Defaults auto-detect and are printed at the top of every run, so you always
know what was used:

```
device: cuda (NVIDIA GeForce RTX 4080 Laptop GPU, sm_89, 11.6 GB)  precision: bf16  tf32: True
batch size: 8  dataloader workers: 8  compile: False
```

What is on by default:

- **Device** auto-detects CUDA, falls back to CPU. Never hardcoded.
- **TF32** on for matmul and cudnn, `float32_matmul_precision("high")`.
- **`cudnn.benchmark`** on — shapes are fixed across steps, so the autotuner
  pays for itself after the first batch.
- **AMP**: bf16 where the card supports it, fp16 otherwise. The gradient scaler
  is enabled only for fp16; bf16 has fp32's exponent range and does not need
  it.
- **ADE/FDE are always computed in fp32**, outside autocast. They are reported
  numbers and precision matters for them in a way it does not for the loss.
- **`channels_last_3d`** on the video backbone, so the 3D convs reach the
  tensor-core kernels.
- **DataLoader**: workers auto-scaled from `os.cpu_count()`, `pin_memory`,
  `persistent_workers`, `prefetch_factor=4`, and `non_blocking=True` host-to-
  device copies. The action branch is genuinely GPU-heavy, so the data path has
  to keep up or the GPU starves.

Flags:

| flag | default | notes |
|---|---|---|
| `--device` | auto | `cuda`, `cuda:1`, `cpu` |
| `--batch-size` | auto from VRAM | see the ladder in `device.py` |
| `--num-workers` | `min(8, cpu_count - 2)` | |
| `--amp` / `--no-amp` | on | `--no-amp` forces fp32 |
| `--compile` | **off** | `torch.compile`; see below |

### Bigger GPU

The auto batch size is a deliberately conservative ladder, not a measurement —
it does not know your `max_agents` or your clip length. On a card with real
memory, raise it and watch the `peak N.NN GB` the run prints each epoch:

```bash
# e.g. an A100 80GB
pixi run python -m titan.cli train --data-root data/titan \
  --priors EP+IP+AP --batch-size 48 --num-workers 16
```

Keep pushing until peak VRAM sits around 80-85% of the card, then stop. The
remaining headroom absorbs the eval pass and fragmentation.

If `samples/sec` does not improve when you raise the batch size, you are
data-bound, not compute-bound — raise `--num-workers` (and `prefetch_factor` in
the config) instead. Memory cost per sample scales with `data.max_agents`,
since each agent carries its own video tube; lowering that is the other knob.

`--compile` is off by default because the first call spends real time on
compilation and the pay-off only shows on long runs. For a full training run it
is likely worth it; for a smoke test it is pure overhead. It is off by default
so nothing surprising happens on your first run.

### Out of memory

Symptom:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.10 GiB
(GPU 0; 11.57 GiB total capacity; ...)
```

In order of what to try:

1. `--batch-size 4` (or 2). Biggest lever.
2. Lower `data.max_agents` in the config (default 16). Memory scales with it
   directly — each agent slot is a full video tube through the 3D CNN.
3. Lower `data.clip_frames` (default 16) or `data.crop_size` (default 112).
   Both feed the backbone directly.
4. `model.freeze_backbone: true` — no backbone gradients, large saving, weaker
   action branch.
5. Confirm AMP is actually on. `--no-amp` roughly doubles activation memory.
6. Check nothing else is on the GPU: `nvidia-smi`. A desktop session can hold a
   surprising amount.

Dropping to `--priors EP+IP` removes the video branch entirely and the model
becomes tiny (~400 K parameters). If you are only debugging the trajectory
side, that is much faster to iterate on.

## Troubleshooting

**`no usable TITAN windows found under data/titan`** — the loader found no
clip with a contiguous 30-frame stretch. Either the data is not there yet
(expected: use `--synthetic` or the `smoke` command), or the extraction layout
does not match the tree above.

**`train split has N clips, the release has 400`** — an incomplete download.
Fix the download, or set `data.strict_split_sizes: false` if the subset is
deliberate.

**`fatal error: cuda.h: No such file or directory`** during a backward pass —
torch routes some ops through Triton, which JIT-compiles a small helper that
`#include`s `cuda.h`. The CUDA *runtime* packages alone do not ship that
header. `pixi.toml` pulls in `cuda-cudart-dev` for exactly this reason; if you
are on the pip path and hit it, install the CUDA toolkit headers for your
version.

**`action prior is enabled but the batch has no 'tubes'`** — `data.load_video`
is off while `priors.action` is on. `engine.build_dataset` normally keeps these
consistent; you get this if you construct the dataset by hand.

**Training is extremely slow and `nvidia-smi` shows low GPU utilisation** —
you are data-bound, decoding PNGs. Raise `--num-workers`, and check the images
are on a fast disk rather than a network mount.

**`RuntimeError: device 'cuda' was requested but torch.cuda.is_available() is
False`** — deliberate. Asking explicitly for CUDA on a machine without it is a
mistake worth failing on rather than silently running 100x slower on CPU. Drop
the flag to let it auto-detect, or pass `--device cpu`.

## Layout

```
src/titan/
├── cli.py               train / eval / smoke / ablation / paper commands
├── config.py            dataclass config, YAML loading, the EP/IP/AP switches
├── device.py            device, TF32, AMP dtype, auto batch size and workers
├── engine.py            train and eval loops, AMP, throughput and VRAM logging
├── losses.py            masked smooth-L1 trajectory loss, uncertainty-weighted
│                        action loss, ego MSE
├── metrics.py           ADE / FDE / FIOU, always fp32; shared by both paths
├── baselines.py         constant-velocity and least-squares-linear extrapolation
├── paper.py             the paper's published numbers, clearly labelled as theirs
├── data/
│   ├── schema.py        TITAN CSV columns, action taxonomy, split sizes
│   ├── titan.py         the real dataset loader (untested against real data)
│   ├── ethucy.py        ETH/UCY loader; the public benchmark, separate path
│   ├── synthetic.py     fake data for the smoke test, loudly labelled
│   ├── video.py         per-agent tube cropping and normalisation
│   └── common.py        box conversions, pixel <-> normalised, collate
└── models/
    ├── titan_net.py     the whole network and the prior switches
    ├── action_branch.py r3d_18 backbone + 5 action heads          [AP]
    ├── interaction.py   masked attention with spatial bias        [IP]
    ├── aim.py           Agent Importance Mechanism
    └── traj_lstm.py     LSTM encoder-decoder + social pooling, for ETH/UCY only

scripts/check_gpu.py     the doctor script; run this first
scripts/eval_ethucy.py   the ETH/UCY benchmark; produces the RESULTS.md table
configs/default.yaml     the paper's protocol as defaults
tests/                   schema, metrics, model, device, ETH/UCY
data/datasets/           ETH/UCY, committed (7 MB) so RESULTS.md is verifiable
artifacts/               captured real output from actual runs

pixi.toml, pixi.lock     install path A, fully locked
requirements.txt         install path B, lower bounds
Dockerfile               install path C; not build-verified, see Install
```

Docs: `RESULTS.md` (the ETH/UCY result and how it compares to published
figures), `END_GOAL.md` (what done looks like), `PROGRESS.md` (what is and is
not built), `HANDOFF.md` (design decisions and what to do next).

## References

- Malla, Dariush & Choi. *TITAN: Future Forecast using Action Priors.* CVPR
  2020. [arXiv:2003.13886](https://arxiv.org/abs/2003.13886). The paper this
  reproduces. Dataset: 700 clips, 75,262 annotated frames, 8,592 unique
  pedestrians, 50 hierarchical action labels, recorded in Tokyo.
- Gupta, Johnson, Fei-Fei, Savarese & Alahi. *Social GAN: Socially Acceptable
  Trajectories with Generative Adversarial Networks.* CVPR 2018. A baseline row
  in the paper's table.
- Alahi, Goel, Ramanathan, Robicquet, Fei-Fei & Savarese. *Social LSTM: Human
  Trajectory Prediction in Crowded Spaces.* CVPR 2016. The other baseline row.
- Carreira & Zisserman. *Quo Vadis, Action Recognition? A New Model and the
  Kinetics Dataset.* CVPR 2017. The I3D architecture the paper's action branch
  uses and this repo substitutes for.
- Schöller, Aravantinos, Lay & Knoll. *What the Constant Velocity Model Can
  Teach Us About Pedestrian Motion Prediction.* RA-L / ICRA 2020.
  [arXiv:1903.07933](https://arxiv.org/abs/1903.07933). Why a constant-velocity
  baseline is competitive on ETH/UCY, which is what happens in `RESULTS.md`.
- Pellegrini, Ess, Schindler & Van Gool (ETH, ICCV 2009) and Lerner, Chrysanthou
  & Lischinski (UCY, Eurographics 2007). The two source datasets behind the
  five ETH/UCY scenes in `data/datasets/`.
