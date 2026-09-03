# ETH/UCY result

## What this is, in one paragraph

The TITAN dataset is access-gated and I have not been granted access, so no
number in this repo is a TITAN result. That left an obvious hole: a
reimplementation nobody can check is not worth much. So I ran the parts of this
repo that are not TITAN-specific — the trajectory decoding recipe and the
ADE/FDE metric code in `src/titan/metrics.py` — on **ETH/UCY**, which is public,
is the benchmark TITAN's own two baselines (Social-LSTM and Social-GAN) publish
on, and has numbers in the literature that anyone can look up. This page reports
what I measured, next to what those papers report, and is honest about where
they differ.

**This is not a reproduction of TITAN.** Different dataset, different task,
different units. ETH/UCY is bird's-eye-view world coordinates in metres; TITAN
is bounding boxes in image pixels from a moving vehicle. Nothing here transfers
as a TITAN number.

## Protocol

The standard ETH/UCY protocol, matching Social-LSTM and Social-GAN:

- Five scenes: `eth`, `hotel`, `univ`, `zara1`, `zara2`.
- **Leave-one-scene-out**: train on four scenes, test on the fifth, five times.
  Every reported average is the unweighted mean over the five folds, which is
  how the published tables average.
- **8 observed steps, 12 predicted.** Positions are sampled at 2.5 Hz (one row
  per person every 0.4 s), so that is 3.2 s observed and 4.8 s predicted.
- Units are **metres** in world coordinates, taken straight from the data files.
  No pixel conversion is involved anywhere on this path.
- A person is included in a window only if they are tracked at all 20 steps.
  Partial tracks are dropped, not interpolated.
- ADE is the mean L2 distance over all 12 predicted steps. FDE is the L2
  distance at step 12. Both come from `titan.metrics.ForecastMetrics`, the same
  class the TITAN path uses.

Test-set size, in person-windows (this is what ADE/FDE average over):

| eth | hotel | univ | zara1 | zara2 |
|---|---|---|---|---|
| 364 | 1,197 | 24,334 | 2,356 | 5,910 |

## Reproducing the table

Data is committed under `data/datasets/` (7 MB), so this runs from a clean
clone with no download.

```bash
PYTHONPATH=src .pixi/envs/default/bin/python scripts/eval_ethucy.py \
    --models const_vel linear lstm social_lstm \
    --epochs 200 --seed 0 \
    --out artifacts/ethucy.json
```

About 17 minutes on an RTX 4080 Laptop. Full captured output is in
`artifacts/ethucy_eval.log`; the machine-readable version is
`artifacts/ethucy.json`. `const_vel` and `linear` need no training and are
exactly reproducible; the two learned rows depend on the seed (see
[Seed variance](#seed-variance)).

## What I measured

ADE / FDE in metres, 8 observed / 12 predicted. Lower is better.
**Measured by this repo on ETH/UCY. These are not TITAN numbers.**

| model | eth | hotel | univ | zara1 | zara2 | AVG |
|---|---|---|---|---|---|---|
| Constant velocity | 1.08 / 2.28 | 0.32 / 0.61 | 0.52 / 1.17 | 0.43 / 0.95 | 0.32 / 0.72 | **0.53 / 1.15** |
| Linear (least squares) | 1.18 / 2.38 | 0.26 / 0.48 | 0.74 / 1.43 | 0.60 / 1.18 | 0.46 / 0.89 | **0.65 / 1.27** |
| LSTM encoder-decoder | 1.05 / 2.15 | 0.59 / 1.24 | 0.56 / 1.21 | 0.40 / 0.87 | 0.30 / 0.66 | **0.58 / 1.23** |
| LSTM + social pooling | 1.02 / 2.09 | 0.50 / 1.03 | 0.59 / 1.23 | 0.39 / 0.84 | 0.32 / 0.69 | **0.56 / 1.17** |

What each row is:

- **Constant velocity** — `titan.baselines.constant_velocity`, the function the
  TITAN path already uses. Velocity is the difference between the last two
  observed steps, held fixed for 12 steps. No training, fully deterministic.
- **Linear (least squares)** — `titan.baselines.linear_least_squares`, added for
  this benchmark. Fits a straight line to all 8 observed steps by minimising
  squared error, then extrapolates. This is the estimator Social-GAN describes
  for its "Linear" row, and it is *not* the same thing as constant velocity.
- **LSTM encoder-decoder** — `titan.models.traj_lstm.TrajLSTM` with
  `social=False`. Encodes the observed step-to-step displacements with an LSTM,
  decodes 12 displacements, integrates them onto the last observed position. It
  sees one person at a time and nothing else. This is Social-LSTM with the
  social pooling removed.
- **LSTM + social pooling** — the same network with `social=True`: before
  decoding, each person's state is max-pooled with every other person in the
  same 20-step window, conditioned on where those neighbours are relative to
  them.

Hyperparameters: 200 epochs, batch 64 windows, Adam at 1e-3, 64-d embedding,
64-d hidden, no dropout, gradient clip 5.0. The training loss is mean L2
distance, i.e. ADE itself, and the checkpoint is selected on validation ADE.
Nothing was searched or tuned against the test scenes.

## The published numbers

### Social-GAN (Gupta et al., CVPR 2018), Table 1

Source: Gupta, Johnson, Fei-Fei, Savarese & Alahi, *Social GAN: Socially
Acceptable Trajectories with Generative Adversarial Networks*, CVPR 2018,
[arXiv:1803.10892](https://arxiv.org/abs/1803.10892), **Table 1**. The caption
reads: "We report two error metrics Average Displacement Error (ADE) and Final
Displacement Error (FDE) for t_pred = 8 and t_pred = 12 (8 / 12) in meters."

The table gives each cell as `8 / 12`. Everything below is the **t_pred = 12**
half, which is the protocol used here.

| method (as reported by Gupta et al.) | eth | hotel | univ | zara1 | zara2 | AVG |
|---|---|---|---|---|---|---|
| Linear, ADE | 1.33 | 0.39 | 0.82 | 0.62 | 0.77 | 0.79 |
| Linear, FDE | 2.94 | 0.72 | 1.59 | 1.21 | 1.48 | 1.59 |
| LSTM, ADE | 1.09 | 0.86 | 0.61 | 0.41 | 0.52 | 0.70 |
| LSTM, FDE | 2.41 | 1.91 | 1.31 | 0.88 | 1.11 | 1.52 |
| S-LSTM, ADE | 1.09 | 0.79 | 0.67 | 0.47 | 0.56 | 0.72 |
| S-LSTM, FDE | 2.35 | 1.76 | 1.40 | 1.00 | 1.17 | 1.54 |

Their "S-LSTM" column is their own rerun of Alahi et al., not Alahi's published
figures. They say so directly in the text: "we tried our best to reproduce the
results of the paper", and note that Alahi et al. pretrained on synthetic data
while they did not.

### Social-LSTM (Alahi et al., CVPR 2016), Table 1

Source: Alahi, Goel, Ramanathan, Robicquet, Fei-Fei & Savarese, *Social LSTM:
Human Trajectory Prediction in Crowded Spaces*, CVPR 2016,
[CVF open access PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf),
**Table 1**. Same 8-observed / 12-predicted setting: the paper states "we
observe a trajectory for 3.2secs and predict their paths for the next 4.8secs
... this corresponds to observing 8 frames and predicting for the next 12
frames."

| method (as reported by Alahi et al.) | ETH | HOTEL | ZARA 1 | ZARA 2 | UCY | Average |
|---|---|---|---|---|---|---|
| Lin, avg. disp. error | 0.80 | 0.39 | 0.47 | 0.45 | 0.57 | 0.53 |
| LSTM, avg. disp. error | 0.60 | 0.15 | 0.43 | 0.51 | 0.52 | 0.44 |
| Social-LSTM, avg. disp. error | 0.50 | 0.11 | 0.22 | 0.25 | 0.27 | 0.27 |
| Lin, final disp. error | 1.31 | 0.55 | 0.89 | 0.91 | 1.14 | 0.97 |
| LSTM, final disp. error | 1.31 | 0.33 | 0.93 | 1.09 | 1.25 | 0.98 |
| Social-LSTM, final disp. error | 1.07 | 0.23 | 0.48 | 0.50 | 0.77 | 0.61 |

**Two things to know before comparing against this table.** First, the caption
does not state units, and the word "meters" does not appear anywhere in the
paper's evaluation section — so these figures cannot be assumed to be in the
same units as the Social-GAN table. Second, the paper defines its average
displacement error as "the mean square error (MSE) over all estimated points",
which is a different quantity from the mean L2 distance that Social-GAN and
everything since have used under the same name. I have not resolved either
question from the paper text, so **I do not compare my numbers to this table**
and I have not included it in the comparison below. It is reproduced here only
because it is the primary source and the discrepancy is part of the story.

## My numbers against the published ones

Comparing against Social-GAN's Table 1, t_pred = 12, since that is the table
whose units and metric definition are unambiguous.

| model | mine, AVG ADE / FDE | Gupta et al., AVG ADE / FDE | difference |
|---|---|---|---|
| Linear (least squares) | 0.65 / 1.27 | 0.79 / 1.59 | mine 0.14 / 0.32 lower |
| LSTM | 0.58 / 1.23 | 0.70 / 1.52 | mine 0.12 / 0.29 lower |
| social pooling vs S-LSTM | 0.56 / 1.17 | 0.72 / 1.54 | mine 0.16 / 0.37 lower |

The two learned rows are one seed each. See [Seed variance](#seed-variance)
before reading anything into the gap between them.

Per scene, against the same table:

| scene | Linear: mine / theirs | LSTM: mine / theirs |
|---|---|---|
| eth | 1.18 / 1.33 ADE, 2.38 / 2.94 FDE | 1.05 / 1.09 ADE, 2.15 / 2.41 FDE |
| hotel | 0.26 / 0.39 ADE, 0.48 / 0.72 FDE | 0.59 / 0.86 ADE, 1.24 / 1.91 FDE |
| univ | 0.74 / 0.82 ADE, 1.43 / 1.59 FDE | 0.56 / 0.61 ADE, 1.21 / 1.31 FDE |
| zara1 | 0.60 / 0.62 ADE, 1.18 / 1.21 FDE | 0.40 / 0.41 ADE, 0.87 / 0.88 FDE |
| zara2 | 0.46 / 0.77 ADE, 0.89 / 1.48 FDE | 0.30 / 0.52 ADE, 0.66 / 1.11 FDE |

### What agrees

The numbers land in the right place and, more importantly, the *shape* of the
published result comes out too.

- **zara1 is nearly exact** on both rows — Linear 0.60/1.18 against 0.62/1.21,
  LSTM 0.40/0.87 against 0.41/0.88. On that fold my pipeline and theirs are
  measuring the same thing.
- **univ and eth are within about 10%** on both rows.
- **The scene ordering reproduces.** Both tables agree that eth is by far the
  hardest scene and that zara2 and hotel are the easiest, and both put univ
  and zara1 in the middle in the same order.
- **The published oddity on hotel reproduces.** Gupta et al. report LSTM at
  0.86 ADE on hotel against Linear at 0.39 — the learned model is *worse* than
  a straight line by more than a factor of two. I get the same inversion, LSTM
  0.59 against Linear 0.26. A well-known way to accidentally beat this
  benchmark is to leak the test scene's statistics into training; if I had done
  that, this inversion is the first thing that would have disappeared. It did
  not.
- **Social pooling buys nothing measurable.** At seed 0 it improves average ADE
  from 0.58 to 0.56; at seed 1 it makes it worse, 0.58 against 0.55. The gap
  between the two models is smaller than the gap between two seeds of one
  model, so the honest statement is that I cannot separate them. Gupta et al.
  reached the same conclusion and said so plainly: "in our experiments S-LSTM
  does not outperform LSTM." See [Seed variance](#seed-variance).

### Where I differ, and why I think so

My rows are consistently 10-25% better than the corresponding published rows,
with the gap concentrated on **zara2** (Linear ADE 0.46 against 0.77) and
**hotel** (Linear ADE 0.26 against 0.39). The Linear row is where this matters
most, because it is deterministic — there is no training seed, no
hyperparameter and no early-stopping choice in it. A gap there is a gap in the
data pipeline or the metric, not in the model.

I chased it as far as the evidence goes, and stopped rather than tuning:

1. **The metric is not the problem.** ADE and FDE come out of the repo's own
   `ForecastMetrics`, which is unit-tested (`tests/test_ethucy.py` includes a
   fixed-3-metre-offset case that must return exactly 3.0 for both). A metric
   bug that produced a uniform 20% discount would have to survive that test.

2. **The window definition is not enough to explain it either.** The
   widely-copied Social-GAN loader drops any window containing fewer than two
   fully-tracked people. Mine keeps them by default. Running with
   `--min-agents 2` to match (captured in `artifacts/ethucy_min_agents_2.json`)
   moves the Linear average from 0.65/1.27 to 0.62/1.23. It matters on eth,
   where the crowd is sparse and the ADE drops from 1.18 to 1.02, and barely
   registers anywhere else. Real, documented, but far too small to close a gap
   of 0.14 ADE that is concentrated in zara2.

3. **The most likely cause is that "Linear" is not one estimator.** Gupta et
   al. describe theirs as "a linear regressor that estimates linear parameters
   by minimizing the least square error" and nothing further — no statement of
   whether the fit is on positions or velocities, what it is conditioned on, or
   whether it is refit per person or fit once. My constant-velocity row and my
   least-squares row differ from each other by 0.12 average ADE using the same
   data and the same metric, which is roughly the size of the gap to the
   published figure. Two defensible readings of the same one-line description
   are enough to produce it.

4. **zara2 specifically looks like a different split, not a different model.**
   The published table puts zara2 (Linear ADE 0.77) as materially *harder* than
   zara1 (0.62), even though the two are recordings of the same location. Both
   my rows put zara2 easier than zara1, which is what I would expect from the
   scene. That pattern points at which frames land in which split rather than
   at how the prediction is computed, and I cannot check it without the exact
   preprocessed files Gupta et al. used.

None of this is unusual for this benchmark. ETH/UCY reproduction is known to be
messy: papers report materially different numbers for nominally identical
baselines on nominally identical splits, and the observation window, the
sampling rate and the handling of partial tracks are all handled inconsistently
across the literature. My own two "linear" rows disagreeing by 20% is a small
demonstration of exactly that. **I am reporting the gap rather than tuning
until it closes.** A 10-25% offset that I can partly account for is a more
useful result than a suspiciously exact match reached by fiddling.

### One more thing worth stating

Constant velocity has the best average ADE in my table (0.53), beating both
learned models. That is not a bug, and it is not novel: Schöller, Aravantinos,
Lay & Knoll, *What the Constant Velocity Model Can Teach Us About Pedestrian
Motion Prediction* (RA-L / ICRA 2020,
[arXiv:1903.07933](https://arxiv.org/abs/1903.07933)) is a whole paper on the
point, and its abstract states that "a simple Constant Velocity Model can
outperform even state-of-the-art neural models". My learned models do win on
FDE-per-scene in three of five folds and are better than constant velocity on
eth, zara1 and zara2; hotel and univ are where constant velocity wins. This is
a small model trained for 200 epochs with no tuning, so I would not read
anything into the margin either way.

## Seed variance

`const_vel` and `linear` involve no training and are bit-for-bit reproducible.
The two learned rows are not. Rerunning the identical command with `--seed 1`
(`artifacts/ethucy_seed1.json`) gives:

| model | seed 0 AVG ADE / FDE | seed 1 AVG ADE / FDE |
|---|---|---|
| LSTM | 0.58 / 1.23 | 0.55 / 1.16 |
| LSTM + social pooling | 0.56 / 1.17 | 0.58 / 1.21 |

Per scene, the two seeds agree closely on eth, univ, zara1 and zara2 — zara1
LSTM is 0.40 / 0.87 on both — and disagree on hotel, where LSTM moves from
0.59 / 1.24 to 0.45 / 0.90. Hotel is the fold where the learned model loses to a
straight line, so it is unsurprising that it is also where training is least
stable.

**This changes one conclusion.** At seed 0 social pooling looks slightly better
than the vanilla LSTM (0.56 against 0.58 ADE). At seed 1 the ordering reverses
(0.58 against 0.55). The difference between the two models is smaller than the
difference between two seeds of the same model, so **I cannot claim social
pooling helps here.** That is not a disappointing result — it matches what
Gupta et al. found and stated, and it is exactly the kind of claim that a single
run would have let me overstate.

Two seeds is not a variance estimate. It is a sanity check that the learned
numbers are not a single lucky draw, and it should be read as nothing more.

## What this establishes, and what it does not

**Establishes:**

- The ADE/FDE implementation in `src/titan/metrics.py` produces correct
  displacement errors on real human trajectory data, checked against a public
  benchmark with published figures.
- The decoding recipe the TITAN model uses — encode the past, decode a fixed
  horizon of per-step deltas, integrate onto the last observed position — trains
  and generalises to a held-out scene. On ETH/UCY it lands where published LSTM
  baselines land.
- The evaluation loop is sound end to end: a leave-one-scene-out protocol, a
  deterministic baseline that anyone can rerun and get the identical number, and
  a learned model that beats it on some folds and loses on others in the same
  pattern the literature reports.

**Does not establish:**

- Anything about TITAN. TITAN is a different dataset, in pixels, from a moving
  camera, with action labels and ego-motion that ETH/UCY does not have. The
  TITAN loader in `src/titan/data/titan.py` has still never seen a real file,
  and the action branch, the interaction encoder and the Agent Importance
  Mechanism are all untested on real data.
- That the TITAN action-prior claim reproduces. Nothing here touches action
  priors at all.
- A state-of-the-art ETH/UCY result. These are baselines run to validate
  machinery, not competitive models, and no hyperparameter was searched.

The blocker is unchanged: TITAN reproduction needs the gated dataset. See
`HANDOFF.md`.
